import os
import sqlite3
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify, render_template, redirect, session, url_for
from werkzeug.security import generate_password_hash
import hashlib

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Configuration
DATABASE = 'tasks.db'
DOMAIN = os.environ.get('DOMAIN', 'http://localhost:5001')
SMTP_HOST = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
SMTP_USER = os.environ.get('SMTP_USER', '')
SMTP_PASS = os.environ.get('SMTP_PASS', '')
FROM_EMAIL = os.environ.get('FROM_EMAIL', SMTP_USER)


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS magic_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            token TEXT UNIQUE NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            used INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            assigned_by TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    conn.commit()
    conn.close()


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        conn = get_db()
        user = conn.execute('SELECT is_admin FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        conn.close()
        if not user or not user['is_admin']:
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function


def send_magic_link(email, token):
    link = f"{DOMAIN}/auth/{token}"

    msg = MIMEMultipart()
    msg['From'] = FROM_EMAIL
    msg['To'] = email
    msg['Subject'] = 'Your login link for Make Arjo Work'

    body = f'''
    Click here to log in: {link}

    This link expires in 15 minutes.

    If you didn't request this, you can ignore this email.
    '''
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(FROM_EMAIL, email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Email error: {e}")
        # In development, just print the link
        print(f"\n>>> MAGIC LINK: {link}\n", flush=True)
        return True  # Return True in dev mode so testing works


@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').lower().strip()
        if not email:
            return render_template('login.html', error='Email is required')

        # Create magic link
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(minutes=15)

        conn = get_db()
        conn.execute(
            'INSERT INTO magic_links (email, token, expires_at) VALUES (?, ?, ?)',
            (email, token, expires_at)
        )
        conn.commit()
        conn.close()

        send_magic_link(email, token)
        return render_template('login.html', success=True)

    return render_template('login.html')


@app.route('/auth/<token>')
def authenticate(token):
    conn = get_db()
    link = conn.execute(
        'SELECT * FROM magic_links WHERE token = ? AND used = 0 AND expires_at > ?',
        (token, datetime.now())
    ).fetchone()

    if not link:
        conn.close()
        return render_template('login.html', error='Invalid or expired link')

    # Mark link as used
    conn.execute('UPDATE magic_links SET used = 1 WHERE id = ?', (link['id'],))

    # Get or create user
    user = conn.execute('SELECT * FROM users WHERE email = ?', (link['email'],)).fetchone()
    if not user:
        cursor = conn.execute('INSERT INTO users (email) VALUES (?)', (link['email'],))
        user_id = cursor.lastrowid
    else:
        user_id = user['id']

    conn.commit()
    conn.close()

    session['user_id'] = user_id
    session['email'] = link['email']

    return redirect(url_for('dashboard'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    conn.close()
    return render_template('dashboard.html', user=user)


@app.route('/api/tasks', methods=['GET'])
@login_required
def get_tasks():
    status = request.args.get('status')
    conn = get_db()

    if status:
        tasks = conn.execute(
            'SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC',
            (status,)
        ).fetchall()
    else:
        tasks = conn.execute('SELECT * FROM tasks ORDER BY created_at DESC').fetchall()

    conn.close()
    return jsonify([dict(t) for t in tasks])


@app.route('/api/tasks', methods=['POST'])
@login_required
def create_task():
    data = request.json
    title = data.get('title')
    description = data.get('description', '')
    assigned_by = session.get('email', 'Unknown')

    if not title:
        return jsonify({'error': 'Title is required'}), 400

    conn = get_db()
    cursor = conn.execute(
        'INSERT INTO tasks (title, description, assigned_by) VALUES (?, ?, ?)',
        (title, description, assigned_by)
    )
    task_id = cursor.lastrowid
    task = conn.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone()
    conn.commit()
    conn.close()

    return jsonify(dict(task)), 201


@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
@login_required
def update_task(task_id):
    data = request.json

    conn = get_db()
    task = conn.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone()

    if not task:
        conn.close()
        return jsonify({'error': 'Task not found'}), 404

    title = data.get('title', task['title'])
    description = data.get('description', task['description'])
    status = data.get('status', task['status'])

    conn.execute(
        'UPDATE tasks SET title = ?, description = ?, status = ?, updated_at = ? WHERE id = ?',
        (title, description, status, datetime.now(), task_id)
    )
    conn.commit()

    updated_task = conn.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone()
    conn.close()

    return jsonify(dict(updated_task))


@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
@login_required
def delete_task(task_id):
    conn = get_db()
    conn.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
    conn.commit()
    conn.close()
    return '', 204


@app.route('/api/users/make-admin', methods=['POST'])
@admin_required
def make_admin():
    data = request.json
    email = data.get('email', '').lower().strip()

    conn = get_db()
    conn.execute('UPDATE users SET is_admin = 1 WHERE email = ?', (email,))
    conn.commit()
    conn.close()

    return jsonify({'success': True})


# CLI command to make first admin
@app.cli.command('make-admin')
def make_first_admin():
    import sys
    if len(sys.argv) < 2:
        print("Usage: flask make-admin <email>")
        return
    email = sys.argv[1].lower().strip()
    conn = get_db()
    conn.execute('INSERT OR IGNORE INTO users (email, is_admin) VALUES (?, 1)', (email,))
    conn.execute('UPDATE users SET is_admin = 1 WHERE email = ?', (email,))
    conn.commit()
    conn.close()
    print(f"Made {email} an admin")


if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5001)
