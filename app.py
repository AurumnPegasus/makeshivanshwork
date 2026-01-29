import os
import re
import sqlite3
import secrets
import smtplib
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify, render_template, redirect, session, url_for, g
from werkzeug.security import generate_password_hash
import hashlib
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Session configuration - stay logged in for 30 days
app.config['SESSION_COOKIE_SECURE'] = True  # HTTPS only in production
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JS access
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

# Configuration
USE_CLOUD_SQL = os.environ.get('USE_CLOUD_SQL', 'false').lower() == 'true'
CLOUD_SQL_CONNECTION = os.environ.get('CLOUD_SQL_CONNECTION', '')  # e.g., project:region:instance
DB_USER = os.environ.get('DB_USER', 'appuser')
DB_PASS = os.environ.get('DB_PASS', '')
DB_NAME = os.environ.get('DB_NAME', 'makearjowork')
DATABASE = 'tasks.db'  # SQLite fallback for local dev

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL = 'gemini-3-pro-preview'
DOMAIN = os.environ.get('DOMAIN', 'http://localhost:5001')
SMTP_HOST = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
SMTP_USER = os.environ.get('SMTP_USER', '')
SMTP_PASS = os.environ.get('SMTP_PASS', '')
FROM_EMAIL = os.environ.get('FROM_EMAIL', SMTP_USER)

class PgRowWrapper:
    """Wrapper to make psycopg2 rows dict-accessible like sqlite3.Row"""
    def __init__(self, row, columns):
        self._data = dict(zip(columns, row))

    def __getitem__(self, key):
        return self._data[key]

    def keys(self):
        return self._data.keys()


def get_db():
    if USE_CLOUD_SQL:
        import psycopg2
        # Cloud Run provides Unix socket at /cloudsql/CONNECTION_NAME
        conn = psycopg2.connect(
            host=f'/cloudsql/{CLOUD_SQL_CONNECTION}',
            user=DB_USER,
            password=DB_PASS,
            dbname=DB_NAME,
        )
        return conn
    else:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        return conn


def execute_query(conn, query, params=None):
    """Execute query with proper placeholder handling for both SQLite and PostgreSQL"""
    if USE_CLOUD_SQL:
        # Convert ? placeholders to %s for PostgreSQL
        query = query.replace('?', '%s')
    cursor = conn.cursor()
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)
    return cursor


def fetchone(cursor):
    """Fetch one row with proper dict-like access"""
    row = cursor.fetchone()
    if row is None:
        return None
    if USE_CLOUD_SQL:
        columns = [desc[0] for desc in cursor.description]
        return PgRowWrapper(row, columns)
    return row


def fetchall(cursor):
    """Fetch all rows with proper dict-like access"""
    rows = cursor.fetchall()
    if USE_CLOUD_SQL:
        columns = [desc[0] for desc in cursor.description]
        return [PgRowWrapper(row, columns) for row in rows]
    return rows


def init_db():
    conn = get_db()
    if USE_CLOUD_SQL:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                is_admin INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS magic_links (
                id SERIAL PRIMARY KEY,
                email TEXT NOT NULL,
                token TEXT UNIQUE NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                used INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                assigned_by TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                user_email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    else:
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

            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                user_email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        cursor = execute_query(conn, 'SELECT is_admin FROM users WHERE id = ?', (session['user_id'],))
        user = fetchone(cursor)
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
        execute_query(conn,
            'INSERT INTO magic_links (email, token, expires_at) VALUES (?, ?, ?)',
            (email, token, expires_at)
        )
        conn.commit()
        conn.close()

        send_magic_link(email, token)
        return render_template('login.html', success=True, email=email)

    return render_template('login.html')


@app.route('/auth/<token>')
def authenticate(token):
    conn = get_db()
    cursor = execute_query(conn,
        'SELECT * FROM magic_links WHERE token = ? AND used = 0 AND expires_at > ?',
        (token, datetime.now())
    )
    link = fetchone(cursor)

    if not link:
        conn.close()
        return render_template('login.html', error='Invalid or expired link')

    # Mark link as used
    execute_query(conn, 'UPDATE magic_links SET used = 1 WHERE id = ?', (link['id'],))

    # Get or create user
    cursor = execute_query(conn, 'SELECT * FROM users WHERE email = ?', (link['email'],))
    user = fetchone(cursor)
    if not user:
        cursor = execute_query(conn, 'INSERT INTO users (email) VALUES (?) RETURNING id', (link['email'],))
        if USE_CLOUD_SQL:
            row = cursor.fetchone()
            user_id = row[0]
        else:
            user_id = cursor.lastrowid
    else:
        user_id = user['id']

    conn.commit()
    conn.close()

    session.permanent = True  # Stay logged in for 30 days
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
    cursor = execute_query(conn, 'SELECT * FROM users WHERE id = ?', (session['user_id'],))
    user = fetchone(cursor)
    conn.close()
    return render_template('dashboard.html', user=user)


@app.route('/api/tasks', methods=['GET'])
@login_required
def get_tasks():
    status = request.args.get('status')
    conn = get_db()

    if status:
        cursor = execute_query(conn,
            'SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC',
            (status,)
        )
    else:
        cursor = execute_query(conn, 'SELECT * FROM tasks ORDER BY created_at DESC')
    tasks = fetchall(cursor)

    conn.close()
    return jsonify([dict(t._data) if hasattr(t, '_data') else dict(t) for t in tasks])


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
    if USE_CLOUD_SQL:
        cursor = execute_query(conn,
            'INSERT INTO tasks (title, description, assigned_by) VALUES (?, ?, ?) RETURNING id',
            (title, description, assigned_by)
        )
        task_id = cursor.fetchone()[0]
    else:
        cursor = execute_query(conn,
            'INSERT INTO tasks (title, description, assigned_by) VALUES (?, ?, ?)',
            (title, description, assigned_by)
        )
        task_id = cursor.lastrowid
    cursor = execute_query(conn, 'SELECT * FROM tasks WHERE id = ?', (task_id,))
    task = fetchone(cursor)
    conn.commit()
    conn.close()

    return jsonify(dict(task._data) if hasattr(task, '_data') else dict(task)), 201


@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
@login_required
def update_task(task_id):
    data = request.json

    conn = get_db()
    cursor = execute_query(conn, 'SELECT * FROM tasks WHERE id = ?', (task_id,))
    task = fetchone(cursor)

    if not task:
        conn.close()
        return jsonify({'error': 'Task not found'}), 404

    title = data.get('title', task['title'])
    description = data.get('description', task['description'])
    status = data.get('status', task['status'])

    execute_query(conn,
        'UPDATE tasks SET title = ?, description = ?, status = ?, updated_at = ? WHERE id = ?',
        (title, description, status, datetime.now(), task_id)
    )
    conn.commit()

    cursor = execute_query(conn, 'SELECT * FROM tasks WHERE id = ?', (task_id,))
    updated_task = fetchone(cursor)
    conn.close()

    return jsonify(dict(updated_task._data) if hasattr(updated_task, '_data') else dict(updated_task))


@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
@login_required
def delete_task(task_id):
    conn = get_db()
    execute_query(conn, 'DELETE FROM tasks WHERE id = ?', (task_id,))
    conn.commit()
    conn.close()
    return '', 204


@app.route('/api/users/make-admin', methods=['POST'])
@admin_required
def make_admin():
    data = request.json
    email = data.get('email', '').lower().strip()

    conn = get_db()
    execute_query(conn, 'UPDATE users SET is_admin = 1 WHERE email = ?', (email,))
    conn.commit()
    conn.close()

    return jsonify({'success': True})


# Get shared chat history
@app.route('/api/chat/history', methods=['GET'])
@login_required
def get_chat_history():
    conn = get_db()
    cursor = execute_query(conn,
        'SELECT role, content, user_email, created_at FROM chat_history ORDER BY id ASC LIMIT 100'
    )
    messages = fetchall(cursor)
    conn.close()
    return jsonify([dict(m._data) if hasattr(m, '_data') else dict(m) for m in messages])


# Clear chat history
@app.route('/api/chat/clear', methods=['POST'])
@login_required
def clear_chat():
    conn = get_db()
    execute_query(conn, 'DELETE FROM chat_history')
    conn.commit()
    conn.close()
    return jsonify({'success': True})




# Chat with Gemini AI
@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'Message required'}), 400

    conn = get_db()

    # Save user message to shared history
    execute_query(conn,
        'INSERT INTO chat_history (role, content, user_email) VALUES (?, ?, ?)',
        ('user', user_message, session.get('email', 'Unknown'))
    )
    conn.commit()

    # Get all chat history
    cursor = execute_query(conn,
        'SELECT role, content FROM chat_history ORDER BY id ASC LIMIT 50'
    )
    history = fetchall(cursor)

    # Get current tasks for context
    cursor = execute_query(conn, 'SELECT * FROM tasks ORDER BY created_at DESC LIMIT 20')
    tasks = fetchall(cursor)
    conn.close()

    tasks_context = "\n".join([
        f"- [#{t['id']}] [{t['status']}] {t['title']} (by {t['assigned_by']})"
        for t in tasks
    ]) if tasks else "No tasks yet."

    system_prompt = f"""You are the AI assistant for "Make Arjo Work" - a shared task tracker where people assign work to Arjo.

## Current Tasks
{tasks_context}

## Your Capabilities
You can fully manage Arjo's tasks. Use these action blocks:

**Add a task:**
```action
{{"action": "add", "title": "Task title", "description": "Optional details"}}
```

**Update a task (change title, description, or status):**
```action
{{"action": "update", "id": 1, "title": "New title", "status": "in_progress"}}
```
Status options: pending, in_progress, done

**Delete a task:**
```action
{{"action": "delete", "id": 1}}
```

**Mark task done:**
```action
{{"action": "done", "id": 1}}
```

## Guidelines
- Be concise and direct - this is a CLI interface
- Use **bold** and `code` for emphasis
- When users want to modify tasks, do it immediately with an action block
- You can perform multiple actions in one response
- Always confirm what you did after an action"""

    # Build Gemini messages
    gemini_messages = []
    for msg in history:
        role = "user" if msg['role'] == 'user' else "model"
        gemini_messages.append({
            "role": role,
            "parts": [{"text": msg['content']}]
        })

    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": GEMINI_API_KEY},
            json={
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": gemini_messages,
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 2048,
                }
            },
            timeout=60
        )

        if response.status_code != 200:
            return jsonify({"error": f"Gemini API error: {response.text}"}), 500

        result = response.json()
        ai_response = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')

        # Save assistant response to shared history
        conn = get_db()
        execute_query(conn,
            'INSERT INTO chat_history (role, content, user_email) VALUES (?, ?, ?)',
            ('assistant', ai_response, 'assistant')
        )
        conn.commit()

        # Process all action blocks
        actions_performed = []
        action_blocks = re.findall(r'```action\s*(.*?)\s*```', ai_response, re.DOTALL)

        for action_json in action_blocks:
            try:
                action_data = json.loads(action_json.strip())
                action_type = action_data.get('action')

                if action_type == 'add':
                    if USE_CLOUD_SQL:
                        cursor = execute_query(conn,
                            'INSERT INTO tasks (title, description, assigned_by) VALUES (?, ?, ?) RETURNING id',
                            (action_data.get('title'), action_data.get('description', ''), session.get('email', 'Unknown'))
                        )
                        task_id = cursor.fetchone()[0]
                    else:
                        cursor = execute_query(conn,
                            'INSERT INTO tasks (title, description, assigned_by) VALUES (?, ?, ?)',
                            (action_data.get('title'), action_data.get('description', ''), session.get('email', 'Unknown'))
                        )
                        task_id = cursor.lastrowid
                    cursor = execute_query(conn, 'SELECT * FROM tasks WHERE id = ?', (task_id,))
                    task = fetchone(cursor)
                    conn.commit()
                    actions_performed.append({
                        'type': 'added',
                        'task': dict(task._data) if hasattr(task, '_data') else dict(task)
                    })

                elif action_type == 'update':
                    task_id = action_data.get('id')
                    cursor = execute_query(conn, 'SELECT * FROM tasks WHERE id = ?', (task_id,))
                    task = fetchone(cursor)
                    if task:
                        title = action_data.get('title', task['title'])
                        description = action_data.get('description', task['description'])
                        status = action_data.get('status', task['status'])
                        execute_query(conn,
                            'UPDATE tasks SET title = ?, description = ?, status = ?, updated_at = ? WHERE id = ?',
                            (title, description, status, datetime.now(), task_id)
                        )
                        conn.commit()
                        cursor = execute_query(conn, 'SELECT * FROM tasks WHERE id = ?', (task_id,))
                        updated = fetchone(cursor)
                        actions_performed.append({
                            'type': 'updated',
                            'task': dict(updated._data) if hasattr(updated, '_data') else dict(updated)
                        })

                elif action_type == 'done':
                    task_id = action_data.get('id')
                    execute_query(conn,
                        'UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?',
                        ('done', datetime.now(), task_id)
                    )
                    conn.commit()
                    cursor = execute_query(conn, 'SELECT * FROM tasks WHERE id = ?', (task_id,))
                    task = fetchone(cursor)
                    if task:
                        actions_performed.append({
                            'type': 'done',
                            'task': dict(task._data) if hasattr(task, '_data') else dict(task)
                        })

                elif action_type == 'delete':
                    task_id = action_data.get('id')
                    cursor = execute_query(conn, 'SELECT * FROM tasks WHERE id = ?', (task_id,))
                    task = fetchone(cursor)
                    if task:
                        task_dict = dict(task._data) if hasattr(task, '_data') else dict(task)
                        execute_query(conn, 'DELETE FROM tasks WHERE id = ?', (task_id,))
                        conn.commit()
                        actions_performed.append({
                            'type': 'deleted',
                            'task': task_dict
                        })

            except Exception as e:
                print(f"Failed to process action: {e}")

        conn.close()

        return jsonify({
            "response": ai_response,
            "actions": actions_performed
        })

    except requests.Timeout:
        return jsonify({"error": "Request timed out"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# CLI command to make first admin
@app.cli.command('make-admin')
def make_first_admin():
    import sys
    if len(sys.argv) < 2:
        print("Usage: flask make-admin <email>")
        return
    email = sys.argv[1].lower().strip()
    conn = get_db()
    if USE_CLOUD_SQL:
        execute_query(conn, 'INSERT INTO users (email, is_admin) VALUES (?, 1) ON CONFLICT (email) DO UPDATE SET is_admin = 1', (email,))
    else:
        execute_query(conn, 'INSERT OR IGNORE INTO users (email, is_admin) VALUES (?, 1)', (email,))
        execute_query(conn, 'UPDATE users SET is_admin = 1 WHERE email = ?', (email,))
    conn.commit()
    conn.close()
    print(f"Made {email} an admin")


if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5001)
