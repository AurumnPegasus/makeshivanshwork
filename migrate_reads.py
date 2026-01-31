#!/usr/bin/env python3
"""One-time migration to create reads table in Cloud SQL."""

import os
os.environ['USE_CLOUD_SQL'] = 'true'

from app import get_db, execute_query

def migrate():
    conn = get_db()
    cursor = conn.cursor()

    try:
        # Check if table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'reads'
            );
        """)
        exists = cursor.fetchone()[0]

        if exists:
            print("reads table already exists")
        else:
            print("Creating reads table...")
            cursor.execute('''
                CREATE TABLE reads (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    url TEXT,
                    author TEXT,
                    notes TEXT,
                    status TEXT DEFAULT 'unread',
                    added_by TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            print("reads table created successfully!")

    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == '__main__':
    migrate()
