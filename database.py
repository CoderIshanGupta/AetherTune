import sqlite3
from datetime import datetime

DB_NAME = "aethertune.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            activity TEXT,
            liked BOOLEAN,
            timestamp DATETIME
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_tolerance (
            age INTEGER,
            activity TEXT,
            adjustment REAL,
            last_updated DATETIME,
            PRIMARY KEY (age, activity)
        )
    """)

    conn.commit()
    conn.close()