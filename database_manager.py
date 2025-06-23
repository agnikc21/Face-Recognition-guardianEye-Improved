import sqlite3
import os
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path="data/FaceBase.db"):
        self.db_path = db_path
        self._ensure_data_dir()
        self._init_database()
    
    def _ensure_data_dir(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS People (
                    ID INTEGER PRIMARY KEY,
                    Name TEXT NOT NULL,
                    Age INTEGER,
                    Gender TEXT,
                    Notes TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def get_next_id(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(ID) FROM People")
            result = cursor.fetchone()[0]
            return 1 if result is None else result + 1
    
    def insert_person(self, person_id, name, age, gender, notes=""):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO People (ID, Name, Age, Gender, Notes)
                VALUES (?, ?, ?, ?, ?)
            """, (person_id, name, age, gender, notes))
            conn.commit()
    
    def get_person(self, person_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM People WHERE ID = ?", (person_id,))
            return cursor.fetchone()
    
    def get_all_people(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM People ORDER BY ID")
            return cursor.fetchall()
    
    def delete_person(self, person_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM People WHERE ID = ?", (person_id,))
            conn.commit()
