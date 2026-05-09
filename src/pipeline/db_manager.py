import sqlite3
import datetime

DB_PATH = "local_hospital_records.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patient_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            patient_id TEXT,
            diagnosis TEXT,
            confidence REAL,
            routing_tier TEXT,
            full_report TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_record(patient_id, diagnosis, report, confidence=0.0, routing_tier="N/A", ood_flag=False):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO patient_logs (timestamp, patient_id, diagnosis, confidence, routing_tier, full_report)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, patient_id, diagnosis, confidence, routing_tier, report))
    
    conn.commit()
    conn.close()
    print(f"[SQLITE] Record permanently saved to {DB_PATH}")

# Initialize the database file the first time this script is imported
init_db()
