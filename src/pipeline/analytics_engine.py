import sqlite3
import datetime
import json

class AnalyticsEngine:
    def __init__(self, db_path="hospital_edge.db"):
        self.db_path = db_path
        print(f"[SYSTEM] Analytics Engine initialized. Connected to: {self.db_path}")

    def _execute_query(self, query, params=()):
        """Helper to safely execute SQL and return dictionary results."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Returns dict-like rows
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_total_scans_processed(self):
        """Computes the total volume of scans handled by the Jetson."""
        query = "SELECT COUNT(*) as total_scans FROM patient_scans"
        result = self._execute_query(query)
        return result[0]['total_scans'] if result else 0

    def get_modality_breakdown(self):
        """Breaks down the processing volume by medical modality."""
        query = """
            SELECT modality, COUNT(*) as count 
            FROM patient_scans 
            GROUP BY modality 
            ORDER BY count DESC
        """
        return self._execute_query(query)

    def get_confidence_trends(self):
        """
        Calculates the average AI confidence per modality.
        This proves to the hospital that the AI remains accurate over time.
        """
        query = """
            SELECT modality, 
                   ROUND(AVG(confidence), 2) as avg_confidence,
                   MIN(confidence) as lowest_confidence
            FROM patient_scans 
            GROUP BY modality
        """
        return self._execute_query(query)

    def get_anomaly_rejections(self):
        """Counts how many times the OOD Bouncer protected the system."""
        query = """
            SELECT COUNT(*) as rejected_count 
            FROM patient_scans 
            WHERE status = 'REJECTED_OOD'
        """
        result = self._execute_query(query)
        return result[0]['rejected_count'] if result else 0

    def generate_full_report(self):
        """Compiles all queries into a single JSON response for Ahmed's API."""
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "hardware_node": "Jetson Nano (Edge)",
            "total_volume": self.get_total_scans_processed(),
            "anomalies_blocked": self.get_anomaly_rejections(),
            "modality_breakdown": self.get_modality_breakdown(),
            "accuracy_trends": self.get_confidence_trends()
        }

# ==========================================
# MOCK DATABASE BUILDER & TEST BED
# ==========================================
def build_mock_db(db_path):
    """Creates a fake database so M1 can test before M3 finishes the real one."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Create Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patient_scans (
                id INTEGER PRIMARY KEY,
                modality TEXT,
                diagnosis TEXT,
                confidence REAL,
                status TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute("DELETE FROM patient_scans") # Clear old test data
        
        # Insert Fake Historical Data
        mock_data = [
            ("mri", "Glioma", 98.2, "PROCESSED"),
            ("mri", "Normal", 99.1, "PROCESSED"),
            ("chest", "Pneumonia", 87.5, "PROCESSED"),
            ("chest", "Normal", 92.0, "PROCESSED"),
            ("retinal", "Diabetic Retinopathy", 81.4, "PROCESSED"),
            ("mri", "Unknown", 45.0, "REJECTED_OOD"), # The Bouncer caught a bad image
            ("skin", "Melanoma", 94.7, "PROCESSED")
        ]
        cursor.executemany('''
            INSERT INTO patient_scans (modality, diagnosis, confidence, status) 
            VALUES (?, ?, ?, ?)
        ''', mock_data)
        conn.commit()

if __name__ == "__main__":
    TEST_DB = "hospital_edge_test.db"
    
    print("============================================================")
    print("[SYSTEM] Building Mock SQLite Database for Testing...")
    build_mock_db(TEST_DB)
    
    print("[SYSTEM] Booting Analytics Engine...")
    print("============================================================\n")
    
    engine = AnalyticsEngine(db_path=TEST_DB)
    
    # Generate the payload that will be sent to the front-end dashboard
    report = engine.generate_full_report()
    
    print("[API PAYLOAD] /analytics endpoint response:")
    print(json.dumps(report, indent=4))
