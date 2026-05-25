"""
active_learning_auto.py — WORKING VERSION for M3

Purpose:
- Poll active_learning_queue for PENDING records
- Rank by uncertainty_trigger
- Auto-call /correct for top-K eligible cases
- Used by /federated_sync later

This version is Windows-safe:
- No Unicode symbols that crash cp1252 console
- Self-test works with the current DatabaseManager
"""

import sys
import time
import json
import logging
import argparse
import threading
import requests
from datetime import datetime, timedelta

try:
    from db_manager import DatabaseManager
except ImportError:
    DatabaseManager = None
    print("[WARNING] db_manager.py not found.")


FLASK_BASE_URL = "http://127.0.0.1:5000"
POLL_INTERVAL_S = 60
TOP_K_AUTO_CORRECT = 3
AUTO_CORRECT_THRESHOLD_H = 24
AUTO_URGENT_UNCERTAINTY = 15.0
AUTO_LABEL = "REVIEW_REQUIRED"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("active_learning_auto.log", mode="a", encoding="utf-8"),
    ]
)

log = logging.getLogger("al_auto")


class ActiveLearningAutomation:
    def __init__(self, db=None, flask_url: str = FLASK_BASE_URL):
        self.db = db if db is not None else (DatabaseManager() if DatabaseManager else None)
        self.flask_url = flask_url
        self.running = False
        self._thread = None
        self.stats = {
            "polls": 0,
            "auto_corrected": 0,
            "errors": 0,
        }

    def run_once(self) -> dict:
        self.stats["polls"] += 1
        log.info(f"[AL_AUTO] Poll #{self.stats['polls']} - checking queue...")

        if self.db is None:
            log.error("[AL_AUTO] No DB connection.")
            self.stats["errors"] += 1
            return {"triggered": 0, "error": "no_db"}

        try:
            pending = self.db.get_pending_queue()
        except Exception as e:
            log.error(f"[AL_AUTO] DB queue fetch failed: {e}")
            self.stats["errors"] += 1
            return {"triggered": 0, "error": str(e)}

        if not pending:
            log.info("[AL_AUTO] Queue empty - nothing to process.")
            return {"triggered": 0, "pending": 0}

        ranked = sorted(
            pending,
            key=lambda r: r.get("uncertainty_trigger", 0) or 0,
            reverse=True
        )

        top_unc = ranked[0].get("uncertainty_trigger", 0) or 0
        log.info(f"[AL_AUTO] {len(pending)} pending items. Top uncertainty: {top_unc:.2f}%")

        triggered = 0

        for record in ranked[:TOP_K_AUTO_CORRECT]:
            if self._should_auto_correct(record):
                ok = self._post_correct(record)
                if ok:
                    triggered += 1
                    self.stats["auto_corrected"] += 1

        log.info(f"[AL_AUTO] Pass complete - triggered {triggered} corrections.")
        return {"triggered": triggered, "pending": len(pending)}

    def _should_auto_correct(self, record: dict) -> bool:
        uncertainty = float(record.get("uncertainty_trigger", 0.0) or 0.0)

        if uncertainty >= AUTO_URGENT_UNCERTAINTY:
            log.info(
                f"[AL_AUTO] record_id={record.get('record_id')} eligible: "
                f"uncertainty={uncertainty:.2f}% >= {AUTO_URGENT_UNCERTAINTY}%"
            )
            return True

        queued_str = record.get("queued_at", "")
        if queued_str:
            try:
                queued_at = datetime.fromisoformat(str(queued_str))
                age_h = (datetime.utcnow() - queued_at).total_seconds() / 3600

                if age_h > AUTO_CORRECT_THRESHOLD_H:
                    log.info(
                        f"[AL_AUTO] record_id={record.get('record_id')} eligible: "
                        f"age={age_h:.1f}h > {AUTO_CORRECT_THRESHOLD_H}h threshold"
                    )
                    return True
            except Exception:
                pass

        return False

    def _post_correct(self, record: dict) -> bool:
        record_id = record["record_id"]

        payload = {
            "record_id": record_id,
            "corrected_label": AUTO_LABEL,
            "notes": (
                f"Auto-escalated by active_learning_auto.py. "
                f"Uncertainty: {record.get('uncertainty_trigger', 0):.2f}%. "
                f"Requires radiologist review."
            ),
        }

        try:
            resp = requests.post(
                f"{self.flask_url}/correct",
                json=payload,
                timeout=30
            )

            if resp.status_code == 200:
                log.info(f"[AL_AUTO] /correct succeeded for record_id={record_id}")
                return True

            log.warning(
                f"[AL_AUTO] /correct returned {resp.status_code} "
                f"for record_id={record_id}: {resp.text[:200]}"
            )
            self.stats["errors"] += 1
            return False

        except requests.exceptions.ConnectionError:
            log.warning(
                f"[AL_AUTO] Flask API not reachable at {self.flask_url}. "
                f"Is flask_api.py running?"
            )
            self.stats["errors"] += 1
            return False

        except Exception as e:
            log.error(f"[AL_AUTO] /correct failed for record_id={record_id}: {e}")
            self.stats["errors"] += 1
            return False

    def start(self):
        if self.running:
            log.warning("[AL_AUTO] Already running.")
            return

        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        log.info(
            f"[AL_AUTO] Started - polling every {POLL_INTERVAL_S}s, "
            f"top-{TOP_K_AUTO_CORRECT} per pass."
        )

    def stop(self):
        self.running = False
        log.info("[AL_AUTO] Stopped.")

    def _loop(self):
        while self.running:
            try:
                self.run_once()
            except Exception as e:
                log.error(f"[AL_AUTO] Unhandled error in poll loop: {e}")
                self.stats["errors"] += 1

            time.sleep(POLL_INTERVAL_S)

    def get_stats(self) -> dict:
        return dict(self.stats)


def trigger_top3_corrections(db=None, flask_url: str = FLASK_BASE_URL) -> dict:
    al = ActiveLearningAutomation(db=db, flask_url=flask_url)
    return al.run_once()


def _run_self_test():
    print("\n" + "=" * 55)
    print("  active_learning_auto.py - M3 Day 7 Self-Test")
    print("=" * 55)

    al = ActiveLearningAutomation(db=None, flask_url="http://127.0.0.1:5000")

    urgent = {
        "record_id": 1,
        "uncertainty_trigger": 16.0,
        "queued_at": "",
    }

    stale = {
        "record_id": 2,
        "uncertainty_trigger": 3.0,
        "queued_at": (datetime.utcnow() - timedelta(hours=25)).isoformat(),
    }

    fresh = {
        "record_id": 3,
        "uncertainty_trigger": 3.0,
        "queued_at": datetime.utcnow().isoformat(),
    }

    assert al._should_auto_correct(urgent), "High uncertainty record should be eligible"
    assert al._should_auto_correct(stale), "Stale record should be eligible"
    assert not al._should_auto_correct(fresh), "Fresh low-uncertainty record should NOT be eligible"

    print("  [OK] _should_auto_correct() logic correct")

    al.db = None
    result = al.run_once()
    assert result.get("error") == "no_db", f"Expected no_db error, got: {result}"

    print("  [OK] run_once() handles missing DB gracefully")

    stats = al.get_stats()
    assert "polls" in stats and "auto_corrected" in stats and "errors" in stats
    print("  [OK] get_stats() returns expected keys")

    al.start()
    time.sleep(0.2)
    al.stop()
    print("  [OK] start()/stop() daemon thread lifecycle OK")

    print("\n  All tests passed.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Active Learning Automation - FYP M3"
    )

    parser.add_argument("--run", action="store_true", help="Start background polling daemon")
    parser.add_argument("--once", action="store_true", help="Run a single pass and exit")
    parser.add_argument("--test", action="store_true", help="Run self-test")
    parser.add_argument("--flask-url", default=FLASK_BASE_URL)

    args = parser.parse_args()

    if args.test:
        _run_self_test()

    elif args.once:
        db = DatabaseManager() if DatabaseManager else None
        al = ActiveLearningAutomation(db=db, flask_url=args.flask_url)
        result = al.run_once()
        print(json.dumps(result, indent=2))

    elif args.run:
        db = DatabaseManager() if DatabaseManager else None
        al = ActiveLearningAutomation(db=db, flask_url=args.flask_url)
        al.start()

        log.info("[AL_AUTO] Running. Ctrl+C to stop.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            al.stop()

    else:
        parser.print_help()
