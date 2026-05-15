"""
active_learning_auto.py  —  FYP: Accelerated Medical Imaging using ML Architectures
Ahmed Rasheed (M3) — Day 7 Deliverable

Automates the Active Learning Queue:
    - Polls active_learning_queue for PENDING records every N seconds
    - Ranks by uncertainty_trigger (highest first)
    - Auto-calls /correct POST endpoint for top-K cases that have been
      in the queue longer than AUTO_CORRECT_THRESHOLD_HOURS (default: 24h)
      OR immediately if confidence < AUTO_URGENT_CONFIDENCE (default: 30%)

This is the "federated sync" side of M3's Day 7 work — M1 runs FedAvg
aggregation server, M3 automates the active learning loop.

Usage:
    python active_learning_auto.py --run          # background daemon
    python active_learning_auto.py --once         # single pass
    python active_learning_auto.py --test         # self-test

Endpoints used:
    POST http://127.0.0.1:5000/correct
    GET  http://127.0.0.1:5000/queue
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
import requests
from datetime import datetime, timedelta

# ── Local imports ──────────────────────────────────────────────────────────────
try:
    from db_manager import DatabaseManager
except ImportError:
    DatabaseManager = None
    print("[WARNING] db_manager.py not found.")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

FLASK_BASE_URL             = "http://127.0.0.1:5000"
POLL_INTERVAL_S            = 60          # Check queue every 60 seconds
TOP_K_AUTO_CORRECT         = 3          # Auto-correct top-3 uncertain cases (Day 7)
AUTO_CORRECT_THRESHOLD_H   = 24         # Hours before auto-correcting stale items
AUTO_URGENT_UNCERTAINTY    = 15.0       # % — auto-escalate if uncertainty this high
AUTO_LABEL                 = "REVIEW_REQUIRED"   # Placeholder label for auto-corrections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("active_learning_auto.log", mode="a"),
    ]
)
log = logging.getLogger("al_auto")


# ══════════════════════════════════════════════════════════════════════════════
#  QUEUE MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class ActiveLearningAutomation:
    """
    Polls the active_learning_queue and auto-triggers /correct for the
    top-K highest-uncertainty PENDING records.

    Day 7 spec: "Automate active learning loop: auto-trigger /correct
                 for top 3 highest-uncertainty cases."
    """

    def __init__(self, db=None, flask_url: str = FLASK_BASE_URL):
        self.db        = db or (DatabaseManager() if DatabaseManager else None)
        self.flask_url = flask_url
        self.running   = False
        self._thread   = None
        self.stats     = {
            "polls":        0,
            "auto_corrected": 0,
            "errors":       0,
        }

    # ── Single-pass logic ─────────────────────────────────────────────────────

    def run_once(self) -> dict:
        """
        One full pass:
          1. Fetch PENDING queue items from DB
          2. Rank by uncertainty_trigger DESC
          3. Apply auto-correction rules to top-K candidates
          4. POST /correct for each eligible record

        Returns summary dict.
        """
        self.stats["polls"] += 1
        log.info(f"[AL_AUTO] Poll #{self.stats['polls']} — checking queue...")

        if self.db is None:
            log.error("[AL_AUTO] No DB connection.")
            return {"triggered": 0, "error": "no_db"}

        try:
            pending = self.db.get_pending_queue()
        except Exception as e:
            log.error(f"[AL_AUTO] DB queue fetch failed: {e}")
            self.stats["errors"] += 1
            return {"triggered": 0, "error": str(e)}

        if not pending:
            log.info("[AL_AUTO] Queue empty — nothing to process.")
            return {"triggered": 0, "pending": 0}

        # Sort by uncertainty DESC — highest uncertainty first
        ranked = sorted(pending, key=lambda r: r.get("uncertainty_trigger", 0),
                        reverse=True)

        log.info(f"[AL_AUTO] {len(pending)} pending items. "
                 f"Top uncertainty: {ranked[0]['uncertainty_trigger']:.2f}%")

        triggered = 0
        for record in ranked[:TOP_K_AUTO_CORRECT]:
            if self._should_auto_correct(record):
                ok = self._post_correct(record)
                if ok:
                    triggered += 1
                    self.stats["auto_corrected"] += 1

        log.info(f"[AL_AUTO] Pass complete — triggered {triggered} corrections.")
        return {"triggered": triggered, "pending": len(pending)}

    def _should_auto_correct(self, record: dict) -> bool:
        """
        Eligibility rules for auto-correction:
          - Uncertainty above AUTO_URGENT_UNCERTAINTY → always eligible
          - Item has been PENDING > AUTO_CORRECT_THRESHOLD_H → eligible
        """
        uncertainty = record.get("uncertainty_trigger", 0.0)
        if uncertainty >= AUTO_URGENT_UNCERTAINTY:
            log.info(f"[AL_AUTO] record_id={record['record_id']} eligible: "
                     f"uncertainty={uncertainty:.2f}% ≥ {AUTO_URGENT_UNCERTAINTY}%")
            return True

        queued_str = record.get("queued_at", "")
        if queued_str:
            try:
                queued_at = datetime.fromisoformat(queued_str)
                age_h = (datetime.utcnow() - queued_at).total_seconds() / 3600
                if age_h > AUTO_CORRECT_THRESHOLD_H:
                    log.info(f"[AL_AUTO] record_id={record['record_id']} eligible: "
                             f"age={age_h:.1f}h > {AUTO_CORRECT_THRESHOLD_H}h threshold")
                    return True
            except Exception:
                pass

        return False

    def _post_correct(self, record: dict) -> bool:
        """POST /correct for a single queue record."""
        record_id = record["record_id"]
        payload   = {
            "record_id":       record_id,
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
                log.info(f"[AL_AUTO] ✓ /correct succeeded for record_id={record_id}")
                return True
            else:
                log.warning(f"[AL_AUTO] /correct returned {resp.status_code} "
                            f"for record_id={record_id}: {resp.text[:200]}")
                self.stats["errors"] += 1
                return False
        except requests.exceptions.ConnectionError:
            log.warning(f"[AL_AUTO] Flask API not reachable at {self.flask_url}. "
                        f"Is flask_api.py running?")
            self.stats["errors"] += 1
            return False
        except Exception as e:
            log.error(f"[AL_AUTO] /correct failed for record_id={record_id}: {e}")
            self.stats["errors"] += 1
            return False

    # ── Daemon loop ───────────────────────────────────────────────────────────

    def start(self):
        """Start background polling thread."""
        if self.running:
            log.warning("[AL_AUTO] Already running.")
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info(f"[AL_AUTO] Started — polling every {POLL_INTERVAL_S}s, "
                 f"top-{TOP_K_AUTO_CORRECT} per pass.")

    def stop(self):
        self.running = False
        log.info("[AL_AUTO] Stopped.")

    def _loop(self):
        while self.running:
            try:
                self.run_once()
            except Exception as e:
                log.error(f"[AL_AUTO] Unhandled error in poll loop: {e}")
            time.sleep(POLL_INTERVAL_S)

    def get_stats(self) -> dict:
        return dict(self.stats)


# ══════════════════════════════════════════════════════════════════════════════
#  /federated_sync helper  (M3 Day 7 — called from flask_api.py)
# ══════════════════════════════════════════════════════════════════════════════

def trigger_top3_corrections(db=None, flask_url: str = FLASK_BASE_URL) -> dict:
    """
    One-shot helper for the /federated_sync endpoint:
    immediately auto-corrects the top-3 highest-uncertainty pending cases.

    Called from flask_api.py POST /federated_sync.
    """
    al = ActiveLearningAutomation(db=db, flask_url=flask_url)
    return al.run_once()


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

def _run_self_test():
    print("\n" + "="*55)
    print("  active_learning_auto.py — M3 Day 7 Self-Test")
    print("="*55)

    # Test without real DB/Flask — just verify logic paths
    al = ActiveLearningAutomation(db=None, flask_url="http://127.0.0.1:5000")

    # Test _should_auto_correct
    urgent = {"record_id": 1, "uncertainty_trigger": 16.0, "queued_at": ""}
    stale  = {"record_id": 2, "uncertainty_trigger": 3.0,
               "queued_at": (datetime.utcnow() - timedelta(hours=25)).isoformat()}
    fresh  = {"record_id": 3, "uncertainty_trigger": 3.0,
               "queued_at": datetime.utcnow().isoformat()}

    assert al._should_auto_correct(urgent), "High uncertainty record should be eligible"
    assert al._should_auto_correct(stale),  "Stale record should be eligible"
    assert not al._should_auto_correct(fresh), "Fresh low-unc record should NOT be eligible"

    print("  ✓ _should_auto_correct() logic correct")

    # Test run_once with no DB
    result = al.run_once()
    assert "error" in result
    print("  ✓ run_once() handles missing DB gracefully")

    # Test stats
    stats = al.get_stats()
    assert "polls" in stats and "auto_corrected" in stats
    print("  ✓ get_stats() returns expected keys")

    # Test start/stop
    al.start()
    time.sleep(0.2)
    al.stop()
    print("  ✓ start()/stop() daemon thread lifecycle OK")

    print("\n  All tests passed.")
    print("="*55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Active Learning Automation — FYP (M3 Day 7)"
    )
    parser.add_argument("--run",   action="store_true",
                        help="Start background polling daemon")
    parser.add_argument("--once",  action="store_true",
                        help="Run a single pass and exit")
    parser.add_argument("--test",  action="store_true",
                        help="Run self-test")
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