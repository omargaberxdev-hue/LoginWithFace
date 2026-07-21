# metrics.py
import sqlite3, time, threading
from collections import defaultdict

_stage_durations = defaultdict(list)
_lock = threading.Lock()

def record_duration(stage, seconds):
    with _lock:
        _stage_durations[stage].append(seconds)

def setup_metrics(db_path="metrics.db", interval=30):
    """Also a missing connection -- flush_to_db existed before but nothing
    ever called it, so record_duration() was accumulating into a dict
    that was never read."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS stage_metrics (stage TEXT, count INT, avg_s REAL, ts REAL)")
    conn.commit()

    def loop():
        while True:
            time.sleep(interval)
            with _lock:
                for stage, vals in list(_stage_durations.items()):
                    if vals:
                        conn.execute("INSERT INTO stage_metrics VALUES (?,?,?,?)",
                                     (stage, len(vals), sum(vals) / len(vals), time.time()))
                        vals.clear()
                conn.commit()
    threading.Thread(target=loop, daemon=True).start()