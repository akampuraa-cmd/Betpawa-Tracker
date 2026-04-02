"""
Web dashboard for the Betpawa Tracker (Flask).

Replaces the Tkinter GUI: scraper controls, AI train/predict, results table, log.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Deque, List, Optional

from flask import Flask, jsonify, render_template, request

import config
from ai_model import BetpawaAI
from data_manager import DataManager
from scraper import ScraperScheduler, run_historical_backfill, run_scrape_session

MAX_LOG_LINES = 1000
MAX_TABLE_ROWS = 50


class LogBuffer:
    """Thread-safe ring buffer for dashboard log lines."""

    def __init__(self, maxlen: int = MAX_LOG_LINES) -> None:
        self._lines: Deque[str] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, msg: str) -> None:
        with self._lock:
            self._lines.append(msg)

    def get_lines(self, limit: int = 200) -> List[str]:
        with self._lock:
            if limit <= 0:
                return list(self._lines)
            return list(self._lines)[-limit:]


class DashboardState:
    def __init__(self) -> None:
        self.db = DataManager()
        self.ai = BetpawaAI()
        self.scheduler: Optional[ScraperScheduler] = None
        self.log = LogBuffer(MAX_LOG_LINES)
        self.background_threads: List[threading.Thread] = []
        self.backfill_running = False
        self.backfill_progress = 0.0
        self.backfill_status = "History: Idle"
        self.training = False
        self.last_prediction = ""
        self._thread_lock = threading.Lock()

    def append_log(self, msg: str) -> None:
        self.log.append(msg)


state = DashboardState()


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")

    @app.route("/")
    def index() -> str:
        return render_template("dashboard.html")

    @app.route("/api/status")
    def api_status() -> Any:
        sched = state.scheduler
        running = sched is not None and sched.is_running()
        secs = sched.seconds_until_next() if sched and running else 0
        if running and secs == 0:
            scraper_label = "Scraping"
        elif running:
            scraper_label = "Scheduled"
        elif sched is not None:
            scraper_label = "Stopped"
        else:
            scraper_label = "Idle"

        n = state.db.count_results()
        source_counts = state.db.count_results_by_source()
        summary = state.ai.summary()
        ck = state.ai.checkpoint_info() or {}

        return jsonify(
            {
                "scheduler_running": running,
                "countdown_seconds": secs,
                "scraper_label": scraper_label,
                "total_results": n,
                "source_live": source_counts.get("live", 0),
                "source_historical": source_counts.get("historical", 0),
                "backfill_running": state.backfill_running,
                "backfill_progress": state.backfill_progress,
                "backfill_status": state.backfill_status,
                "training": state.training,
                "checkpoint_exists": bool(ck.get("exists")),
                "checkpoint_size_bytes": ck.get("size_bytes"),
                "ga_generation": summary["ga_generation"],
                "ga_best_fitness": summary["ga_best_fitness"],
                "ql_steps": summary["ql_steps"],
                "ql_accuracy": summary["ql_accuracy"],
                "ql_epsilon": summary["ql_epsilon"],
                "last_prediction": state.last_prediction,
            }
        )

    @app.route("/api/results")
    def api_results() -> Any:
        limit = min(MAX_TABLE_ROWS, max(1, request.args.get("limit", 50, type=int)))
        rows = state.db.get_recent_results(limit)
        out = []
        for row in rows:
            ts_short = row["timestamp"][:16].replace("T", " ")
            outcome_label = config.RESULT_LABELS.get(row["outcome"], "?")
            out.append(
                {
                    "id": row["id"],
                    "timestamp": ts_short,
                    "source": row["source"],
                    "team_home": row["team_home"],
                    "team_away": row["team_away"],
                    "ft_home": row["ft_home"],
                    "ft_away": row["ft_away"],
                    "outcome": row["outcome"],
                    "outcome_label": outcome_label,
                    "raw_result": row["raw_result"][:80],
                }
            )
        return jsonify({"results": out})

    @app.route("/api/logs")
    def api_logs() -> Any:
        n = request.args.get("limit", 200, type=int) or 200
        return jsonify({"lines": state.log.get_lines(min(n, MAX_LOG_LINES))})

    def _safe_background(target) -> None:
        def _wrap() -> None:
            try:
                target()
            except Exception as exc:
                state.append_log(f"❌ Background task error: {exc}")

        t = threading.Thread(target=_wrap, daemon=True)
        with state._thread_lock:
            state.background_threads.append(t)
        t.start()

    @app.route("/api/scheduler/start", methods=["POST"])
    def api_scheduler_start() -> Any:
        if state.scheduler and state.scheduler.is_running():
            state.append_log("Scheduler already running.")
            return jsonify({"ok": True, "message": "Already running"})
        state.scheduler = ScraperScheduler(state.db, log_callback=state.append_log)
        state.scheduler.start()
        state.append_log("⚡ Scheduler started – next scrape at clock boundary")
        return jsonify({"ok": True})

    @app.route("/api/scheduler/stop", methods=["POST"])
    def api_scheduler_stop() -> Any:
        if state.scheduler:
            state.scheduler.stop()
        state.append_log("Scheduler stopped.")
        return jsonify({"ok": True})

    @app.route("/api/scrape", methods=["POST"])
    def api_scrape() -> Any:
        state.append_log("🔄 Running immediate scrape session …")

        def _run() -> None:
            run_scrape_session(state.db, log_callback=state.append_log)

        _safe_background(_run)
        return jsonify({"ok": True})

    @app.route("/api/backfill", methods=["POST"])
    def api_backfill() -> Any:
        if state.backfill_running:
            return jsonify({"ok": False, "message": "Backfill already running"}), 409

        state.backfill_running = True
        state.backfill_progress = 0.0
        state.backfill_status = "History: Starting …"
        state.append_log("🗂 Starting historical backfill …")

        def _progress(completed: int, total: int, status: str) -> None:
            state.backfill_status = f"History: {status}"
            if total > 0:
                state.backfill_progress = (completed / total) * 100.0
            else:
                state.backfill_progress = 0.0

        def _run() -> None:
            try:
                inserted = run_historical_backfill(
                    state.db,
                    log_callback=state.append_log,
                    progress_callback=_progress,
                )
                state.backfill_progress = 100.0
                state.backfill_status = f"History: Done ({inserted} imported)"
            finally:
                state.backfill_running = False

        _safe_background(_run)
        return jsonify({"ok": True})

    @app.route("/api/train", methods=["POST"])
    def api_train() -> Any:
        if state.training:
            return jsonify({"ok": False, "message": "Training already in progress"}), 409

        def _run() -> None:
            state.training = True
            try:
                outcomes = state.db.get_outcomes()
                goals = state.db.get_goals_series()
                n = len(outcomes)
                if n < 3:
                    state.append_log(
                        f"⚠  Need at least 3 results to train (have {n}). Run the scraper first."
                    )
                    return
                state.append_log(f"🧬 Training GA on {n} results …")

                def ga_cb(*, generation: int, best_fitness: float) -> None:
                    if generation % 10 == 0:
                        state.append_log(
                            f"  GA gen {generation:>4} | best fitness {best_fitness:.3f}"
                        )

                def ql_cb(*, step: int, reward: float, epsilon: float, q_table_size: int) -> None:
                    if step % 20 == 0:
                        state.append_log(
                            f"  QL step {step:>4} | reward {reward:+.0f} | ε={epsilon:.3f}"
                        )

                state.ai.train(outcomes, goals, ga_callback=ga_cb, ql_callback=ql_cb)
                summary = state.ai.summary()
                state.append_log(
                    f"✅ Training complete | GA gen {summary['ga_generation']} "
                    f"fit={summary['ga_best_fitness']} | "
                    f"QL acc={summary['ql_accuracy']:.2%}"
                )
            finally:
                state.training = False

        _safe_background(_run)
        return jsonify({"ok": True})

    @app.route("/api/predict", methods=["POST"])
    def api_predict() -> Any:
        outcomes = state.db.get_outcomes()
        goals = state.db.get_goals_series()
        if len(outcomes) < 2:
            state.append_log("⚠  Not enough data to predict yet.")
            return jsonify({"ok": False, "message": "Not enough data"})

        upcoming = state.db.get_upcoming_matches(1)
        odds = None
        if upcoming:
            m = upcoming[0]
            odds = (m["home_odds"], m["draw_odds"], m["away_odds"])

        result = state.ai.predict(outcomes, goals, odds)
        ga_label = result.get("ga_label", "N/A")
        ga_conf = result.get("ga_confidence", 0.0)
        ql_label = result.get("ql_label", "N/A")
        lstm_label = result.get("lstm_label", "N/A")
        lstm_conf = result.get("lstm_confidence", 0.0)
        consensus = result.get("consensus_label", "N/A")

        label = (
            f"GA: {ga_label} ({ga_conf:.0%}) | "
            f"QL: {ql_label} | "
            f"LSTM: {lstm_label} ({lstm_conf:.0%}) | "
            f"Consensus: {consensus}"
        )
        state.last_prediction = f"{consensus}"
        state.append_log(f"Prediction → {label}")
        return jsonify({"ok": True, "detail": label, "consensus": str(consensus)})

    @app.route("/api/save", methods=["POST"])
    def api_save() -> Any:
        if state.ai.save():
            info = state.ai.checkpoint_info() or {}
            state.append_log(f"💾 Saved AI checkpoint to {info.get('path')}")
            return jsonify({"ok": True})
        state.append_log("⚠  No checkpoint path configured.")
        return jsonify({"ok": False})

    @app.route("/api/reload", methods=["POST"])
    def api_reload() -> Any:
        if state.ai.load():
            summary = state.ai.summary()
            state.append_log(
                f"↻ Reloaded AI checkpoint | GA gen {summary['ga_generation']} | "
                f"QL steps {summary['ql_steps']}"
            )
            return jsonify({"ok": True})
        state.append_log("⚠  No checkpoint found or load failed.")
        return jsonify({"ok": False})

    @app.route("/api/clear-results", methods=["POST"])
    def api_clear_results() -> Any:
        state.db.clear_all_results()
        state.append_log("🗑 All match results have been deleted.")
        return jsonify({"ok": True})

    return app


def run(host: Optional[str] = None, port: Optional[int] = None) -> None:
    h = host if host is not None else getattr(config, "WEB_HOST", "127.0.0.1")
    p = port if port is not None else int(getattr(config, "WEB_PORT", 5050))
    app = create_app()
    print(f"\n  Betpawa dashboard: http://{h}:{p}/\n")
    app.run(host=h, port=p, threaded=True, use_reloader=False)


if __name__ == "__main__":
    run()
