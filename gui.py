"""
Tkinter GUI for the Betpawa Tracker application.

Layout
------
┌────────────────────────────────────────────────────────────┐
│ Title bar                                                  │
├──────────────────────────┬─────────────────────────────────┤
│ Scraper Panel            │ AI Panel                        │
│  – Status / countdown    │  – GA stats                     │
│  – [Start] [Stop] [Now]  │  – QL stats                     │
│                          │  – [Train AI] [Predict]         │
├──────────────────────────┴─────────────────────────────────┤
│ Results Table (scrollable)                                  │
├────────────────────────────────────────────────────────────┤
│ Log window (scrollable text)                               │
└────────────────────────────────────────────────────────────┘
"""

import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from datetime import datetime
from typing import Optional

import config
from data_manager import DataManager
from scraper import ScraperScheduler, run_scrape_session, run_historical_backfill
from ai_model import BetpawaAI

# ── Colour palette ─────────────────────────────────────────────────────────────
BG_DARK = "#1e1e2e"
BG_MID = "#2a2a3c"
BG_LIGHT = "#313145"
FG_WHITE = "#cdd6f4"
FG_GREEN = "#a6e3a1"
FG_RED = "#f38ba8"
FG_YELLOW = "#f9e2af"
FG_BLUE = "#89b4fa"
FG_PURPLE = "#cba6f7"
FG_GREY = "#6c7086"
FONT_MAIN = ("Consolas", 10)
FONT_BOLD = ("Consolas", 10, "bold")
FONT_TITLE = ("Consolas", 14, "bold")
FONT_SMALL = ("Consolas", 9)


MAX_LOG_LINES = 1000
MAX_TABLE_ROWS = 100


class BetpawaTrackerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.db = DataManager()
        self.ai = BetpawaAI()
        self.scheduler: Optional[ScraperScheduler] = None
        self._background_threads: list[threading.Thread] = []
        self._backfill_running = False

        root.title("Betpawa MUN Tracker — AI Edition")
        root.configure(bg=BG_DARK)
        root.geometry("1100x780")
        root.minsize(900, 640)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._refresh_loop()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Title ────────────────────────────────────────────────────────────
        title = tk.Label(
            self.root,
            text="⚽  Betpawa MUN Virtual-Sports Tracker",
            font=FONT_TITLE,
            bg=BG_DARK,
            fg=FG_PURPLE,
            pady=8,
        )
        title.pack(fill=tk.X)

        # ── Top row: Scraper | AI ─────────────────────────────────────────────
        top_frame = tk.Frame(self.root, bg=BG_DARK)
        top_frame.pack(fill=tk.X, padx=10, pady=4)

        self._build_scraper_panel(top_frame)
        self._build_ai_panel(top_frame)

        # ── Results table ─────────────────────────────────────────────────────
        self._build_results_table()

        # ── Log panel ─────────────────────────────────────────────────────────
        self._build_log_panel()

    # ── Scraper panel ─────────────────────────────────────────────────────────

    def _build_scraper_panel(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(
            parent,
            text=" 🔄 Scraper ",
            bg=BG_MID,
            fg=FG_BLUE,
            font=FONT_BOLD,
        )
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.scraper_status_var = tk.StringVar(value="Idle")
        tk.Label(
            frame, textvariable=self.scraper_status_var,
            bg=BG_MID, fg=FG_GREEN, font=FONT_BOLD,
        ).pack(anchor="w", padx=8, pady=(6, 2))

        self.countdown_var = tk.StringVar(value="")
        tk.Label(
            frame, textvariable=self.countdown_var,
            bg=BG_MID, fg=FG_YELLOW, font=FONT_SMALL,
        ).pack(anchor="w", padx=8)

        self.total_results_var = tk.StringVar(value="Stored results: 0")
        tk.Label(
            frame, textvariable=self.total_results_var,
            bg=BG_MID, fg=FG_WHITE, font=FONT_SMALL,
        ).pack(anchor="w", padx=8, pady=(2, 6))

        self.backfill_status_var = tk.StringVar(value="History: Idle")
        tk.Label(
            frame, textvariable=self.backfill_status_var,
            bg=BG_MID, fg=FG_PURPLE, font=FONT_SMALL,
        ).pack(anchor="w", padx=8, pady=(0, 4))

        self.checkpoint_status_var = tk.StringVar(value="Checkpoint: Unknown")
        tk.Label(
            frame, textvariable=self.checkpoint_status_var,
            bg=BG_MID, fg=FG_BLUE, font=FONT_SMALL,
        ).pack(anchor="w", padx=8, pady=(0, 4))

        self.backfill_progress_var = tk.DoubleVar(value=0.0)
        self.backfill_progress = ttk.Progressbar(
            frame,
            orient="horizontal",
            length=220,
            mode="determinate",
            variable=self.backfill_progress_var,
            maximum=100,
        )
        self.backfill_progress.pack(anchor="w", padx=8, pady=(0, 6))

        btn_row = tk.Frame(frame, bg=BG_MID)
        btn_row.pack(pady=6)
        self._btn(btn_row, "▶ Start", FG_GREEN, self._start_scheduler).pack(side=tk.LEFT, padx=4)
        self._btn(btn_row, "■ Stop", FG_RED, self._stop_scheduler).pack(side=tk.LEFT, padx=4)
        self._btn(btn_row, "⚡ Scrape Now", FG_YELLOW, self._scrape_now).pack(side=tk.LEFT, padx=4)
        self.backfill_btn = self._btn(btn_row, "🗂 Backfill History", FG_PURPLE, self._backfill_history)
        self.backfill_btn.pack(side=tk.LEFT, padx=4)

    # ── AI panel ──────────────────────────────────────────────────────────────

    def _build_ai_panel(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(
            parent,
            text=" 🧠 AI (GA + Q-Learning) ",
            bg=BG_MID,
            fg=FG_PURPLE,
            font=FONT_BOLD,
        )
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # GA stats
        ga_frame = tk.LabelFrame(frame, text=" Genetic Algorithm ", bg=BG_MID, fg=FG_BLUE, font=FONT_SMALL)
        ga_frame.pack(fill=tk.X, padx=6, pady=4)

        self.ga_gen_var = tk.StringVar(value="Generation: 0")
        self.ga_fit_var = tk.StringVar(value="Best Fitness: —")
        tk.Label(ga_frame, textvariable=self.ga_gen_var, bg=BG_MID, fg=FG_WHITE, font=FONT_SMALL).pack(anchor="w", padx=6)
        tk.Label(ga_frame, textvariable=self.ga_fit_var, bg=BG_MID, fg=FG_GREEN, font=FONT_SMALL).pack(anchor="w", padx=6, pady=(0, 4))

        # QL stats
        ql_frame = tk.LabelFrame(frame, text=" Q-Learning ", bg=BG_MID, fg=FG_YELLOW, font=FONT_SMALL)
        ql_frame.pack(fill=tk.X, padx=6, pady=4)

        self.ql_steps_var = tk.StringVar(value="Steps: 0")
        self.ql_acc_var = tk.StringVar(value="Accuracy: —")
        self.ql_eps_var = tk.StringVar(value="ε = 0.20")
        tk.Label(ql_frame, textvariable=self.ql_steps_var, bg=BG_MID, fg=FG_WHITE, font=FONT_SMALL).pack(anchor="w", padx=6)
        tk.Label(ql_frame, textvariable=self.ql_acc_var, bg=BG_MID, fg=FG_GREEN, font=FONT_SMALL).pack(anchor="w", padx=6)
        tk.Label(ql_frame, textvariable=self.ql_eps_var, bg=BG_MID, fg=FG_YELLOW, font=FONT_SMALL).pack(anchor="w", padx=6, pady=(0, 4))

        # Prediction display
        self.prediction_var = tk.StringVar(value="Prediction: —")
        tk.Label(
            frame, textvariable=self.prediction_var,
            bg=BG_MID, fg=FG_PURPLE, font=FONT_BOLD,
        ).pack(pady=4)

        btn_row = tk.Frame(frame, bg=BG_MID)
        btn_row.pack(pady=6)
        self._btn(btn_row, "🧬 Train AI", FG_BLUE, self._train_ai).pack(side=tk.LEFT, padx=4)
        self._btn(btn_row, "🔮 Predict", FG_PURPLE, self._predict).pack(side=tk.LEFT, padx=4)
        self._btn(btn_row, "💾 Save", FG_GREEN, self._save_ai).pack(side=tk.LEFT, padx=4)
        self._btn(btn_row, "↻ Reload", FG_YELLOW, self._reload_ai).pack(side=tk.LEFT, padx=4)

    # ── Results table ─────────────────────────────────────────────────────────

    def _build_results_table(self) -> None:
        frame = tk.LabelFrame(
            self.root,
            text=" 📊 Recent Results ",
            bg=BG_DARK,
            fg=FG_GREEN,
            font=FONT_BOLD,
        )
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # Header frame for table tools (like the delete button)
        tool_frame = tk.Frame(frame, bg=BG_DARK)
        tool_frame.pack(fill=tk.X, padx=4, pady=2)
        self._btn(tool_frame, "🗑 Clear Results", FG_RED, self._clear_results).pack(side=tk.RIGHT)

        columns = ("id", "timestamp", "source", "home", "score", "away", "outcome", "raw")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=8)

        # Style the treeview
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background=BG_LIGHT,
            foreground=FG_WHITE,
            fieldbackground=BG_LIGHT,
            rowheight=22,
            font=FONT_SMALL,
        )
        style.configure("Treeview.Heading", background=BG_MID, foreground=FG_BLUE, font=FONT_BOLD)
        style.map("Treeview", background=[("selected", BG_MID)])

        headers = {
            "id": ("#", 40),
            "timestamp": ("Timestamp", 140),
            "source": ("Source", 80),
            "home": ("Home", 80),
            "score": ("FT Score", 90),
            "away": ("Away", 80),
            "outcome": ("Result", 70),
            "raw": ("Raw", 200),
        }
        for col, (text, width) in headers.items():
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor="center")

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Row colour tags
        self.tree.tag_configure("win", foreground=FG_GREEN)
        self.tree.tag_configure("draw", foreground=FG_YELLOW)
        self.tree.tag_configure("loss", foreground=FG_RED)
        self.tree.tag_configure("historical", background="#2b233a")
        self.tree.tag_configure("live", background=BG_LIGHT)

    # ── Log panel ─────────────────────────────────────────────────────────────

    def _build_log_panel(self) -> None:
        frame = tk.LabelFrame(
            self.root,
            text=" 📋 Backend Log ",
            bg=BG_DARK,
            fg=FG_GREY,
            font=FONT_BOLD,
        )
        frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.log_text = scrolledtext.ScrolledText(
            frame,
            height=6,
            bg=BG_LIGHT,
            fg=FG_WHITE,
            font=FONT_SMALL,
            state=tk.DISABLED,
            wrap=tk.WORD,
        )
        self.log_text.pack(fill=tk.X, padx=4, pady=4)

    # ── Widget helper ─────────────────────────────────────────────────────────

    @staticmethod
    def _btn(parent, text: str, fg: str, cmd) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=cmd,
            bg=BG_LIGHT,
            fg=fg,
            font=FONT_SMALL,
            relief=tk.FLAT,
            cursor="hand2",
            padx=8,
            pady=4,
            activebackground=BG_MID,
            activeforeground=fg,
        )

    # ── Logging helper ────────────────────────────────────────────────────────

    def _append_log(self, msg: str) -> None:
        """Append a message to the log widget (thread-safe), trimming to MAX_LOG_LINES."""
        def _do():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, msg + "\n")
            # Trim log to MAX_LOG_LINES
            line_count = int(self.log_text.index("end-1c").split(".")[0])
            if line_count > MAX_LOG_LINES:
                self.log_text.delete("1.0", f"{line_count - MAX_LOG_LINES}.0")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)
        self.root.after(0, _do)

    # ── Scraper controls ──────────────────────────────────────────────────────

    def _start_scheduler(self) -> None:
        if self.scheduler and self.scheduler.is_running():
            self._append_log("Scheduler already running.")
            return
        self.scheduler = ScraperScheduler(self.db, log_callback=self._append_log)
        self.scheduler.start()
        self.scraper_status_var.set("Running ▶")
        self._append_log("⚡ Scheduler started – next scrape in 0 s (running now)")

    def _stop_scheduler(self) -> None:
        if self.scheduler:
            self.scheduler.stop()
        self.scraper_status_var.set("Stopped ■")
        self.countdown_var.set("")
        self._append_log("Scheduler stopped.")

    def _scrape_now(self) -> None:
        self._append_log("🔄 Running immediate scrape session …")
        t = threading.Thread(
            target=self._safe_run,
            args=(lambda: run_scrape_session(self.db, self._append_log),),
            daemon=True,
        )
        self._background_threads.append(t)
        t.start()

    def _backfill_history(self) -> None:
        if self._backfill_running:
            self._append_log("Historical backfill is already running.")
            return

        if not messagebox.askyesno(
            "Backfill History",
            "Import historical MUN results from Betpawa into the database?\nThis may take a few minutes.",
        ):
            return

        self._backfill_running = True
        self.backfill_btn.configure(state=tk.DISABLED)
        self.backfill_progress_var.set(0)
        self.backfill_status_var.set("History: Starting …")

        def _progress(completed: int, total: int, status: str) -> None:
            def _update() -> None:
                self.backfill_status_var.set(f"History: {status}")
                if total > 0:
                    self.backfill_progress_var.set((completed / total) * 100)
                else:
                    self.backfill_progress_var.set(0)
            self.root.after(0, _update)

        def _run() -> None:
            inserted = run_historical_backfill(
                self.db,
                log_callback=self._append_log,
                progress_callback=_progress,
            )

            def _finish() -> None:
                self._backfill_running = False
                self.backfill_btn.configure(state=tk.NORMAL)
                self.backfill_progress_var.set(100)
                self.backfill_status_var.set(f"History: Done ({inserted} imported)")

            self.root.after(0, _finish)

        self._append_log("🗂 Starting historical backfill …")
        t = threading.Thread(target=self._safe_run, args=(_run,), daemon=True)
        self._background_threads.append(t)
        t.start()

    # ── AI controls ───────────────────────────────────────────────────────────

    def _train_ai(self) -> None:
        def _run():
            outcomes = self.db.get_outcomes()
            goals = self.db.get_goals_series()
            n = len(outcomes)
            if n < 3:
                self._append_log(
                    f"⚠  Need at least 3 results to train (have {n}). "
                    "Run the scraper first."
                )
                return

            self._append_log(f"🧬 Training GA on {n} results …")

            def ga_cb(generation, best_fitness):
                if generation % 10 == 0:
                    self._append_log(
                        f"  GA gen {generation:>4} | best fitness {best_fitness:.3f}"
                    )
                self.root.after(0, lambda: (
                    self.ga_gen_var.set(f"Generation: {generation}"),
                    self.ga_fit_var.set(f"Best Fitness: {best_fitness:.4f}"),
                ))

            def ql_cb(step, reward, epsilon, q_table_size):
                if step % 20 == 0:
                    self._append_log(
                        f"  QL step {step:>4} | reward {reward:+.0f} | ε={epsilon:.3f}"
                    )
                self.root.after(0, lambda: (
                    self.ql_steps_var.set(f"Steps: {step}"),
                    self.ql_acc_var.set(f"Accuracy: {self.ai.ql.accuracy:.2%}"),
                    self.ql_eps_var.set(f"ε = {epsilon:.3f}"),
                ))

            self.ai.train(outcomes, goals, ga_callback=ga_cb, ql_callback=ql_cb)
            summary = self.ai.summary()
            self._append_log(
                f"✅ Training complete | GA gen {summary['ga_generation']} "
                f"fit={summary['ga_best_fitness']} | "
                f"QL acc={summary['ql_accuracy']:.2%}"
            )

        t = threading.Thread(target=self._safe_run, args=(_run,), daemon=True)
        self._background_threads.append(t)
        t.start()

    def _predict(self) -> None:
        outcomes = self.db.get_outcomes()
        goals = self.db.get_goals_series()
        if len(outcomes) < 2:
            self._append_log("⚠  Not enough data to predict yet.")
            return

        # Fetch latest upcoming odds
        upcoming = self.db.get_upcoming_matches(1)
        odds = None
        if upcoming:
            match = upcoming[0]
            odds = (match['home_odds'], match['draw_odds'], match['away_odds'])

        result = self.ai.predict(outcomes, goals, odds)
        ga_label = result.get('ga_label', 'N/A')
        ga_conf = result.get('ga_confidence', 0.0)
        ql_label = result.get('ql_label', 'N/A')
        lstm_label = result.get('lstm_label', 'N/A')
        lstm_conf = result.get('lstm_confidence', 0.0)
        consensus = result.get('consensus_label', 'N/A')

        label = (
            f"GA: {ga_label} ({ga_conf:.0%}) | "
            f"QL: {ql_label} | "
            f"LSTM: {lstm_label} ({lstm_conf:.0%}) | "
            f"Consensus: {consensus}"
        )
        self.prediction_var.set(f"🔮 {consensus}")
        self._append_log(f"Prediction → {label}")

    # ── Thread safety helpers ─────────────────────────────────────────────────

    def _safe_run(self, func) -> None:
        """Wrapper that catches exceptions in background threads and logs them."""
        try:
            func()
        except Exception as exc:
            self.root.after(0, self._reset_backfill_state)
            self._append_log(f"❌ Background task error: {exc}")

    def _reset_backfill_state(self) -> None:
        self._backfill_running = False
        if hasattr(self, "backfill_btn"):
            self.backfill_btn.configure(state=tk.NORMAL)
        if hasattr(self, "backfill_status_var"):
            self.backfill_status_var.set("History: Idle")

    def _on_close(self) -> None:
        """Gracefully shut down scheduler and threads before closing."""
        if self.scheduler:
            self.scheduler.stop()
        for t in self._background_threads:
            t.join(timeout=5)
        self.root.destroy()

    # ── Refresh loop ──────────────────────────────────────────────────────────

    def _refresh_loop(self) -> None:
        """Update the table, stats, and countdown every second."""
        try:
            self._refresh_table()
            self._refresh_stats()
            self._refresh_countdown()
        except Exception:
            pass
        self.root.after(1000, self._refresh_loop)

    def _clear_results(self) -> None:
        """Clear all stored results after confirmation."""
        if messagebox.askyesno(
            "Clear Data",
            "Are you sure you want to delete all stored match results?\nThis cannot be undone."
        ):
            self.db.clear_all_results()
            # Clear GUI table
            for iid in self.tree.get_children():
                self.tree.delete(iid)
            self._append_log("🗑 All match results have been deleted.")

    def _refresh_table(self) -> None:
        rows = self.db.get_recent_results(50)
        existing_ids = {
            int(self.tree.item(iid, "values")[0])
            for iid in self.tree.get_children()
        }
        for row in rows:
            if row["id"] in existing_ids:
                continue
            outcome_label = config.RESULT_LABELS.get(row["outcome"], "?")
            tag = {
                config.RESULT_WIN: "win",
                config.RESULT_DRAW: "draw",
                config.RESULT_LOSS: "loss",
            }.get(row["outcome"], "")
            source_tag = row["source"] if row["source"] in ("live", "historical") else "live"
            score = f"{row['ft_home']} - {row['ft_away']}"
            ts_short = row["timestamp"][:16].replace("T", " ")
            self.tree.insert(
                "", 0,
                values=(
                    row["id"],
                    ts_short,
                    row["source"].title(),
                    row["team_home"],
                    score,
                    row["team_away"],
                    outcome_label,
                    row["raw_result"][:40],
                ),
                tags=(tag, source_tag),
            )

        # Prune table to MAX_TABLE_ROWS
        children = self.tree.get_children()
        if len(children) > MAX_TABLE_ROWS:
            for iid in children[MAX_TABLE_ROWS:]:
                self.tree.delete(iid)

    def _refresh_stats(self) -> None:
        n = self.db.count_results()
        source_counts = self.db.count_results_by_source()
        self.total_results_var.set(
            f"Stored results: {n}  |  Live: {source_counts['live']}  |  Historical: {source_counts['historical']}"
        )

        summary = self.ai.summary()
        self.ga_gen_var.set(f"Generation: {summary['ga_generation']}")
        if summary["ga_best_fitness"] > 0:
            self.ga_fit_var.set(f"Best Fitness: {summary['ga_best_fitness']:.4f}")
        self.ql_steps_var.set(f"Steps: {summary['ql_steps']}")
        if summary["ql_steps"] > 0:
            self.ql_acc_var.set(f"Accuracy: {summary['ql_accuracy']:.2%}")
        self.ql_eps_var.set(f"ε = {summary['ql_epsilon']:.3f}")

        checkpoint_info = self.ai.checkpoint_info() or {}
        if checkpoint_info.get("exists", False):
            size = checkpoint_info.get("size_bytes")
            size_str = f" ({size} bytes)" if size is not None else ""
            self.checkpoint_status_var.set(f"Checkpoint: Saved{size_str}")
        else:
            self.checkpoint_status_var.set("Checkpoint: Missing")

    def _save_ai(self) -> None:
        if self.ai.save():
            info = self.ai.checkpoint_info() or {}
            self._append_log(f"💾 Saved AI checkpoint to {info.get('path')}")
            self._refresh_stats()
        else:
            self._append_log("⚠  No checkpoint path configured.")

    def _reload_ai(self) -> None:
        if self.ai.load():
            summary = self.ai.summary()
            self._append_log(
                f"↻ Reloaded AI checkpoint | GA gen {summary['ga_generation']} | QL steps {summary['ql_steps']}"
            )
            self._refresh_stats()
        else:
            self._append_log("⚠  No checkpoint found or load failed.")

    def _refresh_countdown(self) -> None:
        if self.scheduler and self.scheduler.is_running():
            secs = self.scheduler.seconds_until_next()
            if secs == 0:
                self.countdown_var.set("Scraping in progress …")
                self.scraper_status_var.set("Scraping … 🔄")
            else:
                mins, s = divmod(secs, 60)
                self.countdown_var.set(f"Next scrape in {mins:02d}:{s:02d}")
                self.scraper_status_var.set("Scheduled ▶")


# ── Entry point ────────────────────────────────────────────────────────────────

def launch():
    root = tk.Tk()
    app = BetpawaTrackerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
