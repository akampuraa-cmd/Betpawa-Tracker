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
from scraper import ScraperScheduler, run_scrape_session
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


class BetpawaTrackerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.db = DataManager()
        self.ai = BetpawaAI()
        self.scheduler: Optional[ScraperScheduler] = None

        root.title("Betpawa MUN Tracker — AI Edition")
        root.configure(bg=BG_DARK)
        root.geometry("1100x780")
        root.minsize(900, 640)

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

        btn_row = tk.Frame(frame, bg=BG_MID)
        btn_row.pack(pady=6)
        self._btn(btn_row, "▶ Start", FG_GREEN, self._start_scheduler).pack(side=tk.LEFT, padx=4)
        self._btn(btn_row, "■ Stop", FG_RED, self._stop_scheduler).pack(side=tk.LEFT, padx=4)
        self._btn(btn_row, "⚡ Scrape Now", FG_YELLOW, self._scrape_now).pack(side=tk.LEFT, padx=4)

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

        columns = ("id", "timestamp", "home", "score", "away", "outcome", "raw")
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
        """Append a message to the log widget (thread-safe)."""
        def _do():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, msg + "\n")
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
        threading.Thread(
            target=lambda: run_scrape_session(self.db, self._append_log),
            daemon=True,
        ).start()

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

        threading.Thread(target=_run, daemon=True).start()

    def _predict(self) -> None:
        outcomes = self.db.get_outcomes()
        goals = self.db.get_goals_series()
        if len(outcomes) < 2:
            self._append_log("⚠  Not enough data to predict yet.")
            return
        result = self.ai.predict(outcomes, goals)
        label = (
            f"GA: {result['ga_label']} ({result['ga_confidence']:.0%}) | "
            f"QL: {result['ql_label']} | "
            f"Consensus: {result['consensus_label']}"
        )
        self.prediction_var.set(f"🔮 {result['consensus_label']}")
        self._append_log(f"Prediction → {label}")

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
            score = f"{row['ft_home']} - {row['ft_away']}"
            ts_short = row["timestamp"][:16].replace("T", " ")
            self.tree.insert(
                "", 0,
                values=(
                    row["id"],
                    ts_short,
                    row["team_home"],
                    score,
                    row["team_away"],
                    outcome_label,
                    row["raw_result"][:40],
                ),
                tags=(tag,),
            )

    def _refresh_stats(self) -> None:
        n = self.db.count_results()
        self.total_results_var.set(f"Stored results: {n}")

        summary = self.ai.summary()
        self.ga_gen_var.set(f"Generation: {summary['ga_generation']}")
        if summary["ga_best_fitness"] > 0:
            self.ga_fit_var.set(f"Best Fitness: {summary['ga_best_fitness']:.4f}")
        self.ql_steps_var.set(f"Steps: {summary['ql_steps']}")
        if summary["ql_steps"] > 0:
            self.ql_acc_var.set(f"Accuracy: {summary['ql_accuracy']:.2%}")
        self.ql_eps_var.set(f"ε = {summary['ql_epsilon']:.3f}")

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
