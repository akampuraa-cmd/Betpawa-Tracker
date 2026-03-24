"""
Command-line interface for the Betpawa Tracker.

Usage examples
--------------
# Start the scheduler (runs continuously every 5 minutes)
python cli.py start

# Run a single scrape session right now
python cli.py scrape

# Show the last N stored results
python cli.py results --count 20

# Train the AI on all stored data
python cli.py train --ga-generations 50

# Predict the next MUN match outcome
python cli.py predict

# Show scraper + AI status summary
python cli.py status
"""

import argparse
import sys
import time
import signal
import logging
from typing import List

import config
from data_manager import DataManager
from scraper import run_scrape_session, ScraperScheduler
from ai_model import BetpawaAI

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Colour helpers (ANSI) ─────────────────────────────────────────────────────
_RESET = "\033[0m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_PURPLE = "\033[95m"
_BOLD = "\033[1m"


def _c(text: str, colour: str) -> str:
    """Wrap text with an ANSI colour code."""
    return f"{colour}{text}{_RESET}"


def _print_header(title: str) -> None:
    print(_c(f"\n{'─' * 60}", _CYAN))
    print(_c(f"  {title}", _BOLD))
    print(_c(f"{'─' * 60}", _CYAN))


# ── Sub-command handlers ──────────────────────────────────────────────────────

def cmd_start(args, db: DataManager) -> None:
    """Start the scraper scheduler (runs until Ctrl-C)."""
    _print_header("Betpawa MUN Tracker — Scheduler")
    print(f"  Scraping every {config.SCRAPE_INTERVAL_SECONDS // 60} min "
          f"for {config.SCRAPE_SESSION_DURATION_SECONDS} s each session")
    print(f"  Press {_c('Ctrl-C', _YELLOW)} to stop.\n")

    def _log(msg: str) -> None:
        print(f"  {msg}")

    scheduler = ScraperScheduler(db, log_callback=_log)
    scheduler.start()

    # Keep the main thread alive; handle Ctrl-C gracefully
    try:
        while scheduler.is_running():
            secs = scheduler.seconds_until_next()
            if secs > 0:
                mins, s = divmod(secs, 60)
                print(
                    f"\r  {_c('Next scrape in', _CYAN)} {mins:02d}:{s:02d}   ",
                    end="",
                    flush=True,
                )
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n")
        logger.info("Stopping scheduler …")
        scheduler.stop()
        print(_c("  Scheduler stopped.", _YELLOW))


def cmd_scrape(args, db: DataManager) -> None:
    """Run a single scrape session immediately."""
    _print_header("Betpawa MUN Tracker — Single Scrape")

    def _log(msg: str) -> None:
        print(f"  {msg}")

    inserted = run_scrape_session(db, log_callback=_log)
    colour = _GREEN if inserted > 0 else _YELLOW
    print(_c(f"\n  Done — {inserted} new result(s) stored.", colour))


def cmd_results(args, db: DataManager) -> None:
    """Print recent stored results."""
    n: int = args.count
    _print_header(f"Last {n} MUN Results")

    rows = db.get_recent_results(n)
    if not rows:
        print(_c("  No results stored yet. Run `scrape` first.", _YELLOW))
        return

    # Header
    print(
        f"  {'#':>4}  {'Timestamp':<18}  {'Home':<6}  {'FT':^7}  {'Away':<6}  "
        f"{'Outcome':<6}"
    )
    print("  " + "─" * 56)
    for row in rows:
        outcome_label = config.RESULT_LABELS.get(row["outcome"], "?")
        colour = {
            config.RESULT_WIN: _GREEN,
            config.RESULT_DRAW: _YELLOW,
            config.RESULT_LOSS: _RED,
        }.get(row["outcome"], _RESET)
        ts = row["timestamp"][:16].replace("T", " ")
        score = f"{row['ft_home']}-{row['ft_away']}"
        print(
            f"  {row['id']:>4}  {ts:<18}  {row['team_home']:<6}  "
            f"{score:^7}  {row['team_away']:<6}  "
            f"{_c(outcome_label, colour)}"
        )

    total = db.count_results()
    print(f"\n  Total stored: {total}")


def cmd_train(args, db: DataManager) -> None:
    """Train both AI models on all stored data."""
    _print_header("Betpawa AI — Training")

    outcomes = db.get_outcomes()
    goals = db.get_goals_series()
    n = len(outcomes)

    if n < 3:
        print(_c(f"  ⚠  Need at least 3 results to train (have {n}).", _YELLOW))
        return

    print(f"  Training on {n} results …\n")
    ai = BetpawaAI()
    gens: int = args.ga_generations

    def ga_cb(generation, best_fitness):
        bar = "█" * int(best_fitness * 20) + "░" * (20 - int(best_fitness * 20))
        print(
            f"\r  GA gen {generation:>4}/{gens}  |  [{bar}]  "
            f"fitness={best_fitness:.4f}   ",
            end="",
            flush=True,
        )

    def ql_cb(step, reward, epsilon, q_table_size):
        pass  # Q-learning is fast; we print a summary at the end

    ai.train(outcomes, goals, ga_generations=gens, ga_callback=ga_cb, ql_callback=ql_cb)
    print()  # newline after progress bar

    summary = ai.summary()
    print(f"\n  {_c('Genetic Algorithm', _PURPLE)}")
    print(f"    Generations:  {summary['ga_generation']}")
    print(f"    Best fitness: {summary['ga_best_fitness']:.4f}")
    print(f"\n  {_c('Q-Learning', _YELLOW)}")
    print(f"    Steps:        {summary['ql_steps']}")
    print(f"    Accuracy:     {summary['ql_accuracy']:.2%}")
    print(f"    ε (epsilon):  {summary['ql_epsilon']:.4f}")
    print(f"    Q-table size: {summary['ql_q_table_size']}")


def cmd_predict(args, db: DataManager) -> None:
    """Predict the next MUN match outcome."""
    _print_header("Betpawa AI — Prediction")

    outcomes = db.get_outcomes()
    goals = db.get_goals_series()

    if len(outcomes) < 2:
        print(_c("  ⚠  Not enough data to predict yet.", _YELLOW))
        return

    ai = BetpawaAI()
    # Quick re-train so the prediction reflects all available data
    if len(outcomes) >= 3:
        ai.train(
            outcomes, goals,
            ga_generations=min(50, config.GA_GENERATIONS),
        )

    result = ai.predict(outcomes, goals)
    colour = {
        config.RESULT_WIN: _GREEN,
        config.RESULT_DRAW: _YELLOW,
        config.RESULT_LOSS: _RED,
    }.get(result["consensus"], _RESET)

    print(f"\n  Based on {len(outcomes)} historical results:\n")
    print(
        f"  Genetic Algorithm  →  {_c(result['ga_label'], _PURPLE)}  "
        f"(confidence {result['ga_confidence']:.0%})"
    )
    print(f"  Q-Learning         →  {_c(result['ql_label'], _YELLOW)}")
    print(f"\n  {_c('Consensus prediction', _BOLD)}:  {_c(result['consensus_label'], colour)}")


def cmd_status(args, db: DataManager) -> None:
    """Show a brief system status."""
    _print_header("Betpawa Tracker — Status")

    total = db.count_results()
    print(f"  Stored results : {_c(str(total), _GREEN)}")
    print(f"  Database       : {config.DATABASE_PATH}")
    print(f"  Target URL     : {config.TARGET_URL}")
    print(f"  Team           : {config.TEAM_NAME}")
    print(f"  Scrape interval: every {config.SCRAPE_INTERVAL_SECONDS // 60} min")
    print(f"  Session length : {config.SCRAPE_SESSION_DURATION_SECONDS} s")
    print(f"  Poll interval  : every {config.SCRAPE_POLL_INTERVAL_SECONDS} s")

    if total >= 3:
        outcomes = db.get_outcomes()
        wins = outcomes.count(config.RESULT_WIN)
        draws = outcomes.count(config.RESULT_DRAW)
        losses = outcomes.count(config.RESULT_LOSS)
        print(f"\n  Form summary   : {_c(f'W{wins}', _GREEN)} "
              f"{_c(f'D{draws}', _YELLOW)} {_c(f'L{losses}', _RED)}")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="betpawa-tracker",
        description=(
            "Betpawa MUN Virtual-Sports Tracker — "
            "web scraper + AI (Genetic Algorithm + Q-Learning)"
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # start
    sub.add_parser("start", help="Start the scheduler (runs every 5 min continuously)")

    # scrape
    sub.add_parser("scrape", help="Run a single scrape session immediately")

    # results
    p_res = sub.add_parser("results", help="Show the last N stored results")
    p_res.add_argument("--count", type=int, default=20, metavar="N",
                       help="How many results to show (default: 20)")

    # train
    p_train = sub.add_parser("train", help="Train both AI models on stored data")
    p_train.add_argument(
        "--ga-generations", type=int, default=config.GA_GENERATIONS, metavar="N",
        help=f"Number of GA generations (default: {config.GA_GENERATIONS})",
    )

    # predict
    sub.add_parser("predict", help="Predict the next MUN match outcome")

    # status
    sub.add_parser("status", help="Show status and stats")

    return parser


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    db = DataManager()

    handlers = {
        "start": cmd_start,
        "scrape": cmd_scrape,
        "results": cmd_results,
        "train": cmd_train,
        "predict": cmd_predict,
        "status": cmd_status,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        handler(args, db)
    except KeyboardInterrupt:
        print(_c("\nInterrupted.", _YELLOW))
    return 0


if __name__ == "__main__":
    sys.exit(main())
