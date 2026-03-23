"""
SQLite database manager for the Betpawa Tracker application.

Stores scraped MUN match results and exposes helpers used by the AI and GUI.
"""

import sqlite3
import threading
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import config


class DataManager:
    """Thread-safe SQLite manager for match-result records."""

    def __init__(self, db_path: str = config.DATABASE_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    # ── Schema setup ─────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create tables if they don't already exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS results (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,
                    team_home       TEXT    NOT NULL,
                    team_away       TEXT    NOT NULL,
                    ht_home         INTEGER,
                    ht_away         INTEGER,
                    ft_home         INTEGER NOT NULL,
                    ft_away         INTEGER NOT NULL,
                    mun_goals       INTEGER NOT NULL,
                    opp_goals       INTEGER NOT NULL,
                    outcome         INTEGER NOT NULL,   -- 0=Loss 1=Draw 2=Win
                    raw_result      TEXT    NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scrape_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT NOT NULL,
                    status      TEXT NOT NULL,
                    message     TEXT
                )
                """
            )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Result insertion ─────────────────────────────────────────────────────

    def insert_result(
        self,
        team_home: str,
        team_away: str,
        ht_home: Optional[int],
        ht_away: Optional[int],
        ft_home: int,
        ft_away: int,
        raw_result: str,
    ) -> Optional[int]:
        """
        Persist a match result.  Returns the new row id, or None if this exact
        raw_result was already stored in the last 5 minutes (deduplication).
        """
        # Determine which side is MUN
        if team_home.upper() == config.TEAM_NAME:
            mun_goals = ft_home
            opp_goals = ft_away
        else:
            mun_goals = ft_away
            opp_goals = ft_home

        if mun_goals > opp_goals:
            outcome = config.RESULT_WIN
        elif mun_goals == opp_goals:
            outcome = config.RESULT_DRAW
        else:
            outcome = config.RESULT_LOSS

        now = datetime.now(timezone.utc).isoformat()

        with self._lock, self._connect() as conn:
            # Deduplicate: same raw_result within the last 5 minutes
            existing = conn.execute(
                """
                SELECT id FROM results
                WHERE raw_result = ?
                  AND timestamp >= datetime('now', '-5 minutes')
                LIMIT 1
                """,
                (raw_result,),
            ).fetchone()
            if existing:
                return None

            cursor = conn.execute(
                """
                INSERT INTO results
                    (timestamp, team_home, team_away,
                     ht_home, ht_away, ft_home, ft_away,
                     mun_goals, opp_goals, outcome, raw_result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    team_home,
                    team_away,
                    ht_home,
                    ht_away,
                    ft_home,
                    ft_away,
                    mun_goals,
                    opp_goals,
                    outcome,
                    raw_result,
                ),
            )
            return cursor.lastrowid

    # ── Scrape log ───────────────────────────────────────────────────────────

    def log_scrape(self, status: str, message: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO scrape_log (timestamp, status, message) VALUES (?, ?, ?)",
                (now, status, message),
            )

    # ── Data retrieval ───────────────────────────────────────────────────────

    def get_all_results(self) -> List[sqlite3.Row]:
        """Return all stored results, newest first."""
        with self._lock, self._connect() as conn:
            return conn.execute(
                "SELECT * FROM results ORDER BY id DESC"
            ).fetchall()

    def get_recent_results(self, n: int = 20) -> List[sqlite3.Row]:
        """Return the most recent *n* results (oldest → newest)."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM results ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return list(reversed(rows))

    def get_outcomes(self, n: Optional[int] = None) -> List[int]:
        """Return a list of outcome integers (oldest → newest)."""
        query = "SELECT outcome FROM results ORDER BY id ASC"
        params: Tuple = ()
        if n is not None:
            query = (
                "SELECT outcome FROM results "
                "ORDER BY id DESC LIMIT ?"
            )
            params = (n,)

        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        outcomes = [r["outcome"] for r in rows]
        if n is not None:
            outcomes = list(reversed(outcomes))
        return outcomes

    def get_goals_series(self, n: Optional[int] = None) -> List[Tuple[int, int]]:
        """Return list of (mun_goals, opp_goals) tuples (oldest → newest)."""
        query = "SELECT mun_goals, opp_goals FROM results ORDER BY id ASC"
        params: Tuple = ()
        if n is not None:
            query = (
                "SELECT mun_goals, opp_goals FROM results "
                "ORDER BY id DESC LIMIT ?"
            )
            params = (n,)

        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        series = [(r["mun_goals"], r["opp_goals"]) for r in rows]
        if n is not None:
            series = list(reversed(series))
        return series

    def count_results(self) -> int:
        with self._lock, self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]

    def get_scrape_log(self, n: int = 50) -> List[sqlite3.Row]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scrape_log ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return list(reversed(rows))
