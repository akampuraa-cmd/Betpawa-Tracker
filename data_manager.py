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
                    raw_result      TEXT    NOT NULL,
                    source          TEXT    NOT NULL DEFAULT 'live'
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS upcoming_matches (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT    NOT NULL,
                    team_home   TEXT    NOT NULL,
                    team_away   TEXT    NOT NULL,
                    home_odds   REAL,
                    draw_odds   REAL,
                    away_odds   REAL,
                    source      TEXT    NOT NULL DEFAULT 'upcoming'
                )
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(results)").fetchall()
            }
            if "source" not in columns:
                conn.execute(
                    "ALTER TABLE results ADD COLUMN source TEXT NOT NULL DEFAULT 'live'"
                )
            if "season_id" not in columns:
                conn.execute("ALTER TABLE results ADD COLUMN season_id INTEGER")
            if "matchday" not in columns:
                conn.execute("ALTER TABLE results ADD COLUMN matchday INTEGER")

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
        source: str = "live",
        deduplicate: bool = True,
        season_id: Optional[int] = None,
        matchday: Optional[int] = None,
    ) -> Optional[int]:
        """
        Persist a match result.

        Returns the new row id, or None when the live-scrape deduplication
        rule rejects the insert.
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
            if deduplicate and source == "live":
                # Live scraping revisits the same fixture several times while
                # scores are updating. Keep dedup on for that path only.
                existing = conn.execute(
                    """
                    SELECT id FROM results
                    WHERE team_home = ? AND team_away = ?
                      AND ft_home = ? AND ft_away = ?
                      AND raw_result = ?
                      AND timestamp >= datetime('now', '-10 minutes')
                    LIMIT 1
                    """,
                    (team_home, team_away, ft_home, ft_away, raw_result),
                ).fetchone()
                if existing:
                    return None

            if source == "historical" and season_id is not None and matchday is not None:
                existing_historical = conn.execute(
                    """
                    SELECT id FROM results
                    WHERE season_id = ? AND matchday = ? AND team_home = ? AND team_away = ? AND source = 'historical'
                    LIMIT 1
                    """,
                    (season_id, matchday, team_home, team_away)
                ).fetchone()
                if existing_historical:
                    return None

            cursor = conn.execute(
                """
                INSERT INTO results
                    (timestamp, team_home, team_away,
                     ht_home, ht_away, ft_home, ft_away,
                     mun_goals, opp_goals, outcome, raw_result, source, season_id, matchday)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    source,
                    season_id,
                    matchday,
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

    def count_results_by_source(self) -> dict[str, int]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT source, COUNT(*) AS total FROM results GROUP BY source"
            ).fetchall()
        counts = {"live": 0, "historical": 0}
        for row in rows:
            counts[row["source"]] = row["total"]
        return counts

    def clear_all_results(self) -> None:
        """Delete all match results from the database."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM results")
            conn.commit()

    def get_scrape_log(self, n: int = 50) -> List[sqlite3.Row]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scrape_log ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return list(reversed(rows))

    # ── Upcoming matches ─────────────────────────────────────────────────────

    def insert_upcoming_match(
        self,
        team_home: str,
        team_away: str,
        home_odds: Optional[float],
        draw_odds: Optional[float],
        away_odds: Optional[float],
        source: str = "upcoming",
    ) -> Optional[int]:
        """Persist an upcoming match with odds."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO upcoming_matches
                    (timestamp, team_home, team_away, home_odds, draw_odds, away_odds, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (now, team_home, team_away, home_odds, draw_odds, away_odds, source),
            )
            return cursor.lastrowid

    def get_upcoming_matches(self, n: Optional[int] = None) -> List[sqlite3.Row]:
        """Return upcoming matches, newest first."""
        query = "SELECT * FROM upcoming_matches ORDER BY id DESC"
        params: Tuple = ()
        if n is not None:
            query += " LIMIT ?"
            params = (n,)
        with self._lock, self._connect() as conn:
            return conn.execute(query, params).fetchall()

    def clear_upcoming_matches(self) -> None:
        """Delete all upcoming matches from the database."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM upcoming_matches")
            conn.commit()
