"""
Web scraper for the Betpawa virtual-sports page.

Scheduling behaviour
--------------------
* A scraping *session* starts every SCRAPE_INTERVAL_SECONDS (5 min).
* Each session lasts SCRAPE_SESSION_DURATION_SECONDS (30 s), polling the
  live page every SCRAPE_POLL_INTERVAL_SECONDS (5 s).
* Only results that contain TEAM_NAME ("MUN") are stored.

Result parsing
--------------
The page shows strings like:  "(0 - 0)0 - 3"
  - half-time score  → inside ( … )
  - full-time score  → outside ( … )
We always store both but use the full-time score for AI training.
"""

import re
import time
import threading
import logging
from datetime import datetime
from typing import Callable, List, Optional, Tuple

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from webdriver_manager.chrome import ChromeDriverManager

import config
from data_manager import DataManager

logger = logging.getLogger(__name__)

# Matches "(ht_home - ht_away)ft_home - ft_away"  (spaces optional)
_RESULT_RE = re.compile(
    r"\(\s*(\d+)\s*-\s*(\d+)\s*\)\s*(\d+)\s*-\s*(\d+)"
)


# ── Result parsing ────────────────────────────────────────────────────────────

def parse_result(raw: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Parse a result string of the form "(ht_h - ht_a)ft_h - ft_a".

    Returns (ht_home, ht_away, ft_home, ft_away) or None if not matched.
    """
    m = _RESULT_RE.search(raw)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))


# ── Selenium helpers ─────────────────────────────────────────────────────────

def _build_driver() -> webdriver.Chrome:
    """Create a headless Chrome WebDriver."""
    opts = Options()
    if config.HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--log-level=3")
    # Suppress ChromeDriver logs
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(30)
    return driver


# ── Core scraping logic ───────────────────────────────────────────────────────

def _scrape_page(driver: webdriver.Chrome) -> List[dict]:
    """
    Parse the current page source and return a list of match dicts that
    involve TEAM_NAME.  Each dict has keys:
        team_home, team_away, raw_result,
        ht_home, ht_away, ft_home, ft_away
    """
    results: List[dict] = []

    # The page renders a table / card layout.
    # We look for any element whose text contains the team name and then walk
    # up / sideways to find the score element.
    # Strategy: grab all text nodes, find rows containing TEAM_NAME.
    try:
        # Wait for at least one score element to appear
        WebDriverWait(driver, config.PAGE_LOAD_WAIT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
        )
    except TimeoutException:
        logger.warning("Page load timed out")
        return results

    # Try various CSS patterns the site may use
    candidates = _find_match_rows(driver)
    for row_text in candidates:
        parsed = _parse_row(row_text)
        if parsed:
            results.append(parsed)

    return results


def _find_match_rows(driver: webdriver.Chrome) -> List[str]:
    """
    Return raw text of page sections that contain TEAM_NAME and a score.
    Falls back to scanning the full page body text line by line.
    """
    team = config.TEAM_NAME

    # 1) Try to find structured match elements
    row_texts: List[str] = []
    for selector in (
        "div.match",
        "div.event",
        "div.fixture",
        "tr",
        "li.match-item",
        "div.score-row",
    ):
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for el in elements:
                txt = el.text.strip()
                if team in txt.upper() and _RESULT_RE.search(txt):
                    row_texts.append(txt)
        except Exception:
            pass

    if row_texts:
        return row_texts

    # 2) Fallback: scan the full visible text line by line
    try:
        body_text = driver.find_element(By.TAG_NAME, "body").text
    except Exception:
        return []

    lines: List[str] = []
    # Group consecutive lines into blocks of ~3 (home, score, away)
    all_lines = [ln.strip() for ln in body_text.splitlines() if ln.strip()]
    for i, line in enumerate(all_lines):
        # Look for a line that contains a result pattern near the team name
        block = " ".join(all_lines[max(0, i - 2): i + 3])
        if team in block.upper() and _RESULT_RE.search(block):
            lines.append(block)

    return lines


def _parse_row(row_text: str) -> Optional[dict]:
    """
    Try to extract team names and score from a row / block of text.
    Returns a dict or None.
    """
    parsed = parse_result(row_text)
    if not parsed:
        return None

    ht_home, ht_away, ft_home, ft_away = parsed

    # Extract team names: any token that is purely alphabetic (≥2 chars)
    # and is not a digit-containing token
    words = re.split(r"[\s\-\(\)]+", row_text)
    team_tokens = [
        w.upper()
        for w in words
        if re.match(r"^[A-Za-z]{2,}$", w) and w.upper() != "VS"
    ]

    if len(team_tokens) < 2:
        # Can't determine teams – use placeholder
        if config.TEAM_NAME not in row_text.upper():
            return None
        team_home = config.TEAM_NAME
        team_away = "OPP"
    else:
        team_home = team_tokens[0]
        team_away = team_tokens[1]

    if config.TEAM_NAME not in (team_home, team_away):
        return None

    return {
        "team_home": team_home,
        "team_away": team_away,
        "raw_result": row_text.strip(),
        "ht_home": ht_home,
        "ht_away": ht_away,
        "ft_home": ft_home,
        "ft_away": ft_away,
    }


# ── Scraping session ──────────────────────────────────────────────────────────

def run_scrape_session(
    db: DataManager,
    log_callback: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Open the browser, poll the page for SCRAPE_SESSION_DURATION_SECONDS,
    save any new results to the database.

    Returns the number of new rows inserted.
    """
    def _log(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        full = f"[{ts}] {msg}"
        logger.info(msg)
        db.log_scrape("INFO", msg)
        if log_callback:
            log_callback(full)

    _log("Starting scrape session …")
    driver: Optional[webdriver.Chrome] = None
    inserted = 0

    try:
        driver = _build_driver()
        _log(f"Loading {config.TARGET_URL}")
        driver.get(config.TARGET_URL)

        session_end = time.time() + config.SCRAPE_SESSION_DURATION_SECONDS
        poll = config.SCRAPE_POLL_INTERVAL_SECONDS

        while time.time() < session_end:
            try:
                matches = _scrape_page(driver)
                if matches:
                    for m in matches:
                        row_id = db.insert_result(
                            team_home=m["team_home"],
                            team_away=m["team_away"],
                            ht_home=m.get("ht_home"),
                            ht_away=m.get("ht_away"),
                            ft_home=m["ft_home"],
                            ft_away=m["ft_away"],
                            raw_result=m["raw_result"],
                        )
                        if row_id:
                            inserted += 1
                            _log(
                                f"  ✔ New result #{row_id}: "
                                f"{m['team_home']} {m['ft_home']}-{m['ft_away']} "
                                f"{m['team_away']}  [{m['raw_result']}]"
                            )
                else:
                    _log("  – No MUN results found this poll")

                remaining = session_end - time.time()
                sleep_time = min(poll, max(0, remaining))
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except WebDriverException as exc:
                _log(f"  WebDriver error during poll: {exc}")
                break

        _log(f"Session complete – {inserted} new result(s) stored.")

    except WebDriverException as exc:
        msg = f"WebDriver failed to start: {exc}"
        _log(msg)
        db.log_scrape("ERROR", msg)
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    return inserted


# ── Scheduler ─────────────────────────────────────────────────────────────────

class ScraperScheduler:
    """
    Runs run_scrape_session() every SCRAPE_INTERVAL_SECONDS in a background
    thread.  The first session starts immediately.
    """

    def __init__(
        self,
        db: DataManager,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        self.db = db
        self.log_callback = log_callback
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.next_run: Optional[float] = None  # unix timestamp

    def start(self) -> None:
        """Start the background scheduler thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the scheduler to stop after the current session."""
        self._stop_event.set()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _loop(self) -> None:
        interval = config.SCRAPE_INTERVAL_SECONDS
        while not self._stop_event.is_set():
            self.next_run = None
            run_scrape_session(self.db, self.log_callback)
            if self._stop_event.is_set():
                break
            self.next_run = time.time() + interval
            # Sleep in small increments so we can respond to stop quickly
            while time.time() < self.next_run:
                if self._stop_event.is_set():
                    return
                time.sleep(1)

    def seconds_until_next(self) -> int:
        """Return seconds until the next scrape session (0 if running)."""
        if self.next_run is None:
            return 0
        remaining = int(self.next_run - time.time())
        return max(0, remaining)
