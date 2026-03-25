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

import config
from data_manager import DataManager

logger = logging.getLogger(__name__)

# Matches "(ht_home - ht_away)ft_home - ft_away"  (spaces optional)
_RESULT_RE = re.compile(
    r"\(\s*(\d+)\s*-\s*(\d+)\s*\)\s*(\d+)\s*-\s*(\d+)"
)
_SEASON_RE = re.compile(r"Season\s*#(\d+)", re.IGNORECASE)
_MATCHDAY_FIXTURE_RE = re.compile(
    r"(?P<team_home>[A-Z]{2,})\s*-\s*(?P<team_away>[A-Z]{2,})\s*"
    r"\(\s*(?P<ht_home>\d+)\s*-\s*(?P<ht_away>\d+)\s*\)\s*"
    r"(?P<ft_home>\d+)\s*-\s*(?P<ft_away>\d+)"
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
    """Create a headless Brave (Chromium) WebDriver."""
    opts = Options()
    opts.binary_location = config.BRAVE_BINARY_PATH
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

    service = Service()
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


def _scrape_upcoming_page(driver: webdriver.Chrome) -> List[dict]:
    """
    Scrape the upcoming page for MUN matches and their odds.
    Returns a list of match dicts with odds.
    """
    matches: List[dict] = []

    try:
        driver.get(config.UPCOMING_URL)
        WebDriverWait(driver, config.PAGE_LOAD_WAIT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
        )
    except TimeoutException:
        logger.warning("Upcoming page load timed out")
        return matches

    # Find match elements (similar to live page)
    team = config.TEAM_NAME
    for selector in (
        "div.match",
        "div.event",
        "div.fixture",
        "tr",
        "li.match-item",
    ):
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for el in elements:
                txt = el.text.strip()
                if team in txt.upper():
                    parsed = _parse_upcoming_row(el)
                    if parsed:
                        matches.append(parsed)
        except Exception:
            pass

    return matches


def _parse_upcoming_row(element) -> Optional[dict]:
    """
    Parse a match element for teams and odds.
    """
    txt = element.text.strip()
    words = re.split(r"[\s\-\(\)]+", txt)
    team_tokens = [
        w.upper()
        for w in words
        if re.match(r"^[A-Za-z]{2,}$", w) and w.upper() != "VS"
    ]

    if len(team_tokens) < 2 or config.TEAM_NAME not in team_tokens:
        return None

    team_home = team_tokens[0]
    team_away = team_tokens[1]

    # Extract odds – this is placeholder; need to inspect the page for actual selectors
    home_odds = draw_odds = away_odds = None
    try:
        odds_elements = element.find_elements(By.CSS_SELECTOR, config.ODDS_SELECTOR)
        if len(odds_elements) >= 3:
            home_odds = float(odds_elements[0].text.strip())
            draw_odds = float(odds_elements[1].text.strip())
            away_odds = float(odds_elements[2].text.strip())
    except (ValueError, IndexError):
        pass

    return {
        "team_home": team_home,
        "team_away": team_away,
        "home_odds": home_odds,
        "draw_odds": draw_odds,
        "away_odds": away_odds,
    }


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


def parse_matchday_text(page_text: str) -> List[dict]:
    """Extract TEAM_NAME fixtures from a rendered historical matchday page."""
    results: List[dict] = []
    normalized = re.sub(r"\s+", " ", page_text.upper())

    for match in _MATCHDAY_FIXTURE_RE.finditer(normalized):
        team_home = match.group("team_home")
        team_away = match.group("team_away")
        if config.TEAM_NAME not in (team_home, team_away):
            continue

        ht_home = int(match.group("ht_home"))
        ht_away = int(match.group("ht_away"))
        ft_home = int(match.group("ft_home"))
        ft_away = int(match.group("ft_away"))

        results.append(
            {
                "team_home": team_home,
                "team_away": team_away,
                "raw_result": (
                    f"{team_home} - {team_away} "
                    f"({ht_home} - {ht_away}){ft_home} - {ft_away}"
                ),
                "ht_home": ht_home,
                "ht_away": ht_away,
                "ft_home": ft_home,
                "ft_away": ft_away,
            }
        )

    return results


def _wait_for_rendered_body_text(
    driver: webdriver.Chrome,
    required_text: Optional[str] = None,
    min_length: int = 100,
    timeout: Optional[int] = None,
) -> str:
    """Wait until the page body contains rendered text instead of the app shell."""
    deadline = time.time() + (timeout or config.PAGE_LOAD_WAIT)
    last_text = ""

    while time.time() < deadline:
        try:
            body_text = driver.find_element(By.TAG_NAME, "body").text.strip()
        except Exception:
            body_text = ""

        if body_text:
            last_text = body_text
            if len(body_text) >= min_length:
                if required_text is None or required_text.upper() in body_text.upper():
                    return body_text
        time.sleep(1)

    return last_text


def discover_recent_season_ids(
    driver: webdriver.Chrome,
    limit: int = config.HISTORICAL_SEASONS_LIMIT,
) -> List[int]:
    """Read the visible season list from the Betpawa Results page."""
    driver.get(config.RESULTS_URL)
    body_text = _wait_for_rendered_body_text(driver, required_text="Season #")

    season_ids: List[int] = []
    for raw in _SEASON_RE.findall(body_text):
        season_id = int(raw)
        if season_id not in season_ids:
            season_ids.append(season_id)
        if len(season_ids) >= limit:
            break

    return season_ids


def run_historical_backfill(
    db: DataManager,
    seasons: int = config.HISTORICAL_SEASONS_LIMIT,
    matchdays: int = config.HISTORICAL_MATCHDAYS,
    start_season_id: Optional[int] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> int:
    """Import historical TEAM_NAME results from the results/matchday pages."""

    def _log(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        full = f"[{ts}] {msg}"
        logger.info(msg)
        db.log_scrape("INFO", msg)
        if log_callback:
            log_callback(full)

    _log("Starting historical backfill …")
    driver: Optional[webdriver.Chrome] = None
    inserted = 0

    def _progress(completed: int, total: int, status: str) -> None:
        if progress_callback:
            progress_callback(completed, total, status)

    try:
        driver = _build_driver()

        if start_season_id is None:
            season_ids = discover_recent_season_ids(driver, limit=seasons)
        else:
            season_ids = [
                season_id
                for season_id in range(start_season_id, start_season_id - seasons, -1)
                if season_id > 0
            ]

        if not season_ids:
            _log("No season IDs were discovered from the Results page.")
            _progress(0, 0, "No seasons found")
            return 0

        _log("Discovered seasons: " + ", ".join(f"#{season_id}" for season_id in season_ids))
        total_steps = len(season_ids) * matchdays
        completed_steps = 0
        _progress(0, total_steps, "Preparing historical import")

        # Insert oldest to newest so AI training sees a natural history.
        for season_id in reversed(season_ids):
            _log(f"Season #{season_id} …")
            for matchday in range(1, matchdays + 1):
                completed_steps += 1
                url = config.MATCHDAY_URL_TEMPLATE.format(
                    season_id=season_id,
                    matchday=matchday,
                )
                _progress(
                    completed_steps - 1,
                    total_steps,
                    f"Season #{season_id} • Matchday {matchday:02d}",
                )
                driver.get(url)
                body_text = _wait_for_rendered_body_text(
                    driver,
                    required_text=f"MATCH DAY: {matchday:02d}",
                    min_length=150,
                )
                fixtures = parse_matchday_text(body_text)

                if not fixtures:
                    _log(f"  Matchday {matchday:02d}: no {config.TEAM_NAME} fixture found")
                    _progress(
                        completed_steps,
                        total_steps,
                        f"Season #{season_id} • Matchday {matchday:02d} complete",
                    )
                    continue

                for fixture in fixtures:
                    row_id = db.insert_result(
                        team_home=fixture["team_home"],
                        team_away=fixture["team_away"],
                        ht_home=fixture["ht_home"],
                        ht_away=fixture["ht_away"],
                        ft_home=fixture["ft_home"],
                        ft_away=fixture["ft_away"],
                        raw_result=fixture["raw_result"],
                        source="historical",
                        deduplicate=False,
                    )
                    if row_id:
                        inserted += 1
                        _log(
                            f"  Matchday {matchday:02d}: stored #{row_id} "
                            f"{fixture['team_home']} {fixture['ft_home']}-{fixture['ft_away']} "
                            f"{fixture['team_away']}"
                        )
                _progress(
                    completed_steps,
                    total_steps,
                    f"Season #{season_id} • Matchday {matchday:02d} complete",
                )

        _log(f"Historical backfill complete – {inserted} result(s) imported.")
        _progress(total_steps, total_steps, f"Historical import complete ({inserted} results)")

    except WebDriverException as exc:
        msg = f"Historical backfill failed: {exc}"
        _log(msg)
        db.log_scrape("ERROR", msg)
        _progress(0, 0, f"Historical import failed: {exc}")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    return inserted


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
        max_polls = getattr(config, "SCRAPE_MAX_POLLS", 8)
        poll_count = 0
        latest_matches: List[dict] = []

        # Poll repeatedly; only the last successful poll is treated as
        # the final result (scores update live every ~5 s on the site).
        while time.time() < session_end and poll_count < max_polls:
            try:
                poll_count += 1
                matches = _scrape_page(driver)
                if matches:
                    latest_matches = matches
                    _log(f"  Poll {poll_count}/{max_polls} – {len(matches)} MUN result(s) (updating …)")
                else:
                    _log(f"  Poll {poll_count}/{max_polls} – No MUN results found")

                remaining = session_end - time.time()
                sleep_time = min(poll, max(0, remaining))
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except WebDriverException as exc:
                _log(f"  WebDriver error during poll: {exc}")
                break

        # Store only the final (last) scraped results as definitive
        if latest_matches:
            for m in latest_matches:
                row_id = db.insert_result(
                    team_home=m["team_home"],
                    team_away=m["team_away"],
                    ht_home=m.get("ht_home"),
                    ht_away=m.get("ht_away"),
                    ft_home=m["ft_home"],
                    ft_away=m["ft_away"],
                    raw_result=m["raw_result"],
                    source="live",
                )
                if row_id:
                    inserted += 1
                    _log(
                        f"  ✔ Final result #{row_id}: "
                        f"{m['team_home']} {m['ft_home']}-{m['ft_away']} "
                        f"{m['team_away']}  [{m['raw_result']}]"
                    )

        # Scrape upcoming matches after live results
        upcoming_matches = _scrape_upcoming_page(driver)
        upcoming_inserted = 0
        for m in upcoming_matches:
            row_id = db.insert_upcoming_match(
                team_home=m["team_home"],
                team_away=m["team_away"],
                home_odds=m.get("home_odds"),
                draw_odds=m.get("draw_odds"),
                away_odds=m.get("away_odds"),
                source="upcoming",
            )
            if row_id:
                upcoming_inserted += 1
                _log(f"  Upcoming: {m['team_home']} vs {m['team_away']} (odds: {m.get('home_odds')}/{m.get('draw_odds')}/{m.get('away_odds')})")

        _log(f"Session complete ({poll_count} polls) – {inserted} new result(s), {upcoming_inserted} upcoming match(es) stored.")

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
        # Wait until the next clock-aligned 5-minute mark before first scrape
        self._wait_until_next_boundary(interval)
        while not self._stop_event.is_set():
            self.next_run = None
            run_scrape_session(self.db, self.log_callback)
            if self._stop_event.is_set():
                break
            # Schedule next run at the next clock-aligned boundary
            self._wait_until_next_boundary(interval)

    def _wait_until_next_boundary(self, interval: int) -> None:
        """Sleep until the next clock-aligned boundary (e.g. XX:00, XX:05, XX:10)."""
        now = time.time()
        # Seconds past the current interval boundary
        remainder = now % interval
        wait = interval - remainder
        # If we're within 2 seconds of a boundary, skip to the next one
        if wait < 2:
            wait += interval
        self.next_run = now + wait
        if self.log_callback:
            from datetime import datetime as _dt
            target = _dt.fromtimestamp(self.next_run).strftime("%H:%M:%S")
            self.log_callback(f"⏰ Next scrape aligned to {target}")
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
