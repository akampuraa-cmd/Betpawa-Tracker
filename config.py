"""
Configuration constants for the Betpawa Tracker application.
"""

# ── Target URL & team ────────────────────────────────────────────────────────
TARGET_URL = (
    "https://www.betpawa.ug/virtual-sports"
    "?virtualTab=live&leagueId=7794"
)
RESULTS_URL = (
    "https://www.betpawa.ug/virtual-sports"
    "?virtualTab=results&leagueId=7794"
)
MATCHDAY_URL_TEMPLATE = (
    "https://www.betpawa.ug/virtual-sports/matchday/{season_id}"
    "?matchday={matchday}&leagueId=7794"
)
UPCOMING_URL = (
    "https://www.betpawa.ug/virtual-sports"
    "?virtualTab=upcoming&leagueId=7794"
)
TEAM_NAME = "MUN"

# ── Odds parsing ─────────────────────────────────────────────────────────────
ODDS_SELECTOR = "div.odds"  # Placeholder; adjust based on page inspection
# How often (seconds) to start a new scraping session
SCRAPE_INTERVAL_SECONDS = 300       # 5 minutes

# How long (seconds) each scraping session lasts (8 refreshes × ~5 s + buffer)
SCRAPE_SESSION_DURATION_SECONDS = 50

# How often (seconds) to poll within a session (page refreshes every ~5 s)
SCRAPE_POLL_INTERVAL_SECONDS = 5

# Maximum number of polls per session (the site refreshes scores ~8 times)
SCRAPE_MAX_POLLS = 8
HISTORICAL_SEASONS_LIMIT = 10
HISTORICAL_MATCHDAYS = 38

# ── Database ─────────────────────────────────────────────────────────────────
DATABASE_PATH = "betpawa_tracker.db"

# ── AI – Genetic Algorithm ───────────────────────────────────────────────────
GA_POPULATION_SIZE = 50
GA_GENERATIONS = 100
GA_MUTATION_RATE = 0.1
GA_CROSSOVER_RATE = 0.7
GA_ELITISM_COUNT = 5
# Number of recent results used as features
GA_FEATURE_WINDOW = 5

# ── AI – Q-Learning (Reinforcement Learning) ─────────────────────────────────
QL_LEARNING_RATE = 0.1
QL_DISCOUNT_FACTOR = 0.9
QL_EPSILON = 0.2          # exploration rate (ε-greedy)
QL_EPSILON_DECAY = 0.995
QL_EPSILON_MIN = 0.01
# State window: how many recent results to encode as the RL state
QL_STATE_WINDOW = 3

# ── AI – LSTM ────────────────────────────────────────────────────────────────
LSTM_SEQUENCE_LENGTH = 10
LSTM_HIDDEN_SIZE = 32
LSTM_NUM_LAYERS = 2
LSTM_EPOCHS = 50
LSTM_LEARNING_RATE = 0.005

# ── Result encoding ──────────────────────────────────────────────────────────
# Maps a match result to a numeric label for the AI
RESULT_WIN = 2
RESULT_DRAW = 1
RESULT_LOSS = 0

# Human-readable label map
RESULT_LABELS = {RESULT_WIN: "Win", RESULT_DRAW: "Draw", RESULT_LOSS: "Loss"}

# ── Selenium ─────────────────────────────────────────────────────────────────
# Seconds to wait for the page to load before starting to parse
PAGE_LOAD_WAIT = 10
# Run browser in headless mode (no visible browser window)
HEADLESS = True

# Path to Brave browser binary (Chromium-based, uses ChromeDriver)
BRAVE_BINARY_PATH = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
