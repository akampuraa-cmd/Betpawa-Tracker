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

# ── AI persistence ──────────────────────────────────────────────────────────
MODEL_CHECKPOINT_PATH = "betpawa_ai_checkpoint.pkl"

# ── AI – Genetic Algorithm ───────────────────────────────────────────────────
GA_POPULATION_SIZE = 300
GA_GENERATIONS = 800
GA_MUTATION_RATE = 0.08
GA_MUTATION_RATE_FLOOR = 0.005
GA_MUTATION_RATE_CEILING = 0.30
GA_EARLY_STOP_PATIENCE = 60
GA_EARLY_STOP_MIN_DELTA = 0.00005
GA_CROSSOVER_RATE = 0.75
GA_ELITISM_COUNT = 10
# Number of recent results used as features
GA_FEATURE_WINDOW = 15

# ── AI – Q-Learning (Reinforcement Learning) ─────────────────────────────────
QL_LEARNING_RATE = 0.15
QL_DISCOUNT_FACTOR = 0.95
QL_EPSILON = 0.3          # exploration rate (ε-greedy) — higher start for broader exploration
QL_EPSILON_DECAY = 0.9985
QL_EPSILON_MIN = 0.005
QL_REPLAY_PASSES = 5
QL_REPLAY_MIN_TD_ERROR = 0.00005
QL_REPLAY_EARLY_STOP_PATIENCE = 3
# State window: how many recent results to encode as the RL state
QL_STATE_WINDOW = 4

# ── AI – LSTM ────────────────────────────────────────────────────────────────
LSTM_SEQUENCE_LENGTH = 25
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 3
LSTM_EPOCHS = 400
LSTM_LEARNING_RATE = 0.002
LSTM_DROPOUT = 0.25
LSTM_BATCH_SIZE = 16
LSTM_VALIDATION_SPLIT = 0.15
LSTM_EARLY_STOP_PATIENCE = 20
LSTM_MIN_DELTA = 0.00005
LSTM_LR_FACTOR = 0.5
LSTM_LR_PATIENCE = 6

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
# Retry navigation when Betpawa returns the app shell or a partial render.
PAGE_FETCH_RETRIES = 3
PAGE_FETCH_RETRY_DELAY_SECONDS = 2
PAGE_FETCH_MIN_BODY_LENGTH = 100
# Run browser in headless mode (no visible browser window)
HEADLESS = True

# Path to Brave browser binary (Chromium-based, uses ChromeDriver)
BRAVE_BINARY_PATH = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"

# ── Web dashboard ────────────────────────────────────────────────────────────
WEB_HOST = "127.0.0.1"
WEB_PORT = 5050
