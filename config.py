"""
Configuration constants for the Betpawa Tracker application.
"""

# ── Target URL & team ────────────────────────────────────────────────────────
TARGET_URL = (
    "https://www.betpawa.ug/virtual-sports"
    "?virtualTab=live&leagueId=7794"
)
TEAM_NAME = "MUN"

# ── Scraper timing ───────────────────────────────────────────────────────────
# How often (seconds) to start a new scraping session
SCRAPE_INTERVAL_SECONDS = 300       # 5 minutes

# How long (seconds) each scraping session lasts
SCRAPE_SESSION_DURATION_SECONDS = 30

# How often (seconds) to poll within a session (page refreshes every ~5 s)
SCRAPE_POLL_INTERVAL_SECONDS = 5

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
# Run Chrome in headless mode (no visible browser window)
HEADLESS = True
