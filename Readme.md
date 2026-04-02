# Betpawa MUN Virtual-Sports Tracker

An AI-powered web scraper that tracks MUN (Manchester United) results from
[Betpawa virtual sports](https://www.betpawa.ug/virtual-sports?virtualTab=live&leagueId=7794)
and predicts the next match outcome using a **Genetic Algorithm** and
**Q-Learning (Reinforcement Learning)**.

---

## Screenshot

![GUI Screenshot](https://github.com/user-attachments/assets/2cf7da5c-41e9-4049-aa99-e0db5456bcee)

---

## Features

| Feature | Detail |
|---|---|
| 🔄 Web scraper | Brave/Selenium scraper; runs every **5 minutes** on clock boundaries and polls for up to **50 seconds** |
| 🗂 Historical backfill | Imports up to **10 previous seasons** from Betpawa Results/Matchday pages into the same AI training database |
| 🗄️ Database | SQLite – stores every MUN result with half-time and full-time scores |
| 🧬 Genetic Algorithm | Evolves a linear classifier over match-history features |
| 🤖 Q-Learning | Tabular RL agent that learns from the reward of each correct/wrong prediction |
| 🖥️ GUI | Tkinter desktop app showing live scraper status, results table, AI stats, and backend log |
| ⌨️ CLI | Full command-line interface (`start`, `scrape`, `results`, `train`, `predict`, `backfill`, `status`) |

---

## Result format

The scraper parses result strings of the form:

```
(0 - 0)0 - 3
 ↑ half-time  ↑ full-time (used by AI)
```

---

## Project structure

```
Betpawa-Tracker/
├── main.py           # Entry point – launches GUI or delegates to CLI
├── config.py         # All tunable constants (URL, intervals, AI hyper-params)
├── scraper.py        # Selenium scraper + scheduler
├── data_manager.py   # SQLite database manager
├── ai_model.py       # Genetic Algorithm + Q-Learning AI
├── gui.py            # Tkinter GUI
├── cli.py            # argparse CLI
├── tests.py          # Unit tests (45 tests)
└── requirements.txt  # Python dependencies
```

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> Tkinter is required for the GUI. Install it with your OS package manager if needed:
> ```bash
> sudo apt install python3-tk   # Debian/Ubuntu
> ```

### 2. Brave / ChromeDriver

The scraper uses **Brave Browser** in headless mode via Selenium's built-in
driver management. Update `BRAVE_BINARY_PATH` in `config.py` if Brave is
installed in a different location.

---

## Usage

### GUI (default)

```bash
python main.py
```

Launches the Tkinter desktop application. Use the **▶ Start** button to begin
the 5-minute scheduler, **⚡ Scrape Now** for an immediate session, and
**🧬 Train AI** / **🔮 Predict** once enough results are collected.

### CLI

```bash
# Start the continuous 5-minute scheduler
python main.py --cli start

# One-shot scrape
python main.py --cli scrape

# Show last 20 results
python main.py --cli results --count 20

# Train both AI models
python main.py --cli train --ga-generations 100

# Predict next MUN result
python main.py --cli predict

# Import historical seasons and matchdays for AI training
python main.py --cli backfill --seasons 10 --matchdays 38

# Status summary
python main.py --cli status
```

You can also call `cli.py` directly:

```bash
python cli.py start
python cli.py predict
```

---

## Configuration (`config.py`)

| Constant | Default | Description |
|---|---|---|
| `TARGET_URL` | betpawa.ug URL | Page to scrape |
| `RESULTS_URL` | betpawa.ug Results URL | Results page used to discover historical seasons |
| `TEAM_NAME` | `"MUN"` | Team to track |
| `SCRAPE_INTERVAL_SECONDS` | `300` | Seconds between sessions (5 min) |
| `SCRAPE_SESSION_DURATION_SECONDS` | `50` | How long each live session runs |
| `SCRAPE_POLL_INTERVAL_SECONDS` | `5` | Poll frequency within a session |
| `HISTORICAL_SEASONS_LIMIT` | `10` | Default number of seasons to import for backfill |
| `HISTORICAL_MATCHDAYS` | `38` | Default number of matchdays per season to import |
| `GA_POPULATION_SIZE` | `50` | GA population size |
| `GA_GENERATIONS` | `100` | Default training generations |
| `GA_MUTATION_RATE` | `0.1` | GA mutation rate |
| `QL_LEARNING_RATE` | `0.1` | Q-learning α |
| `QL_DISCOUNT_FACTOR` | `0.9` | Q-learning γ |
| `QL_EPSILON` | `0.2` | Initial exploration rate |
| `PAGE_FETCH_RETRIES` | `3` | Retries when a page renders partial or app-shell content |
| `PAGE_FETCH_RETRY_DELAY_SECONDS` | `2` | Base delay between page-load retries |
| `PAGE_FETCH_MIN_BODY_LENGTH` | `100` | Minimum rendered body length before parsing starts |
| `HEADLESS` | `True` | Run Brave headless |

---

## Tests

```bash
python -m unittest tests -v
```

50 unit tests covering result parsing, database CRUD, AI feature engineering,
GA/QL mechanics, and CLI argument parsing.
