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
| 🔄 Web scraper | Selenium-based scraper; runs every **5 minutes** for **30 seconds** (polling every 5 s) |
| 🗄️ Database | SQLite – stores every MUN result with half-time and full-time scores |
| 🧬 Genetic Algorithm | Evolves a linear classifier over match-history features |
| 🤖 Q-Learning | Tabular RL agent that learns from the reward of each correct/wrong prediction |
| 🖥️ GUI | Tkinter desktop app showing live scraper status, results table, AI stats, and backend log |
| ⌨️ CLI | Full command-line interface (`start`, `scrape`, `results`, `train`, `predict`, `status`) |

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

### 2. Chrome / ChromeDriver

The scraper uses **Google Chrome** in headless mode.
[`webdriver-manager`](https://github.com/SergeyPirogov/webdriver_manager) downloads
the correct ChromeDriver automatically on first run.

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
| `TEAM_NAME` | `"MUN"` | Team to track |
| `SCRAPE_INTERVAL_SECONDS` | `300` | Seconds between sessions (5 min) |
| `SCRAPE_SESSION_DURATION_SECONDS` | `30` | How long each session runs |
| `SCRAPE_POLL_INTERVAL_SECONDS` | `5` | Poll frequency within a session |
| `GA_POPULATION_SIZE` | `50` | GA population size |
| `GA_GENERATIONS` | `100` | Default training generations |
| `GA_MUTATION_RATE` | `0.1` | GA mutation rate |
| `QL_LEARNING_RATE` | `0.1` | Q-learning α |
| `QL_DISCOUNT_FACTOR` | `0.9` | Q-learning γ |
| `QL_EPSILON` | `0.2` | Initial exploration rate |
| `HEADLESS` | `True` | Run Chrome headless |

---

## Tests

```bash
python -m unittest tests -v
```

45 unit tests covering result parsing, database CRUD, AI feature engineering,
GA/QL mechanics, and CLI argument parsing.
