# Agentic Newsvendor Study

An experimental platform for studying human-AI collaboration in newsvendor decision making.

## Quick Start

```bash
# Install dependencies with uv
uv sync

# Run the dashboard
uv run streamlit run dashboard/app.py
```

Or without uv:
```bash
pip install -e .
streamlit run dashboard/app.py
```

## Project Structure

```
AgenticNewsvendor/
├── src/                      # Core Python modules
│   ├── dgp/                  # Data generating process
│   │   ├── demand_model.py   # Demand computation
│   │   └── products.py       # Product configurations
│   ├── agent/                # AI agent system
│   │   └── responder.py      # Question-answer handling
│   ├── experiment/           # Experiment management
│   │   └── scenario_loader.py # Load scenario configs
│   └── tracking/             # Data collection
│       ├── models.py         # Data models
│       └── storage.py        # File-based storage
├── dashboard/                # Streamlit UI
│   └── app.py                # Main application
├── data/
│   ├── scenarios/            # Scenario definitions
│   │   └── scenario_backlog.yaml
│   └── results/              # Participant data (auto-created)
└── docs/                     # Documentation
    ├── EXPERIMENT_DESIGN_MASTER.md
    └── STUDY_PLAN_AND_ARCHITECTURE.md
```

## Scenarios

10 curated scenarios covering:
- **Products**: Fresh Salad, Ice Cream, Ready Meals, Bakery
- **Scenario types**: Trust AI (5) + Override Needed (5)
- **Hidden factors**: Sports events, school holidays, festivals, weather changes

## Data Storage

Results are stored as:
- **JSON**: Individual session files in `data/results/sessions/`
- **Parquet**: Aggregated export for analysis

## Development

```bash
# Run tests
python -c "
from src.experiment import ScenarioLoader
from pathlib import Path
loader = ScenarioLoader(Path('data/scenarios/scenario_backlog.yaml'))
print(f'Loaded {len(loader.load_all_scenarios())} scenarios')
"

# Export data
python -c "
from src.tracking import FileStorage
from pathlib import Path
storage = FileStorage(Path('data/results'))
path = storage.export_to_parquet()
print(f'Exported to {path}')
"
```

## Documentation

See `docs/` for:
- Study design and architecture
- Scenario specifications
- Research questions

