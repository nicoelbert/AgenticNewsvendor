# AgenticNewsvendor Research Project

A streamlined research project for newsvendor problem analysis featuring synthetic data generation, a Streamlit dashboard, and a lightweight chat backend.

## 🎯 Project Overview

This project combines:
- **Data Generation**: Synthetic demand data creation for newsvendor problems
- **Chat Backend**: Minimal LangGraph logic ready for future LLM integration
- **Dashboard**: Interactive Streamlit dashboard for visualization and planning
- **Research Focus**: Clean, modular architecture suitable for experimentation

## 📁 Project Structure

```
AgenticNewsvendor/
├── data_generation/          # Synthetic data utilities
├── dashboard/               # Streamlit dashboard code
│   └── mvp_streamlit_dashboard.py  # MVP dashboard entry point
├── agentic_system/         # Chat backend logic (LangGraph)
│   └── langraph_backend.py
├── notebooks/             # Jupyter notebooks for experiments
├── data/                  # Generated data samples
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure LLM access (optional for forecast assistant)
cp config/private_config.example.yaml config/private_config.yaml
# then edit config/private_config.yaml with your Anthropic API key
```

### 2. Data Generation

```python
from data_generation import DataGenerator

# Create data generator
data_gen = DataGenerator(seed=42)

# Generate demand data
demand_data = data_gen.generate_demand_data(n_products=5, n_periods=50)

# Generate time series
ts_data = data_gen.generate_time_series('2024-01-01', '2024-12-31')
```

### 3. Create an experiment run

```python
from pathlib import Path
from data_generation import load_experiment

load_experiment(Path("config/experiment_run_001.yaml"))
# Parquet files written to the configured output directory
```

### 4. Dashboard

```bash
# Launch Streamlit dashboard
streamlit run dashboard/mvp_streamlit_dashboard.py
```

### 5. Jupyter Development

```bash
# Start Jupyter
jupyter notebook notebooks/data_exploration.ipynb
```

## 🔬 Research Features

### Data Generation
- **Time Series**: Synthetic time series with trend, seasonality, and noise
- **Demand Data**: Multi-product demand patterns for newsvendor analysis
- **Export Formats**: CSV, JSON, and Parquet support

### Dashboard
- **Lets-Plot visualizations** rendered in Streamlit
- **Chat prototype** backed by a lightweight LangGraph heuristic
- **Planning form** for forecasting and order inputs
- **Wide layout** combining assistant chat, plots, and planning tools

## 📊 Key Components

### 1. Data Generator (`data_generation/`)
- `DataGenerator.generate_demand_data()`: Create newsvendor demand datasets
- `DataGenerator.generate_time_series()`: Generate time series with patterns
- `DataGenerator.export_data()`: Export in multiple formats
- `load_experiment()`: Build product/model scenarios from YAML configs and persist outputs

### 2. Dashboard (`dashboard/`)
- `mvp_streamlit_dashboard.py`: Streamlit MVP combining chat, plots, and planning form
- Leverages `agentic_system/langraph_backend.py` for simple rule-based chat replies

## 🔧 Development Workflow

### For Testing and Prototyping:
1. Use Jupyter notebooks in `notebooks/` directory
2. Import project modules directly
3. Iterate quickly with interactive development

### For Production:
1. Extend data-generation utilities in `data_generation/`
2. Enhance the Streamlit dashboard with new components and logic
3. Integrate a full LLM-backed LangGraph agent when ready

## 📦 Dependencies

### Core Scientific Computing:
- NumPy, Pandas, SciPy
- Scikit-learn, Statsmodels

### Visualization:
- Matplotlib, Seaborn, Plotly
- Streamlit for dashboard
- Marimo for interactive notebooks/dashboards

### AI/ML:
- LangChain for agentic systems
- OpenAI integration support

### Development:
- Jupyter for interactive development
- Pytest for testing

## 🐳 Future: Docker Deployment

Docker configuration will be added for:
- Containerized development environment
- Streamlit dashboard deployment  
- Multi-service orchestration

## 📈 Research Applications

This framework supports research in:
- **Newsvendor Optimization**: Classical and modern approaches
- **Data-Driven Decision Making**: Interactive analysis and visualization
- **Chat-Augmented Workflows**: Prototype interactions powered by LangGraph heuristics

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏗️ Project Status

- ✅ **Core Architecture**: Data generation + dashboard modules in place
- ✅ **Data Generation**: Synthetic data creation ready
- ✅ **Dashboard Framework**: Streamlit integration prepared
- ✅ **Chat Prototype**: Rule-based LangGraph loop connected
- ✅ **Jupyter Integration**: Development notebooks available
- 🚧 **LLM Integration**: Hook in a real model when ready
- 🚧 **Advanced Optimization**: Enhanced algorithms planned

---

**Happy researching!** 🎉
