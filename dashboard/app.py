"""
Agentic Newsvendor Experiment - Main Streamlit App
Clean, research-appropriate design with chat interface
"""

import random
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment import ScenarioLoader
from src.tracking import FileStorage, ParticipantSession

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
SCENARIOS_PATH = DATA_DIR / "scenarios" / "scenario_backlog.yaml"
RESULTS_DIR = DATA_DIR / "results"

# Store locations
STORES = [
    "EDEKA Trabold, Würzburg Zellerau",
    "Tegut, Würzburg Sanderau",
]

# Page config
st.set_page_config(
    page_title="Grocery Ordering Study",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS - clean, tight layout with larger fonts
st.markdown(
    """
<style>
    /* Larger base font */
    html, body, .main, p, div { font-size: 16px !important; }

    /* Tight container spacing */
    .main .block-container {
        padding: 0.5rem 1rem 1rem 1rem;
        max-width: 1400px;
    }

    /* Kill all excess gaps */
    .element-container { margin-bottom: 0.2rem !important; }
    .stMarkdown { margin-bottom: 0 !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.3rem !important; }

    /* Section label */
    .section-label {
        font-size: 11px !important;
        font-weight: 700;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0 0 6px 0;
    }

    /* Scenario card */
    .scenario-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 12px 16px;
        border-radius: 10px;
        color: white;
        margin-bottom: 0;
    }
    .scenario-card h3 { margin: 0; font-weight: 600; font-size: 18px !important; }
    .scenario-card p { margin: 4px 0 0 0; opacity: 0.9; font-size: 14px !important; }

    /* Cost box */
    .cost-box {
        background: #f5f5f5;
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 14px !important;
    }

    /* Narrative */
    .narrative-box {
        background: #f8f9fa;
        padding: 14px 16px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0 0 12px 0;
        line-height: 1.55;
        font-size: 15px !important;
        color: #333;
    }

    /* Chat bubbles */
    .chat-bubble-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 14px;
        border-radius: 14px 14px 4px 14px;
        margin: 4px 0;
        max-width: 90%;
        margin-left: auto;
        font-size: 14px !important;
    }
    .chat-bubble-ai {
        background: #f0f0f0;
        color: #1a1a2e;
        padding: 10px 14px;
        border-radius: 14px 14px 14px 4px;
        margin: 4px 0;
        max-width: 90%;
        font-size: 14px !important;
        line-height: 1.5;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px 8px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-value { font-size: 28px !important; font-weight: 700; color: #667eea; }
    .metric-label { font-size: 11px !important; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 15px !important;
        border-radius: 8px;
    }
    .stButton > button:hover { box-shadow: 0 3px 10px rgba(102, 126, 234, 0.35); }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Data table - larger & bolder */
    .dataframe { font-size: 15px !important; }
    .dataframe th {
        padding: 12px 16px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    .dataframe td {
        padding: 10px 16px !important;
        font-weight: 500 !important;
    }
    /* Make index column (row headers) stand out */
    .dataframe th.row_heading {
        background-color: #f0f0f0 !important;
        font-weight: 600 !important;
        color: #444 !important;
    }

    /* Number inputs */
    .stNumberInput label { font-size: 14px !important; font-weight: 500; }
    .stNumberInput > div > div > input { padding: 10px; font-size: 16px !important; }

    /* Hide branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Kill gaps */
    .row-widget { margin-bottom: 0 !important; }
    hr { margin: 8px 0 !important; border-color: #e0e0e0; }

    /* Secondary buttons (suggestions) */
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 1.5px dashed #aaa !important;
        color: #555 !important;
        padding: 8px 16px !important;
        font-weight: 400 !important;
        font-size: 13px !important;
        border-radius: 20px !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(102, 126, 234, 0.1) !important;
        border-color: #667eea !important;
        border-style: solid !important;
        color: #667eea !important;
    }

    /* Container borders - tighter */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        padding: 10px !important;
        border-radius: 8px !important;
    }

    /* Technical log styling */
    .tech-log {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px;
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
        font-size: 12px !important;
        line-height: 1.5;
    }
    .tech-log-header {
        color: #64748b;
        font-size: 10px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
        padding-bottom: 6px;
        border-bottom: 1px solid #e2e8f0;
    }
    .tech-log-section {
        margin-bottom: 10px;
    }
    .tech-log-title {
        color: #475569;
        font-weight: 600;
        font-size: 11px !important;
        margin-bottom: 4px;
    }
    .tech-log-row {
        display: flex;
        justify-content: space-between;
        padding: 3px 0;
        border-bottom: 1px dotted #e2e8f0;
    }
    .tech-log-key {
        color: #64748b;
    }
    .tech-log-value {
        color: #1e293b;
        font-weight: 500;
    }
    .tech-log-coeff {
        color: #7c3aed;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.page = "welcome"
        st.session_state.session = None
        st.session_state.scenario_loader = None
        st.session_state.current_scenario = None
        st.session_state.current_trial = None
        st.session_state.chat_history = []
        st.session_state.storage = FileStorage(RESULTS_DIR)
        st.session_state.store_location = STORES[0]

        # Check for quick-start mode via query param: ?dashboard=true
        params = st.query_params
        if params.get("dashboard") == "true":
            # Auto-setup: random store, create session, skip to main task
            st.session_state.store_location = random.choice(STORES)
            st.session_state.session = ParticipantSession()
            loader = ScenarioLoader(SCENARIOS_PATH)
            st.session_state.scenario_loader = loader
            st.session_state.session.scenario_order = loader.get_randomized_order()
            st.session_state.page = "main_task"


def load_scenario_loader():
    """Load scenario loader if not already loaded."""
    if st.session_state.scenario_loader is None:
        st.session_state.scenario_loader = ScenarioLoader(SCENARIOS_PATH)
    return st.session_state.scenario_loader


def create_demand_chart(
    demand_history: list, ai_forecast: int = None, target_weekday: str = "Fr"
) -> go.Figure:
    """Create a clean, high-end demand chart with AI prediction."""
    days = ["Sa", "So", "Mo", "Di", "Mi", "Do", "Fr", "Sa", "So", "Mo", "Di", "Mi", "Do", "Fr"]

    fig = go.Figure()

    # Add area fill for historical data
    fig.add_trace(
        go.Scatter(
            x=list(range(len(demand_history))),
            y=demand_history,
            fill="tozeroy",
            fillcolor="rgba(102, 126, 234, 0.1)",
            line=dict(color="rgba(102, 126, 234, 0.3)", width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Add main line for historical data
    fig.add_trace(
        go.Scatter(
            x=list(range(len(demand_history))),
            y=demand_history,
            mode="lines+markers",
            line=dict(color="#667eea", width=3),
            marker=dict(size=8, color="#667eea", line=dict(width=2, color="white")),
            name="Nachfrage",
            hovertemplate="<b>%{text}</b><br>Nachfrage: %{y} Einheiten<extra></extra>",
            text=days[: len(demand_history)],
        )
    )

    # Use AI forecast if provided, else use last known value
    forecast_value = ai_forecast if ai_forecast is not None else demand_history[-1]

    # Calculate y-axis range with padding for KI label
    all_values = demand_history + [forecast_value]
    y_max = max(all_values) * 1.15  # 15% padding above max for label
    y_min = 0

    # Add dashed line from last actual point to AI prediction
    fig.add_trace(
        go.Scatter(
            x=[len(demand_history) - 1, len(demand_history)],
            y=[demand_history[-1], forecast_value],
            mode="lines",
            line=dict(color="#764ba2", width=2, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Add AI prediction marker (open circle)
    fig.add_trace(
        go.Scatter(
            x=[len(demand_history)],
            y=[forecast_value],
            mode="markers+text",
            marker=dict(size=16, color="#764ba2", symbol="circle-open", line=dict(width=3)),
            name="KI-Prognose",
            text=["KI-Prognose"],
            textposition="top center",
            textfont=dict(size=10, color="#764ba2", family="Arial"),
            hovertemplate=f"<b>{target_weekday} (KI-Prognose)</b><br>Vorhersage: {forecast_value} Einheiten<extra></extra>",
        )
    )

    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            ticktext=days[: len(demand_history)] + [target_weekday],
            tickvals=list(range(len(demand_history) + 1)),
            tickfont=dict(size=11, color="#666"),
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#666"),
            gridcolor="rgba(0,0,0,0.05)",
            zeroline=False,
            range=[y_min, y_max],
        ),
        showlegend=False,
        hovermode="x unified",
    )

    return fig


def render_chat_message(role: str, content: str):
    """Render a chat message bubble."""
    if role == "user":
        st.markdown(f'<div class="chat-bubble-user">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-ai">{content}</div>', unsafe_allow_html=True)


def page_welcome():
    """Welcome/consent page."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            '<p class="main-header">🛒 Bestellentscheidungen im Supermarkt</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="sub-header">Eine Studie zur Mensch-KI-Zusammenarbeit</p>',
            unsafe_allow_html=True,
        )

        st.markdown("""
        ### Willkommen!

        In dieser Studie übernehmen Sie die Rolle eines **Filialleiters**, der frische
        Lebensmittel für einen Supermarkt bestellt. Ihr Ziel ist es, durch die richtige
        Bestellmenge den **Gewinn zu maximieren**.

        Sie werden:
        - Informationen über Wetter, Wochentag und besondere Ereignisse sehen
        - Prognosen und Empfehlungen von einem **KI-Assistenten** erhalten
        - Entscheiden, wie viele Einheiten Sie bestellen möchten

        Der KI-Assistent kann Ihnen helfen, seine Vorhersagen zu verstehen – aber er weiß
        nicht alles. Nutzen Sie Ihr eigenes Urteilsvermögen!

        **Dauer**: Ca. 20-25 Minuten

        ---
        """)

        # Store selection
        st.session_state.store_location = st.selectbox(
            "Wählen Sie Ihren Standort:",
            STORES,
            index=0,
        )

        st.markdown("---")

        if st.button("▶️ Studie starten", use_container_width=True, type="primary"):
            st.session_state.session = ParticipantSession()
            loader = load_scenario_loader()
            st.session_state.session.scenario_order = loader.get_randomized_order()
            st.session_state.page = "instructions"
            st.rerun()


def page_instructions():
    """Instructions page."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<p class="main-header">📋 Anleitung</p>', unsafe_allow_html=True)

        st.markdown("""
        ### Ihre Aufgabe

        Sie bestellen **frische Lebensmittel** (Salat, Eis, Fertiggerichte, Backwaren).
        Diese Produkte sind verderblich – nicht verkaufte Ware verliert an Wert.

        ### Kostenstruktur

        | | Bedeutung |
        |---|---|
        | **Einkaufspreis** | Was Sie pro Einheit bezahlen |
        | **Verkaufspreis** | Was Kunden zahlen |
        | **Restwert** | Was Sie für unverkaufte Ware erhalten |

        **Ihr Gewinn hängt ab von:**
        - ✅ Verkaufte Einheiten = Verkaufspreis − Einkaufspreis
        - ❌ Unverkaufte Einheiten = Restwert − Einkaufspreis (meist Verlust)
        - ❌ Fehlmengen = Entgangener Gewinn

        ### Der KI-Assistent

        Die KI liefert:
        - Eine **Nachfrageprognose** (erwartete Verkaufsmenge)
        - Eine **Bestellempfehlung**

        Sie können der KI **Fragen stellen**, um ihre Logik zu verstehen.
        Die KI ist hilfreich, aber sie kennt nicht alle Faktoren!

        ---
        """)

        st.markdown("### Verständnisfragen")

        q1 = st.radio(
            "Was passiert mit unverkaufter Ware?",
            [
                "Wird zum vollen Preis am nächsten Tag verkauft",
                "Wird entsorgt oder zum reduzierten Restwert verkauft",
                "Wird an den Lieferanten zurückgegeben",
            ],
            index=None,
            key="q1",
        )

        q2 = st.radio(
            "Hat der KI-Assistent Zugang zu ALLEN Informationen?",
            ["Ja", "Nein"],
            index=None,
            key="q2",
        )

        q3 = st.radio(
            "Was ist Ihr Ziel?",
            [
                "Der KI-Empfehlung immer genau folgen",
                "Den Gewinn durch die richtige Bestellmenge maximieren",
                "So wenig wie möglich bestellen",
            ],
            index=None,
            key="q3",
        )

        correct = (
            q1 == "Wird entsorgt oder zum reduzierten Restwert verkauft"
            and q2 == "Nein"
            and q3 == "Den Gewinn durch die richtige Bestellmenge maximieren"
        )

        if q1 and q2 and q3 and not correct:
            st.warning("Einige Antworten sind nicht korrekt. Bitte lesen Sie die Anleitung erneut.")

        st.markdown("---")

        if st.button(
            "Weiter zu den Szenarien →",
            use_container_width=True,
            type="primary",
            disabled=not correct,
        ):
            st.session_state.page = "main_task"
            st.rerun()


def page_main_task():
    """Main task page with chat interface."""
    from src.agent import AgentResponder
    from src.dgp import DemandModel, ProductConfig
    from src.tracking import TrialRecord

    session = st.session_state.session
    loader = load_scenario_loader()

    if session.current_trial >= len(session.scenario_order):
        st.session_state.page = "complete"
        st.rerun()
        return

    scenario_id = session.scenario_order[session.current_trial]
    scenario = loader.load_scenario(scenario_id)

    # Initialize trial if needed
    if (
        st.session_state.current_trial is None
        or st.session_state.current_trial.scenario_id != scenario_id
    ):
        st.session_state.current_trial = TrialRecord(
            participant_id=session.participant_id,
            scenario_id=scenario_id,
            trial_number=session.current_trial + 1,
            start_time=datetime.now(),
            product=scenario.product,
            ai_forecast=scenario.ai_forecast,
            ai_recommendation=scenario.ai_recommendation,
            true_expected_demand=scenario.true_expected_demand,
            actual_demand=scenario.actual_demand,
            optimal_order=scenario.optimal_order,
        )
        st.session_state.chat_history = []

    agent = AgentResponder(scenario.full_config)

    # Progress bar with inline text
    progress = (session.current_trial + 1) / len(session.scenario_order)
    st.progress(progress)
    st.caption(f"Szenario {session.current_trial + 1} von {len(session.scenario_order)}")

    # === PREPARE DATA FIRST ===
    visible = scenario.visible_features
    last_friday_demand = (
        scenario.demand_history[6]
        if len(scenario.demand_history) > 6
        else scenario.demand_history[-1]
    )
    today_demand = scenario.demand_history[-1]

    # Calculate deltas
    temp_last_week = visible.get("temperature", 20) - 4
    temp_today = visible.get("temperature", 20) - 2
    temp_tomorrow = visible.get("temperature", 20)

    demand_change = today_demand - last_friday_demand
    demand_indicator = (
        f"▲ +{demand_change}"
        if demand_change > 0
        else (f"▼ {demand_change}" if demand_change < 0 else "—")
    )

    temp_change = temp_tomorrow - temp_today
    temp_indicator = (
        f"▲ +{temp_change}°"
        if temp_change > 0
        else (f"▼ {temp_change}°" if temp_change < 0 else "—")
    )

    # Weekday mapping
    weekday_map = {
        "Montag": "Mo",
        "Dienstag": "Di",
        "Mittwoch": "Mi",
        "Donnerstag": "Do",
        "Freitag": "Fr",
        "Samstag": "Sa",
        "Sonntag": "So",
        "Monday": "Mo",
        "Tuesday": "Di",
        "Wednesday": "Mi",
        "Thursday": "Do",
        "Friday": "Fr",
        "Saturday": "Sa",
        "Sunday": "So",
    }
    weekday_raw = visible.get("weekday", "Fr")
    weekday_short = weekday_map.get(
        weekday_raw, weekday_raw[:2] if len(weekday_raw) > 2 else weekday_raw
    )

    # Build table
    conditions_df = pd.DataFrame(
        {
            "": ["Nachfrage", "Temperatur", "Regen", "Wochentag"],
            "Vorwoche": [f"{last_friday_demand}", f"{temp_last_week}°C", "Nein", weekday_short],
            "Heute": [f"{today_demand}", f"{temp_today}°C", "Nein", "—"],
            "Morgen": [
                "?",
                f"{temp_tomorrow}°C",
                "Ja" if visible.get("rain") else "Nein",
                weekday_short,
            ],
            "Δ": [demand_indicator, temp_indicator, "—", "—"],
        }
    )

    def color_delta(val):
        if "▲" in str(val):
            return "color: #16a34a; font-weight: bold"
        elif "▼" in str(val):
            return "color: #dc2626; font-weight: bold"
        return "color: #999"

    styled_df = (
        conditions_df.set_index("")
        .style.map(color_delta, subset=["Δ"])
        .set_properties(**{"font-size": "14px", "font-weight": "500"})
        .set_properties(
            subset=["Morgen"],
            **{"background-color": "#f3e8ff", "font-weight": "600", "color": "#7c3aed"},
        )
        .set_properties(
            subset=["Vorwoche", "Heute"], **{"background-color": "#f8fafc", "color": "#475569"}
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#1e293b"),
                        ("color", "white"),
                        ("font-weight", "700"),
                        ("padding", "10px 12px"),
                        ("font-size", "11px"),
                        ("text-transform", "uppercase"),
                        ("letter-spacing", "0.5px"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [("padding", "10px 12px"), ("border-bottom", "1px solid #e2e8f0")],
                },
                {
                    "selector": "th.row_heading",
                    "props": [
                        ("background-color", "#f1f5f9"),
                        ("color", "#334155"),
                        ("font-weight", "600"),
                        ("text-transform", "none"),
                    ],
                },
                {
                    "selector": "th.col3",
                    "props": [("background-color", "#7c3aed")],
                },  # Morgen header purple
            ]
        )
    )

    # === HEADER: Scenario card (full width) ===
    st.markdown(
        f"""
    <div class="scenario-card">
        <h3>🛒 {scenario.product_display_name}</h3>
        <p>{scenario.date} · {st.session_state.store_location}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # === SITUATION (full width) ===
    st.markdown('<p class="section-label">Situation</p>', unsafe_allow_html=True)
    narrative = scenario.narrative.replace("Munich", "Würzburg").replace("München", "Würzburg")
    narrative = narrative.replace(
        "suburban supermarket", st.session_state.store_location.split(",")[0]
    )
    st.markdown(f'<div class="narrative-box">{narrative}</div>', unsafe_allow_html=True)

    # === MAIN 2-COLUMN LAYOUT ===
    col_left, col_right = st.columns([5, 4], gap="small")

    # --- LEFT COLUMN: Table+AI (top) + Chart (bottom) ---
    with col_left:
        # Table + AI side by side
        st.markdown(
            '<p class="section-label">Bedingungen & KI-Assistent</p>', unsafe_allow_html=True
        )
        tbl_col, ai_col = st.columns([4, 2], gap="small")
        with tbl_col:
            st.dataframe(styled_df, use_container_width=True, hide_index=False)
        with ai_col:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{scenario.ai_forecast}</div>
                <div class="metric-label">KI-Prognose</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{scenario.ai_recommendation}</div>
                <div class="metric-label">KI-Empfehlung</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Chart below
        st.markdown(
            '<p class="section-label" style="margin-top:12px;">Nachfrageverlauf (14 Tage)</p>',
            unsafe_allow_html=True,
        )
        fig = create_demand_chart(
            scenario.demand_history, ai_forecast=scenario.ai_forecast, target_weekday=weekday_short
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Technical Log (collapsible) - shows model documentation
        with st.expander("📋 Modelldokumentation", expanded=False):
            # Map beta names to German labels
            beta_labels = {
                "temperature": ("Temperatur", f"{visible.get('temperature', 20)}°C"),
                "rain": ("Regen", "Ja" if visible.get("rain") else "Nein"),
                "weekday_friday": ("Wochentag Fr", "Ja" if visible.get("weekday") in ["Freitag", "Friday"] else "Nein"),
                "weekday_saturday": ("Wochentag Sa", "Ja" if visible.get("weekday") in ["Samstag", "Saturday"] else "Nein"),
                "weekday_sunday": ("Wochentag So", "Ja" if visible.get("weekday") in ["Sonntag", "Sunday"] else "Nein"),
                "promotion": ("Aktion", "Ja" if visible.get("promotion") else "Nein"),
            }

            # Build coefficient display
            coeff_rows = ""
            for beta_name, coeff_value in scenario.visible_betas.items():
                if beta_name in beta_labels:
                    label, current_val = beta_labels[beta_name]
                    sign = "+" if coeff_value >= 0 else ""
                    coeff_rows += f'''
                    <div class="tech-log-row">
                        <span class="tech-log-key">{label}</span>
                        <span class="tech-log-value">{current_val}</span>
                        <span class="tech-log-coeff">{sign}{coeff_value:.1f}</span>
                    </div>'''

            # Calculate confidence interval (±15%)
            ci_low = int(scenario.ai_forecast * 0.85)
            ci_high = int(scenario.ai_forecast * 1.15)

            st.markdown(
                f'''
                <div class="tech-log">
                    <div class="tech-log-header">PROGNOSEMODELL v2.3 · Training: POS 2022-2024</div>

                    <div class="tech-log-section">
                        <div class="tech-log-title">INPUT-FEATURES & KOEFFIZIENTEN</div>
                        <div class="tech-log-row">
                            <span class="tech-log-key">Basisnachfrage</span>
                            <span class="tech-log-value">—</span>
                            <span class="tech-log-coeff">{scenario.base_level:.0f}</span>
                        </div>
                        {coeff_rows}
                    </div>

                    <div class="tech-log-section">
                        <div class="tech-log-title">OUTPUT</div>
                        <div class="tech-log-row">
                            <span class="tech-log-key">prognose</span>
                            <span class="tech-log-value">{scenario.ai_forecast} Einheiten</span>
                            <span class="tech-log-coeff"></span>
                        </div>
                        <div class="tech-log-row">
                            <span class="tech-log-key">konfidenz_70</span>
                            <span class="tech-log-value">[{ci_low}, {ci_high}]</span>
                            <span class="tech-log-coeff"></span>
                        </div>
                    </div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

    # --- RIGHT COLUMN: Decision+Cost (top) + Chat (bottom) ---
    with col_right:
        # Decision + Cost side by side
        st.markdown('<p class="section-label">Ihre Entscheidung</p>', unsafe_allow_html=True)
        dec_col, cost_col = st.columns([1, 1], gap="small")

        with dec_col:
            forecast = st.number_input(
                "Prognose",
                min_value=1,
                max_value=500,
                value=None,
                key=f"forecast_{scenario_id}",
                label_visibility="collapsed",
                placeholder="Prognose",
            )
            order = st.number_input(
                "Bestellung",
                min_value=0,
                max_value=500,
                value=None,
                key=f"order_{scenario_id}",
                label_visibility="collapsed",
                placeholder="Bestellung",
            )
            can_submit = forecast is not None and order is not None
            if st.button(
                "Weiter →",
                use_container_width=True,
                type="primary",
                disabled=not can_submit,
                key=f"submit_{scenario_id}",
            ):
                trial = st.session_state.current_trial
                trial.participant_forecast = forecast
                trial.participant_order = order
                trial.end_time = datetime.now()
                trial.compute_metrics()
                product_config = ProductConfig(
                    name=scenario.product,
                    display_name=scenario.product_display_name,
                    base_level=0,
                    price=scenario.price,
                    cost=scenario.cost,
                    salvage=scenario.salvage,
                    profit_per_unit=scenario.profit_per_unit,
                    loss_per_unit=scenario.loss_per_unit,
                    noise_std=0,
                )
                model = DemandModel(product_config)
                trial.profit = model.compute_profit(order, scenario.actual_demand)
                trial.optimal_profit = model.compute_profit(
                    scenario.optimal_order, scenario.actual_demand
                )
                session.add_trial(trial)
                st.session_state.storage.save_session(session)
                st.session_state.current_trial = None
                st.session_state.chat_history = []
                st.rerun()

        with cost_col:
            st.markdown(
                f"""
            <div class="cost-box">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span>Einkauf:</span><span style="font-weight:600;">€{scenario.cost:.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span>Verkauf:</span><span style="font-weight:600;">€{scenario.price:.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; border-top:1px solid #ddd; padding-top:6px;">
                    <span style="color:#2a2;">Gewinn:</span><span style="font-weight:700; color:#2a2;">+€{scenario.profit_per_unit:.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#c00;">Verlust:</span><span style="font-weight:700; color:#c00;">−€{scenario.loss_per_unit:.2f}</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Chat interface (still inside col_right)
        st.markdown(
            '<p class="section-label" style="margin-top:12px;">Fragen an die KI</p>',
            unsafe_allow_html=True,
        )

        # Get questions first
        questions = agent.get_available_questions()

        # Grey chat box using container with border
        chat_container = st.container(border=True)
        with chat_container:
            # Chat history display
            if st.session_state.chat_history:
                for msg in st.session_state.chat_history:
                    render_chat_message(msg["role"], msg["content"])

        # Generate numeric-based responses
        def get_numeric_response(q_id: str) -> str:
            uncertainty = scenario.ai_forecast * 0.15  # ±15%
            low = int(scenario.ai_forecast - uncertainty)
            high = int(scenario.ai_forecast + uncertainty)

            responses = {
                "explain_forecast": f"""Meine Prognose von **{scenario.ai_forecast} Einheiten** setzt sich zusammen aus:

• Basisnachfrage: ~{int(scenario.ai_forecast * 0.7)} Einheiten
• Temperatureffekt ({visible.get("temperature", 20)}°C): +{int(scenario.ai_forecast * 0.15)} Einheiten
• Wochentageffekt ({visible.get("weekday", "N/A")}): +{int(scenario.ai_forecast * 0.08)} Einheiten
• Sonstige Faktoren: +{int(scenario.ai_forecast * 0.07)} Einheiten""",
                "confidence": f"""Meine **Konfidenz** für diese Prognose:

• Erwartete Nachfrage: **{scenario.ai_forecast}** Einheiten
• 70%-Konfidenzintervall: **{low} - {high}** Einheiten
• Historische Genauigkeit: ±{int(uncertainty)} Einheiten in {int(70 + scenario.ai_forecast % 10)}% der Fälle

Die Unsicherheit kommt hauptsächlich von täglichen Schwankungen.""",
                "what_factors": f"""Mein Modell berücksichtigt folgende **Faktoren**:

✅ **Sichtbar für mich:**
• Temperatur (aktuell: {visible.get("temperature", 20)}°C) → Koeffizient: +0.8 pro °C
• Regen (aktuell: {"Ja" if visible.get("rain") else "Nein"}) → Koeffizient: -5.0
• Wochentag (aktuell: {visible.get("weekday", "N/A")}) → Koeffizient: +4.0
• Preis (aktuell: €{visible.get("price", 0):.2f})

❌ **Nicht verfügbar:**
• Lokale Veranstaltungen
• Schulferien
• Wettbewerberaktionen""",
                "why_order_more": f"""Ich empfehle **{scenario.ai_recommendation}** statt {scenario.ai_forecast} Einheiten wegen der **Kostenasymmetrie**:

• Entgangener Gewinn pro Fehlmenge: **€{scenario.profit_per_unit:.2f}**
• Verlust pro unverkaufter Einheit: **€{scenario.loss_per_unit:.2f}**

Da Fehlmengen teurer sind ({scenario.profit_per_unit:.2f} > {scenario.loss_per_unit:.2f}),
ist es besser, etwas mehr zu bestellen. Die optimale Bestellmenge liegt beim
**{int(scenario.profit_per_unit / (scenario.profit_per_unit + scenario.loss_per_unit) * 100)}. Perzentil** der erwarteten Nachfrage.""",
                "what_missing": f"""Folgende Informationen habe ich **nicht**:

❌ Lokale Veranstaltungen (Sport, Festivals, Konzerte)
❌ Schulferienkalender
❌ Aktivitäten von Wettbewerbern
❌ Kurzfristige Wetteränderungen
❌ Besondere Kundenanfragen

Wenn Sie von solchen Faktoren wissen, sollten Sie meine Prognose von
**{scenario.ai_forecast}** Einheiten entsprechend anpassen.""",
            }

            # Specific questions
            if "school_holiday" in q_id or "holiday" in q_id:
                return f"""Ich habe **keinen Zugang** zu Schulferienkalendern.

Meine Prognose von {scenario.ai_forecast} Einheiten basiert nur auf Wetter und Wochentag.

Schulferien können die Nachfrage um **+20-40%** beeinflussen, besonders bei:
• Eis und Süßwaren (Kinder zu Hause)
• Familienprodukte

Falls Schulferien sind, empfehle ich eine Anpassung nach oben."""

            if "football" in q_id or "match" in q_id or "sport" in q_id:
                return f"""Ich habe **keine Daten** zu Sportveranstaltungen.

Bei Fußballspielen kann die Nachfrage variieren:
• Fertiggerichte: **+30-50%** (Spielabend zu Hause)
• Snacks & Getränke: **+20-40%**
• Frische Salate: **+5-10%** (geringerer Effekt)

Meine aktuelle Prognose von {scenario.ai_forecast} Einheiten berücksichtigt dies nicht."""

            if "festival" in q_id:
                return f"""Ich tracke **keine lokalen Veranstaltungen** wie Festivals.

Straßenfeste in der Nähe können bedeuten:
• Mehr Laufkundschaft: **+15-30%** Nachfrage
• Aber auch: Konkurrenz durch Essensstände

Meine Prognose von {scenario.ai_forecast} Einheiten geht von einem normalen Tag aus."""

            if "market" in q_id or "farmers" in q_id:
                return f"""Ich habe **keine Informationen** über Wochenmärkte.

Ein nahegelegener Markt mit Bäckerei-/Gemüseständen kann bedeuten:
• Weniger Kunden für ähnliche Produkte: **-20-30%**
• Besonders betroffen: Brot, frisches Obst/Gemüse

Meine Prognose von {scenario.ai_forecast} könnte zu hoch sein, wenn ein Markt stattfindet."""

            if "weather" in q_id:
                return f"""Ich verwende die **offizielle Wettervorhersage** von heute Morgen:
• Temperatur: {visible.get("temperature", 20)}°C
• Regen: {"Ja" if visible.get("rain") else "Nein"}

⚠️ Ich aktualisiere mich nicht in Echtzeit. Falls Sie aktuellere
Wetterinformationen haben, sollten Sie diese berücksichtigen.

Regen reduziert die Salat-Nachfrage um ca. **-15%**."""

            return responses.get(q_id, agent.answer_question(q_id))

        # German question suggestions (chatbot-style)
        german_questions = {
            "explain_forecast": "Wie kam die Prognose zustande?",
            "confidence": "Wie sicher bist du?",
            "what_factors": "Welche Faktoren?",
            "why_order_more": "Warum mehr bestellen?",
            "what_missing": "Was weißt du nicht?",
            "consider_school_holiday": "Schulferien?",
            "consider_football_match": "Sportevents?",
            "consider_street_festival": "Fest in der Nähe?",
            "consider_farmers_market": "Wochenmarkt?",
            "consider_weather": "Wetter aktuell?",
        }

        # Chat container with suggestions
        with chat_container:
            # Suggestion chips - full width, subtle dashed style
            if not st.session_state.chat_history:
                st.markdown(
                    '<p style="color:#666; font-size:0.75rem; margin:0.3rem 0;">Vorschläge:</p>',
                    unsafe_allow_html=True,
                )

            # Full-width suggestion buttons
            available_qs = questions[:5]
            for q in available_qs:
                q_text = german_questions.get(q.id, q.text)
                btn_key = f"q_{q.id}_{scenario_id}"
                if st.button(
                    f"💬 {q_text}", key=btn_key, use_container_width=True, type="secondary"
                ):
                    st.session_state.chat_history.append({"role": "user", "content": q_text})
                    response = get_numeric_response(q.id)
                    st.session_state.chat_history.append({"role": "ai", "content": response})
                    st.session_state.current_trial.questions_asked.append(q.id)
                    st.session_state.current_trial.question_timestamps.append(
                        datetime.now().isoformat()
                    )
                    st.rerun()

            # Text input for custom questions
            user_input = st.text_input(
                "Eigene Frage",
                placeholder="Oder eigene Frage eingeben...",
                key=f"chat_input_{scenario_id}",
                label_visibility="collapsed",
            )
            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append(
                    {
                        "role": "ai",
                        "content": f"Diese Frage kann ich leider nicht beantworten. Nutzen Sie die Vorschläge oben für Details zu meiner Prognose von {scenario.ai_forecast} Einheiten.",
                    }
                )
                st.rerun()


def page_complete():
    """Completion page."""
    session = st.session_state.session
    session.completed = True
    session.end_time = datetime.now()
    st.session_state.storage.save_session(session)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<p class="main-header">✅ Studie abgeschlossen!</p>', unsafe_allow_html=True)
        st.markdown("### Vielen Dank für Ihre Teilnahme!")

        st.markdown("---")

        total_profit = sum(t.profit or 0 for t in session.trials)
        total_optimal = sum(t.optimal_profit or 0 for t in session.trials)
        questions_total = sum(len(t.questions_asked) for t in session.trials)

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Szenarien", len(session.trials))
        with col_s2:
            st.metric("Gesamtgewinn", f"€{total_profit:.2f}")
        with col_s3:
            st.metric("Fragen gestellt", questions_total)

        if total_optimal > 0:
            efficiency = (total_profit / total_optimal) * 100
            st.metric(
                "Effizienz",
                f"{efficiency:.1f}%",
                help="Ihr Gewinn im Vergleich zum optimalen Gewinn",
            )

        st.markdown(f"""
        ---
        **Teilnehmer-ID**: `{session.participant_id}`

        Bei Fragen wenden Sie sich an das Forschungsteam.
        """)


def main():
    init_session_state()

    if st.session_state.page == "welcome":
        page_welcome()
    elif st.session_state.page == "instructions":
        page_instructions()
    elif st.session_state.page == "main_task":
        page_main_task()
    elif st.session_state.page == "complete":
        page_complete()


if __name__ == "__main__":
    main()
