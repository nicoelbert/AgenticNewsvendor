import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from lets_plot import (
    LetsPlot,
    aes,
    element_text,
    facet_wrap,
    geom_line,
    geom_point,
    geom_text,
    ggplot,
    ggsize,
    ggtitle,
    layer_tooltips,
    scale_color_manual,
    scale_linetype_manual,
    theme,
    theme_minimal,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from agentic_system.langraph_backend import ChatMessage
from config.llm_loader import load_llm, load_prompt_template

LetsPlot.setup_html()

st.set_page_config(page_title="Inventory Planning Dashboard", layout="wide")

st.title("Inventory Planning Overview")
st.caption(
    "Experiment with setting optimal order quantities for perishable goods, supported by a forecast model and assistant explanations."
)

CONFIG_DEFAULT = ROOT_DIR / "config" / "experiment_run_001.yaml"


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, (int,)):
        return str(value)
    if value is None:
        return "—"
    return str(value)


def format_markdown_table(rows: List[List[Any]], headers: List[str]) -> str:
    if not rows:
        return "_No data available._"

    formatted_rows = [[_format_value(item) for item in row] for row in rows]
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in formatted_rows]
    return "\n".join([header_line, separator, *body_lines])


@st.cache_resource(show_spinner=False)
def get_prompt_template() -> str:
    return load_prompt_template("system_prompt_standard")


@st.cache_resource(show_spinner=False)
def get_llm():
    return load_llm()


def build_observations_table(data: pd.DataFrame) -> str:
    history = data[data["phase"] == "history"].tail(7)
    forecast = data[data["phase"] == "forecast"].head(7)

    rows: List[List[Any]] = []
    for row in history.itertuples():
        rows.append(
            [
                "Actual",
                row.date.strftime("%b %d"),
                row.demand,
            ]
        )
    for row in forecast.itertuples():
        rows.append(
            [
                "Forecast",
                row.date.strftime("%b %d"),
                row.forecast,
            ]
        )
    return format_markdown_table(rows, ["Type", "Date", "Value"])


def build_model_table(data: pd.DataFrame, product_cfg: Dict[str, Any]) -> str:
    feature_order: List[str] = []
    for key in ("visible_features", "hidden_features", "noise_features"):
        for feature in product_cfg.get(key, []):
            if feature not in feature_order:
                feature_order.append(feature)

    latest_row = data.iloc[-1]
    rows: List[List[Any]] = []
    betas = product_cfg.get("betas", {})
    for feature in feature_order:
        rows.append(
            [
                feature.replace("_", " ").title(),
                latest_row.get(feature, "—"),
                f"{betas.get(feature, 0.0):+.2f}",
            ]
        )

    base_level = product_cfg.get("base_level")
    if base_level is not None:
        rows.append(["Intercept", "—", f"{base_level:+.2f}"])

    return format_markdown_table(rows, ["Feature", "Current Value", "Coefficient"])


def invoke_assistant(system_prompt: str, user_message: str) -> str:
    try:
        llm = get_llm()
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        )
        content = getattr(response, "content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                else:
                    text_parts.append(str(part))
            content = "".join(text_parts)
        if not content:
            content = str(response)
        return content
    except Exception as exc:  # noqa: BLE001
        return f"Assistant error: {exc}"


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


@st.cache_data
def load_experiment_config(path_str: str) -> Dict:
    path = _resolve_path(path_str)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["experiment"]
    cfg["config_path"] = str(path)
    cfg["output_dir"] = str(_resolve_path(cfg["output_dir"]))
    return cfg


@st.cache_data
def load_assignment_data(output_dir: str, product: str, model: str) -> pd.DataFrame:
    file_path = Path(output_dir) / f"{product}_{model}.parquet"
    df = pd.read_parquet(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def format_product_name(product: str) -> str:
    return product.replace("_", " ").title()


def prepare_facet_frame(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    weekday_labels = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    rows = []
    for row in df.itertuples():
        weekday_label = weekday_labels[row.date.weekday()]
        rows.append(
            {
                "date": row.date,
                "value": row.demand if row.phase == "history" else None,
                "series": "Demand",
                "line_type": "Actual",
                "phase": row.phase,
                "weekday_label": weekday_label,
            }
        )
        rows.append(
            {
                "date": row.date,
                "value": row.forecast,
                "series": "Demand",
                "line_type": "Forecast",
                "phase": row.phase,
                "weekday_label": weekday_label,
            }
        )
        for feature in feature_columns:
            value = getattr(row, feature, None)
            rows.append(
                {
                    "date": row.date,
                    "value": value,
                    "series": feature.replace("_", " ").title(),
                    "line_type": "Feature",
                    "phase": row.phase,
                    "weekday_label": weekday_label,
                }
            )
    facet_df = pd.DataFrame(rows)
    facet_df = facet_df.dropna(subset=["value"])
    return facet_df


def make_facet_plot(facet_df: pd.DataFrame, title: str) -> Tuple[str, int]:
    if facet_df.empty:
        return ggplot().to_html(), 1

    tooltip = (
        layer_tooltips()
        .format("@date", "%b %d")
        .format("@value", ".2f")
        .line("@series")
        .line("Type: @line_type")
        .line("Phase: @phase")
        .line("Weekday: @weekday_label")
    )

    n_facets = facet_df["series"].nunique()
    weekday_df = (
        facet_df[facet_df["series"] == "Demand"]
        .drop_duplicates(subset=["date"], keep="last")
        .copy()
    )
    if not weekday_df.empty:
        demand_values = weekday_df["value"].astype(float)
        demand_min = demand_values.min()
        demand_max = demand_values.max()
        buffer = max(1.0, 0.05 * (demand_max - demand_min))
        weekday_df["weekday_y"] = demand_min - buffer
    else:
        weekday_df["weekday_y"] = 0.0

    color_palette = {"Actual": "#264653", "Forecast": "#e76f51", "Feature": "#2a9d8f"}
    linetype_palette = {"Actual": "solid", "Forecast": "dashed", "Feature": "solid"}

    plot = (
        ggplot(
            facet_df,
            aes(
                "date",
                "value",
                color="line_type",
                linetype="line_type",
                group="line_type",
            ),
        )
        + geom_line(size=1.3, tooltips=tooltip)
        + geom_point(size=2.4, tooltips=tooltip, alpha=0.85, show_legend=False)
        + geom_text(
            data=weekday_df,
            mapping=aes("date", "weekday_y", label="weekday_label"),
            color="#6c757d",
            size=10,
        )
        + facet_wrap("series", ncol=1, scales="free_y")
        + theme_minimal()
        + theme(
            text=element_text(size=15),
            axis_text_x=element_text(size=13),
            axis_text_y=element_text(size=13),
            strip_text=element_text(size=14),
            legend_text=element_text(size=13),
            legend_title=element_text(size=13),
            title=element_text(size=18),
            legend_position="top",
        )
        + scale_color_manual(values=color_palette)
        + scale_linetype_manual(values=linetype_palette)
        + ggsize(1080, max(340, 220 * n_facets))
        + ggtitle(title)
    )
    return plot.to_html(), n_facets


with st.sidebar:
    st.header("Scenario Loader")
    config_input = st.text_input(
        "Experiment config",
        value=str(CONFIG_DEFAULT),
        help="Path to the YAML file defining products, models, and horizons.",
    )
    try:
        config = load_experiment_config(config_input)
    except FileNotFoundError:
        st.error("Config file not found. Update the path to continue.")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load config: {exc}")
        st.stop()

    assignments = [(item["product"], item["model"]) for item in config["assignments"]]
    assignment_labels = [
        f"{format_product_name(prod)} · {model.upper()}" for prod, model in assignments
    ]
    selection = st.selectbox("Scenario", assignment_labels, index=0)
    selected_product, selected_model = assignments[assignment_labels.index(selection)]

    product_cfg = config["products"][selected_product]
    available_features = product_cfg.get("visible_features", [])
    selected_features = st.multiselect(
        "Show features",
        options=available_features,
        default=available_features,
        help="Toggle which contextual signals appear below the demand plot.",
    )

    st.caption(
        f"Output directory: `{config['output_dir']}`\n\n"
        "Re-run the generator notebook to refresh the Parquet files."
    )

data = load_assignment_data(config["output_dir"], selected_product, selected_model)
horizon_cfg = config["horizon"]
history_days = horizon_cfg["history_days"]
forecast_days = horizon_cfg["forecast_days"]
product_label = format_product_name(selected_product)
date_min = data["date"].min()
date_max = data["date"].max()

history_end = data[data["phase"] == "history"]["date"].max()
timeframe_text = (
    f"{date_min.strftime('%b %d')} – {date_max.strftime('%b %d, %Y')}"
    if pd.notna(date_min) and pd.notna(date_max)
    else "Configured horizon"
)

forecast_defaults = data[data["phase"] == "forecast"]["forecast"]
forecast_default_value = (
    float(forecast_defaults.mean()) if not forecast_defaults.empty else 0.0
)

history_data = data[data["phase"] == "history"]
if not history_data.empty:
    demand_nonzero = history_data["demand"].replace(0, np.nan)
    pct_errors = (
        history_data["forecast"] - history_data["demand"]
    ).abs() / demand_nonzero
    pct_errors = pct_errors.dropna()
    mean_pct_deviation = (
        float(pct_errors.mean() * 100) if not pct_errors.empty else None
    )
else:
    mean_pct_deviation = None

next_day_row = data[data["phase"] == "forecast"].head(1)
next_day_forecast = (
    float(next_day_row["forecast"].iloc[0]) if not next_day_row.empty else None
)
next_day_features = {
    feat: float(next_day_row[feat].iloc[0])
    for feat in selected_features
    if feat in next_day_row.columns and not next_day_row.empty
}

purchase_price = float(product_cfg.get("purchase_price", 0.0))
selling_price = float(product_cfg.get("selling_price", 0.0))
salvage_value = float(product_cfg.get("salvage_value", 0.0))
underage_cost = selling_price - purchase_price
overage_cost = purchase_price - salvage_value

neutral_color = "#222222"
model_color = "#e76f51"
feature_color = "#2a9d8f"

planner_cols = st.columns([0.65, 0.35], gap="large")

with planner_cols[0]:
    info_cols = st.columns([0.18, 0.18, 0.18, 0.18, 0.28], gap="small")
    info_cols[0].markdown(
        f"<div style='color:{neutral_color}; font-size:0.85rem;'>Purchase</div>"
        f"<div style='color:{neutral_color}; font-size:1.6rem; font-weight:600;'>€{purchase_price:.2f}</div>",
        unsafe_allow_html=True,
    )
    info_cols[1].markdown(
        f"<div style='color:{neutral_color}; font-size:0.85rem;'>Selling</div>"
        f"<div style='color:{neutral_color}; font-size:1.6rem; font-weight:600;'>€{selling_price:.2f}</div>",
        unsafe_allow_html=True,
    )
    forecast_value = (
        f"{next_day_forecast:.1f}" if next_day_forecast is not None else "—"
    )
    info_cols[2].markdown(
        f"<div style='color:{model_color}; font-size:0.85rem;'>Forecast</div>"
        f"<div style='color:{model_color}; font-size:1.6rem; font-weight:600;'>{forecast_value}</div>",
        unsafe_allow_html=True,
    )
    info_cols[3].markdown(
        f"<div style='color:{model_color}; font-size:0.85rem;'>Order</div>"
        f"<div style='color:{model_color}; font-size:1.6rem; font-weight:600;'>{forecast_value}</div>",
        unsafe_allow_html=True,
    )
    deviation = f"{mean_pct_deviation:.1f}%" if mean_pct_deviation is not None else "—"
    info_cols[4].markdown(
        f"<div style='color:{model_color}; font-size:0.85rem;'>Mean % deviation</div>"
        f"<div style='color:{model_color}; font-size:1.6rem; font-weight:600;'>{deviation}</div>",
        unsafe_allow_html=True,
    )

    feature_cols = st.columns(max(1, len(next_day_features)), gap="small")
    for idx, (feat, value) in enumerate(next_day_features.items()):
        feature_cols[idx].markdown(
            f"<div style='color:{feature_color}; font-size:0.8rem;'>{feat.replace('_', ' ').title()}</div>"
            f"<div style='color:{feature_color}; font-size:1.3rem; font-weight:600;'>{value:.1f}</div>",
            unsafe_allow_html=True,
        )
    if not next_day_features:
        st.markdown(
            f"<div style='color:{feature_color}; font-size:0.85rem;'>Next-day features</div>"
            "<div style='font-size:1.2rem; color:#666;'>No feature preview available.</div>",
            unsafe_allow_html=True,
        )

with planner_cols[1]:
    st.markdown(
        "<div style='font-size:0.95rem; font-weight:600; margin-bottom:0.6rem;'>Planning Input</div>",
        unsafe_allow_html=True,
    )
    with st.form("planning_form", clear_on_submit=False):
        expected_cols = st.columns(2, gap="small")
        with expected_cols[0]:
            expected_demand = st.number_input(
                "Expected demand",
                min_value=0.0,
                value=0.0,
                step=5.0,
            )
        with expected_cols[1]:
            order_quantity = st.number_input(
                "Order quantity",
                min_value=0.0,
                value=0.0,
                step=5.0,
            )
        submitted = st.form_submit_button("Submit", use_container_width=True)

if submitted:
    st.success(
        f"Submitted: order {order_quantity:.0f} vs. expected demand {expected_demand:.0f}."
    )

chart_col, chat_col = st.columns([0.7, 0.3], gap="medium")

with chart_col:
    st.subheader(f"Demand & Context · {product_label}")
    st.caption(
        f"{history_days} days actual + {forecast_days} days forecast · {selected_model.upper()} model\n"
        f"{timeframe_text}"
    )
    facet_features = [feat for feat in selected_features if feat in data.columns]
    facet_df = prepare_facet_frame(data, facet_features)
    facet_html, facet_count = make_facet_plot(
        facet_df,
        f"{product_label}: Demand & Signals",
    )
    chart_height = max(380, 230 * facet_count)
    st.components.v1.html(facet_html, height=chart_height, scrolling=False)

    with st.expander("Scenario data table", expanded=False):
        table_features = ["demand", "forecast"]
        table_features += [feat for feat in selected_features if feat in data.columns]
        table_features = [feat for feat in table_features if feat in data.columns]
        if table_features:
            table_matrix = data.set_index("date")[table_features].T
            month_labels = [idx.strftime("%b") for idx in table_matrix.columns]
            day_labels = [idx.strftime("%d") for idx in table_matrix.columns]
            table_matrix.columns = pd.MultiIndex.from_arrays([month_labels, day_labels])

            def _format_value_table(val):
                try:
                    return f"{float(val):.1f}"
                except (TypeError, ValueError):
                    return val

            rounded_matrix = table_matrix.applymap(_format_value_table)
            rounded_matrix.index = [
                name.replace("_", " ").title() for name in rounded_matrix.index
            ]
            st.dataframe(rounded_matrix)
        else:
            st.info("No features selected for the table view.")

with chat_col:
    st.markdown("### 🤖 Chat")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Welcome! I can explain how the model derived this forecast. Ask away!",
            }
        ]

    observations_table = build_observations_table(data)
    model_table_str = build_model_table(data, product_cfg)
    system_template = get_prompt_template()

    chat_history = cast(List[ChatMessage], st.session_state.chat_history)

    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask the forecast assistant…")
    if user_input:
        chat_history.append({"role": "user", "content": user_input})
        system_prompt = system_template.format(
            observations_table=observations_table,
            model_table=model_table_str,
        )
        assistant_reply = invoke_assistant(system_prompt, user_input)
        chat_history.append({"role": "assistant", "content": assistant_reply})
        st.session_state.chat_history = chat_history
        st.rerun()
