"""Getaround Delay Analysis Dashboard.

Interactive dashboard to help PM decide on minimum delay threshold between rentals.
Answers key questions about delay impact and revenue implications.
Includes a pricing prediction section powered by the ML API.
"""

import logging
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config.settings import get_settings

# Setup
DATA_PATH = (
    Path(__file__).parent.parent.parent / "data" / "get_around_delay_analysis.csv"
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Getaround Delay Analysis",
    page_icon="GA",
    layout="wide",
)

st.title("Getaround Delay Analysis Dashboard")
st.markdown(
    """
When using Getaround, drivers book cars for a specific time period.
Late returns at checkout can generate high friction for the next driver
if the car was supposed to be rented again on the same day.

To mitigate this, we evaluate implementing a **minimum delay between
two consecutive rentals**. The Product Manager needs to decide:

- **Threshold**: how long should the minimum delay be?
- **Scope**: should it apply to all cars or only Connect cars?

Use the **sidebar controls** to adjust threshold and scope, and explore
how different scenarios impact rentals across sections 1 to 4.
"""
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and cache the delay analysis data.

    Returns:
        DataFrame with rental delay data.
    """
    return pd.read_csv(DATA_PATH)


@st.cache_data
def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare derived datasets for analysis.

    Args:
        df: Raw rental data.

    Returns:
        Tuple of (ended_rentals, consecutive_rentals, consecutive_with_delay).
    """
    # Ended rentals with delay info
    ended = df[df["state"] == "ended"].copy()
    ended["is_late"] = ended["delay_at_checkout_in_minutes"] > 0

    # Rentals with a previous rental on the same car
    consecutive = df[df["previous_ended_rental_id"].notna()].copy()

    # Merge to get previous rental delay info
    prev_delays = df[["rental_id", "delay_at_checkout_in_minutes"]].copy()
    prev_delays.columns = ["previous_ended_rental_id", "previous_delay"]
    consecutive = consecutive.merge(
        prev_delays, on="previous_ended_rental_id", how="left"
    )

    # Calculate effective time and problematic flag
    consecutive["effective_time"] = consecutive[
        "time_delta_with_previous_rental_in_minutes"
    ] - consecutive["previous_delay"].fillna(0)
    consecutive["is_problematic"] = consecutive["effective_time"] < 0

    return ended, consecutive, consecutive[consecutive["is_problematic"]]


def simulate_threshold(data: pd.DataFrame, threshold_minutes: int, scope: str) -> dict:
    """Simulate the impact of a minimum delay threshold.

    Args:
        data: DataFrame with consecutive rentals.
        threshold_minutes: Minimum required gap between rentals.
        scope: 'All rentals', 'Connect only', or 'Mobile only'.

    Returns:
        Dict with simulation results.
    """
    if scope == "Connect only":
        subset = data[data["checkin_type"] == "connect"].copy()
    elif scope == "Mobile only":
        subset = data[data["checkin_type"] == "mobile"].copy()
    else:
        subset = data.copy()

    if len(subset) == 0:
        return {
            "affected_rentals": 0,
            "affected_pct": 0.0,
            "total_problematic": 0,
            "solved_cases": 0,
            "solved_pct": 0.0,
            "unsolved_cases": 0,
        }

    # Rentals affected (scheduled gap < threshold)
    affected = subset[
        subset["time_delta_with_previous_rental_in_minutes"] < threshold_minutes
    ]

    # Problematic cases in subset
    problematic = subset[subset["is_problematic"]]

    # Solved = problematic cases with time_delta < threshold
    solved = problematic[
        problematic["time_delta_with_previous_rental_in_minutes"] < threshold_minutes
    ]

    return {
        "affected_rentals": len(affected),
        "affected_pct": len(affected) / len(subset) * 100 if len(subset) > 0 else 0,
        "total_problematic": len(problematic),
        "solved_cases": len(solved),
        "solved_pct": len(solved) / len(problematic) * 100
        if len(problematic) > 0
        else 0,
        "unsolved_cases": len(problematic) - len(solved),
    }


# Load data
try:
    df = load_data()
except FileNotFoundError:
    st.error("Data file not found. Ensure data/get_around_delay_analysis.csv exists.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

ended, consecutive, problematic_cases = prepare_data(df)

# Sidebar - Controls
st.sidebar.header("Configuration")
scope = st.sidebar.selectbox(
    "Scope",
    options=["All rentals", "Connect only", "Mobile only"],
    help="Which rental types to include in analysis",
)
st.sidebar.caption("Applied to sections 1-4.")
threshold_minutes = st.sidebar.slider(
    "Minimum delay threshold (minutes)",
    min_value=0,
    max_value=720,
    value=60,
    step=15,
    help="Proposed minimum time between consecutive rentals",
)
st.sidebar.caption("Applied to sections 3-4.")

# Filter data based on scope
if scope == "Connect only":
    df_filtered = df[df["checkin_type"] == "connect"]
    ended_filtered = ended[ended["checkin_type"] == "connect"]
    consecutive_filtered = consecutive[consecutive["checkin_type"] == "connect"]
elif scope == "Mobile only":
    df_filtered = df[df["checkin_type"] == "mobile"]
    ended_filtered = ended[ended["checkin_type"] == "mobile"]
    consecutive_filtered = consecutive[consecutive["checkin_type"] == "mobile"]
else:
    df_filtered = df
    ended_filtered = ended
    consecutive_filtered = consecutive

# Active filters banner
_filter_parts = []
if scope != "All rentals":
    _filter_parts.append(f"Scope: **{scope}**")
if threshold_minutes > 0:
    _filter_parts.append(f"Threshold: **{threshold_minutes} min**")
if _filter_parts:
    st.info("Active filters: " + " | ".join(_filter_parts))
else:
    st.info("No active filters.")

# =============================================================================
# Section 1: Overview
# =============================================================================
st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Rentals", f"{len(df_filtered):,}")

with col2:
    ended_count = len(df_filtered[df_filtered["state"] == "ended"])
    ended_pct = ended_count / len(df_filtered) * 100 if len(df_filtered) > 0 else 0
    st.metric("Completed", f"{ended_count:,}", f"{ended_pct:.1f}%")

with col3:
    canceled_count = len(df_filtered[df_filtered["state"] == "canceled"])
    canceled_pct = (
        canceled_count / len(df_filtered) * 100 if len(df_filtered) > 0 else 0
    )
    st.metric("Canceled", f"{canceled_count:,}", f"{canceled_pct:.1f}%")

with col4:
    consecutive_count = len(consecutive_filtered)
    consecutive_pct = (
        consecutive_count / len(df_filtered) * 100 if len(df_filtered) > 0 else 0
    )
    st.metric(
        "Consecutive Rentals", f"{consecutive_count:,}", f"{consecutive_pct:.1f}%"
    )

# Rental type distribution chart
if scope == "All rentals":
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        type_counts = df["checkin_type"].value_counts()
        fig_type = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Rentals by Checkin Type",
            hole=0.4,
        )
        fig_type.update_traces(textinfo="label+percent")
        st.plotly_chart(fig_type, use_container_width=True)
        st.caption(
            "Mobile dominates (~80% of rentals). "
            "Connect (20%) shows a lower problematic "
            "rate on consecutive rentals "
            "(8.5% vs 14.5% for Mobile)."
        )

    with col_chart2:
        state_counts = df["state"].value_counts()
        fig_state = px.pie(
            values=state_counts.values,
            names=state_counts.index,
            title="Rentals by State",
            hole=0.4,
        )
        fig_state.update_traces(textinfo="label+percent")
        st.plotly_chart(fig_state, use_container_width=True)
        st.caption(
            "~48% of rentals are returned on time or early. 52.1% are returned late, suggesting delays are the norm rather than the exception."
        )

# =============================================================================
# Section 2: Late Returns Analysis
# =============================================================================
st.header("2. Late Returns Analysis")

# Late return statistics
delay_data = ended_filtered["delay_at_checkout_in_minutes"].dropna()
late_count = (delay_data > 0).sum()
on_time_count = (delay_data <= 0).sum()
late_pct = late_count / len(delay_data) * 100 if len(delay_data) > 0 else 0

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Late Returns", f"{late_count:,}", f"{late_pct:.1f}%")

with col2:
    median_delay = delay_data.median() if len(delay_data) > 0 else 0
    st.metric("Median Delay", f"{median_delay:.0f} min")

with col3:
    mean_delay = delay_data.mean() if len(delay_data) > 0 else 0
    st.metric("Mean Delay", f"{mean_delay:.0f} min")

# Visualizations
col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    # Pie chart: on-time vs late
    fig_late = px.pie(
        values=[on_time_count, late_count],
        names=["On time or early", "Late"],
        title="Return Timing Distribution",
        hole=0.4,
        color_discrete_sequence=["#2ecc71", "#e74c3c"],
    )
    fig_late.update_traces(textinfo="label+percent")
    st.plotly_chart(fig_late, use_container_width=True)
    st.caption(
        "Among ended rentals, only ~48% are on time. Canceled rentals (not shown) reduce the effective fleet utilization further."
    )

with col_viz2:
    # Histogram of delay distribution
    delay_capped = delay_data.clip(-120, 360)
    fig_hist = px.histogram(
        delay_capped,
        nbins=50,
        title="Delay Distribution (capped at -2h to +6h)",
        labels={"value": "Delay (minutes)", "count": "Number of rentals"},
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
    fig_hist.update_layout(showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(
        "Median delay is +9 min but the mean reaches ~60 min due to heavy right-tail outliers. Most delays are under 30 min."
    )

# Stats by rental type (only if showing all)
if scope == "All rentals":
    st.subheader("Late Returns by Checkin Type")

    late_by_type = ended.groupby("checkin_type").agg(
        total=("rental_id", "count"),
        late_count=("is_late", "sum"),
        mean_delay=("delay_at_checkout_in_minutes", "mean"),
        median_delay=("delay_at_checkout_in_minutes", "median"),
    )
    late_by_type["late_rate"] = late_by_type["late_count"] / late_by_type["total"] * 100
    late_by_type = late_by_type.round(1)

    st.dataframe(
        late_by_type[
            ["total", "late_count", "late_rate", "mean_delay", "median_delay"]
        ],
        use_container_width=True,
    )
    st.caption(
        "Mobile has a higher late rate (54.7%) than Connect (41.6%). Connect's digital flow likely reduces handoff friction."
    )

# =============================================================================
# Section 3: Impact on Next Rental
# =============================================================================
st.header("3. Impact on Next Driver")

st.markdown(
    """
    Analysis of consecutive rentals on the same car
    and how delays affect the next driver.
    """
)

# Key metrics
problematic_in_scope = consecutive_filtered[consecutive_filtered["is_problematic"]]
prob_count = len(problematic_in_scope)
prob_pct = (
    prob_count / len(consecutive_filtered) * 100 if len(consecutive_filtered) > 0 else 0
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Consecutive Rentals",
        f"{len(consecutive_filtered):,}",
        f"{len(consecutive_filtered) / len(df_filtered) * 100:.1f}% of total"
        if len(df_filtered) > 0
        else "0%",
    )

with col2:
    time_delta = consecutive_filtered["time_delta_with_previous_rental_in_minutes"]
    median_gap = time_delta.median() if len(time_delta) > 0 else 0
    st.metric("Median Gap Between Rentals", f"{median_gap:.0f} min")

with col3:
    st.metric(
        "Problematic Cases",
        f"{prob_count:,}",
        f"{prob_pct:.1f}% of consecutive",
    )

# Time between consecutive rentals
col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    fig_delta = px.histogram(
        time_delta.clip(0, 720),
        nbins=40,
        title="Scheduled Gap Between Consecutive Rentals",
        labels={"value": "Time Delta (minutes)", "count": "Count"},
    )
    fig_delta.add_vline(
        x=threshold_minutes,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {threshold_minutes}min",
    )
    fig_delta.update_layout(showlegend=False)
    st.plotly_chart(fig_delta, use_container_width=True)
    st.caption(
        "Median scheduled gap between consecutive rentals is ~3 hours (180 min). Most owners already set large buffers."
    )

with col_viz2:
    # Wait time for impacted drivers
    if len(problematic_in_scope) > 0:
        wait_time = -problematic_in_scope["effective_time"]
        fig_wait = px.histogram(
            wait_time.clip(0, 240),
            nbins=30,
            title="Wait Time for Impacted Drivers",
            labels={"value": "Wait Time (minutes)", "count": "Count"},
            color_discrete_sequence=["#e74c3c"],
        )
        fig_wait.update_layout(showlegend=False)
        st.plotly_chart(fig_wait, use_container_width=True)
        st.caption(
            "Impacted next drivers wait a median of "
            "~27 min. 30% wait over 60 min, directly "
            "causing cancellations and poor experience."
        )
    else:
        st.info("No problematic cases in the selected scope.")

# Problematic rate by type
if scope == "All rentals":
    st.subheader("Problematic Cases by Checkin Type")

    prob_by_type = consecutive.groupby("checkin_type").agg(
        total=("rental_id", "count"),
        problematic=("is_problematic", "sum"),
    )
    prob_by_type["rate_pct"] = (
        prob_by_type["problematic"] / prob_by_type["total"] * 100
    ).round(1)

    st.dataframe(prob_by_type, use_container_width=True)
    st.caption(
        "Only 8.6% of rentals have a consecutive "
        "booking, yet 11.8% of those are problematic "
        "(218 cases). Connect: 8.5%, Mobile: 14.5%."
    )

# =============================================================================
# Section 4: Threshold Simulation
# =============================================================================
st.header("4. Threshold Impact Simulation")

st.markdown(
    f"""
    Simulating impact of **{threshold_minutes}-minute** threshold on **{scope}**.
    """
)

# Current threshold simulation
sim_result = simulate_threshold(consecutive, threshold_minutes, scope)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Affected Rentals",
        f"{sim_result['affected_rentals']:,}",
        f"{sim_result['affected_pct']:.1f}%",
    )

with col2:
    st.metric(
        "Total Problematic",
        f"{sim_result['total_problematic']:,}",
    )

with col3:
    st.metric(
        "Problems Solved",
        f"{sim_result['solved_cases']:,}",
        f"{sim_result['solved_pct']:.1f}%",
    )

with col4:
    st.metric(
        "Unsolved Cases",
        f"{sim_result['unsolved_cases']:,}",
    )

# Revenue impact estimation
st.subheader("Revenue Impact Estimation")

total_rentals = len(df)
affected_pct_total = sim_result["affected_rentals"] / total_rentals * 100

st.info(
    f"""
    **Share of owner revenue potentially affected:** {affected_pct_total:.2f}%

    - Consecutive rentals represent \
{len(consecutive) / total_rentals * 100:.1f}% of all rentals
    - With {threshold_minutes}-minute threshold on {scope.lower()}:
      {sim_result["affected_rentals"]:,} rentals would need to be rescheduled
    """
)

# Comparison table for different thresholds
st.subheader("Threshold Comparison")

thresholds = [15, 30, 60, 90, 120, 180, 240, 360]
comparison_data = []

for t in thresholds:
    result = simulate_threshold(consecutive, t, scope)
    comparison_data.append(
        {
            "Threshold (min)": t,
            "Affected Rentals": result["affected_rentals"],
            "Affected %": f"{result['affected_pct']:.1f}%",
            "Problems Solved": result["solved_cases"],
            "Solved %": f"{result['solved_pct']:.1f}%",
            "Unsolved": result["unsolved_cases"],
        }
    )

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Trade-off visualization
st.subheader("Trade-off: Affected vs Solved")

tradeoff_data = []
for t in thresholds:
    for s in ["All rentals", "Connect only"]:
        result = simulate_threshold(consecutive, t, s)
        tradeoff_data.append(
            {
                "threshold": t,
                "scope": s,
                "affected_pct": result["affected_pct"],
                "solved_pct": result["solved_pct"],
            }
        )

tradeoff_df = pd.DataFrame(tradeoff_data)

fig_tradeoff = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["All Rentals", "Connect Only"],
)

for i, s in enumerate(["All rentals", "Connect only"]):
    scope_data = tradeoff_df[tradeoff_df["scope"] == s]

    fig_tradeoff.add_trace(
        go.Scatter(
            x=scope_data["threshold"],
            y=scope_data["affected_pct"],
            name="Affected %",
            mode="lines+markers",
            line=dict(color="#e74c3c"),
            showlegend=(i == 0),
        ),
        row=1,
        col=i + 1,
    )
    fig_tradeoff.add_trace(
        go.Scatter(
            x=scope_data["threshold"],
            y=scope_data["solved_pct"],
            name="Solved %",
            mode="lines+markers",
            line=dict(color="#2ecc71"),
            showlegend=(i == 0),
        ),
        row=1,
        col=i + 1,
    )

fig_tradeoff.update_xaxes(title_text="Threshold (minutes)")
fig_tradeoff.update_yaxes(title_text="Percentage")
fig_tradeoff.update_layout(height=400)

st.plotly_chart(fig_tradeoff, use_container_width=True)
st.caption(
    "Higher thresholds resolve more conflicts but cancel more revenue-generating rentals. The optimal threshold balances both."
)

# =============================================================================
# Section 5: Recommendations
# =============================================================================
st.header("5. Key Findings and Recommendations")

st.markdown(
    """
    ### Data Insights

    | Metric | Value |
    |--------|-------|
    | Total rentals | {:,} |
    | Mobile checkin share | {:.1f}% |
    | Connect checkin share | {:.1f}% |
    | Late return rate | {:.1f}% |
    | Consecutive rentals | {:.1f}% of total |
    | Problematic cases | {:.1f}% of consecutive |

    ### Recommendations

    | Strategy | Threshold | Scope | Trade-off |
    |----------|-----------|-------|-----------|
    | Conservative | 60 min | Connect | ~70% solved, low impact |
    | Balanced | 120 min | Connect | ~86% solved, moderate |
    | Aggressive | 180 min | All | ~90% solved, high impact |

    ### Key Takeaways

    1. **Connect-only scope is safer**: Affects only 20% of rentals with automated
       checkin, enabling easier threshold enforcement

    2. **60-minute threshold is a good starting point**: Minimal revenue impact with
       meaningful improvement in driver experience

    3. **Problematic cases are relatively rare**: At a 60-min Connect threshold,
       less than 1% of all rentals are affected

    4. **Mobile rentals have more volume but higher problematic rate**: Consider
       phased rollout starting with Connect
    """.format(
        len(df),
        len(df[df["checkin_type"] == "mobile"]) / len(df) * 100,
        len(df[df["checkin_type"] == "connect"]) / len(df) * 100,
        (ended["is_late"].sum() / len(ended) * 100),
        len(consecutive) / len(df) * 100,
        len(problematic_cases) / len(consecutive) * 100 if len(consecutive) > 0 else 0,
    )
)

# =============================================================================
# Section 6: Pricing Prediction
# =============================================================================
st.header("6. Pricing Prediction")
st.markdown(
    "Use the form below to estimate the daily rental price "
    "for a car based on its features."
)

st.info(
    "Model: GradientBoosting | Test RMSE: 16.14 EUR | R2: 0.762\n"
    "Selected as best among 3 models (Linear Regression, Random Forest, "
    "Gradient Boosting) via RandomizedSearchCV (5-fold CV, optimizing RMSE). "
    "Baseline (mean predictor): 33.56 RMSE."
)

settings = get_settings()
api_url = settings.api_url

st.markdown(f"API Documentation: [{api_url}/docs]({api_url}/docs)")

with st.form("prediction_form"):
    # Row 1 - Main characteristics
    r1c1, r1c2, r1c3 = st.columns(3)

    with r1c1:
        model_key = st.selectbox(
            "Brand",
            options=[
                "Alfa Romeo",
                "Audi",
                "BMW",
                "Citroen",
                "Ferrari",
                "Fiat",
                "Ford",
                "Honda",
                "KIA",
                "Lamborghini",
                "Lexus",
                "Maserati",
                "Mazda",
                "Mercedes",
                "Mini",
                "Mitsubishi",
                "Nissan",
                "Opel",
                "PGO",
                "Peugeot",
                "Porsche",
                "Renault",
                "SEAT",
                "Subaru",
                "Suzuki",
                "Toyota",
                "Volkswagen",
                "Yamaha",
            ],
            index=19,  # Peugeot
        )

    with r1c2:
        fuel = st.selectbox(
            "Fuel",
            options=[
                "diesel",
                "petrol",
                "hybrid_petrol",
                "electro",
            ],
            index=0,
        )

    with r1c3:
        car_type = st.selectbox(
            "Car Type",
            options=[
                "sedan",
                "hatchback",
                "suv",
                "van",
                "estate",
                "convertible",
                "coupe",
                "subcompact",
            ],
            index=0,
        )

    # Row 2 - Specifications
    r2c1, r2c2, r2c3 = st.columns(3)

    with r2c1:
        mileage = st.number_input(
            "Mileage (km)",
            min_value=0,
            max_value=500_000,
            value=100_000,
            step=5_000,
        )

    with r2c2:
        engine_power = st.number_input(
            "Engine Power (hp)",
            min_value=10,
            max_value=500,
            value=120,
            step=10,
        )

    with r2c3:
        paint_color = st.selectbox(
            "Paint Color",
            options=[
                "black",
                "white",
                "grey",
                "silver",
                "blue",
                "red",
                "beige",
                "brown",
                "green",
                "orange",
            ],
            index=0,
        )

    # Row 3 - Equipment
    st.markdown("**Equipment**")
    r3c1, r3c2, r3c3, r3c4 = st.columns(4)

    with r3c1:
        private_parking = st.checkbox("Private parking", value=False)
        has_gps = st.checkbox("GPS", value=False)

    with r3c2:
        has_ac = st.checkbox("Air conditioning", value=True)
        automatic = st.checkbox("Automatic transmission", value=False)

    with r3c3:
        has_connect = st.checkbox("Getaround Connect", value=False)
        has_regulator = st.checkbox("Speed regulator", value=False)

    with r3c4:
        winter_tires = st.checkbox("Winter tires", value=False)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    payload = {
        "cars": [
            {
                "model_key": model_key,
                "mileage": mileage,
                "engine_power": engine_power,
                "fuel": fuel,
                "paint_color": paint_color,
                "car_type": car_type,
                "private_parking_available": private_parking,
                "has_gps": has_gps,
                "has_air_conditioning": has_ac,
                "automatic_car": automatic,
                "has_getaround_connect": has_connect,
                "has_speed_regulator": has_regulator,
                "winter_tires": winter_tires,
            }
        ]
    }

    try:
        response = httpx.post(
            f"{api_url}/predict",
            json=payload,
            timeout=10.0,
        )
        response.raise_for_status()
        result = response.json()
        price = result["prediction"][0]
        st.success(f"Estimated daily rental price: {price} EUR/day")
    except httpx.ConnectError:
        st.error(
            f"Cannot connect to API at {api_url}. Ensure the API server is running."
        )
    except httpx.HTTPStatusError as exc:
        st.error(f"API error (HTTP {exc.response.status_code}): {exc.response.text}")
    except Exception as exc:
        logger.exception("Prediction request failed")
        st.error(f"Prediction failed: {exc}")

# Footer
st.markdown("---")
st.caption("Dashboard built for Getaround PM team")
