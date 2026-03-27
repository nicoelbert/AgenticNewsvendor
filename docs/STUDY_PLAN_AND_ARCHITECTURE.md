# Study Plan & Software Architecture

**Document Version**: 1.1
**Last Updated**: 2024-02-09
**Status**: Draft - Iteration 2 (addressing review feedback)

---

## Table of Contents

1. [Study Overview](#1-study-overview)
2. [Study Design](#2-study-design)
3. [Data Generating Process](#3-data-generating-process)
4. [Scenario Backlog](#4-scenario-backlog)
5. [Agent Architecture](#5-agent-architecture)
6. [Frontend Dashboard](#6-frontend-dashboard)
7. [Result Tracking](#7-result-tracking)
8. [Technical Architecture](#8-technical-architecture)
9. [Implementation Phases](#9-implementation-phases)
10. [Open Questions](#10-open-questions)

---

## 1. Study Overview

### 1.1 Research Question

**How can agentic AI help humans calibrate when to use their domain knowledge vs. when to trust ML forecasts in newsvendor decisions?**

### 1.2 Core Hypothesis

Participants who interact with an AI agent that transparently explains its capabilities and limitations will make better-calibrated decisions than those without AI assistance.

### 1.3 Key Constructs

| Construct | Operationalization |
|-----------|-------------------|
| **Appropriate Reliance** | Adjusting when AI is wrong, following when AI is right |
| **Domain Knowledge Use** | Incorporating hidden factors AI doesn't see |
| **Decision Quality** | Profit vs. optimal profit |
| **Trust Calibration** | Match between AI capability and participant reliance |

---

## 2. Study Design

### 2.1 Task Structure

| Aspect | Specification |
|--------|---------------|
| **Role** | Store manager ordering fresh grocery products |
| **Products** | Fresh salad, ice cream, ready meals, bakery (4 categories) |
| **Decisions per scenario** | 2: (1) Demand forecast, (2) Order quantity |
| **Scenarios per participant** | 10-12 |
| **Feedback** | None between scenarios (no path dependency) |
| **Total duration** | ~20-25 minutes |

### 2.2 Scenario Structure

Each scenario contains:

```
1. HEADER
   - Product name, date, location context

2. SITUATION BRIEFING (~150 words)
   - Weather conditions (visible to AI)
   - Calendar info (visible to AI)
   - Price/promotion status (visible to AI)
   - Local events/context (HIDDEN from AI)
   - Irrelevant noise details

3. DEMAND HISTORY
   - 14-day time series chart

4. CONDITIONS TABLE
   - Last week same day | Today | Tomorrow
   - Feature values for comparison

5. AI OUTPUT
   - Forecast: X units
   - Recommended order: Y units

6. INTERACTION AREA
   - Question mechanism (TBD: predefined vs. free)
   - AI response display

7. DECISION INPUT
   - "Your demand forecast: [___] units"
   - "Your order quantity: [___] units"

8. COST INFO BOX (collapsible)
   - Purchase cost, selling price, salvage value
```

### 2.3 Experimental Conditions

**Status**: TBD - Options under consideration:

| Condition | Description |
|-----------|-------------|
| **No AI** | Participant sees only data, no AI forecast |
| **AI Forecast Only** | AI gives forecast, no interaction |
| **AI + Questions** | Full interaction capability |

### 2.4 Participant Flow

```
1. CONSENT & INSTRUCTIONS (3 min)
   - Study purpose (generic)
   - Task explanation
   - AI assistant introduction
   - Cost structure explanation

2. PRACTICE SCENARIO (2 min)
   - One guided scenario
   - Explanation of interface

3. MAIN TASK (15-18 min)
   - 10-12 scenarios
   - Randomized order
   - No feedback between scenarios

4. POST-TASK SURVEY (3 min)
   - Trust in AI scales
   - Perceived usefulness
   - Demographics
   - Shopping habits (for domain knowledge proxy)
```

---

## 3. Data Generating Process

### 3.1 True Demand Model

```python
TRUE_DEMAND = (
    base_level
    + Σ (beta_visible[i] × feature_visible[i])   # AI sees these
    + Σ (beta_hidden[j] × feature_hidden[j])     # AI doesn't see
    + noise ~ N(0, sigma)
)
```

### 3.2 AI Forecast Model

```python
AI_FORECAST = (
    base_level
    + Σ (beta_visible[i] × feature_visible[i])   # Only visible features
    + estimation_noise ~ N(0, small_sigma)       # Small model error
)

AI_ORDER_RECOMMENDATION = newsvendor_optimal(
    forecast=AI_FORECAST,
    underage_cost=selling_price - purchase_cost,
    overage_cost=purchase_cost - salvage_value
)
```

### 3.3 Feature Specification

#### Visible Features (AI knows)

| Feature | Type | Range | Coefficient (β) |
|---------|------|-------|-----------------|
| `temperature` | Continuous | 5-35°C | Product-dependent |
| `rain` | Binary | 0/1 | Product-dependent |
| `weekday` | Categorical | Mon-Sun | Product-dependent |
| `price` | Continuous | Product-specific | Negative |
| `promotion` | Binary | 0/1 | Positive |
| `historical_avg` | Continuous | Product-specific | ~1.0 (anchor) |

#### Hidden Features (AI doesn't know)

| Feature | Type | Typical Effect | Products Affected |
|---------|------|----------------|-------------------|
| `sports_event` | Binary | +15-40% | Ready meals (high), Ice cream (medium), Salad (low) |
| `school_holiday` | Binary | +20-60% | Ice cream (high), Ready meals (low) |
| `local_festival` | Binary | +20-50% | All products |
| `competitor_closed` | Binary | +15-30% | All products |
| `weather_wrong` | Binary | Varies | Weather-sensitive products |

#### Noise Features (Zero effect, mentioned in scenario)

| Feature | Example Mentions |
|---------|------------------|
| `parking_status` | "Parking lot being repainted" |
| `staff_changes` | "New assistant manager started" |
| `unrelated_news` | "Local team won championship last week" |

### 3.4 Product Specifications

#### Fresh Salad

```yaml
fresh_salad:
  base_level: 45
  price: 3.20
  cost: 1.60
  salvage: 0.40

  visible_betas:
    temperature: 0.8      # +0.8 units per °C above 20
    rain: -5.0            # -5 units if raining
    weekday_friday: 4.0   # +4 units on Friday
    weekday_saturday: 6.0 # +6 units on Saturday
    promotion: 8.0        # +8 units if on promotion

  hidden_betas:
    sports_event: 3.0     # Small effect (+3 units)
    school_holiday: 4.0   # Small effect
    local_festival: 6.0   # Medium effect
    competitor_closed: 5.0

  weather_sensitivity: HIGH
  event_sensitivity: LOW
  noise_std: 4.0
```

#### Ice Cream

```yaml
ice_cream:
  base_level: 35
  price: 4.99
  cost: 2.50
  salvage: 0.00  # Cannot resell

  visible_betas:
    temperature: 1.5      # Very sensitive to heat
    rain: -8.0            # Strong negative
    weekday_friday: 5.0
    weekday_saturday: 10.0
    weekday_sunday: 8.0
    promotion: 12.0

  hidden_betas:
    sports_event: 8.0     # Medium effect
    school_holiday: 20.0  # Very high (kids home)
    local_festival: 15.0  # High effect
    competitor_closed: 8.0

  weather_sensitivity: VERY_HIGH
  event_sensitivity: MEDIUM
  noise_std: 5.0
```

#### Ready Meals

```yaml
ready_meals:
  base_level: 55
  price: 5.99
  cost: 3.00
  salvage: 1.00

  visible_betas:
    temperature: -0.3     # Slight negative (people cook less in heat)
    rain: 4.0             # Positive (people stay in)
    weekday_friday: 12.0  # Strong weekend prep
    weekday_saturday: 8.0
    promotion: 15.0

  hidden_betas:
    sports_event: 25.0    # Very high (game night!)
    school_holiday: -5.0  # Slight negative (families cook)
    local_festival: 5.0   # Low effect
    competitor_closed: 10.0

  weather_sensitivity: LOW
  event_sensitivity: VERY_HIGH
  noise_std: 5.0
```

#### Fresh Bakery

```yaml
fresh_bakery:
  base_level: 50
  price: 3.49
  cost: 1.75
  salvage: 0.35  # Breadcrumbs, donations

  visible_betas:
    temperature: 0.2      # Minor effect
    rain: -2.0            # Slight negative
    weekday_friday: 3.0
    weekday_saturday: 15.0  # Weekend brunch!
    weekday_sunday: 12.0
    promotion: 10.0

  hidden_betas:
    sports_event: 5.0     # Tailgating snacks
    school_holiday: 8.0   # Family breakfasts
    local_festival: 10.0  # Community events
    local_market: -12.0   # Competition from farmers market

  weather_sensitivity: LOW
  event_sensitivity: MEDIUM
  noise_std: 4.0
```

### 3.5 Optimal Order Calculation

```python
def calculate_optimal_order(forecast, underage_cost, overage_cost, demand_std):
    """
    Newsvendor optimal order quantity.

    Critical ratio = Cu / (Cu + Co)
    Where: Cu = underage cost (lost profit)
           Co = overage cost (loss on unsold)
    """
    critical_ratio = underage_cost / (underage_cost + overage_cost)
    z_score = scipy.stats.norm.ppf(critical_ratio)
    optimal_order = forecast + z_score * demand_std
    return optimal_order
```

---

## 4. Scenario Backlog

### 4.1 Scenario Design Principles

1. **Curated, not random**: Each scenario is hand-designed for specific learning/test purpose
2. **Balanced**: ~50% "trust AI" scenarios, ~50% "override needed"
3. **Varied products**: Each product appears 2-3 times
4. **Varied hidden factors**: Different events/contexts across scenarios
5. **Noise included**: All scenarios have some irrelevant details

### 4.2 Scenario Matrix

| ID | Product | AI Quality | Hidden Factor | Expected Action | Difficulty |
|----|---------|------------|---------------|-----------------|------------|
| S01 | Salad | Good | None | Trust AI | Easy |
| S02 | Ice Cream | Under-forecasts | School holiday starts | Override ↑ (+40%) | Medium |
| S03 | Ready Meals | Good | None | Trust AI | Easy |
| S04 | Bakery | Under-forecasts | Local festival | Override ↑ (+20%) | Medium |
| S05 | Salad | Over-forecasts | Rain coming (AI has wrong weather) | Override ↓ (-15%) | Hard |
| S06 | Ice Cream | Good | None (noise: parking lot) | Trust AI | Medium |
| S07 | Ready Meals | Under-forecasts | Champions League final | Override ↑ (+45%) | Medium |
| S08 | Bakery | Over-forecasts | Farmers market Saturday | Override ↓ (-25%) | Medium |
| S09 | Salad | Good | Football match (low effect for salad) | Small adjust (+5%) | Hard |
| S10 | Ice Cream | Under-forecasts | Festival + hot day | Override ↑ (+50%) | Medium |
| S11 | Ready Meals | Good | School holiday (low effect) | Trust AI | Hard |
| S12 | Bakery | Good | None | Trust AI | Easy |

### 4.3 Scenario Template

```yaml
scenario:
  id: "S02"
  product: "ice_cream"
  date: "Friday, July 18th"
  location: "Munich suburban supermarket"

  # Visible features (AI knows)
  visible:
    temperature: 28
    rain: false
    weekday: "Friday"
    price: 4.99
    promotion: false

  # Hidden features (AI doesn't know)
  hidden:
    school_holiday: true  # Summer holidays begin
    sports_event: false
    local_festival: false

  # Noise (mentioned but zero effect)
  noise:
    - "The ice cream freezer was serviced last week"
    - "A new flavor (mango sorbet) was added to the range"

  # Scenario narrative (~150 words)
  narrative: |
    You manage the frozen desserts section at a suburban supermarket
    in Munich. Tomorrow is Friday, July 18th - the first day of
    summer school holidays in Bavaria.

    The weather forecast shows a warm and sunny day with temperatures
    reaching 28°C. No rain expected. It's a regular Friday with
    standard pricing on all ice cream products.

    The store's ice cream freezer was serviced last week and is
    working perfectly. You recently added a new mango sorbet flavor
    to the premium range.

    Families are preparing for the holiday period. You've noticed
    more children in the store this week as schools wind down.

  # Pre-computed values
  ai_forecast: 65
  ai_order_recommendation: 72
  true_expected_demand: 91  # +40% for school holiday
  optimal_order: 98
  actual_demand: 88  # true_expected + noise realization

  # AI responses
  ai_responses:
    what_factors: |
      My forecast considers:
      • Temperature (28°C): +12 units above baseline
      • Friday effect: +5 units
      • No rain: +0 units (baseline)
      • Historical average for this period: ~48 units

      Total forecast: 65 units

    confidence: |
      I'm moderately confident in this forecast. Similar conditions
      in the past have shown demand within ±10 units of my prediction
      about 70% of the time.

    school_holiday: |
      I don't have information about school schedules or holiday
      periods. My forecast is based on weather, calendar patterns,
      and historical sales data.

    why_order_72: |
      I recommend ordering 72 units (7 more than forecast) because
      running out of ice cream is costly - you lose €2.49 profit
      per missed sale. Unsold units are a total loss (€2.50 cost,
      no salvage). The cost asymmetry suggests ordering above forecast.
```

### 4.4 Full Scenario Specifications

*[To be expanded: Full narrative and AI responses for each of 12 scenarios]*

---

## 5. Agent Architecture

### 5.1 Agent Capabilities

| Capability | Description | Implementation |
|------------|-------------|----------------|
| **Explain forecast** | Break down forecast by feature contributions | Compute from coefficients × values |
| **State confidence** | Express uncertainty range | Based on noise_std |
| **Admit limitations** | Say what it doesn't know | Fixed list of hidden features |
| **Answer "Did you consider X?"** | Yes/No based on feature visibility | Lookup in visible_features |
| **Explain recommendation** | Why order differs from forecast | Newsvendor logic explanation |

### 5.2 Response Generation

**Approach**: Template-based with variable injection (no LLM for responses)

```python
class AgentResponder:
    def __init__(self, scenario_config):
        self.visible_features = scenario_config['visible_features']
        self.hidden_features = scenario_config['hidden_features']
        self.betas = scenario_config['betas']
        self.values = scenario_config['feature_values']

    def explain_forecast(self) -> str:
        """Generate explanation of forecast components."""
        explanation = "My forecast considers:\n"
        for feature in self.visible_features:
            contribution = self.betas[feature] * self.values[feature]
            explanation += f"• {feature}: {contribution:+.0f} units\n"
        return explanation

    def did_you_consider(self, factor: str) -> str:
        """Answer whether a factor was considered."""
        if factor in self.visible_features:
            return f"Yes, {factor} is included in my model. [explanation]"
        elif factor in self.hidden_features:
            return f"No, I don't have data on {factor}. My forecast is based on [visible features]."
        else:
            return f"I'm not sure what you mean by {factor}. Could you clarify?"
```

### 5.3 Question Handling

**Option A: Predefined Questions (Current lean)**

```yaml
questions:
  - id: "explain_forecast"
    text: "How did you calculate this forecast?"
    always_available: true

  - id: "confidence"
    text: "How confident are you?"
    always_available: true

  - id: "explain_order"
    text: "Why do you recommend ordering more/less than the forecast?"
    always_available: true

  - id: "what_factors"
    text: "What factors does your model consider?"
    always_available: true

  - id: "limitations"
    text: "What don't you know about?"
    always_available: true

  # Scenario-specific (shown only when relevant context exists)
  - id: "consider_event"
    text: "Did you consider the [event]?"
    dynamic: true  # [event] replaced with scenario-specific term

  - id: "consider_holiday"
    text: "Did you account for the school holiday?"
    conditional: "school_holiday mentioned in scenario"
```

**Option B: Free Text (Alternative)**

- Use LLM to interpret question
- Match to predefined response templates
- Risk: Hallucination, unpredictable responses

**Decision**: TBD (likely Option A for control)

### 5.4 Agent State Machine

```
                    ┌─────────────────┐
                    │  SCENARIO START │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  SHOW CONTEXT   │
                    │  + AI FORECAST  │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     ┌─────────────────┐          ┌─────────────────┐
     │  USER ASKS      │◄────────►│  USER ENTERS    │
     │  QUESTION       │          │  DECISIONS      │
     └────────┬────────┘          └────────┬────────┘
              │                             │
              ▼                             │
     ┌─────────────────┐                    │
     │  AGENT RESPONDS │                    │
     └────────┬────────┘                    │
              │                             │
              └──────────────┬──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  LOG RESULTS    │
                    │  NEXT SCENARIO  │
                    └─────────────────┘
```

---

## 6. Frontend Dashboard

### 6.1 Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Framework** | Streamlit | Rapid prototyping, Python-native |
| **Visualization** | Plotly / Lets-Plot | Interactive charts |
| **State Management** | Streamlit session_state | Simple, built-in |
| **Hosting** | Streamlit Cloud / Heroku | Easy deployment |

### 6.2 Page Structure

```
/
├── pages/
│   ├── 01_consent.py           # Consent form
│   ├── 02_instructions.py      # Task instructions
│   ├── 03_practice.py          # Practice scenario
│   ├── 04_main_task.py         # Main experiment (10-12 scenarios)
│   └── 05_survey.py            # Post-task survey
└── app.py                      # Entry point, routing
```

### 6.3 Main Task UI Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCENARIO 3 of 12                                              [Progress ▓▓▓░░░░░░░░░]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PRODUCT: Fresh Salad · Friday, June 14th                           │   │
│  │  Munich suburban supermarket, near Allianz Arena                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         SITUATION                                    │   │
│  │                                                                      │   │
│  │  You manage fresh produce at a supermarket in Munich, located near  │   │
│  │  the Allianz Arena stadium. Tomorrow is Friday, June 14th.          │   │
│  │                                                                      │   │
│  │  Weather forecast shows 28°C and sunny. Bayern Munich plays a       │   │
│  │  Champions League match tomorrow evening. Salads are at regular     │   │
│  │  price (€3.20). The parking lot is being repainted this week.       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌────────────────────────────────┐  ┌────────────────────────────────┐   │
│  │     DEMAND HISTORY (14 days)   │  │       CONDITIONS               │   │
│  │                                │  │                                │   │
│  │   55│        ●      ●          │  │  │         │Last Fri│Tomorrow│   │
│  │     │      ●  ●   ●  ●         │  │  │Temp     │  24°C  │  28°C  │   │
│  │   45│    ●      ●      ●  [?]  │  │  │Rain     │   No   │   No   │   │
│  │     │  ●                       │  │  │Price    │ €3.20  │ €3.20  │   │
│  │   35│────────────────────────  │  │  │Promo    │   No   │   No   │   │
│  │     F S S M T W T F S S M T W T│  │                                │   │
│  └────────────────────────────────┘  └────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  🤖 AI ASSISTANT                                                    │   │
│  │                                                                      │   │
│  │  Forecast: 52 units     │     Recommended Order: 55 units           │   │
│  │                                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │  Ask a question:                                             │    │   │
│  │  │  ○ How did you calculate this forecast?                      │    │   │
│  │  │  ○ How confident are you?                                    │    │   │
│  │  │  ○ What factors does your model consider?                    │    │   │
│  │  │  ○ Did you consider the football match?                      │    │   │
│  │  │  ○ Why order more than the forecast?                         │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                                                                      │   │
│  │  [AI Response area - shows response when question clicked]          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  YOUR DECISIONS                                                      │   │
│  │                                                                      │   │
│  │  What demand do YOU expect?     [________] units                    │   │
│  │                                                                      │   │
│  │  How many units will you ORDER? [________] units                    │   │
│  │                                                                      │   │
│  │                                          [Submit & Continue →]       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ COST INFO (click to expand) ───────────────────────────────────────┐   │
│  │  Purchase: €1.60 | Sell: €3.20 | Salvage: €0.40                     │   │
│  │  Profit if sold: €1.60 | Loss if unsold: €1.20                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.4 UI Components

| Component | Implementation | Notes |
|-----------|----------------|-------|
| **Scenario header** | st.header + st.progress | Show progress |
| **Situation text** | st.markdown in container | Scrollable if long |
| **Demand chart** | plotly line chart | 14 days, interactive |
| **Conditions table** | st.dataframe or HTML | 3-column comparison |
| **AI forecast box** | st.info or custom CSS | Prominent display |
| **Question buttons** | st.radio or st.button group | Single select |
| **AI response** | st.markdown in container | Appears on question click |
| **Decision inputs** | st.number_input | Validation: positive integers |
| **Cost info** | st.expander | Collapsed by default |
| **Submit button** | st.button | Disabled until both inputs filled |

---

## 7. Result Tracking

### 7.1 Data Schema

#### Participant Record

```python
@dataclass
class ParticipantRecord:
    participant_id: str          # UUID
    condition: str               # Experimental condition
    start_time: datetime
    end_time: datetime
    completed: bool
    demographics: Dict
    survey_responses: Dict
```

#### Trial Record

```python
@dataclass
class TrialRecord:
    participant_id: str
    scenario_id: str
    trial_number: int            # 1-12

    # Timestamps
    scenario_start_time: datetime
    scenario_end_time: datetime
    time_spent_seconds: float

    # Scenario info
    product: str
    ai_forecast: float
    ai_recommendation: float
    true_demand: float
    optimal_order: float

    # Participant decisions
    participant_forecast: float
    participant_order: float

    # Derived metrics
    forecast_adjustment: float   # participant - AI
    order_adjustment: float      # participant - AI recommendation
    forecast_error: float        # participant - true demand
    profit: float
    optimal_profit: float
    profit_loss: float           # optimal - actual

    # Interaction data
    questions_asked: List[str]
    question_timestamps: List[datetime]
    question_order: List[str]
```

### 7.2 Storage

| Option | Technology | Pros | Cons |
|--------|------------|------|------|
| **SQLite** | sqlite3 | Simple, no server | Single-user |
| **PostgreSQL** | psycopg2 | Robust, concurrent | Setup overhead |
| **Firebase** | firebase-admin | Real-time, hosted | Vendor lock-in |
| **CSV/Parquet** | pandas | Simple export | No concurrency |

**Recommendation**: SQLite for pilot, PostgreSQL for main study

### 7.3 Export Format

```csv
participant_id,scenario_id,trial_number,product,ai_forecast,ai_recommendation,
participant_forecast,participant_order,true_demand,optimal_order,forecast_adjustment,
order_adjustment,profit,optimal_profit,questions_asked,time_spent_seconds
```

---

## 8. Technical Architecture

### 8.1 Project Structure

```
AgenticNewsvendor/
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── config/
│   ├── __init__.py
│   ├── settings.py              # App settings
│   ├── llm_loader.py            # API key loading (reuse pattern)
│   ├── private_config.yaml      # Secrets (gitignored)
│   ├── private_config.example.yaml
│   └── prompts.yaml             # Agent response templates
│
├── data/
│   ├── scenarios/               # Curated scenario definitions
│   │   ├── scenario_backlog.yaml
│   │   └── scenarios/
│   │       ├── S01_salad_baseline.yaml
│   │       ├── S02_icecream_holiday.yaml
│   │       └── ...
│   ├── experiments/             # Generated experiment data
│   └── results/                 # Collected participant data
│
├── src/
│   ├── __init__.py
│   │
│   ├── dgp/                     # Data Generating Process
│   │   ├── __init__.py
│   │   ├── demand_model.py      # True demand generation
│   │   ├── ai_model.py          # AI forecast generation
│   │   ├── features.py          # Feature definitions
│   │   └── products.py          # Product configurations
│   │
│   ├── agent/                   # Agent system
│   │   ├── __init__.py
│   │   ├── responder.py         # Response generation
│   │   ├── templates.py         # Response templates
│   │   └── questions.py         # Question definitions
│   │
│   ├── experiment/              # Experiment management
│   │   ├── __init__.py
│   │   ├── scenario_loader.py   # Load scenario configs
│   │   ├── session_manager.py   # Manage participant sessions
│   │   └── randomization.py     # Scenario ordering
│   │
│   └── tracking/                # Data collection
│       ├── __init__.py
│       ├── models.py            # Data models (dataclasses)
│       ├── database.py          # Database operations
│       └── export.py            # Export to CSV/Parquet
│
├── dashboard/                   # Streamlit frontend
│   ├── app.py                   # Main entry point
│   ├── components/              # Reusable UI components
│   │   ├── __init__.py
│   │   ├── scenario_display.py
│   │   ├── demand_chart.py
│   │   ├── ai_interaction.py
│   │   └── decision_input.py
│   ├── pages/
│   │   ├── 01_consent.py
│   │   ├── 02_instructions.py
│   │   ├── 03_practice.py
│   │   ├── 04_main_task.py
│   │   └── 05_survey.py
│   └── style/
│       └── custom.css
│
├── tests/
│   ├── test_dgp.py
│   ├── test_agent.py
│   └── test_scenarios.py
│
├── scripts/
│   ├── generate_scenarios.py    # Generate scenario data
│   ├── validate_scenarios.py    # Check scenario consistency
│   └── export_results.py        # Export collected data
│
└── docs/
    ├── EXPERIMENT_DESIGN_MASTER.md
    ├── STUDY_PLAN_AND_ARCHITECTURE.md  # This file
    └── STUDY_DESIGN_OPTIONS.md
```

### 8.2 Key Interfaces

#### DGP Interface

```python
# src/dgp/demand_model.py

class DemandModel:
    def __init__(self, product_config: ProductConfig):
        self.product = product_config

    def compute_true_demand(
        self,
        visible_features: Dict[str, float],
        hidden_features: Dict[str, float],
        noise_seed: int = None
    ) -> float:
        """Compute true demand from all features."""
        pass

    def compute_ai_forecast(
        self,
        visible_features: Dict[str, float]
    ) -> float:
        """Compute AI forecast from visible features only."""
        pass

    def compute_optimal_order(
        self,
        forecast: float,
        demand_std: float
    ) -> float:
        """Compute newsvendor optimal order."""
        pass
```

#### Agent Interface

```python
# src/agent/responder.py

class AgentResponder:
    def __init__(self, scenario: ScenarioConfig):
        self.scenario = scenario

    def get_available_questions(self) -> List[Question]:
        """Return list of questions available for this scenario."""
        pass

    def answer_question(self, question_id: str) -> str:
        """Generate response for a question."""
        pass

    def explain_forecast(self) -> str:
        """Explain how forecast was computed."""
        pass

    def explain_recommendation(self) -> str:
        """Explain order recommendation logic."""
        pass
```

#### Session Interface

```python
# src/experiment/session_manager.py

class SessionManager:
    def __init__(self, db: Database):
        self.db = db

    def create_session(self, condition: str) -> str:
        """Create new participant session, return participant_id."""
        pass

    def get_scenario_order(self, participant_id: str) -> List[str]:
        """Get randomized scenario order for participant."""
        pass

    def record_trial(self, trial: TrialRecord) -> None:
        """Save trial data."""
        pass

    def complete_session(self, participant_id: str, survey: Dict) -> None:
        """Mark session complete, save survey data."""
        pass
```

### 8.3 Configuration Management

```python
# config/settings.py

from pydantic import BaseSettings

class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite:///data/results/experiment.db"

    # Experiment
    n_scenarios: int = 12
    practice_scenario_id: str = "practice_01"
    randomize_scenarios: bool = True

    # UI
    debug_mode: bool = False
    show_optimal_feedback: bool = False  # For pilot only

    class Config:
        env_file = ".env"
```

---

## 9. Implementation Phases

### Phase 1: Core DGP & Scenarios (Week 1)

- [ ] Implement DemandModel class
- [ ] Define product configurations (YAML)
- [ ] Create 12 scenario specifications
- [ ] Write scenario validation script
- [ ] Unit tests for DGP

**Deliverable**: `python scripts/validate_scenarios.py` passes

### Phase 2: Agent System (Week 1-2)

- [ ] Implement AgentResponder class
- [ ] Define response templates
- [ ] Create question definitions
- [ ] Unit tests for agent

**Deliverable**: Agent can answer all question types correctly

### Phase 3: Data Tracking (Week 2)

- [ ] Define data models
- [ ] Implement database layer
- [ ] Create export scripts
- [ ] Test data persistence

**Deliverable**: Full trial record saves and exports

### Phase 4: Frontend Dashboard (Week 2-3)

- [ ] Build page structure
- [ ] Implement scenario display component
- [ ] Implement demand chart
- [ ] Implement AI interaction component
- [ ] Implement decision inputs
- [ ] Connect to backend
- [ ] Style and polish

**Deliverable**: Complete participant flow works end-to-end

### Phase 5: Integration & Pilot (Week 3-4)

- [ ] Integration testing
- [ ] Internal pilot (5-10 participants)
- [ ] Calibrate difficulty
- [ ] Fix bugs
- [ ] Refine scenarios based on feedback

**Deliverable**: Ready for main study

### Phase 6: Main Study (Week 4+)

- [ ] Deploy to production
- [ ] Recruit participants (Prolific)
- [ ] Monitor data collection
- [ ] Export and analyze

---

## 10. Statistical Design

### 10.1 Study Design Type

**Decision**: Within-subjects design (all participants see all scenarios)

| Aspect | Specification |
|--------|---------------|
| **Design** | Single-condition within-subjects (for pre-study) |
| **Participants** | All receive AI + Questions condition |
| **Scenarios** | 10-12 scenarios, randomized order |
| **Analysis** | Within-subject variation across scenario types |

**Rationale**: For initial pre-study, single condition allows us to:
- Understand baseline behavior with AI
- Identify which scenario types are effective
- Calibrate difficulty before adding between-subjects conditions

**Future extension**: Add between-subjects conditions (No AI, AI Only, AI + Questions) in main study.

### 10.2 Sample Size

**Target**: N = 80-100 participants for pre-study

**Justification**:
- 10 scenarios × 80 participants = 800 trial observations
- Sufficient to detect medium effect sizes (d = 0.5) for within-subject comparisons
- Allows subgroup analysis by scenario type (trust vs override)

**Prolific budget estimate**:
- 25 min study × £9/hour = ~£3.75 per participant
- 100 participants = ~£375 + Prolific fees (~£450 total)

### 10.3 Primary Outcome Measures

| Measure | Definition | Analysis |
|---------|------------|----------|
| **Appropriate Adjustment Rate** | % of scenarios where participant correctly trusts/overrides | Primary |
| **Forecast Error** | |Participant forecast - True demand| | Secondary |
| **Order Profit** | Actual profit vs optimal profit | Secondary |
| **Trust Calibration** | Correlation between AI accuracy and participant reliance | Exploratory |

### 10.4 Analysis Plan

```
1. DESCRIPTIVE STATISTICS
   - Mean adjustment by scenario type (trust vs override)
   - Distribution of question-asking behavior
   - Time spent per scenario

2. PRIMARY ANALYSIS
   - Mixed-effects logistic regression: Appropriate adjustment ~ scenario_type + (1|participant)
   - Test: Do participants adjust MORE in "override" scenarios than "trust" scenarios?

3. SECONDARY ANALYSES
   - Learning effects: Does appropriate adjustment improve over trials?
   - Question behavior: Does asking questions predict better decisions?
   - Product effects: Do some products show stronger effects?

4. EXPLORATORY
   - Cluster analysis of participant strategies
   - Qualitative analysis of free-text responses (if any)
```

### 10.5 Pre-registration

**Plan**: Pre-register on OSF before data collection

Contents:
- Hypotheses (directional)
- Primary outcome definition
- Analysis approach
- Exclusion criteria (attention checks, completion threshold)
- Sample size justification

---

## 11. Data Quality & Attention Checks

### 11.1 Comprehension Check (After Instructions)

```
Before starting, please answer these questions:

Q1: If you order MORE than customer demand, what happens to unsold items?
    ○ They are sold at full price tomorrow
    ○ They are discarded or sold at reduced salvage value ← CORRECT
    ○ They are returned to the supplier for refund

Q2: The AI assistant has access to all information about your store.
    ○ True
    ○ False ← CORRECT (AI doesn't know about local events)

Q3: Your goal in this task is to:
    ○ Always follow the AI recommendation exactly
    ○ Maximize your profit by ordering the right amount ← CORRECT
    ○ Order as little as possible to minimize waste

Requirement: Must answer all 3 correctly to proceed (can retry)
```

### 11.2 Attention Checks (Embedded in Scenarios)

**Scenario 4 or 7**: Include obvious instruction

```
SITUATION:
[Normal scenario text...]

To show you're reading carefully, please enter 999 as your
demand forecast for this scenario. Enter any order quantity.
```

**Exclusion**: Participants who fail attention check are excluded from analysis.

### 11.3 Time-Based Quality Filters

| Filter | Threshold | Action |
|--------|-----------|--------|
| **Too fast (scenario)** | < 30 seconds | Flag for review |
| **Too fast (total)** | < 8 minutes for 10 scenarios | Exclude |
| **Too slow (scenario)** | > 10 minutes | Likely abandoned, save partial |
| **Incomplete** | < 8 scenarios completed | Exclude from primary analysis |

### 11.4 Input Validation

| Field | Validation | Error Message |
|-------|------------|---------------|
| Forecast | Integer, 1-500 | "Please enter a number between 1 and 500" |
| Order | Integer, 0-500 | "Please enter a number between 0 and 500" |
| Both required | Non-empty | "Please enter both values before continuing" |

---

## 12. Explicit Design Decisions

### 12.1 DECIDED: Two Decisions Per Scenario

**Decision**: Keep two decisions (forecast + order)

**Rationale**:
1. **Forecast reveals understanding**: Did participant recognize hidden factor?
2. **Order reveals application**: Did they translate understanding to action?
3. **Separates constructs**: Trust in AI prediction vs trust in AI recommendation
4. **Richer data**: Can analyze forecast adjustment AND order adjustment

**Trade-off**: Adds ~20 seconds per scenario. Acceptable for research value.

### 12.2 DECIDED: Predefined Questions (Not Free Text)

**Decision**: Use predefined clickable questions

**Rationale**:
1. **Control**: Exact same stimuli for all participants
2. **No hallucination risk**: Template-based responses
3. **Easier analysis**: Can count which questions asked
4. **Simpler implementation**: No LLM at runtime

**Question set** (6 questions per scenario):
1. "How did you calculate this forecast?" (always)
2. "How confident are you?" (always)
3. "What factors does your model consider?" (always)
4. "Why order more/less than the forecast?" (always)
5. "Did you consider [context-specific]?" (dynamic, based on scenario)
6. "What might you be missing?" (always)

### 12.3 DECIDED: Constant Feature Set (Not Product-Specific)

**Decision**: AI model has same features for all products (Option A from design exploration)

**Visible to AI (all products)**:
- Temperature, rain, weekday, price, promotion, historical average

**Hidden from AI (all products)**:
- Local events, school holidays, competitor actions

**Product differentiation**: Through effect sizes (betas), not feature visibility

**Rationale**:
1. **Simpler for participants**: One model to learn
2. **Foundation model compatible**: Same structure across products
3. **Cleaner research design**: Variation in scenarios, not model structure

### 12.4 DECIDED: No Feedback Between Scenarios

**Decision**: Participants do NOT see actual demand after each scenario

**Rationale**:
1. **Avoid learning from outcomes**: Tests prior beliefs, not learned patterns
2. **No path dependency**: Each scenario is independent
3. **Cleaner analysis**: No carryover effects

**Exception**: May show summary at very end (after all decisions made)

### 12.5 DECIDED: 10 Scenarios (Not 12)

**Decision**: Reduce to 10 scenarios for pre-study

**Rationale**:
1. **Time constraint**: 10 × 2 min = 20 min task (fits in 25 min total)
2. **Sufficient variation**: 5 "trust AI" + 5 "override needed"
3. **Lower dropout**: Shorter is better for Prolific completion

**Scenario balance**:
- 3 × Fresh Salad
- 3 × Ice Cream
- 2 × Ready Meals
- 2 × Bakery

---

## 13. Risk Mitigation

### 13.1 Technical Risks

| Risk | Mitigation |
|------|------------|
| **Streamlit concurrency** | Use PostgreSQL, implement proper session IDs in URL |
| **Data loss** | Auto-save after each scenario to database |
| **Browser issues** | Test in Chrome, Firefox, Safari before launch |
| **Session timeout** | 30-min timeout with recovery link via participant ID |

### 13.2 Design Risks

| Risk | Mitigation |
|------|------------|
| **Ceiling effects** | Vary effect magnitudes, include ambiguous scenarios |
| **Participants don't understand newsvendor** | Comprehension quiz, simple cost framing |
| **Gaming/pattern detection** | Randomize order, vary scenarios |
| **Domain knowledge weak** | Screen for grocery shopping frequency |

### 13.3 Data Quality Risks

| Risk | Mitigation |
|------|------------|
| **Inattentive participants** | Attention checks, time filters |
| **Dropout** | Keep study short (25 min), fair compensation |
| **Incomplete data** | Auto-save per scenario, analyze partial completions separately |

---

## 14. Open Questions (Remaining)

| Question | Status | Notes |
|----------|--------|-------|
| ~~Question interaction~~ | **DECIDED: Predefined** | Section 12.2 |
| ~~Number of scenarios~~ | **DECIDED: 10** | Section 12.5 |
| ~~Feedback between scenarios~~ | **DECIDED: No** | Section 12.4 |
| ~~Experimental conditions~~ | **DECIDED: Single condition (pre-study)** | Section 10.1 |
| Hosting platform | **TBD** | Leaning: Railway or Render with PostgreSQL |
| Participant recruitment | **Prolific confirmed** | Budget ~£450 for N=100 |
| Effect size calibration | **Needs pilot** | Internal pilot with 10 participants first |
| ~~Session timeout handling~~ | **DECIDED: 30 min + recovery** | Section 13.1 |
| Scenario narratives | **IN PROGRESS** | 1 of 10 fully specified |
| AI response templates | **IN PROGRESS** | Need ~60 responses (6 per scenario) |
| IRB/Ethics approval | **TBD** | Check with institution |
| Pre-registration | **PLANNED** | OSF before data collection |

---

## 15. Next Actions (Prioritized)

### Immediate (Before Implementation)

1. [ ] **Write all 10 scenario specifications** - Full narratives + AI responses
2. [ ] **Finalize product configs** - Verify all betas and effect sizes
3. [ ] **Create scenario YAML files** - Machine-readable format
4. [ ] **Choose hosting platform** - Decision needed this week

### Phase 1 Implementation

5. [ ] Implement DGP module (demand_model.py, products.py)
6. [ ] Implement agent responder (templates, question matching)
7. [ ] Build basic Streamlit UI (single scenario flow)
8. [ ] Connect to PostgreSQL for data storage

### Before Pilot

9. [ ] Internal testing (team members)
10. [ ] Pilot with 10 Prolific participants
11. [ ] Calibrate timing and difficulty
12. [ ] Pre-register on OSF

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-02-09 | Initial draft |
| 1.1 | 2024-02-09 | Added: Statistical design, attention checks, explicit decisions, risk mitigation. Resolved 6 TBD items. |

