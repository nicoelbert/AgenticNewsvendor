# AgenticNewsvendor - Project Specification

## 1. Project Understanding (Synthesized from Emails)

### Core Idea: Hybrid Agentic System
A **controlled, interactive experiment** where participants interact with an AI assistant through **predefined clickable questions** rather than free-form chat. This balances:
- **Interactivity** (feels like a real AI system)
- **Control** (no hallucinations, validated outputs)
- **Measurability** (defined question-answer tree)

### Key Design Decisions

| Aspect | Decision |
|--------|----------|
| **Interaction Mode** | Clickable question suggestions (not free-text) |
| **Answer Generation** | Pre-prepared, scenario-specific (scenario × question = answer) |
| **Scenario Complexity** | Simplified (fewer/no time series) |
| **Setting** | TBD: Forecast vs Newsvendor (prepare both) |
| **Design** | Potentially two-stage |

### Research Questions & Metrics
1. **Decision Quality** - How well do participants perform?
2. **Perceived Trust** - Trust in the AI system
3. **Question Types Analysis** - What do participants ask about?
   - Uncertainty
   - Model performance
   - Features used
   - Newsvendor decisions

---

## 2. Open Questions (Need Your Input)

### Q1: Forecast vs Newsvendor Setting?

| Option | Pros | Cons |
|--------|------|------|
| **Forecast Only** | Cleaner task, uncertainty is the focus | Less decision-making complexity |
| **Newsvendor** | Full decision task, realistic | Very routine, uncertainty comes only from forecast |
| **Two-Stage** | Best of both worlds | More complex experiment design |

**Current preference?** _______________

---

### Q2: Scenario Complexity

How much data should participants see?

- [ ] **Minimal**: Single forecast point + key metrics only
- [ ] **Light**: Recent history (7-14 days) as table/simple chart
- [ ] **Moderate**: Time series visualization + feature values
- [ ] **Current state**: Full dashboard with multiple time series

**Note**: Reference studies use very simple inputs (single forecasts, fixed parameters, no time series).

---

### Q3: Question Categories

What types of questions should be in the clickable menu?

Based on literature (algorithmic aversion, overreliance) and your pilots:

| Category | Example Questions |
|----------|-------------------|
| **Uncertainty** | "How confident is the model?" / "What's the forecast range?" |
| **Model Performance** | "How accurate has the model been?" / "What's the error rate?" |
| **Feature Importance** | "Which factors drive this forecast?" / "Why is demand high today?" |
| **Data Explanation** | "What data does the model use?" / "How far back does it look?" |
| **Decision Support** | "What order quantity do you recommend?" / "What's the risk of stockout?" |
| **Counterfactuals** | "What if there's a promotion?" / "What if it rains?" |

**Which categories to include?** _______________

---

### Q4: Experimental Conditions

What conditions are we comparing?

- [ ] **No Agent** vs **Guided Agent** (clickable questions)
- [ ] **No Agent** vs **Guided Agent** vs **Free Agent**
- [ ] **Good Model** vs **Bad Model** (with/without agent)
- [ ] Other: _______________

---

### Q5: Number of Scenarios per Participant

- [ ] 1 scenario (simple, quick)
- [ ] 3-5 scenarios (variety, within-subject)
- [ ] 6+ scenarios (statistical power)

---

## 3. Current Project State vs. Required Changes

### What We Have (Current Codebase)

| Component | Status | Notes |
|-----------|--------|-------|
| Data Generation | Working | Configurable demand simulation with features |
| Experiment Config | Working | YAML-based product/model definitions |
| Streamlit Dashboard | MVP | Full chat + visualization (too complex?) |
| LangGraph Backend | Minimal | Rule-based placeholder, LLM ready |
| LLM Integration | Partial | Anthropic API configured |

### What We Need to Build

| Component | Priority | Description |
|-----------|----------|-------------|
| **Question-Answer Tree** | HIGH | Predefined Q&A pairs per scenario |
| **Simplified UI** | HIGH | Clickable questions, not free-text |
| **Scenario Generator** | MEDIUM | Create distinct scenarios with known "correct" answers |
| **Response Validator** | MEDIUM | Ensure pre-generated answers are accurate |
| **Experiment Platform** | MEDIUM | Prolific/MTurk integration, data collection |
| **Analysis Pipeline** | LOW | Decision quality, trust scales, question analysis |

---

## 4. Proposed Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    EXPERIMENT UI                         │
│  ┌─────────────────┐  ┌─────────────────────────────┐   │
│  │  Scenario View  │  │     Chat Interface          │   │
│  │  - Forecast     │  │  ┌─────────────────────┐    │   │
│  │  - Key metrics  │  │  │ Clickable Questions │    │   │
│  │  - (minimal)    │  │  │ [Q1] [Q2] [Q3] ...  │    │   │
│  │                 │  │  └─────────────────────┘    │   │
│  │                 │  │  ┌─────────────────────┐    │   │
│  │                 │  │  │ Pre-generated Answer│    │   │
│  │                 │  │  │ (scenario-specific) │    │   │
│  │                 │  │  └─────────────────────┘    │   │
│  └─────────────────┘  └─────────────────────────────┘   │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │            Decision Input                        │    │
│  │  [Forecast: ___] or [Order Quantity: ___]       │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    BACKEND                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │  Scenario   │  │  Q&A Tree   │  │  Data Logger    │  │
│  │  Selector   │  │  (JSON/YAML)│  │  (Responses)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Question-Answer Tree Structure (Proposal)

```yaml
scenarios:
  - id: "scenario_001"
    product: "fresh_salad"
    model: "good_model"
    forecast: 42
    actual: 45  # for evaluation
    context:
      recent_avg: 38
      temperature: "warm (28°C)"
      promotion: false
      weekday: "Friday"

    questions:
      - id: "q_uncertainty"
        text: "How confident is the model in this forecast?"
        answer: "The model is moderately confident. Based on similar past situations, actual demand typically falls within ±8 units of the forecast (42 ± 8). The main uncertainty comes from weekend effects."
        category: "uncertainty"

      - id: "q_drivers"
        text: "What factors are driving this forecast?"
        answer: "The forecast of 42 units is primarily driven by: (1) warm temperature (+5 units vs. average), (2) Friday effect (+3 units), and (3) no current promotion. Temperature has the strongest influence on fresh salad demand."
        category: "feature_importance"

      # ... more questions
```

---

## 6. Next Steps

1. **Finalize Design Decisions** (this document)
2. **Create Question Catalog** - All possible questions across categories
3. **Generate Scenarios** - 10-20 distinct scenarios with pre-computed answers
4. **Build Simplified UI** - Clickable questions, minimal visualization
5. **Pilot Test** - Internal testing before Prolific
6. **Run Experiment** - Collect data
7. **Analyze Results** - Decision quality, trust, question patterns

---

## 7. Timeline (TBD)

| Phase | Tasks |
|-------|-------|
| **Phase 1** | Finalize spec, create Q&A catalog |
| **Phase 2** | Build UI, generate scenarios |
| **Phase 3** | Internal pilot |
| **Phase 4** | Main study |

---

## Notes & Discussion

_Add your thoughts here:_

-
-
-

