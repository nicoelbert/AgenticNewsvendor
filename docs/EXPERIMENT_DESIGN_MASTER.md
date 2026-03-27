# Experiment Design - Master Document

**Last Updated**: 2024-02-09
**Status**: In Design Phase

---

## 1. Research Objective

**Core Question**: How can agentic AI help humans calibrate when to use their domain knowledge vs. when to trust ML models?

**Setting**: Newsvendor problem in grocery retail (fresh products)

**Key Insight**: Participants are consumers → they have real domain knowledge about grocery shopping behavior

---

## 2. Task Design

### 2.1 Participant Task

| Aspect | Decision |
|--------|----------|
| **Role** | Store manager ordering fresh products |
| **Decisions per scenario** | 2: (1) Demand forecast, (2) Order quantity |
| **Number of scenarios** | ~10 (TBD) |
| **Feedback between scenarios** | No (avoid path dependency) |
| **Products** | Realistic grocery items (salad, bakery, etc.) |

### 2.2 Information Display

**Scenario text**: ~150 words, flexible, verbose naturalistic briefing containing:
- All feature values as observable facts (weather, day, price, etc.)
- Hidden information (events, holidays) embedded naturally - no causal hints
- Irrelevant details for realism (noise)
- No explicit mention of "AI doesn't know X"

**Demand visualization**: Time series (14 days history)

**Feature display**: Table with 3 columns
| | Last Week Same Day | Today | Tomorrow |
|---|---|---|---|
| Temperature | 24°C | 26°C | 28°C |
| Rain | No | No | No |
| ... | ... | ... | ... |

**Cost structure**: Shown in instructions + separate info box per scenario

### 2.3 AI Interaction

**Status**: TBD - Multiple options under consideration

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A: Predefined questions** | Fixed set of clickable questions | Control, no hallucination | Less natural |
| **B: Free text** | Participant types questions | Natural, realistic | Hallucination risk, harder to analyze |
| **C: Hybrid** | Predefined + one free text option | Balance | Complexity |

**AI Response Style**: Short, factual (CRITICAL: must be factual)

**Key Interaction**: Participant asks "Did you consider X?" → AI answers yes/no based on whether feature is in model

---

## 3. Data Generating Process (DGP)

### 3.1 True Demand Model

```
TRUE_DEMAND = base_level
            + Σ (beta_i × visible_feature_i)
            + Σ (beta_j × hidden_feature_j)
            + noise ~ N(0, σ)
```

### 3.2 Feature Structure

| Category | Examples | Visible to AI? | Effect |
|----------|----------|----------------|--------|
| **Weather** | Temperature, rain | ✓ Yes | Calibrated |
| **Calendar** | Weekday, month | ✓ Yes | Calibrated |
| **Price** | Price, promotion | ✓ Yes | Calibrated |
| **Local Events** | Sports, festival | ✗ No | Domain knowledge |
| **School Calendar** | Holiday, vacation | ✗ No | Domain knowledge |
| **Competitor** | Store closed nearby | ✗ No | Domain knowledge |
| **Irrelevant** | Parking lot work, new manager | N/A | β = 0 (noise) |

### 3.3 Effect Sizes (Betas)

**Decision**: Realistic effects - can be large or subtle depending on product

**To calibrate**: Effect sizes should make domain knowledge valuable but not trivial
- Document calibration process
- Test with pilot data

### 3.4 Noise Level

**Proposal**: σ ≈ 5 units (~10% of mean demand)
**Status**: To be calibrated

### 3.5 AI Model Design

**Status**: OPEN QUESTION - See Section 5

---

## 4. Scenario Design

### 4.1 Scenario Types

| Type | Hidden Feature Active? | AI Accuracy | Optimal Action |
|------|------------------------|-------------|----------------|
| **Trust AI** | No | Good | Follow AI |
| **Override Up** | Yes (positive effect) | Under-forecasts | Adjust upward |
| **Override Down** | Yes (negative effect) | Over-forecasts | Adjust downward |
| **Noise Test** | No (irrelevant info present) | Good | Follow AI (don't over-adjust) |

**Target Balance**: ~50% trust AI, ~50% override

### 4.2 Scenario Content

- Explicit, realistic grocery scenarios
- Observable states only ("there is a football match" not "colleague mentioned")
- No causal hints ("match might increase traffic")
- Domain knowledge applicable (participants are consumers)

### 4.3 Example Scenario

```
You're responsible for ordering fresh salads at a supermarket in Munich,
located near the Allianz Arena. Tomorrow is Friday, June 14th.

The weather forecast shows a warm sunny day reaching 28°C. No rain expected.
It's a regular week - the salads are at standard price (€3.20), no promotions
running.

Bayern Munich has a Champions League home game tomorrow evening. It's not a
school holiday. The store's parking lot is being repainted this week.

Last Friday was a typical day with steady demand throughout the week.
```

**Contains**:
- Temperature: 28°C (visible, positive effect)
- Rain: No (visible)
- Sports event: Yes (HIDDEN, positive effect)
- School holiday: No (hidden, but no effect this scenario)
- Parking lot: Irrelevant noise

---

## 5. AI Model Design Options (Analyzed)

**Critical Question**: How do we design the AI model across scenarios?

### Option A: Single Consistent Model

**Concept**: One fixed AI model across all scenarios. Always sees same features, always blind to same things.

| Aspect | Detail |
|--------|--------|
| **AI always sees** | Weather, price, day of week, historical sales |
| **AI never sees** | Local events, school holidays, competitor actions |
| **What varies** | Scenario context (which hidden factors are active) |

**Pros**:
- Clean experimental design
- Participants learn AI's blind spots → can measure learning
- Reduced cognitive load
- Cleaner statistical analysis

**Cons**:
- May become predictable ("always adjust for events")
- Less realistic (real world has varying AI tools)
- Ceiling effects once learned

**Mitigation**: Vary magnitude of adjustments, include "no hidden factor" scenarios

---

### Option B: Varying Model Quality

**Concept**: 2-3 different AI models with different capabilities. Participant must calibrate trust to model.

| Model | Features | Accuracy | Scenarios |
|-------|----------|----------|-----------|
| **Full Model** | All standard features | Good (±10%) | 4 |
| **Limited Model** | Weather + price only | Moderate (±25%) | 6 |

**Pros**:
- Tests meta-cognition: Can participants assess AI capability?
- Richer behavioral data
- Prevents simple heuristics
- More realistic (different tools have different quality)

**Cons**:
- Higher cognitive load
- Fewer scenarios per model type
- Potential confusion

**Mitigation**: Use only 2 models, label them clearly (Model A vs B), provide feedback

---

### Option C: Product-Specific Models

**Concept**: Each product category has its own AI with distinct capabilities.

| Product | AI Knows | AI Doesn't Know | Consumer Intuition |
|---------|----------|-----------------|-------------------|
| **Fresh Salad** | Temperature, weekday | Fitness events, health campaigns | "Hot day = salad day" |
| **Ice Cream** | Temperature, sunshine | School holidays, festivals | "Kids = ice cream" |
| **Ready Meals** | Weekday, promotions | Sports events, uni schedule | "Game night = quick food" |
| **Bakery** | Weekday, holidays | Local markets, community events | "Weekend brunch" |

**Pros**:
- High ecological validity (real stores do this)
- Leverages genuine consumer knowledge
- Intuitive hidden factors
- More engaging

**Cons**:
- Learning curve per product
- More complex implementation
- Unequal product familiarity across participants

---

### Comparison Matrix

| Criterion | Option A | Option B | Option C |
|-----------|----------|----------|----------|
| **Simplicity** | High | Medium | Medium |
| **Ecological validity** | Medium | Medium | High |
| **Leverages consumer knowledge** | Low | Low | High |
| **Learning measurable** | Yes (one model) | Yes (per model) | Yes (per product) |
| **Cognitive load** | Low | High | Medium |
| **Implementation complexity** | Low | Medium | Medium |

### Decision: Option A (Single Consistent Model)

**Rationale**:
- Cleanest for research design
- Foundation model compatible (same feature structure)
- Simpler for participants (one model to learn)
- Product differentiation via effect sizes, not model structure

**See**: [STUDY_PLAN_AND_ARCHITECTURE.md](STUDY_PLAN_AND_ARCHITECTURE.md) Section 12.3

---

## 6. Experimental Conditions

**Status**: TBD

Potential manipulations:
- AI availability (no AI vs AI available)
- AI explanation depth
- Question interaction mode

---

## 7. Metrics & Evaluation

### 7.1 Primary Outcomes

| Metric | Calculation | Measures |
|--------|-------------|----------|
| Forecast deviation | Participant - AI forecast | Trust in AI prediction |
| Order deviation | Participant - AI recommendation | Trust in AI decision |
| Forecast accuracy | Participant forecast - True demand | Judgment quality |
| Decision quality | Profit vs optimal profit | Overall performance |

### 7.2 Behavioral Measures

- Questions asked (which, how many)
- Time spent per scenario
- Information seeking patterns

### 7.3 Appropriate Reliance

**Key**: Did participant adjust when they SHOULD and follow when they SHOULD?

| Scenario Type | Correct Behavior | Incorrect Behavior |
|---------------|------------------|-------------------|
| Trust AI | Follow AI (±small) | Large unnecessary adjustment |
| Override needed | Meaningful adjustment in right direction | Following AI blindly |

---

## 8. Implementation Pipeline

### Phase 1: DGP & Scenarios
- [ ] Finalize feature structure and betas
- [ ] Create scenario generation code
- [ ] Generate 10-15 scenarios
- [ ] Write scenario texts
- [ ] Pre-generate AI responses

### Phase 2: UI Development
- [ ] Design interface mockup
- [ ] Build Streamlit/survey platform
- [ ] Implement AI interaction (question mechanism TBD)

### Phase 3: Pilot Testing
- [ ] Internal testing
- [ ] Calibrate difficulty
- [ ] Refine scenarios

### Phase 4: Main Study
- [ ] Prolific recruitment
- [ ] Data collection
- [ ] Analysis

---

## 9. Open Questions Log

| Question | Status | Notes |
|----------|--------|-------|
| AI interaction mode (predefined vs free) | TBD | Mark options, decide later |
| AI model design across scenarios | EXPLORING | Subagent analysis in progress |
| Exact number of scenarios | TBD | ~10, depends on complexity |
| Effect size calibration | TBD | Needs pilot testing |
| Noise level calibration | TBD | Proposal: σ ≈ 5 |
| Feedback between scenarios | DECIDED | No feedback (avoid path dependency) |

---

## 10. Change Log

| Date | Change |
|------|--------|
| 2024-02-09 | Initial document created |
| 2024-02-09 | Decided: Option B (Newsvendor) with 2 decisions |
| 2024-02-09 | Decided: Verbose scenarios, observable states |
| 2024-02-09 | Decided: No path dependency (no feedback) |
| 2024-02-09 | Decided: AI responses short and factual |
| 2024-02-09 | Exploring: AI model design options |
| 2024-02-09 | Added 3 AI model options (A: consistent, B: varying quality, C: product-specific) |
| 2024-02-09 | DECIDED: Option A (constant feature set) |
| 2024-02-09 | DECIDED: Predefined questions (not free text) |
| 2024-02-09 | DECIDED: 10 scenarios, single condition for pre-study |
| 2024-02-09 | Created STUDY_PLAN_AND_ARCHITECTURE.md with full specifications |

