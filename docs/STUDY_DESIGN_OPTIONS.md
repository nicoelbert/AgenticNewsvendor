# Study Design Options - Comparison & Recommendation

## Executive Summary

Based on the reference studies (A-J), the key insight is: **published studies use minimal input data**. Most have 2-4 parameters, no time series, and single-decision outputs.

| Reference Study | Input Complexity | Decision |
|-----------------|------------------|----------|
| Study C | Cost structure only (buy €3, sell €12) | Order 0-100 units |
| Study D | 2 binary variables (competitor, sensitivity) | Set price on slider |
| Study F | Shared warehouse rules, demand U[50,150] | Order 0-200 units |
| Study H | 2 suppliers with different reliability/cost | Split order |

---

## Three Design Options

### Option A: Forecast-Focused (Simplest)

**Task**: Participant sees AI forecast → submits their own forecast

**Display**:
```
Product: Winter Jacket
Last Year's November Sales: 847 units
Weather Forecast: Colder than average
Economic Indicator: Consumer confidence UP 3%

AI Forecast: 920 units

YOUR FORECAST: [____] units
```

**Clickable Questions** (7):
1. "How accurate has the AI been historically?"
2. "How did the AI calculate this forecast?"
3. "How reliable is the weather factor?"
4. "What happened last time conditions were similar?"
5. "What's the range of likely outcomes?"
6. "What could make this forecast wrong?"
7. "Should I trust this forecast?"

**Metrics**:
- Forecast deviation from AI (trust measure)
- Questions clicked (information seeking)
- Time to decision

**Pros**: Very simple, matches Study D/E complexity
**Cons**: No ordering decision, limited scope

---

### Option B: Newsvendor-Focused (Recommended for Simplicity)

**Task**: Participant sees costs + AI forecast + AI order recommendation → submits order quantity

**Display**:
```
PRODUCT: Fresh Bakery Items

COST STRUCTURE:
• Purchase: $4/unit
• Sell: $10/unit
• Profit if sold: $6
• Loss if unsold: $4

AI FORECAST: 60 units (range: 40-80)
AI RECOMMENDED ORDER: 68 units

YOUR ORDER: [____] units
```

**Clickable Questions** (7):
1. "Why order more than forecast?" (critical ratio logic)
2. "What's the chance demand exceeds my order?"
3. "What happens if I order exactly 60?"
4. "How confident is the AI?"
5. "What's worst-case if I follow AI?"
6. "What would maximize expected profit?"
7. "Has AI been accurate recently?"

**Metrics**:
- Order deviation from AI recommendation
- Order deviation from optimal (decision quality)
- Questions clicked by category
- Profit achieved

**Pros**: Matches Study C/F/H exactly, clear trust measure, decision quality measurable
**Cons**: Single-stage only

---

### Option C: Two-Stage (Most Comprehensive)

**Stage 1 - Forecast Assessment**:
```
RECENT DEMAND: [52, 38, 41, 44, 43, 47, 45]
Tomorrow: Friday, 28°C, No rain

AI FORECAST: 52 units

YOUR EXPECTATION: [____] units
```

Questions (5): Confidence, drivers, accuracy, range, explanation

**Stage 2 - Order Decision**:
```
AI Forecast: 52 units
Your Expectation: 48 units (from Stage 1)

COSTS: Buy €1.60, Sell €3.20, Salvage €0.40

AI RECOMMENDED ORDER: 55 units

YOUR ORDER: [____] units
```

Questions (5): Why more than forecast, cost tradeoff, risk, conservative option, calculation

**Metrics**:
- Forecast adjustment (Stage 1 trust)
- Order deviation (Stage 2 trust)
- Consistency: Does their order match their forecast belief?
- Decision quality vs. optimal
- Question patterns across both stages

**Pros**: Separates forecast trust from decision trust, richest data
**Cons**: More complex, longer task

---

## Comparison Matrix

| Aspect | Option A (Forecast) | Option B (Newsvendor) | Option C (Two-Stage) |
|--------|---------------------|----------------------|----------------------|
| **Complexity** | Very Low | Low | Medium |
| **Match to refs** | Study D/E | Study C/F/H | Novel but justified |
| **Time/participant** | ~10 min | ~15 min | ~20 min |
| **Trust measures** | Forecast only | Order decision | Both separated |
| **Decision quality** | N/A | Yes | Yes |
| **Questions** | 7 | 7 | 10 (5+5) |
| **Data richness** | Low | Medium | High |

---

## My Recommendation: **Option B (Newsvendor)**

**Why**:

1. **Best match to reference studies**: Studies C, F, H all use this exact structure
2. **Single clear decision**: Order quantity with measurable deviation from AI
3. **Trust operationalized simply**: Do they follow AI recommendation or deviate?
4. **Decision quality measurable**: Compare to theoretical optimal
5. **Reasonable length**: ~15 min per participant
6. **Questions are meaningful**: Critical ratio, uncertainty, risk - all relevant to newsvendor

**Suggested experimental conditions**:

| Condition | Description |
|-----------|-------------|
| **Control** | No AI (just see costs + demand distribution) |
| **AI Forecast Only** | See AI forecast, no order recommendation |
| **Full AI** | See AI forecast + order recommendation + can ask questions |

This tests: Does having an AI recommendation improve decisions? Does access to explanations help?

---

## Implementation Simplicity

For Option B, we need:

```yaml
scenarios:
  - id: "scenario_001"
    product: "fresh_bakery"
    cost_buy: 4
    cost_sell: 10
    demand_low: 40
    demand_high: 80
    ai_forecast: 60
    ai_recommendation: 68
    actual_demand: 55  # for feedback/evaluation

    questions:
      q1:
        text: "Why order more than forecast?"
        answer: "Because stockouts cost more than waste..."
      q2:
        text: "What's the stockout probability?"
        answer: "At 68 units, about 30% chance of stockout..."
      # ... etc
```

**Total scenarios needed**: 10-20 (can vary demand distribution, AI accuracy)

---

## Next Steps (If You Agree with Option B)

1. **Finalize cost parameters** - Keep simple ($4/$10) or use your existing config (€1.60/€3.20)?
2. **Define demand distributions** - Uniform is simplest, matches references
3. **Create scenario variations** - 10-15 distinct scenarios
4. **Write all Q&A pairs** - 7 questions × 15 scenarios = ~105 answers
5. **Build simple UI** - Streamlit or Qualtrics
6. **Pilot internally** - Test flow and timing

---

## Questions for You

1. **Which option appeals most?** A (forecast), B (newsvendor), or C (two-stage)?

2. **Experimental conditions**:
   - 2 conditions (No AI vs Full AI)?
   - 3 conditions (No AI vs AI Forecast vs AI Recommendation)?

3. **Number of scenarios per participant**:
   - 5 (quick, ~10 min)
   - 10 (moderate, ~15 min)
   - 20 (thorough, ~25 min)

4. **Feedback between scenarios**:
   - Show actual demand after each decision?
   - Or no feedback until end?
