# Study Design v2: AI Agents as Documentation Translators

## Research Question

> Do conversational AI interfaces improve human utilization of technical model documentation in decision-making tasks?

## Core Design Principle

**Same information, different access:**
- Both groups have access to identical technical model documentation
- Treatment group additionally has AI chat to query the documentation
- AI acts as **translator/synthesizer**, not information source

---

## Study Structure

### Groups

| Group | Sees | Can Do |
|-------|------|--------|
| **Control** | Technical Log + UI | Read documentation manually |
| **Treatment** | Technical Log + UI + AI Chat | Query AI about documentation |

### Scenarios

**10 scenarios, balanced:**
- 5× Trust AI (no hidden factor, or noise/trap)
- 5× Override needed (clear hidden factor)
  - 3× Increase order
  - 2× Decrease order

**Difficulty = WHETHER to adjust:**
| Difficulty | Description | Example |
|------------|-------------|---------|
| Easy | Clear signal, obvious decision | Festival mentioned + "letztes Jahr 30% mehr verkauft" |
| Medium | Factor mentioned, impact ambiguous | Small local event - does it matter? |
| Hard | Trap/noise - looks important but isn't | Parking renovation mentioned (irrelevant) |

---

## UI Components

### 1. Technical Model Documentation (Both Groups)

```
┌─────────────────────────────────────────────────────────────┐
│ 📋 Modelldokumentation                       [▼ Erweitern]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ MODELL-VERSION: demand_forecast_v2.3                        │
│ TRAINING: POS-Daten 2022-2024                               │
│                                                             │
│ INPUT-FEATURES                                              │
│ ──────────────────────────────────────────────────────────  │
│ Feature              │ Wert        │ Effekt auf Nachfrage   │
│ ─────────────────────┼─────────────┼─────────────────────── │
│ temperatur_celsius   │ 28          │ +1.5 pro °C über 20    │
│ regen                │ Nein        │ -8.0 wenn Ja           │
│ wochentag            │ Freitag     │ +5.0                   │
│ aktion_aktiv         │ Nein        │ +12.0 wenn Ja          │
│ basisnachfrage       │ —           │ 35.0                   │
│                                                             │
│ OUTPUT                                                      │
│ ──────────────────────────────────────────────────────────  │
│ prognose: 65 Einheiten                                      │
│ konfidenzintervall_70: [57, 73]                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. AI Chat Interface (Treatment Only)

**3 Standardized Question Proposals:**

| # | Question | Purpose |
|---|----------|---------|
| 1 | "Welche Daten nutzt das Modell?" | Understand inputs |
| 2 | "Wie berechnet sich die Prognose?" | Understand logic |
| 3 | "Berücksichtigt das Modell [___]?" | Free exploration |

**Free text input:** Participants can type any question

---

## LLM Configuration

### Knowledge Base (per scenario)

```yaml
llm_context:
  # What the model "knows" (from technical log)
  model_features:
    - name: "temperatur_celsius"
      current_value: 28
      coefficient: "+1.5 pro °C über 20"
    - name: "regen"
      current_value: false
      coefficient: "-8.0 wenn true"
    - name: "wochentag"
      current_value: "Freitag"
      coefficient: "+5.0"
    - name: "aktion_aktiv"
      current_value: false
      coefficient: "+12.0 wenn true"

  base_demand: 35
  forecast: 65
  confidence_interval: [57, 73]

  # Cost structure
  costs:
    purchase_price: 2.50
    selling_price: 4.99
    salvage_value: 0.00
    profit_per_unit: 2.49
    loss_per_unit: 2.50

  # Model metadata
  model_version: "demand_forecast_v2.3"
  training_period: "2022-2024"
  data_source: "POS transactions"
```

### System Prompt

```
Du bist ein Assistent, der die technische Modelldokumentation erklärt.

DEINE AUFGABE:
- Erkläre die Modelldokumentation in einfacher Sprache
- Beantworte Fragen zu den genutzten Features und deren Effekten
- Wenn nach einem Feature gefragt wird, prüfe ob es in der Dokumentation steht

DEIN WISSEN (aus der Modelldokumentation):
{model_features}
{costs}
{metadata}

VERHALTEN:
- Bei "Welche Daten nutzt das Modell?" → Liste die Features aus der Doku
- Bei "Wie berechnet sich die Prognose?" → Erkläre: Basis + Feature-Effekte
- Bei "Berücksichtigt das Modell X?" → Prüfe ob X in der Feature-Liste ist
  - Wenn JA: Erkläre den Effekt
  - Wenn NEIN: "X ist nicht in der Modelldokumentation aufgeführt"

WICHTIG:
- Gib NUR Informationen aus der Dokumentation wieder
- Sage NIEMALS "Sie sollten X berücksichtigen"
- Keine Empfehlungen, nur Fakten aus der Doku
```

---

## Scenario Matrix (Revised)

| ID | Product | Hidden Factor | Type | Difficulty | Notes |
|----|---------|---------------|------|------------|-------|
| S01 | Salat | None | Trust | Easy | Baseline |
| S02 | Eis | Schulferien | Override ↑ | Easy | Clear mention in narrative |
| S03 | Fertiggericht | None | Trust | Easy | Baseline |
| S04 | Backwaren | Stadtfest | Override ↑ | Easy | Clear mention + past reference |
| S05 | Salat | Wetterunsicherheit | Override ↓ | Medium | Conflicting weather info |
| S06 | Eis | Noise (Parkplatz) | Trust | Medium | Trap - looks relevant but isn't |
| S07 | Fertiggericht | Fußball CL-Finale | Override ↑ | Easy | Clear mention |
| S08 | Backwaren | Wochenmarkt | Override ↓ | Medium | Mentioned but impact unclear |
| S09 | Salat | Kleines Event | Trust | Hard | Trap - mentioned but minimal effect |
| S10 | Eis | Volksfest | Override ↑ | Easy | Clear large event |

---

## Measurement

### Primary Outcome

**Adjustment Accuracy (binary):**
- ✅ Correct = Adjusted in right direction OR correctly didn't adjust
- ❌ Incorrect = Missed needed adjustment OR over-adjusted on noise

### Secondary Outcomes

1. **Documentation usage:** Did they expand the technical log?
2. **AI interaction:** Number and type of questions asked
3. **Time to decision:** How long per scenario?
4. **Question quality:** Did they ask about the relevant hidden factor?

### Analysis

```
Adjustment_Accuracy ~ Group + Difficulty + (1|Participant) + (1|Scenario)
```

---

## Implementation Tasks

1. [ ] Update `scenario_backlog.yaml` with revised narratives
2. [ ] Add "Technical Log" UI component (collapsible)
3. [ ] Implement 3 standardized question buttons
4. [ ] Add free-text question input with "[___]" placeholder
5. [ ] Create LLM integration with constrained system prompt
6. [ ] Add control group condition (no AI chat)
7. [ ] Add tracking: documentation expansion, question types
8. [ ] Pilot test with 3-5 bachelor students
