# To Be Discussed

## AI Recommendation vs. Forecast Display

**Decision made**: Removed the KI-EMPFEHLUNG (AI order recommendation) from the dashboard. Participants now only see the KI-PROGNOSE (demand forecast) and must determine their own order quantity.

**Chatbot behavior**: The LLM assistant can give conceptual advice about order quantities (e.g. "order a bit above the forecast because stockouts are more costly") but does NOT have access to the calculated recommendation number.

**Open questions**:
- Should the ai_recommendation still be computed and stored in trial records for post-hoc analysis?
- Should there be a treatment condition where some participants DO see the recommendation (A/B design)?
- How much conceptual guidance should the chatbot give? Currently it can explain the cost asymmetry logic but not name a specific number.
