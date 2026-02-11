"""LLM-based responder with constrained knowledge base.

The LLM only knows what's in the model documentation - it acts as a
translator of technical documentation, not as an additional information source.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class ModelDocumentation:
    """Technical documentation that the LLM has access to."""

    # Model metadata
    model_version: str = "demand_forecast_v2.3"
    training_period: str = "2022-2024"
    data_source: str = "POS transactions"

    # Feature configuration
    base_level: float = 0.0
    visible_betas: Dict[str, float] = None

    # Current scenario values
    current_features: Dict[str, Any] = None

    # Forecast output
    forecast: int = 0
    confidence_interval: tuple = (0, 0)

    # Cost structure
    price: float = 0.0
    cost: float = 0.0
    salvage: float = 0.0
    profit_per_unit: float = 0.0
    loss_per_unit: float = 0.0

    def __post_init__(self):
        if self.visible_betas is None:
            self.visible_betas = {}
        if self.current_features is None:
            self.current_features = {}


def build_system_prompt(doc: ModelDocumentation) -> str:
    """Build the constrained system prompt from model documentation."""

    # Format feature list
    feature_lines = []
    beta_labels = {
        "temperature": "Temperatur",
        "rain": "Regen",
        "weekday_friday": "Wochentag (Freitag)",
        "weekday_saturday": "Wochentag (Samstag)",
        "weekday_sunday": "Wochentag (Sonntag)",
        "promotion": "Aktion/Promotion",
    }

    for beta_name, coeff in doc.visible_betas.items():
        label = beta_labels.get(beta_name, beta_name)
        sign = "+" if coeff >= 0 else ""
        feature_lines.append(f"  - {label}: Koeffizient {sign}{coeff:.1f}")

    features_text = "\n".join(feature_lines) if feature_lines else "  (keine Features definiert)"

    # Format current values
    current_lines = []
    for key, value in doc.current_features.items():
        if key == "temperature":
            current_lines.append(f"  - Temperatur: {value}°C")
        elif key == "rain":
            current_lines.append(f"  - Regen: {'Ja' if value else 'Nein'}")
        elif key == "weekday":
            current_lines.append(f"  - Wochentag: {value}")
        elif key == "promotion":
            current_lines.append(f"  - Aktion: {'Ja' if value else 'Nein'}")
        elif key == "price":
            current_lines.append(f"  - Preis: €{value:.2f}")

    current_text = "\n".join(current_lines) if current_lines else "  (keine aktuellen Werte)"

    ci_low, ci_high = doc.confidence_interval

    return f"""Du bist ein Assistent, der die technische Modelldokumentation eines Nachfrage-Prognosemodells erklärt.

DEINE ROLLE:
Du erklärst NUR was in der Modelldokumentation steht. Du bist ein Übersetzer von technischer Dokumentation - KEINE zusätzliche Informationsquelle.

MODELLDOKUMENTATION:
================================================================================
PROGNOSEMODELL {doc.model_version}
Training: {doc.data_source} ({doc.training_period})

INPUT-FEATURES & KOEFFIZIENTEN:
  - Basisnachfrage: {doc.base_level:.0f} Einheiten
{features_text}

AKTUELLE WERTE (dieses Szenario):
{current_text}

OUTPUT:
  - Prognose: {doc.forecast} Einheiten
  - 70%-Konfidenzintervall: [{ci_low}, {ci_high}]

KOSTENSTRUKTUR:
  - Einkaufspreis: €{doc.cost:.2f}
  - Verkaufspreis: €{doc.price:.2f}
  - Restwert (unverkauft): €{doc.salvage:.2f}
  - Gewinn pro verkaufter Einheit: €{doc.profit_per_unit:.2f}
  - Verlust pro unverkaufter Einheit: €{doc.loss_per_unit:.2f}
================================================================================

VERHALTEN:
1. Bei "Welche Daten nutzt das Modell?" → Liste die Features aus der Dokumentation auf
2. Bei "Wie berechnet sich die Prognose?" → Erkläre: Basisnachfrage + Feature-Effekte
3. Bei "Berücksichtigt das Modell X?" → Prüfe ob X in der Feature-Liste ist:
   - Wenn JA: "X ist enthalten, Koeffizient: ..."
   - Wenn NEIN: "X ist nicht in der Modelldokumentation aufgeführt"

WICHTIGE REGELN:
- Gib NUR Informationen aus der Dokumentation oben wieder
- Sage NIEMALS "Sie sollten X berücksichtigen" oder "Vielleicht ist X wichtig"
- Keine Empfehlungen, nur Fakten aus der Doku
- Antworte auf Deutsch
- Halte Antworten kurz und präzise (max 100 Wörter)
- Verwende Markdown für Formatierung (fett für wichtige Begriffe)"""


class LLMResponder:
    """LLM-based responder that only knows model documentation."""

    def __init__(
        self,
        scenario_config: Dict,
        product_config: Dict,
        api_key: Optional[str] = None,
    ):
        """
        Initialize LLM responder with scenario context.

        Args:
            scenario_config: Full scenario configuration
            product_config: Product configuration with betas
            api_key: Anthropic API key (or from env ANTHROPIC_API_KEY)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = None

        if ANTHROPIC_AVAILABLE and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)

        # Build documentation from configs
        visible = scenario_config.get("features", {}).get("visible", {})
        computed = scenario_config.get("computed", {})

        forecast = computed.get("ai_forecast", 0)
        ci_margin = int(forecast * 0.15)

        self.documentation = ModelDocumentation(
            base_level=product_config.get("base_level", 0),
            visible_betas=product_config.get("visible_betas", {}),
            current_features=visible,
            forecast=forecast,
            confidence_interval=(forecast - ci_margin, forecast + ci_margin),
            price=product_config.get("price", 0),
            cost=product_config.get("cost", 0),
            salvage=product_config.get("salvage", 0),
            profit_per_unit=product_config.get("profit_per_unit", 0),
            loss_per_unit=product_config.get("loss_per_unit", 0),
        )

        self.system_prompt = build_system_prompt(self.documentation)
        self.conversation_history: List[Dict] = []

    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self.client is not None

    def ask(self, question: str) -> str:
        """
        Ask a question to the LLM.

        Args:
            question: User's question in German

        Returns:
            LLM response
        """
        if not self.is_available():
            return self._fallback_response(question)

        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": question,
        })

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=self.system_prompt,
                messages=self.conversation_history,
            )

            assistant_message = response.content[0].text

            # Add response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message,
            })

            return assistant_message

        except Exception as e:
            # On error, return fallback and remove failed message
            self.conversation_history.pop()
            return self._fallback_response(question, error=str(e))

    def _fallback_response(self, question: str, error: str = None) -> str:
        """Generate fallback response when LLM is unavailable."""

        question_lower = question.lower()
        doc = self.documentation

        # Feature list
        beta_labels = {
            "temperature": "Temperatur",
            "rain": "Regen",
            "weekday_friday": "Wochentag (Freitag)",
            "weekday_saturday": "Wochentag (Samstag)",
            "weekday_sunday": "Wochentag (Sonntag)",
            "promotion": "Aktion/Promotion",
        }
        feature_list = "\n".join([
            f"• {beta_labels.get(k, k)}"
            for k in doc.visible_betas.keys()
        ])

        if "welche daten" in question_lower or "features" in question_lower:
            return f"""Das Modell nutzt folgende **Eingabedaten**:

{feature_list}

Basisnachfrage: {doc.base_level:.0f} Einheiten"""

        if "berechnet" in question_lower or "prognose" in question_lower:
            ci_low, ci_high = doc.confidence_interval
            return f"""Die Prognose berechnet sich aus:

**Basisnachfrage** + Summe der Feature-Effekte

**Ergebnis:** {doc.forecast} Einheiten
**70%-Konfidenzintervall:** [{ci_low}, {ci_high}]"""

        if "berücksichtigt" in question_lower:
            # Extract the term being checked
            import re
            match = re.search(r"['\"]([^'\"]+)['\"]", question)
            if match:
                term = match.group(1).lower()

                # Check known features
                known_terms = {
                    "temperatur": "temperature",
                    "temp": "temperature",
                    "regen": "rain",
                    "wochentag": "weekday_friday",
                    "aktion": "promotion",
                    "promotion": "promotion",
                }

                for keyword, feature_key in known_terms.items():
                    if keyword in term:
                        if feature_key in doc.visible_betas:
                            coeff = doc.visible_betas[feature_key]
                            return f"""✅ **"{term}"** ist in der Modelldokumentation enthalten.

Koeffizient: {'+' if coeff >= 0 else ''}{coeff:.1f}"""

                return f"""❌ **"{term}"** ist **nicht** in der Modelldokumentation aufgeführt.

Das Modell verwendet nur:
{feature_list}"""

        # Generic fallback
        if error:
            return f"(Fehler bei LLM-Anfrage: {error})\n\nSiehe Modelldokumentation für Details."

        return "Siehe Modelldokumentation für Details zu den Modell-Features und der Prognoseberechnung."

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
