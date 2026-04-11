"""LLM-based chatbot responder for the newsvendor experiment.

Provides a conversational AI assistant backed by Claude that has access to
model documentation, scenario context, and newsvendor domain knowledge.
Guardrails prevent revealing hidden features or making ordering decisions
for the participant.
"""

import os
from dataclasses import dataclass, field
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
    ai_recommendation: int = 0
    confidence_interval: tuple = (0, 0)

    # Cost structure
    price: float = 0.0
    cost: float = 0.0
    salvage: float = 0.0
    profit_per_unit: float = 0.0
    loss_per_unit: float = 0.0

    # Scenario context
    narrative: str = ""
    demand_history: List[int] = field(default_factory=list)

    def __post_init__(self):
        if self.visible_betas is None:
            self.visible_betas = {}
        if self.current_features is None:
            self.current_features = {}


BETA_LABELS = {
    "temperature": "Temperatur",
    "rain": "Regen",
    "weekday_friday": "Wochentag (Freitag)",
    "weekday_saturday": "Wochentag (Samstag)",
    "weekday_sunday": "Wochentag (Sonntag)",
    "promotion": "Aktion/Promotion",
}


def build_system_prompt(doc: ModelDocumentation) -> str:
    """Build the system prompt with full context and guardrails."""

    # Format feature list with coefficients
    feature_lines = []
    for beta_name, coeff in doc.visible_betas.items():
        label = BETA_LABELS.get(beta_name, beta_name)
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

    # Format demand history
    if doc.demand_history:
        history_str = ", ".join(str(d) for d in doc.demand_history)
        avg_demand = sum(doc.demand_history) / len(doc.demand_history)
        history_text = f"  Letzte {len(doc.demand_history)} Tage: [{history_str}]\n  Durchschnitt: {avg_demand:.1f} Einheiten"
    else:
        history_text = "  (keine Nachfragehistorie verfügbar)"

    return f"""Du bist ein hilfreicher KI-Assistent, der einem Filialleiter bei Bestellentscheidungen für frische Lebensmittel hilft. Du erklärst das Nachfrage-Prognosemodell, hilfst beim Verständnis der Daten und unterstützt die Entscheidungsfindung.

AKTUELLE SITUATION:
{doc.narrative if doc.narrative else "(keine Situationsbeschreibung verfügbar)"}

================================================================================
MODELLDOKUMENTATION — PROGNOSEMODELL {doc.model_version}
Training: {doc.data_source} ({doc.training_period})

INPUT-FEATURES & KOEFFIZIENTEN:
  - Basisnachfrage: {doc.base_level:.0f} Einheiten
{features_text}

AKTUELLE WERTE (dieses Szenario):
{current_text}

NACHFRAGEHISTORIE:
{history_text}

PROGNOSE-OUTPUT:
  - KI-Prognose: {doc.forecast} Einheiten
  - 70%-Konfidenzintervall: [{ci_low}, {ci_high}]

KOSTENSTRUKTUR:
  - Einkaufspreis: €{doc.cost:.2f} pro Einheit
  - Verkaufspreis: €{doc.price:.2f} pro Einheit
  - Restwert (unverkauft): €{doc.salvage:.2f} pro Einheit
  - Gewinn pro verkaufter Einheit: €{doc.profit_per_unit:.2f}
  - Verlust pro unverkaufter Einheit: €{doc.loss_per_unit:.2f}
================================================================================

HINTERGRUNDWISSEN ZUR BESTELLENTSCHEIDUNG (Bestellproblem unter Unsicherheit):
- Der Filialleiter muss eine Bestellmenge festlegen, BEVOR die tatsächliche Nachfrage bekannt ist.
- Zu viel bestellt → unverkaufte Ware verliert an Wert (Verlust pro Einheit: €{doc.loss_per_unit:.2f})
- Zu wenig bestellt → entgangener Gewinn durch Fehlmengen (entgangener Gewinn pro Einheit: €{doc.profit_per_unit:.2f})
- Wenn Fehlmengen teurer sind als Überbestände, lohnt es sich eher etwas mehr zu bestellen (und umgekehrt).
- Hilf dem Teilnehmer, selbst über die optimale Bestellmenge nachzudenken, basierend auf der Prognose und der Kostenstruktur.

DEIN VERHALTEN:
- Antworte einfach und verständlich, wie ein erfahrener Kollege — NICHT wie ein Datenanalyst
- Erkläre Zusammenhänge in Alltagssprache (z.B. "Bei warmem Wetter kaufen die Leute mehr" statt "Temperatur-Koeffizient: +1.5 pro Grad")
- Nenne Koeffizienten und Formeln NUR wenn der Nutzer explizit danach fragt
- Hilf dem Filialleiter, die Situation einzuschätzen und eigene Schlüsse zu ziehen
- Du kannst auf die Nachfragehistorie und Muster eingehen

WICHTIGE REGELN:
- Gib NUR Informationen wieder, die in der obigen Dokumentation stehen
- Nenne NIEMALS eine konkrete Bestellzahl als Empfehlung — du kennst keine berechnete Bestellempfehlung. Hilf dem Teilnehmer stattdessen konzeptionell (z.B. "etwas über/unter der Prognose"), selbst zu einer Entscheidung zu kommen
- Erwähne NICHT, dass es versteckte Features oder unbekannte Faktoren geben könnte
- Erfinde KEINE Daten, Statistiken oder Features, die nicht oben dokumentiert sind
- Antworte auf Deutsch
- Halte Antworten KURZ: maximal 2-4 Sätze, nur bei komplexen Fragen bis zu 80 Wörter
- Verwende Markdown-Formatierung NUR sparsam — keine langen Listen oder Überschriften für einfache Antworten"""


class LLMResponder:
    """Conversational LLM chatbot for the newsvendor experiment."""

    def __init__(
        self,
        scenario_config: Dict,
        product_config: Dict,
        narrative: str = "",
        demand_history: Optional[List[int]] = None,
        ai_recommendation: int = 0,
        api_key: Optional[str] = None,
    ):
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
            ai_recommendation=ai_recommendation,
            confidence_interval=(forecast - ci_margin, forecast + ci_margin),
            price=product_config.get("price", 0),
            cost=product_config.get("cost", 0),
            salvage=product_config.get("salvage", 0),
            profit_per_unit=product_config.get("profit_per_unit", 0),
            loss_per_unit=product_config.get("loss_per_unit", 0),
            narrative=narrative,
            demand_history=demand_history or [],
        )

        self.system_prompt = build_system_prompt(self.documentation)
        self.conversation_history: List[Dict] = []

    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self.client is not None

    def ask(self, question: str) -> str:
        """Ask a question to the LLM chatbot."""
        if not self.is_available():
            return self._fallback_response(question)

        self.conversation_history.append({
            "role": "user",
            "content": question,
        })

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=250,
                system=self.system_prompt,
                messages=self.conversation_history,
            )

            assistant_message = response.content[0].text

            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message,
            })

            return assistant_message

        except Exception as e:
            self.conversation_history.pop()
            return self._fallback_response(question, error=str(e))

    def _fallback_response(self, question: str, error: str = None) -> str:
        """Generate fallback response when LLM is unavailable."""
        if error:
            return f"Der KI-Assistent ist momentan nicht erreichbar. Bitte versuchen Sie es erneut.\n\n(Fehler: {error})"

        if not ANTHROPIC_AVAILABLE:
            return "Der KI-Assistent ist nicht verfügbar: Das Python-Paket `anthropic` ist nicht installiert. Bitte `pip install anthropic` ausführen."

        if not self.api_key:
            return "Der KI-Assistent ist nicht verfügbar: Kein API-Schlüssel konfiguriert. Bitte `ANTHROPIC_API_KEY` in der `.env`-Datei setzen."

        return "Der KI-Assistent ist momentan nicht verfügbar."

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
