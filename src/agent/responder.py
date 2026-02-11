"""Agent responder for generating AI assistant responses."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Question:
    """A predefined question that can be asked to the AI."""

    id: str
    text: str
    category: str
    always_available: bool = True


# Standard questions available in all scenarios
STANDARD_QUESTIONS = [
    Question(
        id="explain_forecast",
        text="How did you calculate this forecast?",
        category="explanation",
    ),
    Question(
        id="confidence",
        text="How confident are you in this forecast?",
        category="uncertainty",
    ),
    Question(
        id="what_factors",
        text="What factors does your model consider?",
        category="explanation",
    ),
    Question(
        id="why_order_more",
        text="Why do you recommend ordering more than the forecast?",
        category="recommendation",
    ),
    Question(
        id="what_missing",
        text="What information might you be missing?",
        category="limitations",
    ),
]


class AgentResponder:
    """Handles AI agent responses for a scenario."""

    def __init__(self, scenario_config: Dict):
        """
        Initialize responder with scenario configuration.

        Args:
            scenario_config: Full scenario config from YAML
        """
        self.scenario = scenario_config
        self.ai_responses = scenario_config.get("ai_responses", {})
        self.features = scenario_config.get("features", {})

    def get_available_questions(self) -> List[Question]:
        """Get list of questions available for this scenario."""
        questions = STANDARD_QUESTIONS.copy()

        # Add scenario-specific question based on hidden features
        hidden = self.features.get("hidden", {})

        if hidden.get("school_holiday"):
            questions.append(
                Question(
                    id="consider_school_holiday",
                    text="Did you account for the school holidays?",
                    category="specific",
                    always_available=False,
                )
            )

        if hidden.get("sports_event"):
            questions.append(
                Question(
                    id="consider_football_match",
                    text="Did you consider the football match?",
                    category="specific",
                    always_available=False,
                )
            )

        if hidden.get("local_festival"):
            questions.append(
                Question(
                    id="consider_festival",
                    text="Did you consider the festival happening nearby?",
                    category="specific",
                    always_available=False,
                )
            )

        if hidden.get("local_market"):
            questions.append(
                Question(
                    id="consider_farmers_market",
                    text="Did you account for the farmers market?",
                    category="specific",
                    always_available=False,
                )
            )

        # Check for weather mismatch scenario
        if hidden.get("weather_actually"):
            questions.append(
                Question(
                    id="consider_weather",
                    text="Are you sure about the weather forecast?",
                    category="specific",
                    always_available=False,
                )
            )

        return questions

    def answer_question(self, question_id: str) -> str:
        """
        Get the response for a question.

        Args:
            question_id: ID of the question to answer

        Returns:
            Response text
        """
        # First check if there's a specific response in the scenario
        if question_id in self.ai_responses:
            return self.ai_responses[question_id].strip()

        # Check for alternative key formats
        alt_keys = {
            "consider_school_holiday": "consider_school_holiday",
            "consider_football_match": "consider_football_match",
            "consider_festival": ["consider_festival", "consider_street_festival"],
            "consider_farmers_market": "consider_farmers_market",
            "consider_weather": "consider_weather",
            "consider_specific": "consider_specific",
        }

        if question_id in alt_keys:
            keys = alt_keys[question_id]
            if isinstance(keys, str):
                keys = [keys]
            for key in keys:
                if key in self.ai_responses:
                    return self.ai_responses[key].strip()

        # Fallback generic responses
        fallbacks = {
            "explain_forecast": "My forecast is based on historical patterns, weather conditions, day of week, and pricing. I can provide more details if you ask about specific factors.",
            "confidence": "I have moderate confidence in this forecast. Typical accuracy is within plus or minus 10% of the predicted value.",
            "what_factors": "My model considers: temperature, rainfall, day of week, current price, and promotional status. I also factor in historical sales patterns.",
            "why_order_more": "I recommend ordering slightly above the forecast because stockouts are typically more costly than having some leftover inventory.",
            "what_missing": "I don't have information about local events, school schedules, competitor activities, or real-time changes to weather forecasts.",
        }

        return fallbacks.get(
            question_id, "I don't have specific information about that."
        )

    def get_forecast_explanation(self) -> str:
        """Get the forecast explanation response."""
        return self.answer_question("explain_forecast")

    def get_recommendation_explanation(self) -> str:
        """Get the order recommendation explanation."""
        return self.answer_question("why_order_more")

    def get_limitations(self) -> str:
        """Get what the AI admits it doesn't know."""
        return self.answer_question("what_missing")
