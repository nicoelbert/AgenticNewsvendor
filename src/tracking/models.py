"""Data models for experiment tracking."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from datetime import datetime
import uuid


@dataclass
class TrialRecord:
    """Record of a single scenario trial."""

    participant_id: str
    scenario_id: str
    trial_number: int

    # Timestamps
    start_time: datetime
    end_time: Optional[datetime] = None

    # Scenario info
    product: str = ""
    ai_forecast: float = 0.0
    ai_recommendation: float = 0.0
    true_expected_demand: float = 0.0
    actual_demand: float = 0.0
    optimal_order: float = 0.0

    # Participant decisions
    participant_forecast: Optional[float] = None
    participant_order: Optional[float] = None

    # Interaction data
    questions_asked: List[str] = field(default_factory=list)
    question_timestamps: List[str] = field(default_factory=list)

    # Computed after submission
    forecast_adjustment: Optional[float] = None  # participant - AI
    order_adjustment: Optional[float] = None  # participant - AI rec
    forecast_error: Optional[float] = None  # participant - true
    profit: Optional[float] = None
    optimal_profit: Optional[float] = None

    @property
    def time_spent_seconds(self) -> Optional[float]:
        """Compute time spent on this trial."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def compute_metrics(self) -> None:
        """Compute derived metrics after decisions are made."""
        if self.participant_forecast is not None:
            self.forecast_adjustment = self.participant_forecast - self.ai_forecast
            self.forecast_error = self.participant_forecast - self.true_expected_demand

        if self.participant_order is not None:
            self.order_adjustment = self.participant_order - self.ai_recommendation

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        d = asdict(self)
        # Convert datetime to ISO string
        d["start_time"] = self.start_time.isoformat() if self.start_time else None
        d["end_time"] = self.end_time.isoformat() if self.end_time else None
        d["time_spent_seconds"] = self.time_spent_seconds
        return d


@dataclass
class ParticipantSession:
    """Session data for a participant."""

    participant_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    condition: str = "ai_with_questions"

    # Timestamps
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    # Progress
    current_trial: int = 0
    completed: bool = False
    scenario_order: List[str] = field(default_factory=list)

    # Trials
    trials: List[TrialRecord] = field(default_factory=list)

    # Survey responses (filled at end)
    survey_responses: Dict = field(default_factory=dict)
    demographics: Dict = field(default_factory=dict)

    def add_trial(self, trial: TrialRecord) -> None:
        """Add a completed trial."""
        self.trials.append(trial)
        self.current_trial = len(self.trials)

    def get_trial(self, scenario_id: str) -> Optional[TrialRecord]:
        """Get trial by scenario ID."""
        for trial in self.trials:
            if trial.scenario_id == scenario_id:
                return trial
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "participant_id": self.participant_id,
            "condition": self.condition,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "current_trial": self.current_trial,
            "completed": self.completed,
            "scenario_order": self.scenario_order,
            "trials": [t.to_dict() for t in self.trials],
            "survey_responses": self.survey_responses,
            "demographics": self.demographics,
        }
