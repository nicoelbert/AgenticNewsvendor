"""File-based storage for experiment data."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

from .models import ParticipantSession, TrialRecord


class FileStorage:
    """File-based storage using JSON and Parquet."""

    def __init__(self, base_dir: Path):
        """
        Initialize storage.

        Args:
            base_dir: Directory for storing data files
        """
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / "sessions"
        self.exports_dir = self.base_dir / "exports"

        # Create directories
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, participant_id: str) -> Path:
        """Get path for session file."""
        return self.sessions_dir / f"{participant_id}.json"

    def save_session(self, session: ParticipantSession) -> None:
        """
        Save session to JSON file.

        Args:
            session: Session to save
        """
        path = self._session_path(session.participant_id)
        with open(path, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def load_session(self, participant_id: str) -> Optional[ParticipantSession]:
        """
        Load session from JSON file.

        Args:
            participant_id: Participant ID

        Returns:
            Session or None if not found
        """
        path = self._session_path(participant_id)
        if not path.exists():
            return None

        with open(path, "r") as f:
            data = json.load(f)

        # Reconstruct session
        session = ParticipantSession(
            participant_id=data["participant_id"],
            condition=data["condition"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=(
                datetime.fromisoformat(data["end_time"]) if data["end_time"] else None
            ),
            current_trial=data["current_trial"],
            completed=data["completed"],
            scenario_order=data["scenario_order"],
            survey_responses=data.get("survey_responses", {}),
            demographics=data.get("demographics", {}),
        )

        # Reconstruct trials
        for trial_data in data.get("trials", []):
            trial = TrialRecord(
                participant_id=trial_data["participant_id"],
                scenario_id=trial_data["scenario_id"],
                trial_number=trial_data["trial_number"],
                start_time=datetime.fromisoformat(trial_data["start_time"]),
                end_time=(
                    datetime.fromisoformat(trial_data["end_time"])
                    if trial_data["end_time"]
                    else None
                ),
                product=trial_data.get("product", ""),
                ai_forecast=trial_data.get("ai_forecast", 0),
                ai_recommendation=trial_data.get("ai_recommendation", 0),
                true_expected_demand=trial_data.get("true_expected_demand", 0),
                actual_demand=trial_data.get("actual_demand", 0),
                optimal_order=trial_data.get("optimal_order", 0),
                participant_forecast=trial_data.get("participant_forecast"),
                participant_order=trial_data.get("participant_order"),
                questions_asked=trial_data.get("questions_asked", []),
                question_timestamps=trial_data.get("question_timestamps", []),
                forecast_adjustment=trial_data.get("forecast_adjustment"),
                order_adjustment=trial_data.get("order_adjustment"),
                forecast_error=trial_data.get("forecast_error"),
                profit=trial_data.get("profit"),
                optimal_profit=trial_data.get("optimal_profit"),
            )
            session.trials.append(trial)

        return session

    def list_sessions(self) -> List[str]:
        """List all participant IDs with saved sessions."""
        return [p.stem for p in self.sessions_dir.glob("*.json")]

    def export_to_parquet(self, filename: str = "experiment_data.parquet") -> Path:
        """
        Export all trials to Parquet file.

        Args:
            filename: Output filename

        Returns:
            Path to exported file
        """
        all_trials = []

        for participant_id in self.list_sessions():
            session = self.load_session(participant_id)
            if session:
                for trial in session.trials:
                    trial_dict = trial.to_dict()
                    trial_dict["condition"] = session.condition
                    trial_dict["session_completed"] = session.completed
                    all_trials.append(trial_dict)

        if not all_trials:
            # Create empty DataFrame with expected columns
            df = pd.DataFrame(
                columns=[
                    "participant_id",
                    "scenario_id",
                    "trial_number",
                    "product",
                    "ai_forecast",
                    "ai_recommendation",
                    "participant_forecast",
                    "participant_order",
                    "actual_demand",
                    "optimal_order",
                    "forecast_adjustment",
                    "order_adjustment",
                    "profit",
                    "optimal_profit",
                    "questions_asked",
                    "time_spent_seconds",
                    "condition",
                ]
            )
        else:
            df = pd.DataFrame(all_trials)

        output_path = self.exports_dir / filename
        df.to_parquet(output_path, index=False)
        return output_path

    def export_to_csv(self, filename: str = "experiment_data.csv") -> Path:
        """Export all trials to CSV file."""
        # First export to parquet, then convert
        parquet_path = self.export_to_parquet("temp.parquet")
        df = pd.read_parquet(parquet_path)

        output_path = self.exports_dir / filename
        df.to_csv(output_path, index=False)

        # Clean up temp file
        parquet_path.unlink()

        return output_path

    def get_summary_stats(self) -> Dict:
        """Get summary statistics of collected data."""
        sessions = [self.load_session(pid) for pid in self.list_sessions()]
        sessions = [s for s in sessions if s is not None]

        if not sessions:
            return {
                "total_participants": 0,
                "completed_sessions": 0,
                "total_trials": 0,
                "avg_trials_per_participant": 0,
            }

        total_trials = sum(len(s.trials) for s in sessions)
        completed = sum(1 for s in sessions if s.completed)

        return {
            "total_participants": len(sessions),
            "completed_sessions": completed,
            "total_trials": total_trials,
            "avg_trials_per_participant": (
                total_trials / len(sessions) if sessions else 0
            ),
        }
