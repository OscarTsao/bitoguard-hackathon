from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class OraclePayload:
    user_labels: pd.DataFrame
    scenarios: pd.DataFrame


class OracleClient:
    def __init__(self, oracle_dir: Path) -> None:
        self.oracle_dir = oracle_dir

    def load(self) -> OraclePayload:
        users = pd.read_csv(self.oracle_dir / "users.csv")
        scenarios = pd.read_csv(self.oracle_dir / "scenarios.csv")
        user_labels = users[
            [
                "user_id",
                "hidden_suspicious_label",
                "observed_blacklist_label",
                "scenario_types",
                "evidence_tags",
            ]
        ].copy()
        return OraclePayload(user_labels=user_labels, scenarios=scenarios)
