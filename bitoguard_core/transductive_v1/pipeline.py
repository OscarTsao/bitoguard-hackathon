from __future__ import annotations

from transductive_v1.score import score_transductive_v1
from transductive_v1.train import train_transductive_v1
from transductive_v1.validate import validate_transductive_v1


def run_transductive_v1_pipeline() -> dict[str, object]:
    train_meta = train_transductive_v1()
    validation = validate_transductive_v1()
    predictions = score_transductive_v1()
    return {
        "train_meta": train_meta,
        "validation_report_path": "bitoguard_core/artifacts/transductive_v1/reports/validation_report.json",
        "prediction_rows": int(len(predictions)),
        "decision_rule_type": validation["calibrator"]["selected_rule"]["rule_type"],
    }


def main() -> None:
    print(run_transductive_v1_pipeline())


if __name__ == "__main__":
    main()
