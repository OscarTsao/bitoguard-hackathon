from __future__ import annotations

import numpy as np

from models.ablate_user_holdout import _queue_rate_candidates, _select_best_queue_rate


def test_queue_rate_candidates_include_prevalence_variants() -> None:
    labels = np.array([0] * 90 + [1] * 10)

    rates = _queue_rate_candidates(labels)

    assert 0.05 in rates
    assert 0.1 in rates
    assert 0.2 in rates
    assert all(0.0 < rate < 0.25 for rate in rates)


def test_select_best_queue_rate_prefers_higher_f1() -> None:
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    probabilities = np.array([0.99, 0.98, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07])

    selected = _select_best_queue_rate(labels, probabilities)

    assert selected["selected_rate"] == 0.2
    assert selected["selected_metrics"]["f1"] == 1.0


def test_select_best_queue_rate_uses_smaller_rate_as_tiebreaker() -> None:
    labels = np.array([1] + [0] * 99)
    probabilities = np.linspace(1.0, 0.01, num=100)

    selected = _select_best_queue_rate(labels, probabilities)
    candidate_rates = [row["rate"] for row in selected["candidate_metrics"]]

    assert selected["selected_rate"] == min(candidate_rates)
    assert selected["selected_metrics"]["precision"] == 1.0
