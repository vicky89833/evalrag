import json
from pathlib import Path

from evalrag.core.eval.regression_runner import RegressionRunner, RegressionReport


def test_loads_regression_set(tmp_path):
    f = tmp_path / "set.jsonl"
    f.write_text(json.dumps({"id": "q1", "type": "factual",
                             "question": "Q?", "expected_answer_substrings": ["yes"],
                             "must_refuse": False}) + "\n")
    runner = RegressionRunner(set_path=f, query_fn=lambda q: {"answer": "yes [1]",
                                                              "trust_score": {"overall": 90},
                                                              "latency_ms": 100,
                                                              "cost_usd": 0.01})
    report = runner.run()
    assert isinstance(report, RegressionReport)
    assert report.n_total == 1
    assert report.n_substring_pass == 1


def test_refusal_question_passes_when_system_refuses():
    item = {"id": "q1", "type": "refusal", "question": "Q?",
            "expected_answer_substrings": [], "must_refuse": True}
    runner = RegressionRunner.from_items([item],
        query_fn=lambda q: {"answer": "The document does not contain an answer to that question.",
                            "trust_score": None, "latency_ms": 50, "cost_usd": 0.001})
    rep = runner.run()
    assert rep.n_refusal_pass == 1


def test_compare_to_baseline_emits_deltas(tmp_path):
    baseline = {"avg_trust_overall": 80, "p95_latency_ms": 1000,
                "n_substring_pass_rate": 0.9}
    item = {"id": "q1", "type": "factual", "question": "Q?",
            "expected_answer_substrings": ["yes"], "must_refuse": False}
    runner = RegressionRunner.from_items([item],
        query_fn=lambda q: {"answer": "yes [1]", "trust_score": {"overall": 95},
                            "latency_ms": 500, "cost_usd": 0.01})
    rep = runner.run()
    deltas = rep.compare(baseline)
    assert deltas["avg_trust_overall_delta"] == 15
    assert deltas["p95_latency_ms_delta"] == -500
