"""Run L3 regression set against a live API. Outputs JSON report + delta vs baseline."""
import json
import subprocess
import sys
from pathlib import Path
from uuid import UUID

import httpx

from evalrag.core.eval.regression_runner import RegressionRunner

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "evals" / "regression_set.jsonl"
BASELINE = ROOT / "evals" / "baseline.json"
RESULTS = ROOT / "evals" / "results"
RESULTS.mkdir(exist_ok=True)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: run_regression.py <doc_id>")
        return 2
    doc_id = UUID(sys.argv[1])
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

    def query_fn(q: str) -> dict:
        r = httpx.post("http://localhost:8000/query",
                       json={"doc_id": str(doc_id), "question": q}, timeout=60)
        r.raise_for_status()
        return r.json()

    runner = RegressionRunner(set_path=SET, query_fn=query_fn)
    report = runner.run()
    summary = report.summary()
    out = RESULTS / f"{git_sha}.json"
    out.write_text(json.dumps({"summary": summary,
                               "per_question": report.per_question}, indent=2))
    print(json.dumps(summary, indent=2))

    if BASELINE.exists():
        baseline = json.loads(BASELINE.read_text())
        deltas = report.compare(baseline)
        print("\nDeltas vs baseline:")
        print(json.dumps(deltas, indent=2))
        # Gate: trust regression > 2 points fails CI
        if deltas.get("avg_trust_overall_delta", 0) < -2:
            print("FAIL: trust regression > 2 points")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
