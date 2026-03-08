#!/usr/bin/env python3
import json
from pathlib import Path

POINTS = [
    "p1_conservative",
    "p2_safe",
    "p3_balanced",
    "p4_fast",
    "p5_aggressive",
]
TASKS = ["gsm8k_cot", "humaneval_instruct", "mbpp_instruct", "ifeval"]


def read_dream_avg(root: Path, point: str, task: str):
    p = root / point / task
    cands = sorted(p.glob("step_*/step_stats/fp_stats.json"))
    if not cands:
        return None
    data = json.loads(cands[-1].read_text())
    return float(data.get("avg_forward_passes", data.get("avg_nfe", 0.0)))


def read_llada_avg(root: Path, point: str, task: str):
    p = root / point / task
    cands = sorted(p.glob("step_*/speed/nfe_stats.jsonl"))
    if not cands:
        return None
    lines = [ln for ln in cands[-1].read_text().splitlines() if ln.strip()]
    if not lines:
        return None
    rec = json.loads(lines[-1])
    for key in ("Average NFE", "avg_forward_passes", "avg_nfe"):
        if key in rec:
            return float(rec[key])
    return None


def check_monotonic(vals):
    xs = [v for v in vals if v is not None]
    if len(xs) < 2:
        return None
    return all(xs[i] >= xs[i + 1] for i in range(len(xs) - 1))


def report(model: str, root: Path, reader):
    print(f"\\n## {model} ({root})")
    overall = True
    any_result = False
    for t in TASKS:
        vals = [reader(root, p, t) for p in POINTS]
        any_result = any_result or any(v is not None for v in vals)
        mono = check_monotonic(vals)
        status = "N/A" if mono is None else ("PASS" if mono else "FAIL")
        print(f"- {t}: {status} | " + ", ".join(f"{POINTS[i]}={vals[i]}" for i in range(len(POINTS))))
        if mono is False:
            overall = False
    if not any_result:
        print("(no results found)")
        return 2
    print(f"- overall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    dream_root = Path("/workspace/DAWN/dream/output_dawn_dream_instruct_4bench_5point_limit1")
    llada_root = Path("/workspace/DAWN/llada/output_dawn_llada_instruct_4bench_5point_limit1")

    rd = report("Dream", dream_root, read_dream_avg)
    rl = report("LLaDA", llada_root, read_llada_avg)

    code = 0
    if rd == 1 or rl == 1:
        code = 1
    elif rd == 2 and rl == 2:
        code = 2
    raise SystemExit(code)
