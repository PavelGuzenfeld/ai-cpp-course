"""
Post-Course Assessment — Take this AFTER completing the course.

Run: python3 assessment/post_assessment.py

Compares your score with the pre-assessment to measure growth.
"""

import json
import time
from pathlib import Path

# Same questions as pre_assessment — imported to stay in sync
from pre_assessment import QUESTIONS, run_assessment


def main():
    result = run_assessment("POST-COURSE ASSESSMENT")

    results_path = Path(__file__).parent / "results.json"
    results = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    results["post"] = result

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Compare with pre-assessment if available
    if "pre" in results:
        pre = results["pre"]
        post = result

        print(f"\n{'=' * 60}")
        print(f"  GROWTH COMPARISON")
        print(f"{'=' * 60}")
        print(f"  Pre-course:  {pre['score']}/{pre['total']} ({pre['percentage']:.0f}%)")
        print(f"  Post-course: {post['score']}/{post['total']} ({post['percentage']:.0f}%)")
        improvement = post["percentage"] - pre["percentage"]
        print(f"  Improvement: {'+' if improvement >= 0 else ''}{improvement:.0f} percentage points")
        print()

        # Per-category comparison
        print(f"  {'Category':<20s}  {'Pre':>5s}  {'Post':>5s}  {'Change':>8s}")
        print(f"  {'-' * 20}  {'-' * 5}  {'-' * 5}  {'-' * 8}")

        all_cats = set(list(pre.get("categories", {}).keys()) +
                       list(post.get("categories", {}).keys()))
        for cat in sorted(all_cats):
            pre_cat = pre.get("categories", {}).get(cat, {"correct": 0, "total": 1})
            post_cat = post.get("categories", {}).get(cat, {"correct": 0, "total": 1})
            pre_pct = pre_cat["correct"] / pre_cat["total"] * 100
            post_pct = post_cat["correct"] / post_cat["total"] * 100
            delta = post_pct - pre_pct
            sign = "+" if delta >= 0 else ""
            print(f"  {cat:<20s}  {pre_pct:4.0f}%  {post_pct:4.0f}%  {sign}{delta:5.0f}%")

        print()

        # Which questions improved
        improved = []
        regressed = []
        for q in QUESTIONS:
            qid = str(q["id"])
            pre_ans = pre.get("answers", {}).get(qid, pre.get("answers", {}).get(q["id"]))
            post_ans = post.get("answers", {}).get(qid, post.get("answers", {}).get(q["id"]))
            pre_correct = pre_ans == q["answer"] if pre_ans else False
            post_correct = post_ans == q["answer"] if post_ans else False

            if post_correct and not pre_correct:
                improved.append(q)
            elif not post_correct and pre_correct:
                regressed.append(q)

        if improved:
            print(f"  Questions you now get right ({len(improved)}):")
            for q in improved:
                print(f"    Q{q['id']}: {q['question'][:60]}...")

        if regressed:
            print(f"\n  Questions to review ({len(regressed)}):")
            for q in regressed:
                print(f"    Q{q['id']}: {q['question'][:60]}...")

        print()
    else:
        print("\n  No pre-assessment found. Run pre_assessment.py first for comparison.\n")

    print(f"  Results saved to {results_path}\n")


if __name__ == "__main__":
    main()
