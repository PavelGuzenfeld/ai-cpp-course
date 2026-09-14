"""
Falsifier: "std::sort is always faster than a hand-rolled insertion sort."

Sweeps n, times both algorithms at each size, and finds where (if anywhere)
the claim breaks -- a cheap, ordered-first experiment before trusting an
"optimize everything with std::sort" assumption.

Run:
    python3 falsifier.py
"""
import sys

sys.path.insert(0, ".")

from falsifier_native import time_insertion_sort, time_std_sort  # noqa: E402

CLAIM = "std::sort is always faster than a hand-rolled insertion sort."
SIZES = [4, 8, 16, 32, 64, 128, 256, 1024, 8192]
TRIALS = 200


def sweep(sizes=SIZES, trials=TRIALS):
    results = []
    for n in sizes:
        std_ns = time_std_sort(n, trials)
        ins_ns = time_insertion_sort(n, trials)
        results.append((n, std_ns, ins_ns))
    return results


def find_crossover(results):
    """First n at which std::sort's median time beats insertion sort's.
    None if std::sort never wins across the sizes tested."""
    for n, std_ns, ins_ns in results:
        if std_ns < ins_ns:
            return n
    return None


def render_verdict(results, crossover, claim=CLAIM):
    lines = [
        f"# Verdict: {claim}",
        "",
        "## Measurement",
        "| n | std::sort (ns, median) | insertion sort (ns, median) | winner |",
        "|---|---|---|---|",
    ]
    for n, std_ns, ins_ns in results:
        winner = "std::sort" if std_ns < ins_ns else "insertion sort"
        lines.append(f"| {n} | {std_ns:.0f} | {ins_ns:.0f} | {winner} |")

    lines += ["", "## Decision"]
    if crossover is None:
        lines.append(
            "REFUTED at every size tested -- std::sort never won. "
            "Re-check the harness before trusting this."
        )
    elif crossover == results[0][0]:
        lines.append(f"CONFIRMED for every tested size -- std::sort won even at n={crossover}.")
    else:
        lines.append(
            f"FALSIFIED below n={crossover}: insertion sort wins for smaller inputs. "
            f"'Always faster' is false; the defensible claim is "
            f"'faster above roughly n={crossover}' -- which is exactly why "
            f"libstdc++'s own introsort falls back to insertion sort below "
            f"a small-n threshold."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    results = sweep()
    crossover = find_crossover(results)
    print(render_verdict(results, crossover))


if __name__ == "__main__":
    main()
