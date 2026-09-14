# Lesson 17: Falsifier-First — a Measured "Don't Build It" Is a Result

## Goal

[L10](../ai-cpp-l10/) teaches "profile before you optimise." This lesson is
the step before that: run a cheap experiment before you commit to building
the thing at all, and treat a measured "the claim was wrong" as a completed
result, not a failure. The highest-leverage habit in a real project is
ordering the cheapest disconfirming experiment first — before the design
doc, before the multi-day build, before the accelerator port.

## The Claim

> std::sort is always faster than a hand-rolled insertion sort.

This sounds obviously true — `std::sort` is `O(n log n)`, insertion sort is
`O(n²)`. It is also the kind of plausible-sounding claim that, stated as
"always," is false, and the falsifier is cheaper than trusting it.

## The Falsifier

[`falsifier_native.cpp`](falsifier_native.cpp) times both algorithms across
a range of `n`. [`falsifier.py`](falsifier.py) sweeps the sizes and finds
where the claim breaks:

```bash
python3 falsifier.py
```

```
# Verdict: std::sort is always faster than a hand-rolled insertion sort.

## Measurement
| n | std::sort (ns, median) | insertion sort (ns, median) | winner |
|---|---|---|---|
| 4 | ... | ... | insertion sort |
| ...
| 8192 | ... | ... | std::sort |

## Decision
FALSIFIED below n=<N>: insertion sort wins for smaller inputs. 'Always
faster' is false; the defensible claim is 'faster above roughly n=<N>' --
which is exactly why libstdc++'s own introsort falls back to insertion sort
below a small-n threshold.
```

The exact crossover is hardware-dependent — that's expected, and is why the
verdict names a threshold instead of a universal answer. On one measured
run it landed at n≈128, well above libstdc++'s own internal small-n
threshold (16, where introsort's recursion falls back to insertion sort
mid-partition): calling `std::sort` still pays for partition setup and
recursion machinery that a bare insertion sort loop doesn't, so the
crossover for the *wrapped* call sits higher than the *library's own*
internal one. Both numbers point the same direction: insertion sort wins
somewhere in the small-n regime, which is exactly why libstdc++ has that
fallback at all. The "optimised" algorithm's own authors already ran this
falsifier.

## Ordering the Falsifier First

The pattern this lesson is really teaching:

1. **State the claim as something that can be wrong**, not as a vague
   intuition ("std::sort is better" isn't falsifiable; "std::sort is always
   faster" is).
2. **Write the kill criterion before measuring** — what result would count
   as false? (See [`verdict_template.md`](verdict_template.md).)
3. **Run the cheapest version of the experiment first.** Here that's a
   sweep over array sizes, not a multi-day integration.
4. **Ship the verdict either way.** "We measured that insertion sort wins
   below n≈16, so we're not rewriting the small-buffer path" is a
   deliverable, not an absence of one.
5. **Record a re-entry trigger.** A verdict with no stated condition for
   revisiting it is folklore, not a decision.

## The `verdict.md` Template

[`verdict_template.md`](verdict_template.md) is the shape every falsifier
in this style should produce: Claim, Kill Criterion, Measurement, Decision,
Re-entry Trigger. `falsifier.py`'s own output follows it informally; a
real investigation should fill in the file itself.

## Build and Run

```bash
cd /workspace
colcon build --packages-select nanobind-l17
source install/setup.bash

python3 ai-cpp-l17/falsifier.py
pytest ai-cpp-l17/ -v
```

## What You Learned

- A falsifiable claim names a condition that would prove it wrong; "X is
  better" is not one, "X is always faster than Y" is
- The cheapest disconfirming experiment goes first — before the design,
  before the build, before the accelerator port
- "The claim was false" is a completed result when it's backed by a number,
  not a failure to reach a conclusion
- A verdict needs a re-entry trigger, or it's folklore: nobody can tell
  later whether the decision is still valid
- Even textbook "obviously better" algorithms have a regime where they
  lose — the standard library's own implementation already accounts for it

## Exercises

1. **Falsify a claim of your own**: pick something you believe about this
   course's code ("nanobind is always faster than pybind11 for simple
   getters," say) and write the falsifier before checking whether you're
   right.

2. **Widen the sweep**: extend `SIZES` in `falsifier.py` down to n=1 and n=2.
   Does the crossover behave the way you'd predict, or is there noise at
   the very bottom?

3. **A CPU claim to falsify on hardware you have**: "unified memory always
   removes the copy, so it will be faster than an explicit copy path."
   Measure it (see [L7J](../ai-cpp-l7j/)) and find the case where it is not
   — the source project's own worked example for this lesson.

4. **Write the kill criterion first, actually first**: next time you're
   about to benchmark something, write section 2 of `verdict_template.md`
   before you write any code, then check afterward whether you stuck to it.

## Lesson Files

| File | Description |
|------|-------------|
| [falsifier_native.cpp](falsifier_native.cpp) | Timed `std::sort` vs. insertion sort |
| [CMakeLists.txt](CMakeLists.txt) | CMake build configuration |
| [falsifier.py](falsifier.py) | The sweep, crossover detection, and verdict rendering |
| [verdict_template.md](verdict_template.md) | The shape a falsifier's verdict should take |
| [test_falsifier.py](test_falsifier.py) | Crossover/verdict logic tested against synthetic data |
| [test_integration_falsifier.py](test_integration_falsifier.py) | The real timers, tested for the asymptotic fact only |
