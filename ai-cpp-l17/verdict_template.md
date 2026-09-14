# Verdict: <the claim under test, stated as a falsifiable sentence>

## Claim

<What is being asserted, and why it sounded plausible before you measured it.>

## Kill Criterion

<What result, decided before running anything, would count as "false"?
Written after the measurement, this section is just a rationalization.>

## Measurement

<Hardware, input sizes, iteration count, warmup discarded, statistic used
(median, not mean -- see L6). A table if there's more than one data point.>

## Decision

One of:
- **CONFIRMED** -- the claim held at every size/condition tested.
- **FALSIFIED** -- state exactly where it broke and what the correct,
  narrower claim is.
- **REFUTED** -- the claim never held; state what you'd build instead.

## Re-entry Trigger

<What would have to change -- a new library version, different hardware,
a different input distribution -- for this verdict to be worth re-checking?
A parked decision without one is folklore, not a decision.>
