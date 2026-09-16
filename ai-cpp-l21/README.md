# Lesson 21: Speed-of-Light Budgeting — Part 1, Measure the Machine

> **Part 1 of 2.** This half builds the *machine model*: a measured,
> contention-aware description of the hardware you actually have. Part 2 turns
> that model into per-stage budgets and the measured/SOL ratio that decides
> whether an optimisation is worth starting. Part 1 is useful on its own; Part
> 2 is not useful without it.

## Goal

[L6](../ai-cpp-l6/) teaches *how* to measure. [L10](../ai-cpp-l10/) teaches
*when* to optimise — profile first. Neither answers the question that decides
whether the effort is worth starting at all:

**How fast could this possibly go on this machine, and how far away are we?**

Without that number a profiler tells you where the time went, not whether it
had to go there. You end up tuning a kernel already at 85% of what the memory
system allows, or leaving one at 8% because it "looks fine". The ceiling is
what makes the difference visible, and the ceiling is a property of your
hardware under your contention — not of the datasheet.

## Why the datasheet is the wrong number

A datasheet figure describes one consumer, no operating system, and no
neighbours. Your stage has all three. Three results already measured elsewhere
in this course, each of which would be invisible from a spec sheet:

- [L7](../ai-cpp-l7/): a neighbour launching into the legacy default stream
  costs the victim **88×** in p99 — and the neighbour's own numbers are
  unchanged.
- [L7J](../ai-cpp-l7j/): a 1080p resize is 40% faster on the GPU in isolation
  and **25% slower** than the VIC once six neighbours are loading the GPU.
- [L6](../ai-cpp-l6/): the random-access cache cliff sits at a different size
  on x86 than on an Orin NX, so a working set tuned to fit L1 on a laptop
  spills to L2 on the target.

None of those are things you look up. All of them change a design.

## The machine model

[`machine_model.md`](machine_model.md) is the artifact. Copy it, fill it in
once per platform *configuration* — a power-mode change or a BSP upgrade
invalidates it — and commit it next to the code it describes.

Two rules in that template do the real work:

**Every row carries a `method:` line.** A number without its method is
folklore. Someone will read the row in six months and need to know whether it
was a median or a mean, at what size, under what load.

**Every row passes a responsiveness check.** Vary the thing the row claims to
measure and confirm the number moves. This is not paranoia — L6's sequential
cache lane reads a flat 3.35 ns/access from 4 KB to 256 MB on an Orin NX. It
does not step when the working set leaves L1, L2, L3, or lands in DRAM,
because it is bound by the loop rather than by memory. Quoted as a memory
latency it would be wrong by any amount you like. A number that does not
respond to its own variable is measuring something else.

## Step 1.8: the tax

The section of the model this lesson measures directly, because it is the one
most often left out entirely. A *tax* is the fixed cost of crossing a
boundary — a syscall, an allocation, a lock, a wake, an FFI call. It does not
scale with the work on either side, which is exactly why it disappears from
per-byte reasoning and then dominates the profile.

Build and run:

```bash
cd /workspace
colcon build --packages-select nanobind-l21
source install/setup.bash
python3 ai-cpp-l21/measure_tax.py
pytest ai-cpp-l21/ -v
```

Measured, median of 5 runs, ns per crossing:

| Crossing | x86-64 (i7-12700H) | Orin NX (JP6.2) | ratio |
|---|---|---|---|
| syscall floor (`getpid`) | 109.2 | 304.4 | 2.8× |
| `clock_gettime`, vDSO | 12.4 | 52.1 | 4.2× |
| `clock_gettime`, forced syscall | 136.3 | 381.5 | 2.8× |
| `malloc`+`free`, 64 B | 6.2 | 21.5 | 3.5× |
| `malloc`+`free`+fault, 1 MiB | 603.6 | 704.8 | 1.2× |
| mutex lock+unlock, uncontended | 2.3 | 10.8 | 4.7× |
| pipe round trip, 1 B | 602.7 | 1255.4 | 2.1× |
| socketpair round trip, 1 B | 914.0 | 2395.4 | 2.6× |
| **nanobind call, one per item** | **16.1** | **55.8** | 3.5× |
| **nanobind call, one per batch** | **0.29** | **0.68** | 2.3× |

Read four things out of that table.

**The taxes are 2–5× higher on the Jetson than on the laptop, and they are
higher by more than the clock ratio.** A design that crosses boundaries
frequently does not port; it degrades by a factor you cannot recover by
optimising the work between crossings.

**The vDSO is why a timer is not a syscall.** 12.4 ns against 136.3 ns for the
identical clock read. `clock_gettime` in a hot loop is fine; the same call
forced through the kernel is not. Measure before assuming a libc call is
cheap, and before assuming it is expensive.

**Faulting pages in is most of what a "big allocation" costs.** 64 bytes is
6.2 ns because the allocator has it on a free list. A megabyte touched once
per page is 603.6 ns, and that cost is per-frame if you allocate per-frame —
which is [L7](../ai-cpp-l7/)'s anti-pattern 2 priced.

**The FFI crossing is the one that changes designs.** 16.1 ns per call becomes
0.29 ns per item when 200000 items go through one call: batching removes
**98%** of the per-item cost on x86 and **99%** on the Orin. A tracker that
calls into native code once per bounding box and one that calls once per frame
are not the same program with different constants — they are different
machines.

### Measuring a crossing without fooling yourself

The FFI row is easy to get wrong, so `measure_tax.py` does three things
deliberately:

- It measures the **same total work** both ways — `noop_batch(n)` loops `n`
  times inside C++ — so the difference is the boundary, not the body.
- It reports the **empty Python loop** alongside, because at 8.2 ns/iter on
  x86 the interpreter is half of the 15.8 ns difference. A "crossing cost"
  that silently includes the driving loop is not a crossing cost.
- It takes a **median of repeats**, not a single run or a mean.

If the per-call and per-batch numbers do *not* diverge, the probe is broken —
the compiler has hoisted something, or the body is dominating. `work_per_call`
is in the module for exactly that comparison: give the call a big enough body
and the crossing stops mattering, which is the other half of the rule.

## What this half does not cover

Part 1 of the issue lists eight measurement steps. Three are already measured
by artifacts in this course and the template cross-references them rather than
duplicating:

| Step | Covered by |
|---|---|
| 1.1 inventory of units | [L7J](../ai-cpp-l7j/) engine map; device-node census |
| 1.2 memory system | [L6](../ai-cpp-l6/) `cache_explorer.py` |
| 1.5 contention | [L7](../ai-cpp-l7/) stream probe, [L7J](../ai-cpp-l7j/) engine sweep |
| 1.8 tax | this lesson, `measure_tax.py` |

Still unmeasured, and left as template sections to fill rather than shipped
benchmarks: 1.3 handoff matrix, 1.4 isolated per-unit throughput, 1.6
scheduling substrate, 1.7 power and thermal. Shipping seven half-working
benchmarks would violate this lesson's own `method:` rule on its first page.

## What You Learned

- A machine model is measured, not looked up: datasheets describe one
  consumer, no OS and no neighbours, and you have all three
- Every row needs a `method:` line and a responsiveness check — a number that
  does not move when you vary its own variable is measuring something else
- A tax is a fixed per-crossing cost, so it vanishes from per-byte reasoning
  and then shows up as the profile
- Boundary costs are 2–5× higher on a Jetson than on a laptop, by more than
  the clock ratio, so crossing-heavy designs do not port
- Batching at an FFI boundary removed 98–99% of the per-item cost: cross less
  often rather than speeding up the work between crossings
- Report the driving loop's own cost next to any per-call measurement, or the
  number is partly the interpreter

## Exercises

1. **Fill in `machine_model.md` for your own machine.** Every row gets a
   `method:` line. Stop at the first row you cannot justify and say why.
2. **Break the FFI measurement on purpose.** Give `work_per_call` a large
   enough `inner` that the crossing is noise. At what body size does batching
   stop being worth it on your machine? That threshold, not the 98% figure, is
   the number your design needs.
3. **Find a row that fails its responsiveness check.** Take any figure in your
   model, vary the quantity it claims to measure, and confirm it moves. L6's
   sequential lane is the worked example of one that does not.
4. **Price one real crossing.** Pick a stage in code you own that crosses a
   boundary per item — per box, per detection, per row — and compute what
   moving to one crossing per frame would save, using your own table. Then
   measure it and compare against your estimate.

## Lesson Files

| File | Description |
|------|-------------|
| [machine_model.md](machine_model.md) | The artifact: a fill-in machine model with its rules |
| [tax_bench.cpp](tax_bench.cpp) | Per-crossing cost measurements (syscall, alloc, lock, IPC, FFI) |
| [measure_tax.py](measure_tax.py) | Fills the tax table; owns the FFI-crossing measurement |
| [CMakeLists.txt](CMakeLists.txt) | Build configuration (`NOMINSIZE` so `-O3` is not overridden) |
| [test_tax.py](test_tax.py) | Asserts the orderings the lesson claims, not absolute values |

Depends on [L6](../ai-cpp-l6/) (measurement) and [L10](../ai-cpp-l10/)
(profiling). [L7J](../ai-cpp-l7j/) is recommended for the Jetson lane. Sits
before [L17](../ai-cpp-l17/): you cannot falsify a claim about "fast enough"
without a ceiling to falsify it against.
