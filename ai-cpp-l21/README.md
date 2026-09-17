# Lesson 21: Speed-of-Light Budgeting

> Two disciplines, run in order. **Part 1** builds the *machine model*: a
> measured, contention-aware description of the hardware you actually have,
> done once per platform configuration. **Part 2** turns that model into a
> per-stage floor, a budget, and the ratio that decides whether optimising a
> stage is worth starting at all.

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

## Part 2: from a machine model to a budget

The model is an input, not a deliverable. What it buys you is a *floor* per
stage, and a ratio that says whether optimising that stage is worth starting.

[`sol.py`](sol.py) is the arithmetic; it has no timing in it at all.

### The three floors, and why SOL is their max

```python
compute_floor = ops    / measured_sustained_rate_of_that_unit
memory_floor  = bytes  / bandwidth_available_under_this_graph's_contention
tax_floor     = crossings * measured_cost_per_crossing
node_sol      = max(three) + dispatch + completion
```

`max`, not sum. The floors overlap in time — bytes stream while operations
issue — so adding them would assert that a stage cannot overlap its own
memory traffic with its own compute, which is the opposite of what the
hardware does. Dispatch and completion *are* serial with the work, so those
add.

`ops` and `bytes` are **compulsory** work, counted off the algorithm rather
than off the code. A re-read your implementation happens to do is not
compulsory; it is a design cost, and it gets its own node. That distinction
is what makes the hidden nodes visible.

### Naming the regime is most of the value

Which floor won decides what a fix would even look like, and the three demand
completely different fixes. A tax-bound stage does not get faster because you
optimised the work between crossings — you have to cross less often. `sol.py`
returns the regime alongside the number, and `verdict()` gives a different
instruction for each.

### Budgets are a fraction of SOL, never of the deadline

Budgeting against the deadline hides headroom: a stage sitting at 10% of its
floor still "fits the frame" and nobody looks at it again. 60–80% of SOL is
the usual band, lower for dispatch-dominated nodes.

### Decide by ratio

- **≥ 70%** — at the floor. Only the *graph* can improve it; stop tuning.
- **≤ 30%** — real headroom. Check overhead-bound causes first, because a
  roofline cannot see dispatch, sync or small-work effects.
- **> 100%** — impossible, and the most important case. Nothing beats its own
  speed of light, so a stage that measures faster than its SOL means **the
  machine model is wrong**, not that the stage is excellent. An overstated op
  count, a rate measured on the wrong unit, or work the hardware is not
  actually doing. `verdict()` reports this as `IMPOSSIBLE` rather than as a
  triumph, because celebrating it is how a broken model survives.

### The worked example

[`budget_tracker.py`](budget_tracker.py) budgets one frame of a
preprocess → inference → postprocess → publish loop against an Orin NX model
built from [L6](../ai-cpp-l6/)'s cache numbers and Part 1's tax table:

```
node                           regime     SOL ms  meas ms  of SOL
preprocess                     memory      0.548    3.000     18%
inference                      compute    17.530    9.000    195%
postprocess                    compute     0.033    1.200      3%
publish (one sendto per box)   tax         0.036    4.000      1%
```

Three things to read out of it.

**`publish` is at 1% of its floor and is tax-bound** — one `sendto` per
detection, 100 crossings a frame. A faster serializer buys nothing here; one
message per frame buys everything. That node is the whole reason the tax
table exists.

**`postprocess` at 3% looks similar and is not** — it is compute-bound, so
the fix is a different one, and the regime column is what tells you.

**`inference` reads 195%, which is impossible**, and the example ships that
way on purpose. Either the 7 GFLOP op count or the 400 GFLOP/s sustained rate
is wrong. Until you know which, every other row derived from that machine
model is suspect too. Finding out is Exercise 5 — and noticing it at all is
the habit the lesson is really trying to build.

## What this lesson does not cover

Part 1 lists eight measurement steps. Three are already measured by artifacts
elsewhere in this course, and the template cross-references them rather than
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

From Part 2, step 10 — **gating merges on the budget** — is not implemented
here. `sol.py` gives you the ratio a gate would need, but wiring it into CI so
an over-budget change cannot merge is [L12](../ai-cpp-l12/)'s subject, and the
check only survives if it is automated. `measurements.csv` and
`budget_revisions.md` are likewise conventions to adopt rather than code to
ship: a revision log is only worth anything if a human writes the reason.

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
- A stage's SOL is the **max** of its compute, memory and tax floors, not
  their sum — the floors overlap; only dispatch and completion are serial
- Naming the regime matters more than the number: a tax-bound stage does not
  improve when you speed up the work between its crossings
- Budget against SOL, not against the deadline, or a stage at 10% of its
  floor will keep "fitting the frame" and never get looked at
- Measuring *faster* than SOL is impossible, so it means the machine model is
  wrong — it is the one result you must never accept as good news

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

5. **Find the broken row.** `budget_tracker.py` reports `inference` at 195%
   of SOL, which cannot happen. Work out whether the op count or the
   sustained rate is wrong, fix that input, and say which other rows in the
   table you should now distrust.

6. **Re-derive after fusing.** Fuse `preprocess` into `inference` — one node,
   no intermediate tensor written and re-read. Recompute the SOL. The raw
   time drops; does the *ratio* drop as much? Comparing a new measurement
   against the old ceiling is the silent failure mode this lesson exists to
   prevent.

7. **Budget a stage you own.** Build the `Machine` from your own
   `machine_model.md`, write the nodes for one real pipeline, and check
   `fits_window`. If the floor alone does not fit the deadline, no
   implementation will — say what you would change about the *graph*.

## Lesson Files

| File | Description |
|------|-------------|
| [machine_model.md](machine_model.md) | The artifact: a fill-in machine model with its rules |
| [tax_bench.cpp](tax_bench.cpp) | Per-crossing cost measurements (syscall, alloc, lock, IPC, FFI) |
| [measure_tax.py](measure_tax.py) | Fills the tax table; owns the FFI-crossing measurement |
| [CMakeLists.txt](CMakeLists.txt) | Build configuration (`NOMINSIZE` so `-O3` is not overridden) |
| [test_tax.py](test_tax.py) | Asserts the orderings the lesson claims, not absolute values |
| [sol.py](sol.py) | Part 2: the three floors, regime, budget, ratio and verdict |
| [budget_tracker.py](budget_tracker.py) | Part 2 worked example: one frame of a tracker loop |
| [test_sol.py](test_sol.py) | Exact-value tests for the SOL arithmetic and its thresholds |

Depends on [L6](../ai-cpp-l6/) (measurement) and [L10](../ai-cpp-l10/)
(profiling). [L7J](../ai-cpp-l7j/) is recommended for the Jetson lane. Sits
before [L17](../ai-cpp-l17/): you cannot falsify a claim about "fast enough"
without a ceiling to falsify it against.
