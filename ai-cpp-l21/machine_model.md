# Machine model: &lt;board / configuration&gt;

A machine model is measured, not looked up. Datasheet numbers are ceilings for
a machine that does not exist: they assume one consumer, no OS, and no
neighbours. Fill this in once per *platform configuration* — a power-mode
change or a JetPack upgrade is a new configuration.

Copy this file, fill it in, commit it next to the code it describes.

## Rules for every number in this file

1. **Every row carries a `method:` line.** A number without its method is
   folklore. The method names the tool, the input size, the iteration count,
   and the statistic (median or p99, never a bare mean).
2. **Every row passes a responsiveness check.** Vary the thing the row claims
   to measure and confirm the number moves. A figure that does not respond is
   measuring something else — L6's sequential cache lane reads a flat
   3.35 ns/access from 4 KB to 256 MB on an Orin NX, which means it is bound
   by the loop, not by memory, and must not be quoted as a memory latency.
   Record the check, not just the number.
3. **Sustained, not peak.** Run long enough to reach thermal equilibrium
   (§7) and quote the sustained figure.
4. **Under contention, not in isolation**, wherever the real pipeline will
   have neighbours. See §5.

---

## 1. Configuration header

| Field | Value |
|---|---|
| Board / SoC | |
| OS / BSP version | |
| Power mode | |
| Compiler and flags | |
| Date measured | |

## 2. Execution units

Which units exist *and are reachable from your language*. An engine you
cannot dispatch to is not part of your machine.

| Unit | Present? | How verified | Reachable from C++/Python? |
|---|---|---|---|
| CPU cores | | `lscpu` | |
| GPU | | | |
| DLA | | device nodes, e.g. `/dev/nvhost-ctrl-nvdla*` | |
| Other fixed-function | | | |

> Orin NX example: 2 DLA cores and 1 PVA, from `/dev/nvhost-ctrl-nvdla0`,
> `nvdla1`, `nvhost-ctrl-pva0`. See [L7J](../ai-cpp-l7j/)'s engine map for
> what each unit accepts.

## 3. Memory system

Cache levels, what shares them, and the fabric everything funnels through.

| Level | Size | Measured latency | method: |
|---|---|---|---|
| L1d | | | |
| L2 | | | |
| L3 / SLC | | | |
| DRAM | | | |
| Fabric sustained BW | | | |

Reuse [L6](../ai-cpp-l6/)'s `cache_explorer.py` for the latency rows — it
already reports both sequential and random, and L6's README carries measured
x86 and Orin NX columns to compare against.

**Responsiveness check for this section:** the random column must show a step
where the working set leaves each level. If it does not, the pointer chase is
being predicted and the row is invalid.

## 4. Handoff matrix

For each producer→consumer pair: can the consumer read the producer's output
in place, or does it need a convert or a copy? Every non-in-place cell becomes
an explicit node in your data-flow graph with its own cost.

| from \ to | CPU | GPU | DLA | … |
|---|---|---|---|---|
| CPU | — | | | |
| GPU | | — | | |
| DLA | | | — | |

Layout constraints go here too: an engine that needs block-linear input when
you have pitch-linear is a convert, not a free edge.

## 5. Contention

Isolated numbers are the wrong numbers for any unit with neighbours. Sweep
the offered load and find the knee — the point where everyone's latency
starts rising sharply.

| Stage | Alone | +1 neighbour | +3 | +6 | Knee at |
|---|---|---|---|---|---|
| | | | | | |

Two worked examples in this course, both measured rather than assumed:
[L7](../ai-cpp-l7/)'s `stream_contention_probe.cu` (a neighbour's default-stream
launch costs 88× in the victim's p99) and [L7J](../ai-cpp-l7j/)'s
`engine_contention_bench.sh` (a resize is 40% faster on GPU alone and 25%
slower than VIC under six neighbours).

**Contention is not additive.** Measure the pair; do not add the isolated
numbers.

## 6. Scheduling substrate

| Quantity | Value | method: |
|---|---|---|
| Timer wake jitter | | |
| Context switch | | |
| Core-to-core latency | | |
| Effect of pinning | | |

## 7. Power and thermal

Clocks under sustained load until equilibrium. A benchmark that finishes
before the board heats up reports a number production never sees.

| Load | Clock at 1 s | at 60 s | Throttled? | method: |
|---|---|---|---|---|
| | | | | |

## 8. The tax table

Fixed cost per boundary crossing, independent of the work on either side.
Fill with `measure_tax.py`.

| Crossing | ns | method: |
|---|---|---|
| syscall floor (`getpid`) | | `measure_tax.py`, median of 5 × 200000 |
| `clock_gettime` via vDSO | | |
| `clock_gettime` forced syscall | | |
| `malloc`+`free`, 64 B | | |
| `malloc`+`free`+fault, 1 MiB | | |
| mutex lock+unlock, uncontended | | |
| pipe round trip, 1 B | | |
| socketpair round trip, 1 B | | |
| FFI call, one per item | | |
| FFI call, one per batch | | |

The last two rows are the ones that change designs. Their *difference* is the
crossing cost, and it is why a pipeline that calls into native code once per
bounding box is a different machine from one that calls once per frame.

## 9. Methods appendix

One entry per `method:` reference above: exact command, input size, iteration
count, statistic, and what the responsiveness check varied.
