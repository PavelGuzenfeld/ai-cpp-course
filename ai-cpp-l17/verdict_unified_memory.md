# Verdict: unified memory always removes the copy, so it beats an explicit copy path

## Claim

On a Jetson the CPU and GPU sit behind the same LPDDR controller, so
`cudaMallocManaged` should let both touch one allocation with no transfer at
all. An explicit path — pinned host buffer, device buffer, `cudaMemcpy` each
way — is then doing work that the hardware has made unnecessary. The claim is
that managed memory therefore wins, and that the explicit path only survives
on discrete GPUs.

It sounds plausible because the premise is true: the copy really is
unnecessary on an integrated part. The claim smuggles in a second assertion,
that removing the copy is the only thing that changed.

## Kill Criterion

Pre-registered by the exercise itself ([README](README.md), Exercise 3:
"find the case where it is not"), not invented afterwards: the claim is
false if there is any tested size at which the explicit copy path has a
lower median round-trip time than the managed path, reproducibly across
repeated runs.

"Reproducibly" matters — one inverted row in one run is noise, and the
inverted rows have to be the same rows every time or this is a measurement
bug rather than a finding.

## Measurement

Orin NX, JP6.2 / R36.4.3, CUDA device reports `integrated=1`,
`canMapHostMemory=1`, `managedMemory=1`. Built with the host
`/usr/local/cuda/bin/nvcc -O2 -std=c++17`, run natively — the course Docker
image ships neither `nvcc` nor torch, so this cannot run inside it.

One iteration is the round trip the claim is actually about: CPU writes into
the buffer, GPU scales it, CPU reads one element back. 3 warmup iterations
discarded. 200 iterations per size below 4M elements, 50 at and above it.
Time is mean ms per iteration over that window, and the run was repeated 4
times; the table gives run 2, which is the median run.

| elements | bytes | managed (ms) | explicit (ms) | ratio | claim |
|---|---|---|---|---|---|
| 4 096 | 16 KB | 0.0546 | 0.0428 | 1.28 | **false** |
| 65 536 | 256 KB | 0.0597 | 0.0533 | 1.12 | **false** |
| 1 048 576 | 4 MB | 0.2347 | 0.4297 | 0.55 | holds |
| 4 194 304 | 16 MB | 0.7867 | 1.6106 | 0.49 | holds |
| 16 777 216 | 64 MB | 2.9457 | 6.2814 | 0.47 | holds |

Ratio is managed ÷ explicit; above 1.0 the explicit path won. The two
inverted rows inverted in all 4 runs, at ratios of 1.19–1.29 and 1.06–1.14.
The large-size rows never inverted. The crossover sits between 64 K and 1 M
elements in every run.

## Decision

**FALSIFIED** — at 4 K and 64 K elements the explicit copy path is
consistently 6–29% faster, on exactly the hardware where unified memory was
supposed to be unbeatable.

The narrower claim that survives: *on an integrated Tegra part, managed
memory wins once the payload is large enough that the copy dominates —
around 4 MB here — and loses below that, where the round trip is dominated by
managed memory's page-coherency bookkeeping rather than by moving bytes.*

The mechanism is the giveaway. The explicit path at 16 KB costs 0.0428 ms,
and a 16 KB `memcpy` does not cost 40 µs — nearly all of that is kernel
launch and synchronization, which both paths pay. What the managed path adds
on top is the cost of making the allocation coherent between the two
processors around each `cudaDeviceSynchronize()`, and that cost does not
shrink with the payload. It is a fixed tax, so at small sizes it is the whole
difference, and at 64 MB it disappears into a 6 ms copy.

Which is the same shape as [L16](../ai-cpp-l16/)'s zero-copy result from the
other direction: the technique that removes a size-proportional cost adds a
fixed one, and you only find out which dominates by measuring at a size small
enough for the fixed cost to be visible. A sweep that started at 1 MB would
have confirmed the claim and been wrong.

## Re-entry Trigger

Re-check if any of these change:

- **CUDA or JetPack version.** The bookkeeping cost is a driver implementation
  detail, not an architectural constant. A newer driver could make the
  crossover vanish.
- **Access pattern.** This probe has the CPU touch one element per 1024 and
  the GPU touch all of them. A CPU pass that touches every element, or one
  that never reads back, moves the crossover.
- **Dropping the synchronize.** The tax is charged around synchronization
  points. An async pipeline that overlaps rather than round-trips is a
  different measurement and this verdict does not cover it.
- **A discrete GPU.** Nothing here transfers; `integrated=1` is load-bearing.

Reproduce with [unified_memory_probe.cu](unified_memory_probe.cu).
