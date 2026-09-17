"""Speed-of-light budgeting: derive a per-stage floor, then judge by ratio.

Part 2 of L21. Part 1 measured the machine; this turns that model into a
ceiling for each stage and a verdict on whether optimising it is worth
starting.

Deliberately pure arithmetic with no timing in it. The numbers that come out
are only as good as the machine model that went in, which is why every field
of Machine is something Part 1 makes you measure.
"""

from dataclasses import dataclass
from enum import Enum


class Regime(str, Enum):
    """Which floor dominates. Naming it is the point: the regime decides what
    a fix would even look like, and the three demand different fixes."""

    COMPUTE = "compute"
    MEMORY = "memory"
    TAX = "tax"


@dataclass(frozen=True)
class Machine:
    """Measured capability. Every rate here is sustained and under the
    contention the real pipeline produces -- never a datasheet number."""

    ops_per_s: dict[str, float]
    bytes_per_s: dict[str, float]
    tax_s_per_crossing: float
    dispatch_s: float = 0.0
    completion_s: float = 0.0


@dataclass(frozen=True)
class Node:
    """One stage. ops and byte_count are *compulsory* work read off the
    algorithm, not counted from the code -- a re-read the implementation
    happens to do is a design cost and belongs in its own node."""

    name: str
    unit: str
    ops: float = 0.0
    byte_count: float = 0.0
    crossings: int = 0


@dataclass(frozen=True)
class NodeSol:
    node: Node
    compute_floor_s: float
    memory_floor_s: float
    tax_floor_s: float
    sol_s: float
    regime: Regime


def derive_node_sol(node: Node, machine: Machine) -> NodeSol:
    """SOL is the *max* of the three floors, not their sum.

    The floors overlap in time: bytes stream while operations issue. Adding
    them would claim a stage cannot overlap its own memory traffic with its
    own compute, which is the opposite of what hardware does. Dispatch and
    completion are serial with all of it, so those add.
    """
    if node.unit not in machine.ops_per_s:
        raise KeyError(f"no measured op rate for unit {node.unit!r}")
    if node.unit not in machine.bytes_per_s:
        raise KeyError(f"no measured bandwidth for unit {node.unit!r}")

    compute = node.ops / machine.ops_per_s[node.unit]
    memory = node.byte_count / machine.bytes_per_s[node.unit]
    tax = node.crossings * machine.tax_s_per_crossing

    dominant = max(compute, memory, tax)
    if dominant == tax:
        regime = Regime.TAX
    elif dominant == memory:
        regime = Regime.MEMORY
    else:
        regime = Regime.COMPUTE

    return NodeSol(
        node=node,
        compute_floor_s=compute,
        memory_floor_s=memory,
        tax_floor_s=tax,
        sol_s=dominant + machine.dispatch_s + machine.completion_s,
        regime=regime,
    )


def pipeline_sol_s(nodes: list[Node], machine: Machine) -> float:
    """Sum along the critical path. Callers pass the path, not the graph:
    deciding what is on the critical path is a modelling judgement, not
    something this function can infer from a list."""
    return sum(derive_node_sol(n, machine).sol_s for n in nodes)


def fits_window(nodes: list[Node], machine: Machine, window_s: float,
                margin: float = 0.25) -> bool:
    """Does the floor itself fit the deadline, with margin reserved?

    A False here is not an optimisation problem. No implementation beats its
    own speed of light, so the *graph* has to change -- fewer nodes, fewer
    bytes, or different units.
    """
    if not 0.0 <= margin < 1.0:
        raise ValueError("margin must be in [0, 1)")
    return pipeline_sol_s(nodes, machine) <= window_s * (1.0 - margin)


def budget_s(node_sol: NodeSol, fraction: float = 0.7) -> float:
    """A budget is a fraction of SOL, never a fraction of the deadline.

    Budgeting against the deadline hides headroom: a stage can sit at 10% of
    its floor and still look fine because the frame had room for it.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    return node_sol.sol_s / fraction


def efficiency(node_sol: NodeSol, measured_s: float) -> float:
    """SOL / measured, as a fraction of the achievable ceiling.

    The issue calls this "measured/SOL"; as a percentage-of-achievable it has
    to be this way round, so that at the floor it reads 1.0 rather than 1.0
    meaning nothing.
    """
    if measured_s <= 0.0:
        raise ValueError("measured time must be positive")
    return node_sol.sol_s / measured_s


AT_THE_FLOOR = 0.70
HEADROOM = 0.30


def verdict(node_sol: NodeSol, measured_s: float) -> str:
    """What to do next, which depends on the regime as much as the ratio."""
    eff = efficiency(node_sol, measured_s)
    if eff > 1.0:
        # Nothing beats its own speed of light. This is never a win and never
        # a stage to celebrate: an input to the floor is wrong -- an
        # overstated op count, a rate measured on the wrong unit, or work the
        # hardware is not actually doing.
        return (f"{eff:.0%} of SOL -- IMPOSSIBLE. The machine model is wrong, "
                f"not the stage. Recheck the {node_sol.regime.value} floor's "
                f"inputs before trusting any other row.")
    if eff >= AT_THE_FLOOR:
        return (f"{eff:.0%} of SOL -- at the floor. Only the graph can improve "
                f"this; stop tuning the implementation.")
    if eff > HEADROOM:
        return f"{eff:.0%} of SOL -- some headroom, but not the best target."
    if node_sol.regime is Regime.TAX:
        return (f"{eff:.0%} of SOL, tax-dominated -- cross less often (batch, "
                f"coalesce, shm instead of pipe). Speeding up the work between "
                f"crossings will not move it.")
    return (f"{eff:.0%} of SOL, {node_sol.regime.value}-bound -- real headroom. "
            f"Check overhead-bound causes first; a roofline cannot see dispatch.")
