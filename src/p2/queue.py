"""Queue-reactive queue-position helpers.

This module implements a lightweight queue-position model inspired by
Cont, Kukanov, and Stoikov (2014) and Huang, Lehalle, and Rosenbaum (2015).
The implementation keeps the state space intentionally small so it can be used
inside both the synthetic simulator and the LOBSTER replay path without adding
new dependencies beyond NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Any, Callable, Iterable, Mapping

import numpy as np


QueueRateFn = Callable[[int], float]
_ADD_EVENTS = {"add", "limit", "insert"}
_CANCEL_EVENTS = {"cancel", "delete", "remove"}
_ARRIVAL_EVENTS = {"arrival", "trade", "execute", "market"}
_RESET_EVENTS = {"reset"}
_PLACE_EVENTS = {"place"}
_OWN_CANCEL_EVENTS = {"own_cancel"}


@dataclass(slots=True)
class QueueState:
    """Per-side queue state at a single price level."""

    queue_size: int
    own_position: int | None = None
    total_volume: float = 0.0

    def __post_init__(self) -> None:
        self.queue_size = max(int(self.queue_size), 0)
        if self.total_volume <= 0.0:
            self.total_volume = float(self.queue_size)
        else:
            self.total_volume = max(float(self.total_volume), 0.0)

        if self.own_position is None:
            return

        own_position = max(int(self.own_position), 0)
        self.own_position = min(own_position, self.queue_size)


class QueueReactiveModel:
    """Queue-reactive model with depth-dependent rates."""

    def __init__(
        self,
        arrival_rate_fn: QueueRateFn,
        cancel_rate_fn: QueueRateFn,
        fill_intensity_fn: QueueRateFn,
    ) -> None:
        self.arrival_rate_fn = arrival_rate_fn
        self.cancel_rate_fn = cancel_rate_fn
        self.fill_intensity_fn = fill_intensity_fn
        self.rng = np.random.default_rng()
        self.depth_grid: np.ndarray | None = None
        self.arrival_rates: np.ndarray | None = None
        self.cancel_rates: np.ndarray | None = None
        self.fill_intensities: np.ndarray | None = None

    def simulate(self, state: QueueState, dt: float) -> QueueState:
        """Advance a single queue state by one time step.

        `queue_size` tracks external volume resting at the level and excludes our
        own order. `own_position` is the external volume ahead of us in FIFO
        order, so `own_position=0` means we are at the front of the queue.
        """

        if dt <= 0.0:
            return QueueState(
                queue_size=state.queue_size,
                own_position=state.own_position,
                total_volume=state.total_volume,
            )

        depth = max(int(state.queue_size), 0)
        arrival_rate = max(float(self.arrival_rate_fn(depth)), 0.0)
        cancel_rate = max(float(self.cancel_rate_fn(depth)), 0.0)
        add_rate = max(float(self.fill_intensity_fn(depth)), 0.0)

        arrivals = int(self.rng.poisson(arrival_rate * dt))
        cancels = min(int(self.rng.poisson(cancel_rate * dt)), depth)
        additions = int(self.rng.poisson(add_rate * dt))

        queue_after = max(depth - cancels - min(arrivals, depth), 0) + additions
        own_position = state.own_position
        if own_position is None:
            return QueueState(queue_size=queue_after, own_position=None, total_volume=float(queue_after))

        clamped_position = min(max(int(own_position), 0), depth)
        cancel_ahead = _sample_cancel_ahead(self.rng, depth, clamped_position, cancels)
        position_after_cancel = max(clamped_position - cancel_ahead, 0)
        if arrivals > position_after_cancel:
            return QueueState(queue_size=queue_after, own_position=None, total_volume=float(queue_after))

        position_after = position_after_cancel - arrivals
        return QueueState(
            queue_size=queue_after,
            own_position=min(position_after, queue_after),
            total_volume=float(queue_after),
        )

    @staticmethod
    def fill_probability(
        own_position: int | None,
        queue_depth: int,
        arrival_rate: float,
        cancel_rate: float,
        dt: float,
    ) -> float:
        """Approximate FIFO fill probability over `[t, t + dt]`.

        The approximation uses a Poisson superposition of front-of-queue trades
        and cancellations ahead of us. It is exact when every cancellation
        happens ahead of our order, and conservative otherwise.
        """

        if own_position is None or dt <= 0.0:
            return 0.0

        position = max(int(own_position), 0)
        depth = max(int(queue_depth), 0)
        arrival = max(float(arrival_rate), 0.0)
        cancel = max(float(cancel_rate), 0.0)
        if arrival <= 0.0 and cancel <= 0.0:
            return 0.0

        ahead_share = 0.0 if depth <= 0 else min(position / max(depth, 1), 1.0)
        effective_rate = arrival + cancel * ahead_share
        if effective_rate <= 0.0:
            return 0.0

        threshold = position + 1
        lam = effective_rate * dt
        if lam <= 0.0:
            return 0.0

        # Recurrence keeps the Poisson tail numerically stable for small queue
        # depths without relying on SciPy.
        mass = exp(-lam)
        cdf = mass
        for count in range(1, threshold):
            mass *= lam / count
            cdf += mass
        return float(np.clip(1.0 - cdf, 0.0, 1.0))


def queue_reactive_dynamics(events: Iterable[Mapping[str, Any] | Any], initial_state: QueueState) -> list[QueueState]:
    """Replay an event log into a queue-state trajectory.

    The returned trajectory includes the initial state as the first element.
    """

    states = [_clone_state(initial_state)]
    current = _clone_state(initial_state)
    for raw_event in events:
        event = _as_event_mapping(raw_event)
        kind = _normalize_event_kind(event)
        size = max(int(event.get("size", event.get("count", 1))), 0)

        if kind in _RESET_EVENTS:
            queue_size = max(int(event.get("queue_size", current.queue_size)), 0)
            own_position = event.get("own_position", current.own_position)
            if own_position is not None:
                own_position = min(max(int(own_position), 0), queue_size)
            current = QueueState(
                queue_size=queue_size,
                own_position=own_position,
                total_volume=float(event.get("total_volume", queue_size)),
            )
            states.append(current)
            continue

        if kind in _PLACE_EVENTS:
            current = QueueState(
                queue_size=current.queue_size,
                own_position=current.queue_size,
                total_volume=float(current.queue_size),
            )
            states.append(current)
            continue

        if kind in _OWN_CANCEL_EVENTS:
            current = QueueState(
                queue_size=current.queue_size,
                own_position=None,
                total_volume=float(current.queue_size),
            )
            states.append(current)
            continue

        queue_size = current.queue_size
        own_position = current.own_position

        if kind in _ADD_EVENTS:
            queue_size += size
        elif kind in _CANCEL_EVENTS:
            removed = min(size, queue_size)
            queue_size -= removed
            if own_position is not None and own_position > 0 and removed > 0:
                explicit_ahead = event.get("ahead_size")
                if explicit_ahead is None:
                    pre_cancel_depth = max(queue_size + removed, 1)
                    ahead_share = float(event.get("ahead_fraction", own_position / pre_cancel_depth))
                    explicit_ahead = int(round(removed * np.clip(ahead_share, 0.0, 1.0)))
                own_position = max(own_position - min(int(explicit_ahead), own_position), 0)
        elif kind in _ARRIVAL_EVENTS:
            consumed = min(size, queue_size)
            queue_size -= consumed
            if own_position is not None:
                if size > own_position:
                    own_position = None
                else:
                    own_position = max(own_position - size, 0)

        current = QueueState(
            queue_size=queue_size,
            own_position=own_position,
            total_volume=float(queue_size),
        )
        states.append(current)

    return states


def calibrate_queue_reactive(
    events: Iterable[Mapping[str, Any] | Any],
    depth_grid: Iterable[int],
) -> QueueReactiveModel:
    """Fit depth-conditioned Poisson rates via MLE."""

    grid = np.asarray(sorted({max(int(depth), 0) for depth in depth_grid}), dtype=int)
    if grid.size == 0:
        raise ValueError("depth_grid must contain at least one depth level")

    exposure = np.zeros(grid.size, dtype=float)
    arrival_counts = np.zeros(grid.size, dtype=float)
    cancel_counts = np.zeros(grid.size, dtype=float)
    add_counts = np.zeros(grid.size, dtype=float)

    for raw_event in events:
        event = _as_event_mapping(raw_event)
        depth = max(int(event.get("queue_depth", event.get("depth", 0))), 0)
        idx = _depth_bucket(depth, grid)
        exposure[idx] += max(float(event.get("dt", 0.0)), 0.0)

        arrival_counts[idx] += float(event.get("arrival_count", 0.0))
        cancel_counts[idx] += float(event.get("cancel_count", 0.0))
        add_counts[idx] += float(event.get("add_count", 0.0))

        kind = _normalize_event_kind(event)
        if kind in _ARRIVAL_EVENTS:
            arrival_counts[idx] += float(event.get("count", 1.0))
        elif kind in _CANCEL_EVENTS:
            cancel_counts[idx] += float(event.get("count", 1.0))
        elif kind in _ADD_EVENTS:
            add_counts[idx] += float(event.get("count", 1.0))

    arrival_rates = _rate_mle(arrival_counts, exposure)
    cancel_rates = _rate_mle(cancel_counts, exposure)
    add_rates = _rate_mle(add_counts, exposure)

    model = QueueReactiveModel(
        arrival_rate_fn=_piecewise_rate_fn(grid, arrival_rates),
        cancel_rate_fn=_piecewise_rate_fn(grid, cancel_rates),
        fill_intensity_fn=_piecewise_rate_fn(grid, add_rates),
    )
    model.depth_grid = grid
    model.arrival_rates = arrival_rates
    model.cancel_rates = cancel_rates
    model.fill_intensities = add_rates
    return model


def expected_time_to_fill(
    own_position: int | None,
    queue_depth: int,
    arrival_rate: float,
    cancel_rate: float,
) -> float:
    """Approximate expected time to fill from FIFO position."""

    if own_position is None:
        return float("inf")

    position = max(int(own_position), 0)
    depth = max(int(queue_depth), 0)
    arrival = max(float(arrival_rate), 0.0)
    cancel = max(float(cancel_rate), 0.0)

    if position == 0:
        return float("inf") if arrival <= 0.0 else 1.0 / arrival

    if arrival <= 0.0 and cancel <= 0.0:
        return float("inf")

    ahead_share = 1.0 if depth <= 0 else min(position / max(depth, 1), 1.0)
    progress_rate = arrival + cancel * ahead_share
    if progress_rate <= 0.0 or not isfinite(progress_rate):
        return float("inf")
    if arrival <= 0.0:
        return float("inf")
    return position / progress_rate + 1.0 / arrival


def _sample_cancel_ahead(
    rng: np.random.Generator,
    depth: int,
    own_position: int,
    cancels: int,
) -> int:
    if depth <= 0 or own_position <= 0 or cancels <= 0:
        return 0
    behind = max(depth - own_position, 0)
    return int(rng.hypergeometric(ngood=own_position, nbad=behind, nsample=min(cancels, depth)))


def _depth_bucket(depth: int, grid: np.ndarray) -> int:
    return int(np.clip(np.searchsorted(grid, depth, side="right") - 1, 0, grid.size - 1))


def _piecewise_rate_fn(grid: np.ndarray, values: np.ndarray) -> QueueRateFn:
    def _rate(depth: int) -> float:
        idx = _depth_bucket(max(int(depth), 0), grid)
        return float(values[idx])

    return _rate


def _rate_mle(counts: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    rates = np.zeros_like(counts, dtype=float)
    positive = exposure > 0.0
    rates[positive] = counts[positive] / exposure[positive]
    return rates


def _clone_state(state: QueueState) -> QueueState:
    return QueueState(
        queue_size=state.queue_size,
        own_position=state.own_position,
        total_volume=state.total_volume,
    )


def _as_event_mapping(event: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(event, Mapping):
        return event
    if hasattr(event, "to_dict"):
        maybe_mapping = event.to_dict()
        if isinstance(maybe_mapping, Mapping):
            return maybe_mapping
    if hasattr(event, "__dict__"):
        return vars(event)
    raise TypeError(f"Unsupported queue event type: {type(event)!r}")


def _normalize_event_kind(event: Mapping[str, Any]) -> str:
    explicit = event.get("event")
    if explicit is not None:
        return str(explicit).lower()

    event_type = event.get("event_type")
    if event_type is None:
        return "exposure"

    event_code = int(event_type)
    if event_code == 1:
        return "add"
    if event_code in {2, 3}:
        return "cancel"
    if event_code in {4, 5}:
        return "trade"
    return "exposure"
