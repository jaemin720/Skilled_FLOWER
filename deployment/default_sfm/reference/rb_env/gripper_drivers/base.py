from __future__ import annotations

import abc
import math
import time
from dataclasses import dataclass, fields, replace
from typing import Any

PIKA_STROKE_M = 0.098


def clip_pct(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


@dataclass(frozen=True)
class GripperControlConfig:
    """Policy-to-driver defaults owned by the gripper layer."""

    mode: str = "binary"
    input_range: tuple[float, float] = (-1.0, 1.0)
    output_range: tuple[float, float] = (0.0, 100.0)
    force: float = 100.0
    vel: float = 100.0
    cmd_deadband: float = 0.5
    closed_eps: float = 0.05
    grasp_range: tuple[float, float] = (0.1, 0.9)
    observation_max_age_s: float = 0.25
    binary_close_threshold_m: float | None = None
    binary_open_threshold_m: float | None = None
    binary_closed_width_m: float | None = None
    binary_open_width_m: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.observation_max_age_s) or self.observation_max_age_s <= 0:
            raise ValueError('observation_max_age_s must be positive and finite')
        closed_width, open_width = self.binary_closed_width_m, self.binary_open_width_m
        if closed_width is not None or open_width is not None:
            if closed_width is None or open_width is None:
                raise ValueError("binary_closed_width_m and binary_open_width_m must both be set")
            closed_width, open_width = float(closed_width), float(open_width)
            if not (math.isfinite(closed_width) and math.isfinite(open_width)
                    and 0.0 <= closed_width < open_width):
                raise ValueError("binary widths must satisfy 0 <= binary_closed_width_m < binary_open_width_m")
            object.__setattr__(self, "binary_closed_width_m", closed_width)
            object.__setattr__(self, "binary_open_width_m", open_width)

        close = self.binary_close_threshold_m
        open_ = self.binary_open_threshold_m
        if close is None and open_ is None:
            return
        if close is None or open_ is None:
            raise ValueError("binary_close_threshold_m and binary_open_threshold_m must both be set")
        close, open_ = float(close), float(open_)
        lo, hi = self.input_range
        if not (math.isfinite(close) and math.isfinite(open_)
                and math.isfinite(lo) and math.isfinite(hi)
                and 0.0 <= lo <= close < open_ <= hi):
            raise ValueError("binary thresholds must satisfy 0 <= input_low <= "
                             "binary_close_threshold_m < binary_open_threshold_m <= input_high")
        object.__setattr__(self, "binary_close_threshold_m", close)
        object.__setattr__(self, "binary_open_threshold_m", open_)

    def with_overrides(self, values: dict[str, Any] | None) -> "GripperControlConfig":
        values = values or {}
        field_names = {field.name for field in fields(self)}
        updates = {
            name: values[name]
            for name in field_names
            if name in values
        }
        for name in ("input_range", "output_range", "grasp_range"):
            if name in updates:
                pair = tuple(float(value) for value in updates[name])
                if len(pair) != 2:
                    raise ValueError(f"{name} must contain exactly two values")
                updates[name] = pair
        return replace(self, **updates)


def compute_noop_gripper_action(
    gripper_control: "GripperControlConfig | dict[str, Any] | None",
    target: float = 100.0,
) -> float:
    """The action value that leaves a gripper exactly where it is.

    Zero is *not* that value in ``continuous`` mode: `Rb3Env._send_gripper_command`
    maps an action linearly from ``input_range`` onto ``output_range``, so 0.0
    on the usual [-1, 1] / [0, 100] pair commands a half-closed gripper.  An
    arm the policy is not driving (its action block padded with zeros) would
    therefore squeeze shut on the first step.  This inverts that mapping for
    ``target`` -- 100.0, the value `Rb3Env.reset()` records in
    ``_last_gripper_target`` -- so the command lands inside the deadband and
    no motion is issued.

    ``binary`` uses the midpoint between its close/open thresholds to hold state.
    """
    config = (
        gripper_control
        if isinstance(gripper_control, GripperControlConfig)
        else GripperControlConfig().with_overrides(gripper_control)
    )
    if config.mode != "continuous":
        if config.mode == "binary" and config.binary_close_threshold_m is not None:
            return (config.binary_close_threshold_m + config.binary_open_threshold_m) / 2.0
        return 0.0

    in_lo, in_hi = config.input_range
    out_lo, out_hi = config.output_range
    if out_hi == out_lo:
        raise ValueError("gripper output_range endpoints must differ")

    ratio = (float(target) - out_lo) / (out_hi - out_lo)
    return float(min(max(in_lo + ratio * (in_hi - in_lo), in_lo), in_hi))


class GripperDriver(abc.ABC):
    """Normalized gripper interface.

    All lengths are expressed as 0.0(closed) to 100.0(open) percent.
    Force and speed inputs are also normalized percent values; concrete
    drivers map those values to their native units.
    """

    needs_polling = False
    CONTROL_DEFAULTS = GripperControlConfig()

    def get_control_config(
        self, overrides: dict[str, Any] | None = None
    ) -> GripperControlConfig:
        config = self.CONTROL_DEFAULTS.with_overrides(overrides)
        if config.mode == "binary" and config.binary_closed_width_m is not None:
            # Validate physical limits before any policy or reset command.
            self.width_to_percent(config.binary_closed_width_m)
            self.width_to_percent(config.binary_open_width_m)
        return config

    def width_to_percent(self, width_m: float) -> float:
        """Convert physical opening using this driver's full stroke, not input_range."""
        stroke = getattr(self, "stroke_m", None)
        if stroke is None:
            raise ValueError(f"{type(self).__name__} does not define stroke_m for meter width control")
        stroke = float(stroke)
        width_m = float(width_m)
        if not math.isfinite(stroke) or stroke <= 0.0:
            raise ValueError("gripper stroke_m must be finite and positive")
        if not math.isfinite(width_m) or not 0.0 <= width_m <= stroke:
            raise ValueError(f"gripper width_m must be finite and within [0, {stroke}]")
        return width_m / stroke * 100.0

    def initialize(self) -> None:
        pass

    def poll(self) -> None:
        """Refresh cached state for drivers that require active polling.

        Push-based drivers can keep the default no-op implementation.  A
        polling driver must perform hardware I/O here and keep
        ``read_position()``, ``read_state()``, and ``read_telemetry()``
        non-blocking.
        """
        pass

    @abc.abstractmethod
    def set_position(self, length_pct: float) -> None:
        pass

    @abc.abstractmethod
    def read_position(self) -> float:
        pass

    @abc.abstractmethod
    def read_state(self) -> int:
        """Return 1=reached, 2=object/stall, 0=moving/unknown."""

    def set_speed(self, speed_pct: float) -> None:
        pass

    def set_force(self, force_pct: float) -> None:
        pass

    def open(self) -> None:
        self.set_position(100.0)

    def close(self) -> None:
        self.set_position(0.0)

    def disable(self) -> None:
        """Disable motor output when supported by the backend."""
        pass

    def read_telemetry(self) -> dict[str, Any]:
        return {}

    def read_observation(self) -> dict[str, Any]:
        """Canonical cached position, with measurement provenance and freshness."""
        telemetry = self.read_telemetry()
        age = telemetry.get('measurement_age_sec', telemetry.get('age_sec', 0.))
        measured = bool(telemetry.get('measurement_valid', True))
        position = float(self.read_position())
        return dict(opening_fraction=position / 100.,
                    measured=measured, valid=measured and math.isfinite(position) and 0 <= position <= 100,
                    observed_at=None if age is None else time.time() - age, age_s=age,
                    source='measured' if measured else 'target_estimate', telemetry=telemetry)

    def shutdown(self) -> None:
        pass

    def wait_until_reached(self, timeout: float = 2.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.read_state() in (1, 2):
                return True
            time.sleep(0.01)
        return False
