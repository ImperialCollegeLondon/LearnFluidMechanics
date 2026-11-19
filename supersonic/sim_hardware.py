"""Stand-in hardware so prc.py can run without the Pi rig (dev mode).

Each fake mimics the interface prc.py uses on the real device:
  FakeADC.read_adc(channel, gain)   -> raw ADS1115 counts
  FakeThermocouple.temperature      -> deg C
  fake_pax()                        -> 4 Kulite pressures (as if from LabVIEW over TCP)
  FakeCamera.frame()                -> HxWx3 uint8 RGB array

All share one simple scenario: a repeating tank blowdown (hold at P_MAX,
then exponential decay to atmospheric), so plots, the shock solver and the
video overlay all have something moving to show.
"""
import math
import random
import time

import numpy as np

CYCLE = 30.0   # s, length of one simulated run before the tank "refills"
HOLD = 3.0     # s at full pressure before blowdown starts
P_MAX = 5.0    # bar gauge at start of blowdown
TAU = 6.0      # s, blowdown time constant

_t0 = time.time()


def sim_pressure_bar():
    """Tank gauge pressure (bar) at the current moment of the scenario."""
    t = (time.time() - _t0) % CYCLE
    if t < HOLD:
        return P_MAX
    return P_MAX * math.exp(-(t - HOLD) / TAU)


class FakeADC:
    """ADS1115 replacement. Returns counts that prc.py's conversion maps back to sim pressure."""

    def read_adc(self, channel, gain=1.0):
        p = sim_pressure_bar() * (1.0 if channel == 0 else 0.97) + random.gauss(0, 0.01)
        # Invert prc.py: p = (counts/32767*6.144*50 - 25) * 0.0689 * 0.75
        counts = (p / (0.0689 * 0.75) + 25) / (50 * 6.144) * 32767
        return int(max(-32768, min(32767, counts)))


class FakeThermocouple:
    """MCP9600 replacement. Cools a little as the tank blows down."""

    def __init__(self, base_c=20.0):
        self.base_c = base_c

    @property
    def temperature(self):
        frac = sim_pressure_bar() / P_MAX
        return self.base_c - 8.0 * (1 - frac) + random.gauss(0, 0.05)


def fake_pax():
    """Four axial nozzle-wall pressures, falling along the nozzle."""
    p = sim_pressure_bar()
    return [p * k + random.gauss(0, 0.01) for k in (0.6, 0.35, 0.2, 0.15)]


class FakeCamera:
    """Test pattern (gradient + grid) with a red 'shock' bar that moves with the blowdown."""

    def __init__(self, width, height):
        self.w, self.h = width, height
        base = np.zeros((height, width, 3), np.uint8)
        base[..., 0] = np.linspace(0, 60, width, dtype=np.uint8)[None, :]
        base[..., 1] = np.linspace(0, 60, height, dtype=np.uint8)[:, None]
        base[..., 2] = 40
        base[::40, :, :] = 90
        base[:, ::40, :] = 90
        self.base = base

    def frame(self):
        img = self.base.copy()
        frac = sim_pressure_bar() / P_MAX
        x = int((0.15 + 0.7 * (1 - frac)) * self.w)
        img[:, max(0, x - 3):x + 3] = (255, 80, 80)
        return img
