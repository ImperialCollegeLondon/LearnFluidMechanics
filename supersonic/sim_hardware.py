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
P_MAX = 6.0    # bar gauge in the supply tank at start of blowdown
PLENUM_FRAC = 0.85  # plenum (nozzle P0) gauge pressure as a fraction of tank: valve/pipe losses
TAU = 6.0      # s, blowdown time constant

_t0 = time.time()


def sim_pressure_bar():
    """Supply tank gauge pressure (bar) at the current moment of the scenario."""
    t = (time.time() - _t0) % CYCLE
    if t < HOLD:
        return P_MAX
    return P_MAX * math.exp(-(t - HOLD) / TAU)


class FakeADC:
    """ADS1115 replacement. Returns counts that prc.py's conversion maps back to sim pressure."""

    def read_adc(self, channel, gain=1.0):
        # ch0 = P1 supply tank; ch1 = P2 plenum feeding the nozzle
        p = sim_pressure_bar() * (1.0 if channel == 0 else PLENUM_FRAC) + random.gauss(0, 0.01)
        if channel == 1:
            p -= 0.05  # prc.py adds a 0.05 bar offset to P2
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


P_ATM = 1.01325   # bar, standard atmosphere (matches P_ATM_PA in prc.py)

def plenum_abs_bar():
    """Plenum (nozzle stagnation) absolute pressure, bar."""
    return sim_pressure_bar() * PLENUM_FRAC + P_ATM


# prc.py sets this to a function p0_abs_bar -> 4 Pax readings from nozzle theory,
# so simulated taps show isentropic expansion and a shock moving through as the tank empties.
PAX_MODEL = None


def fake_pax():
    """Four nozzle-wall pressures, as LabVIEW would send them."""
    p0 = plenum_abs_bar()  # nozzle stagnation pressure
    if PAX_MODEL is not None:
        base = PAX_MODEL(p0)
    else:  # crude fallback: falling pressures along the nozzle
        base = [p0 * k for k in (0.53, 0.35, 0.2, 0.13)]
    return [v + random.gauss(0, 0.01) for v in base]


# Fake camera view of the nozzle (camera pixels). prc.py uses these to line up the default overlay.
CAM_THROAT_PX = (110, 300)   # (x, y) of the throat on the flat wall (raw image: contour above it)
CAM_PX_PER_MM = 7.0
NOZZLE = None                # prc.py sets to a nozzle_theory.Nozzle so the fake view shows the real profile
SHOCK_X = None               # prc.py sets to a function p0_abs_bar -> shock x (mm from throat) or None


class FakeCamera:
    """Schlieren-ish side view of the half nozzle: grey walls, dark channel, and a bright
    normal shock at its theoretical position for the current simulated tank pressure."""

    def __init__(self, width, height):
        self.w, self.h = width, height
        tx, ty = CAM_THROAT_PX
        self.x_mm = (np.arange(width) - tx) / CAM_PX_PER_MM
        img = np.full((height, width, 3), 85, np.uint8)            # metal
        img[..., 2] = 95
        self.upper = np.full(width, ty, dtype=int)
        if NOZZLE is not None:
            x_clip = np.clip(self.x_mm, -3.0, NOZZLE.x_exit)
            h_mm = np.array([NOZZLE.height(x) for x in x_clip])
            self.upper = (ty - h_mm * CAM_PX_PER_MM).astype(int)
            beyond = self.x_mm > NOZZLE.x_exit                       # open downstream plenum
            self.upper[beyond] = 0
        rows = np.arange(height)[:, None]
        channel = (rows >= self.upper[None, :]) & (rows < ty)
        img[channel] = (18, 22, 30)
        img[ty:ty + 2, :] = (140, 140, 150)                          # flat wall edge highlight
        self.base = img

    def frame(self):
        img = self.base.copy()
        img += np.random.randint(0, 6, img.shape, dtype=np.uint8)  # sensor noise
        if SHOCK_X is not None:
            xs = SHOCK_X(plenum_abs_bar())
            if xs is not None:
                col = int(CAM_THROAT_PX[0] + xs * CAM_PX_PER_MM)
                if 0 <= col < self.w:
                    top, bot = self.upper[col], CAM_THROAT_PX[1]
                    img[top:bot, max(0, col - 1):col + 2] = (235, 235, 235)
        return img
