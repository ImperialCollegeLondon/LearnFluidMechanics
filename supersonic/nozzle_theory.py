"""1D compressible nozzle theory: geometry, isentropic relations and a normal shock.

Ported from cheang_original/data_animate_kulite.py (geometry, Kulite tap positions and
get_theoretical_pressure), so live theory matches the original post-processing.

Distances are in mm from the throat; pressures in any consistent absolute unit.
"""
import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

GAMMA = 1.4

# Channel height (mm) vs distance from throat (mm). Width is constant, so A/A* = h / h_throat.
# taps_mm: Kulite wall-tap positions from the throat (6 mm apart).
PROFILES = {
    "GEN_M2p74": dict(  # design exit Mach 2.74, A/A* = 20/6 = 3.33 (matches design spreadsheet)
        x=[0., 6.4, 12.8, 19.2, 25.6, 32., 38.4, 44.8, 51.2, 57.6, 64.],
        h=[6., 6.502, 8.4451, 10.4906, 12.3916, 14.359, 16.1275, 17.4878, 18.7439, 19.5813, 20.],
        taps_mm=[0, 6, 12, 18]),
    "GEN_M2p4": dict(
        x=[0., 6.4, 12.8, 19.2, 25.6, 32., 38.4, 44.8, 51.2, 57.6, 64.],
        h=[6., 6.3048, 7.4845, 8.7264, 9.8806, 11.0751, 12.1489, 12.9747, 13.7374, 14.2458, 14.5],
        taps_mm=[6, 12, 18, 24]),
}


class Nozzle:
    def __init__(self, profile="GEN_M2p74"):
        p = PROFILES[profile]
        self.name = profile
        self.x_pts, self.h_pts = np.array(p["x"]), np.array(p["h"])
        self.spline = CubicSpline(self.x_pts, self.h_pts, bc_type=((1, 0.0), (1, 0.0)))
        self.h_throat = self.h_pts[0]
        self.x_exit = self.x_pts[-1]
        self.ae_at = self.h_pts[-1] / self.h_throat
        self.taps_mm = list(p["taps_mm"])

    def height(self, x_mm):
        """Channel height (mm). Upstream of the throat uses the original's converging approximation."""
        return self.h_throat * self.area_ratio(x_mm)

    def area_ratio(self, x_mm):
        """A/A_throat at x (mm from throat)."""
        if x_mm < 0:
            return 1 + x_mm**2 / 4.5
        return float(self.spline(min(x_mm, self.x_exit))) / self.h_throat

    def x_at_area_ratio(self, ar):
        """Diverging-section position (mm) where A/A_throat = ar."""
        return brentq(lambda x: self.area_ratio(x) - ar, 0.0, self.x_exit)


# --- Isentropic & normal-shock relations ---
def am_rel(M, g=GAMMA):
    """A/A* for Mach M."""
    if M <= 0: return float("inf")
    return (1/M) * (2/(g+1) * (1 + (g-1)/2 * M**2))**((g+1)/(2*(g-1)))

def p_isen(M, g=GAMMA, inv=-1):
    """p/p0 for inv=-1 (p0/p for inv=+1)."""
    return (1 + (g-1)/2 * M**2)**(inv*g/(g-1))

def solve_mach(area_ratio, is_supersonic=False, gamma=GAMMA):
    if area_ratio <= 1.0001: return 1.0
    try:
        if is_supersonic: return brentq(lambda M: am_rel(M, gamma) - area_ratio, 1.0, 10.0)
        return brentq(lambda M: am_rel(M, gamma) - area_ratio, 1e-6, 1.0)
    except Exception:
        return 1.0

def solve_nozzle_shock(P0, Pe, At, Ae, gamma=GAMMA):
    """Returns (M1, M2, A_shock). Subsonic throughout -> (M_throat, M_exit, At);
    shock beyond exit -> (M_exit, M_exit, Ae)."""
    def objective(M1, g):
        p1_p01 = p_isen(M1, g, -1)
        p2_p1 = 1 + (2*g/(g+1))*(M1**2-1)
        M2 = np.sqrt((1 + (g-1)/2 * M1**2) / (g*M1**2 - (g-1)/2))
        p02_p2 = p_isen(M2, g, 1)
        Ae_A2star = (Ae/At) * (1/am_rel(M1, g)) * am_rel(M2, g)
        Me = solve_mach(Ae_A2star, False, g)
        return (p_isen(Me, g, -1) * p02_p2 * p2_p1 * p1_p01) - (Pe/P0)

    if P0 <= Pe * 1.0001: return 0.0, 0.0, At
    limit = p_isen(solve_mach(Ae/At, False, gamma), gamma, -1)
    if (Pe/P0) > limit:  # not choked: subsonic everywhere
        try:
            me = math.sqrt(max(0, (2/(gamma-1))*((P0/Pe)**((gamma-1)/gamma)-1)))
            at_astar = am_rel(me, gamma) / (Ae/At)
            return solve_mach(at_astar, False, gamma), me, At
        except Exception:
            return 0.0, 0.0, At
    try:
        m_sup = solve_mach(Ae/At, True, gamma)
        m1 = brentq(objective, 1.0, m_sup, args=(gamma,))
        m2 = np.sqrt((1 + (gamma-1)/2 * m1**2) / (gamma*m1**2 - (gamma-1)/2))
        return m1, m2, am_rel(m1, gamma) * At
    except Exception:
        m_sup = solve_mach(Ae/At, True, gamma)
        return m_sup, m_sup, Ae

def wall_pressure(nozzle, x_list, P0, Pe, gamma=GAMMA):
    """Theoretical static pressure at each x (mm) for stagnation P0 and back pressure Pe
    (same absolute units). Isentropic, with a normal shock where back pressure requires one."""
    x_list = list(x_list)
    if P0 <= Pe * 1.0001:
        return [P0] * len(x_list)
    ae = nozzle.ae_at
    out = []
    if Pe / P0 > p_isen(solve_mach(ae, False), gamma, -1):
        # Subsonic everywhere (throat not choked): effective A* > At
        me = math.sqrt(max(0, (2/(gamma-1))*((P0/Pe)**((gamma-1)/gamma)-1)))
        a_star = 1.0 / (am_rel(me, gamma) / ae)
        for x in x_list:
            out.append(P0 * p_isen(solve_mach(nozzle.area_ratio(x) / a_star, False), gamma, -1))
        return out
    m1, m2, a_ns = solve_nozzle_shock(P0, Pe, 1.0, ae, gamma)
    a2_star = a_ns / am_rel(m2, gamma)      # sonic area for the post-shock flow
    shock_in_nozzle = a_ns < ae - 1e-5
    for x in x_list:
        ar = nozzle.area_ratio(x)
        if x <= 0:
            out.append(P0 * p_isen(solve_mach(ar, False), gamma, -1))
        elif not shock_in_nozzle or ar < a_ns:   # upstream of shock: supersonic isentropic
            out.append(P0 * p_isen(solve_mach(ar, True), gamma, -1))
        else:                                     # downstream: subsonic, reduced stagnation pressure
            P02 = P0 * (1.0 / a2_star)
            out.append(P02 * p_isen(solve_mach(ar / a2_star, False), gamma, -1))
    return out

def shock_x(nozzle, P0, Pe):
    """Theoretical shock position (mm from throat), or None if no shock inside the nozzle."""
    if P0 <= Pe * 1.0001 or Pe / P0 > p_isen(solve_mach(nozzle.ae_at, False), GAMMA, -1):
        return None
    _, _, a_ns = solve_nozzle_shock(P0, Pe, 1.0, nozzle.ae_at)
    if a_ns >= nozzle.ae_at - 1e-5 or a_ns <= 1.0:
        return None
    try:
        return nozzle.x_at_area_ratio(a_ns)
    except Exception:
        return None
