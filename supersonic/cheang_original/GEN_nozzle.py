import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def calculate_mach_2d(Rt, ye, gamma=1.4):
    """Calculates supersonic Mach number from 2D area ratio Ae/At = ye/Rt."""
    area_ratio = ye / Rt
    
    # Isentropic area-Mach relation equation to solve for zero
    def equations(M):
        if M <= 0: return -1.0 # Penalize non-physical/subsonic roots
        term1 = 2 / (gamma + 1)
        term2 = 1 + 0.5 * (gamma - 1) * M**2
        return (1 / M) * (term1 * term2) ** ((gamma + 1) / (2 * (gamma - 1))) - area_ratio

    # Initial guess for supersonic root
    mach_guess = 2.5
    mach_solution, = fsolve(equations, mach_guess)
    return mach_solution

def calculate_ye_2d(Rt, M, gamma=1.4):
    """Calculates required 2D exit height ye from target Mach number and Rt."""
    term1 = 2 / (gamma + 1)
    term2 = 1 + 0.5 * (gamma - 1) * M**2
    area_ratio = (1 / M) * (term1 * term2) ** ((gamma + 1) / (2 * (gamma - 1)))
    return area_ratio * Rt

def generate_nozzle_m27486(num_points=11, ye=20.0, Rt=6.0, l=70.0):
    """
    Generates a nozzle contour for M=2.7486.
    Forces y(0) = Rt exactly and y(l) = ye exactly.
    """
    # 1. Base data points from chart
    x_nodes = np.linspace(0, 1, 11)
    y_25 = np.array([0.38, 0.40, 0.46, 0.55, 0.65, 0.74, 0.82, 0.88, 0.94, 0.98, 1.00])
    y_30 = np.array([0.24, 0.28, 0.43, 0.54, 0.62, 0.72, 0.81, 0.88, 0.94, 0.98, 1.00])

    # 2. Cubic Spline for continuous curvature
    cs25 = CubicSpline(x_nodes, y_25)
    cs30 = CubicSpline(x_nodes, y_30)

    x_grid = np.linspace(0, 1, num_points)
    y25_vals = cs25(x_grid)
    y30_vals = cs30(x_grid)

    # 3. Non-linear interpolation weight specifically tuned for M = 2.7486
    # (2.7486 - 2.5) / (3.0 - 2.5) = 0.4972 base linear weight
    w_base = 0.4972
    w = np.linspace(w_base * 0.7, w_base * 1.2, num_points)
    w = np.clip(w, 0.0, 1.0) # Ensure boundaries stay clean
    
    y_raw = y25_vals * (1 - w) + y30_vals * w
    
    # 4. Rigorous geometric scaling to force boundaries: y(0)=Rt, y(l)=ye
    # Maps the raw curve strictly into the [Rt, ye] physical domain
    y_min, y_max = y_raw[0], y_raw[-1]
    y_final = Rt + (y_raw - y_min) * (ye - Rt) / (y_max - y_min)
    
    x_final = x_grid * l
    return x_final, y_final

# --- Execution Configuration ---
points = 11  # Easily toggle between 11, 21, 51, etc.
mode = 'mach'  # Options: 'geometry' (uses target_ye) or 'mach' (uses target_mach)

target_Rt = 3.0
target_l = 50.0

# Define inputs for both modes
target_ye = 7.25      # Used if mode == 'geometry'
target_mach = 2.3  # Used if mode == 'mach'

# --- Processing Mode ---
if mode == 'geometry':
    calculated_mach = calculate_mach_2d(target_Rt, target_ye)
elif mode == 'mach':
    calculated_mach = target_mach
    target_ye = calculate_ye_2d(target_Rt, target_mach)
else:
    raise ValueError("Invalid mode. Choose 'geometry' or 'mach'.")

x, y = generate_nozzle_m27486(num_points=points, ye=target_ye, Rt=target_Rt, l=target_l)

# --- Verification Print ---
print(f"--- Verification (Mode: {mode}) ---")
print(f"2D Exhaust Mach (M_e) = {calculated_mach:.4f}")
print(f"y(x=0)  = {y[0]:.4f} mm (Target: {target_Rt})")
print(f"y(x={target_l}) = {y[-1]:.4f} mm (Target: {target_ye:.4f})\n")

print("X Coordinates (mm):")
print(np.round(x, 4))
print("\nY Coordinates (mm):")
print(np.round(y, 4))

print(np.round(y/y[-1], 4))

# --- Plotting ---
plt.figure(figsize=(8, 3.5))
plt.plot(x, y, 'go-', markersize=4, label=f'2D Profile ({points} pts)')
plt.axhline(target_Rt, color='r', linestyle='--', alpha=0.5, label=f'Throat Half-Height ({target_Rt}mm)')
plt.axhline(target_ye, color='b', linestyle='--', alpha=0.5, label=f'Exit Half-Height ({target_ye:.2f}mm)')
plt.title(f"Scaled 2D Nozzle Contour (M={calculated_mach:.4f})")
plt.xlabel("Axial Position x (mm)")
plt.ylabel("Transverse Position y (mm)")
plt.grid(True)
plt.legend()
plt.axis('equal')  # Forces 1:1 aspect ratio mapping for x and y axes
plt.tight_layout()
plt.show()