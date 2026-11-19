import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, curve_fit

# 1. Data Entry
distances = np.array([1, 2, 3, 4])
rx95_data = np.array(
    [
        [90.1, 89.8, 90.2],  # Distance 1
        [85.5, 85.5, 85.1],  # Distance 2
        [82.5, 82.4, 82.7],  # Distance 3
        [81.8, 81.3, 81.5],  # Distance 4
    ]
)

means = np.mean(rx95_data, axis=1)
mins = np.min(rx95_data, axis=1)
maxs = np.max(rx95_data, axis=1)
y_errors = np.vstack([means - mins, maxs - means])


# 2. Enhanced Trendline Definition (with scaling and translation)
def enhanced_inverse_square(d, C, d0, offset):
    return C / ((d - d0) ** 2) + offset


# Initial guess to guide the solver [C, d0, offset]
initial_guess = [10.0, -1.0, 80.0]

# Fit the curve
popt, pcov = curve_fit(
    enhanced_inverse_square,
    distances,
    means,
    p0=initial_guess,
    maxfev=10000,
)
C_fit, d0_fit, offset_fit = popt

# Calculate R^2 Error for the new fit
residuals = means - enhanced_inverse_square(distances, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((means - np.mean(means)) ** 2)
r_squared = 1 - (ss_res / ss_tot)


# 3. Find the exact x-intercept where y = 85 dB
target_db = 85.0


def intercept_func(d):
    return enhanced_inverse_square(d, C_fit, d0_fit, offset_fit) - target_db


# Use brentq to find the root between distance bounds 1 and 4
x_intersect = brentq(intercept_func, 1.0, 4.0)


# 4. Plotting
plt.figure()

# Plot experimental data
plt.errorbar(
    distances,
    means,
    yerr=y_errors,
    fmt="o",
    color="darkblue",
    ecolor="red",
    capsize=5,
)

# Plot smooth trendline
x_smooth = np.linspace(0.9, 4.1, 200)
y_smooth = enhanced_inverse_square(x_smooth, C_fit, d0_fit, offset_fit)

# Minimal Change: Using round() to bring back .1f rounding cleanly
trend_label = (
    r"Trendline: $SPL = \frac{" + str(round(C_fit, 1)) + r"}{(d + " + str(round(-d0_fit, 1)) + r")^2} + " + str(round(offset_fit, 1)) + r"$"
)

plt.plot(x_smooth, y_smooth, linestyle="--", color="green", label=trend_label)

# Add horizontal intersection line at 85 dB
plt.axhline(
    y=target_db,
    xmin=0,
    xmax=(x_intersect - 0.9) / (4.1 - 0.9),
    linestyle=":",
    color="purple",
    alpha=0.7,
)

# Add vertical drop line to the x-axis from the intersection point
plt.axvline(
    x=x_intersect,
    ymin=0,
    ymax=(target_db - plt.ylim()[0]) / (plt.ylim()[1] - plt.ylim()[0]),
    linestyle=":",
    color="purple",
    alpha=0.7,
)

# Mark the specific intersection point on the curve
intersect_label = r"85 dB Intersection ($d \approx " + str(round(x_intersect, 1)) + r"$)"
plt.plot(
    x_intersect,
    target_db,
    marker="X",
    color="purple",
    markersize=8,
    label=intersect_label,
)

# Formatting
plt.xlabel("Distance to outlet ($m$)", fontsize=11)
plt.ylabel("Sound Pressure Level (dB)", fontsize=11)

# Modify x-ticks to dynamically include the precise intersection coordinate
current_xticks = list(distances)
if not any(np.isclose(x_intersect, tick, atol=0.1) for tick in current_xticks):
    current_xticks.append(x_intersect)
current_xticks = sorted(current_xticks)

plt.xticks(
    current_xticks,
    [str(round(x, 3)) if x == x_intersect else str(int(x)) for x in current_xticks],
)
plt.grid(True, linestyle=":", alpha=0.5)
plt.legend(loc="upper right", fontsize=9.5)

plt.tight_layout()
plt.show()