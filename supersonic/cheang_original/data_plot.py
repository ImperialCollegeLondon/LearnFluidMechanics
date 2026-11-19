import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.signal import lfilter, lfilter_zi
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D  # Necessary import for 3D projection
from matplotlib.collections import PolyCollection  # Needed for 3D area shading

# Load the dataset
file_path = "data_20260602_171128.csv"
df = pd.read_csv(file_path)

# ==========================================
# USER TUNING CONFIGURATION
# ==========================================
crop_pre_valve = True       # Set to False to keep pre-valve data and preserve the original time scale
plot_raw_mdot = False        # Set to False to plot ONLY filtered mdot and hide the legend
time_step_interval = 5.0    # Step interval in seconds for Figure 4 waterfall plot (e.g., 5.0 or 10.0)
crop_fig4_time = False       # Set to True to truncate the upper time bound of Figure 4
fig4_max_time = 40.0        # Maximum time bound in seconds for Figure 4 (only applies if crop_fig4_time is True)

# ==========================================
# TIME-ALIGNMENT & VALVE_OPENING TRUNCATION
# ==========================================
if crop_pre_valve:
    # Identify where upstream plenum pressure (p2) first increases beyond 1.2 bar
    th_mask = df['p2'] > 1.2
    if th_mask.any():
        idx_valve_open = th_mask.idxmax()
    else:
        # Fallback to absolute max if threshold is never crossed
        idx_valve_open = df['p2'].idxmax()
        
    t_valve_open = df.loc[idx_valve_open, 'time']

    # Remove all data points prior to the valve opening timestamp
    df = df.loc[idx_valve_open:].reset_index(drop=True)

    # Shift all time points so that the valve opening occurs at t = 0
    df['time'] = df['time'] - t_valve_open
    
    # Filter starts immediately at the valve opening on the shifted scale
    t_filter_start = 0.0
else:
    # Filter starts at 12.0 seconds on the original, uncropped scale
    t_filter_start = 12.0

# ==========================================
# RAW MASS FLOW RATE CALCULATION
# ==========================================
# Tank volume: 300 L = 0.3 m^3
V_tank = 0.3  
# Gas constant for air: 287.05 J/(kg*K) (adjust if using a different gas)
R_gas = 287.05  

# Ideal gas law: mass = (P * V) / (R * T)
# Convert p1 from Bar to Pa (1 Bar = 1e5 Pa) and t1 from Celsius to Kelvin (Mass in kg)
df['mass'] = (df['p1'] * 1e5 * V_tank) / (R_gas * (df['t1'] + 273.15))

# Calculate the individual time differences between consecutive timestamps
dt_elements = df['time'].diff()

# First-order backward difference approximation: mdot = -dm/dt
# Negative sign added to flip mass flow to positive (mass depletion)
# Multiplied by 1000 to convert from kg/s to g/s to match the original df units
df['mdot_raw'] = (-df['mass'].diff() / dt_elements) * 1000

# ==========================================
# BIQUAD LOW-PASS FILTER SECTION
# ==========================================
# Adjustable tuning parameters
fc = 0.1     # Cutoff frequency (Hz)
Q = 0.707    # Quality factor

# Automatically calculate sampling frequency from the time column
dt = df['time'].diff().mean()
fs = 1.0 / dt

# Calculate Biquad low-pass filter coefficients
w0 = 2 * np.pi * fc / fs
alpha = np.sin(w0) / (2 * Q)

b0 = (1 - np.cos(w0)) / 2
b1 = 1 - np.cos(w0)
b2 = (1 - np.cos(w0)) / 2
a0 = 1 + alpha
a1 = -2 * np.cos(w0)
a2 = 1 - alpha

# Normalize coefficients by a0
b = [b0 / a0, b1 / a0, b2 / a0]
a = [1.0, a1 / a0, a2 / a0]

# Filter initialization
df['mdot_filtered'] = np.nan

# Mask to apply filter only after the user-specified start time
filter_mask = df['time'] >= t_filter_start

if filter_mask.any():
    num_prior = 20
    # Find the indices prior to t_filter_start
    prior_data = df.loc[df['time'] < t_filter_start, 'mdot_raw'].dropna()
    
    if not prior_data.empty:
        # Take the average of up to 20 values right before t_filter_start
        init_val = prior_data.tail(num_prior).mean()
    else:
        # If no prior data exists (e.g. t_filter_start == 0.0), average the first 20 active values
        init_val = df.loc[filter_mask, 'mdot_raw'].dropna().head(num_prior).mean()
    
    # Check if a valid initial value was found; fallback to 0.0 if data is entirely missing/NaN
    if pd.isna(init_val):
        init_val = 0.0

    # Compute steady-state filter delays scaled by our initial value
    zi = lfilter_zi(b, a) * init_val
    
    # Apply the filter to the target subset using the initialized state
    filtered_values, _ = lfilter(b, a, df.loc[filter_mask, 'mdot_raw'].fillna(method='bfill'), zi=zi)
    df.loc[filter_mask, 'mdot_filtered'] = filtered_values

# ==========================================
# FIGURE 1: Main System Parameters
# ==========================================
fig1, axs1 = plt.subplots(4, 1, sharex=True, figsize=(10, 10))

# 1. Plot Calculated Mass Flow Rate (mdot)
if plot_raw_mdot:
    axs1[0].scatter(df['time'], df['mdot_raw'], color='darkgray', s=2, alpha=0.6, label='raw mass flow rate')
    axs1[0].scatter(df['time'], df['mdot_filtered'], color='navy', s=2, label='filtered mass flow rate')
    axs1[0].legend(loc='upper right')
else:
    axs1[0].scatter(df['time'], df['mdot_filtered'], color='navy', s=2)

axs1[0].set_ylabel(r'Calc. Flow Rate ($\mathrm{g/s}$)')
axs1[0].xaxis.set_minor_locator(AutoMinorLocator())
axs1[0].yaxis.set_minor_locator(AutoMinorLocator())
axs1[0].grid(True, which='major', linestyle='--', alpha=0.6)
axs1[0].grid(True, which='minor', linestyle=':', alpha=0.3)

# 2. Plot Mach
axs1[1].scatter(df['time'], df['M1'], color='darkorchid', s=2)
axs1[1].set_ylabel(r'NS upstream Mach')
axs1[1].xaxis.set_minor_locator(AutoMinorLocator())
axs1[1].yaxis.set_minor_locator(AutoMinorLocator())
axs1[1].grid(True, which='major', linestyle='--', alpha=0.6)
axs1[1].grid(True, which='minor', linestyle=':', alpha=0.3)

# 3. Plot Pressures (p1 and p2)
axs1[2].scatter(df['time'], df['p1'], label='tank', color='crimson', s=2)
axs1[2].scatter(df['time'], df['p2'], label='upstream plenum', color='darkorange', s=2)
axs1[2].set_ylabel('Absolute Pressure (Bar)')
axs1[2].legend(loc='upper right')
axs1[2].xaxis.set_minor_locator(AutoMinorLocator())
axs1[2].yaxis.set_minor_locator(AutoMinorLocator())
axs1[2].grid(True, which='major', linestyle='--', alpha=0.6)
axs1[2].grid(True, which='minor', linestyle=':', alpha=0.3)

# 4. Plot Temperatures (t1 and t2)
axs1[3].scatter(df['time'], df['t1'], label='tank temperature', color='forestgreen', s=2)
axs1[3].scatter(df['time'], df['t2'], label='upstream plenum temperature', color='purple', s=2)
axs1[3].set_ylabel(r'Temperature ($^\circ\mathrm{C}$)')
axs1[3].set_xlabel('Time (s)')
axs1[3].legend(loc='center right')
axs1[3].xaxis.set_minor_locator(AutoMinorLocator())
axs1[3].yaxis.set_minor_locator(AutoMinorLocator())
axs1[3].grid(True, which='major', linestyle='--', alpha=0.6)
axs1[3].grid(True, which='minor', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# FIGURE 2: Normalized Axial Pressures (p/p0)
# ==========================================
fig2, ax2 = plt.subplots(figsize=(10, 4))

# Axial Kulite scatter plots normalized by p2 (stagnation plenum pressure)
ax2.scatter(df['time'], df['pax1'] / df['p2'], label='$p_{ax1}/p_0$, x = 0', color='teal', s=2)
ax2.scatter(df['time'], df['pax2'] / df['p2'], label='$p_{ax2}/p_0$, x = 6', color='chocolate', s=2)
ax2.scatter(df['time'], df['pax3'] / df['p2'], label='$p_{ax3}/p_0$, x = 12', color='cadetblue', s=2)
ax2.scatter(df['time'], df['pax4'] / df['p2'], label='$p_{ax4}/p_0$, x = 18', color='crimson', s=2)

ax2.set_title('Normalized Axial Pressures ($p/p_0$) over Time', fontsize=14)
ax2.set_ylabel('Pressure Ratio ($p/p_0$)')
ax2.set_xlabel('Time (s)')
ax2.legend(loc='upper right')
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.grid(True, which='major', linestyle='--', alpha=0.6)
ax2.grid(True, which='minor', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# FIGURE 3: 3D Temporal Pressure Waterfall Plot (Normalized)
# ==========================================
fig3 = plt.figure(figsize=(11, 6))
ax3 = fig3.add_subplot(111, projection='3d')

# Spatial locations corresponding to the 4 axial transducers from throat
x_locations = [0, 6, 12, 18]
pax_columns = ['pax1', 'pax2', 'pax3', 'pax4']
waterfall_colors = ['teal', 'chocolate', 'cadetblue', 'crimson']

# Determine the structural baseline for shading using the normalized profiles
z_min = min([(df[col] / df['p2']).min() for col in pax_columns])
z_max = max([(df[col] / df['p2']).max() for col in pax_columns])

# Project each individual spatial station's pressure track across the shared timeline
for x_pos, col, color in zip(x_locations, pax_columns, waterfall_colors):
    norm_pressure = df[col] / df['p2']
    
    # 1. Plotting line trace path along spatial track depth (y = constant)
    ax3.plot(df['time'], [x_pos] * len(df), norm_pressure, color=color, alpha=0.9, lw=1.5,
             label=f'x = {x_pos} mm ({col}/p0)')
    
    # 2. Generate polygon coordinates to construct the shaded area underneath the trace line
    times = df['time'].values
    pressures = norm_pressure.values
    
    # Define polygon perimeter vertices: starts at baseline -> follows curve -> closes back at baseline
    verts = []
    path_x = np.concatenate([[times[0]], times, [times[-1]]])
    path_z = np.concatenate([[z_min], pressures, [z_min]])
    verts.append(list(zip(path_x, path_z)))
    
    # Create the polygon collection profile and map it to its specific cross-sectional plane index
    poly = PolyCollection(verts, facecolors=color, alpha=0.15, edgecolors='none')
    ax3.add_collection3d(poly, zs=x_pos, zdir='y')

ax3.set_title('Temporal Axial Pressure Variation (Waterfall $p/p_0$ Profile)', fontsize=14, pad=15)
ax3.set_xlabel('Time (s)', labelpad=10)
ax3.set_ylabel('Distance from Throat (mm)', labelpad=10)
ax3.set_zlabel('Pressure Ratio ($p/p_0$)', labelpad=10)
ax3.set_yticks(x_locations)
ax3.set_zlim(z_min, z_max * 1.05)
ax3.view_init(elev=25, azim=-60)  
ax3.legend(loc='upper left', bbox_to_anchor=(0.02, 0.9))

plt.tight_layout()
plt.show()

# ==========================================
# FIGURE 4: 3D Spatial Pressure Waterfall Plot (Configurable Intervals & Crop)
# ==========================================
fig4 = plt.figure(figsize=(11, 6))
ax4 = fig4.add_subplot(111, projection='3d')

# Identify target intervals regardless of original time cropping using the user variable
min_time = df['time'].min()
max_time = df['time'].max()

# Apply the user defined maximum time limit for Figure 4 if enabled
if crop_fig4_time:
    max_time = min(max_time, fig4_max_time)

target_times = np.arange(np.ceil(min_time), max_time, time_step_interval)

# Generate a colormap gradient for the distinct time slices
colors_fig4 = cm.viridis(np.linspace(0, 1, len(target_times)))

for t_target, color in zip(target_times, colors_fig4):
    # Find the row closest to the target interval time step
    idx = (df['time'] - t_target).abs().idxmin()
    actual_time = df.loc[idx, 'time']
    
    # Extract and normalize the pressure values for this specific time slice across the 4 sensors
    pressures_at_t = (df.loc[idx, pax_columns] / df.loc[idx, 'p2']).values
    
    # 1. Plot the spatial pressure trace (P vs. x) at this specific time depth (y = t)
    ax4.plot(x_locations, [actual_time] * len(x_locations), pressures_at_t, 
             color=color, alpha=0.9, lw=2)
    
    # 2. Add area shading underneath
    verts = []
    path_x = np.concatenate([[x_locations[0]], x_locations, [x_locations[-1]]])
    path_z = np.concatenate([[z_min], pressures_at_t, [z_min]])
    verts.append(list(zip(path_x, path_z)))
    
    poly = PolyCollection(verts, facecolors=color, alpha=0.3, edgecolors='none')
    # Map the polygon onto the depth coordinate corresponding to time
    ax4.add_collection3d(poly, zs=actual_time, zdir='y')

ax4.set_title(f'Spatial Axial Pressure Distribution ($p/p_0$, {time_step_interval}-Second Intervals)', fontsize=14, pad=15)
ax4.set_xlabel('Distance from Throat (mm)', labelpad=10)
ax4.set_ylabel('Time (s)', labelpad=10)
ax4.set_zlabel('Pressure Ratio ($p/p_0$)', labelpad=10)
ax4.set_xticks(x_locations)
ax4.set_ylim(min_time, max_time)  # Lock axis bounds to match the crop explicitly
ax4.set_zlim(z_min, z_max * 1.05)
ax4.view_init(elev=25, azim=-60)

plt.tight_layout()
plt.show()