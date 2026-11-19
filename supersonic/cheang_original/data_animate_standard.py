import os
import math
import numpy as np
import pandas as pd
import matplotlib
# Force Matplotlib to render headlessly to memory buffers
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image, ImageDraw
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import lfilter, lfilter_zi
from scipy.interpolate import CubicSpline

# =====================================================================
# 1. NOZZLE & SHOCK ISENTROPIC FLUID DYNAMICS MATH
# =====================================================================

# New Symmetrical Profile Definition (from image data)
X_NEW = np.array([0., 5., 10., 15., 20., 25., 30., 35., 40., 45., 50.])
Y_NEW = np.array([3., 3.251, 4.2225, 5.2453, 6.1958, 7.1795, 8.0638, 8.7439, 9.3719, 9.7906, 10.])

# Clamped cubic spline with zero start-stop gradients (first derivative = 0)
new_spline = CubicSpline(X_NEW, Y_NEW, bc_type=((1, 0.0), (1, 0.0)))

def am_rel(M, g=1.4):
    if M <= 0: return float('inf')
    return (1/M) * (2/(g+1) * (1 + (g-1)/2 * M**2))**((g+1)/(2*(g-1)))

def p_isen(M, g=1.4, inv=-1):
    return (1 + (g-1)/2 * M**2)**((inv)*g/(g-1))

def solve_mach(area_ratio, is_supersonic=False, gamma=1.4):
    if area_ratio <= 1.0001: return 1.0
    try:
        if is_supersonic: return brentq(lambda M: am_rel(M, gamma) - area_ratio, 1.0, 10.0)
        else: return brentq(lambda M: am_rel(M, gamma) - area_ratio, 1e-6, 1.0)
    except: return 1.0

def get_area_ratio(x, aeat=3.3, xmax=32, profile='old'):
    """Calculates theoretical area ratio based on the chosen nozzle profile."""
    if profile == 'new':
        x_val = abs(x) # Handle negative values symmetrically
        if x_val > 50: x_val = 50.0
        return float(new_spline(x_val) / 3.0) # Divide by throat half-height (3mm)
    else:
        if x < 0: return 1 + (x**2)/4.5
        return math.tanh(2*x/xmax)/math.tanh(2)*(aeat-1) + 1

def solve_shock_angle(M, theta_deg, gamma=1.4):
    """
    Solves the theta-beta-M relation for the weak-shock solution.
    Returns (beta_degrees, is_detached).
    """
    if M <= 1.001:
        return 90.0, True

    mu = math.asin(1.0 / M)
    theta = math.radians(theta_deg)

    def theta_from_beta(beta):
        sin2b = math.sin(beta) ** 2
        num = M ** 2 * sin2b - 1
        den = M ** 2 * (gamma + math.cos(2 * beta)) + 2
        if den == 0:
            return 0.0
        return math.atan(2.0 / math.tan(beta) * num / den)

    res = minimize_scalar(
        lambda b: -theta_from_beta(b),
        bounds=(mu + 1e-5, math.pi / 2 - 1e-5),
        method='bounded'
    )
    beta_max_theta = res.x
    theta_max = theta_from_beta(beta_max_theta)

    if theta >= theta_max:
        return 90.0, True

    beta_rad = brentq(lambda b: theta_from_beta(b) - theta, mu + 1e-5, beta_max_theta - 1e-8)
    return math.degrees(beta_rad), False

def draw_dashed_line(draw, pt1, pt2, fill, width=3, dash_length=10):
    """Helper to draw dashed lines for detached shocks using PIL."""
    x1, y1 = pt1
    x2, y2 = pt2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist == 0: return
    dashes = int(dist / dash_length)
    if dashes == 0:
        draw.line([pt1, pt2], fill=fill, width=width)
        return
    for i in range(dashes):
        if i % 2 == 0:
            sx = x1 + (x2 - x1) * i / dashes
            sy = y1 + (y2 - y1) * i / dashes
            ex = x1 + (x2 - x1) * (i + 1) / dashes
            ey = y1 + (y2 - y1) * (i + 1) / dashes
            draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)

def solve_nozzle_shock(P0, Pe, At, Ae, gamma=1.4):
    def objective(M1, gamma_val):
        p1_p01 = p_isen(M1, gamma_val, -1)
        p2_p1 = 1 + (2*gamma_val/(gamma_val+1))*(M1**2-1)
        M2 = np.sqrt((1 + (gamma_val-1)/2 * M1**2) / (gamma_val*M1**2 - (gamma_val-1)/2))
        p02_p2 = p_isen(M2, gamma_val, 1)
        Ae_A2star = (Ae/At) * (1/am_rel(M1, gamma_val)) * am_rel(M2, gamma_val)
        Me = solve_mach(Ae_A2star, is_supersonic=False, gamma=gamma_val)
        pe_p0e = p_isen(Me, gamma_val, -1)
        return (pe_p0e * p02_p2 * p2_p1 * p1_p01) - (Pe/P0)

    if P0 <= Pe * 1.0001: return 0.0, 0.0, At
    limit = p_isen(solve_mach(Ae/At, False, gamma), gamma, -1)
    if (Pe/P0) > limit:
        try:
            me = math.sqrt(max(0, (2/(gamma-1))*((P0/Pe)**((gamma-1)/gamma)-1)))
            at_astar = am_rel(me, gamma) / (Ae/At)
            return solve_mach(at_astar, False, gamma), me, At
        except: return 0.0, 0.0, At
    try:
        m_sup = solve_mach(Ae/At, True, gamma)
        m1 = brentq(objective, 1.0, m_sup, args=(gamma,))
        m2 = np.sqrt((1 + (gamma-1)/2 * m1**2) / (gamma*m1**2 - (gamma-1)/2))
        return m1, m2, am_rel(m1, gamma) * At
    except:
        m_sup = solve_mach(Ae/At, True, gamma)
        return m_sup, m_sup, Ae

def get_theoretical_pressure(x_eval, P0, Pe, aeat=3.3, xmax=32, profile='old'):
    gamma = 1.4
    At_val = 1.0
    
    if profile == 'new':
        x_eval_abs = np.clip(np.abs(x_eval), 0, 50)
        y_eval = new_spline(x_eval_abs) / 3.0
        Ae_val = new_spline(50) / 3.0
    else:
        y_eval = np.where(x_eval < 0, 1 + (x_eval**2)/4.5, np.tanh(2*x_eval/xmax)/np.tanh(2)*(aeat-1) + 1)
        Ae_val = aeat

    P_eval = np.zeros_like(x_eval)
    if P0 <= Pe * 1.0001: return np.ones_like(x_eval) * P0
    limit = p_isen(solve_mach(Ae_val/At_val, False), gamma, -1)
    
    if (Pe/P0) > limit:
        me = math.sqrt(max(0, (2/(gamma-1))*((P0/Pe)**((gamma-1)/gamma)-1)))
        at_astar = am_rel(me, gamma) / (Ae_val/At_val)
        A_star = At_val / at_astar
        for i, A_local in enumerate(y_eval):
            M_local = solve_mach(A_local/A_star, False)
            P_eval[i] = P0 * p_isen(M_local, gamma, -1)
        return P_eval
        
    try:
        m1, m2, A_NS_ratio = solve_nozzle_shock(P0, Pe, At_val, Ae_val, gamma)
    except:
        m1 = solve_mach(Ae_val/At_val, True)
        m2 = m1
        A_NS_ratio = Ae_val
        
    A2_star = A_NS_ratio / am_rel(m2, gamma)
    for i, (x, A_local) in enumerate(zip(x_eval, y_eval)):
        if x <= 0:
            M_local = solve_mach(A_local/At_val, False)
            P_eval[i] = P0 * p_isen(M_local, gamma, -1)
        else:
            if A_local < A_NS_ratio or A_NS_ratio >= Ae_val - 1e-5:
                M_local = solve_mach(A_local/At_val, True)
                P_eval[i] = P0 * p_isen(M_local, gamma, -1)
            else:
                M_local = solve_mach(A_local/A2_star, False)
                P02 = P0 * (At_val / A2_star)
                P_eval[i] = P02 * p_isen(M_local, gamma, -1)
    return P_eval

# =====================================================================
# 2. MASTER SYNCHRONIZATION AND RENDER ENGINE
# =====================================================================
def generate_synchronized_mp4(csv_path, video_path, output_path, camera_fps=24, speedup=20, video_lag=-2.0, output_fps=24,
                              px_per_mm=6.0, x_throat=285, y_centerline=240, overlay_x_mm=18.0, wedge_half_angle_deg=16.0,
                              nozzle_profile='old'):
    """
    Combines telemetry CSV and raw H.264 camera streams side-by-side.
    """
    print("Loading data files into memory...")
    df = pd.read_csv(csv_path)
    df = df.sort_values(by='time').reset_index(drop=True)
    
    # -----------------------------------------------------------------
    # DETAILED MASS FLOW RATE RECALCULATION & FILTER CO-PROCESSING
    # -----------------------------------------------------------------
    V_tank = 0.3      # Tank volume: 300 L = 0.3 m^3
    R_gas = 287.05    # Gas constant for air J/(kg*K)
    gamma = 1.4
    
    df['mass'] = (df['p1'] * 1e5 * V_tank) / (R_gas * (df['t1'] + 273.15))
    dt_elements = df['time'].diff()
    df['mdot_raw'] = (-df['mass'].diff() / dt_elements) * 1000
    
    fc = 0.1
    Q = 0.707
    dt_mean = df['time'].diff().mean()
    fs = 1.0 / dt_mean

    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * Q)

    b0 = (1 - np.cos(w0)) / 2
    b1 = 1 - np.cos(w0)
    b2 = (1 - np.cos(w0)) / 2
    a0 = 1 + alpha
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha

    b_coeff = [b0 / a0, b1 / a0, b2 / a0]
    a_coeff = [1.0, a1 / a0, a2 / a0]

    th_mask = df['p2'] > 1.2
    if th_mask.any():
        idx_valve_open = th_mask.idxmax()
    else:
        idx_valve_open = df['p2'].idxmax()
        
    num_prior = 20
    idx_filter_start = min(idx_valve_open + num_prior, len(df) - 1)

    init_val = df['mdot_raw'].iloc[idx_valve_open:idx_filter_start].mean()
    if pd.isna(init_val):
        init_val = 0.0

    zi = lfilter_zi(b_coeff, a_coeff) * init_val
    filter_mask = df.index >= idx_filter_start
    
    df['mdot_filtered'] = np.nan
    raw_series_active = df.loc[filter_mask, 'mdot_raw'].fillna(method='bfill')
    
    if not raw_series_active.empty:
        filtered_values, _ = lfilter(b_coeff, a_coeff, raw_series_active, zi=zi)
        df.loc[filter_mask, 'mdot_filtered'] = filtered_values

    df['mdot_final'] = df['mdot']  
    df.loc[filter_mask, 'mdot_final'] = df.loc[filter_mask, 'mdot_filtered']
    
    csv_max_time = df['time'].max()

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=100)
    ax2 = ax1.twinx()

    ax1.set_xlim(0, csv_max_time)
    ax1.set_ylim(0, max(df['p1'].max(), df['p2'].max()) * 1.1)
    ax2.set_ylim(0, 60)

    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Pressure (bar)', color='royalblue', fontsize=11)
    ax2.set_ylabel('Mass Flow Rate (g/s)', color='darkorange', fontsize=11)

    ax1.grid(True, linestyle='--', alpha=0.6)
    
    line_p1, = ax1.plot([], [], label='Tank Pressure (P1)', color='navy', linewidth=2)
    line_p2, = ax1.plot([], [], label='Plenum Pressure (P2)', color='royalblue', linewidth=2)
    line_mdot, = ax2.plot([], [], label='Mass Flow Rate (mdot)', color='darkorange', linewidth=2, linestyle='--')
    
    lines = [line_p1, line_p2, line_mdot]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    title_text = ax1.set_title('', fontsize=12)
    plt.tight_layout()

    dt_per_frame = speedup / output_fps 
    combined_master_frames = []
    output_frame_count = 0

    print(f"Streaming H.264 and synthesizing matching timelines at {speedup}x speedup...")
    
    if nozzle_profile == 'new':
        Ae_At_exit = float(new_spline(50) / 3.0)
    else:
        Ae_At_exit = 3.3
        
    for video_frame_idx, raw_video_frame in enumerate(iio.imiter(video_path, plugin="pyav")):
        
        # -----------------------------------------------------------------
        # STEP A: Timeline Synchronization Calculation
        # -----------------------------------------------------------------
        source_video_time = video_frame_idx / camera_fps
        t_real_corresponding = source_video_time - video_lag
        target_t_real = output_frame_count * dt_per_frame
        
        if target_t_real > csv_max_time:
            print("Reached the end of telemetry data timeframe.")
            break

        if t_real_corresponding < target_t_real:
            continue
            
        while target_t_real < t_real_corresponding and target_t_real <= csv_max_time:
            if t_real_corresponding - target_t_real > dt_per_frame:
                target_t_real = output_frame_count * dt_per_frame
                output_frame_count += 1
                continue
            break

        # -----------------------------------------------------------------
        # STEP B: Render Matplotlib
        # -----------------------------------------------------------------
        idx_csv = np.searchsorted(df['time'], target_t_real)
        idx_csv = min(idx_csv, len(df) - 1)
        
        history_df = df.iloc[:idx_csv + 1]
        
        line_p1.set_data(history_df['time'], history_df['p1'])
        line_p2.set_data(history_df['time'], history_df['p2'])
        line_mdot.set_data(history_df['time'], history_df['mdot_final'].abs())
        
        current_mdot_positive = abs(df["mdot_final"].iloc[idx_csv])
        current_p2 = df["p2"].iloc[idx_csv]
        
        title_text.set_text(f'Telemetry Time: {target_t_real:.2f}s | mdot: {current_mdot_positive:.2f} g/s | P0: {current_p2:.2f} bar')
        
        fig.canvas.draw()
        plot_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        plot_rgb = plot_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4, ))[..., :3]

        target_height = raw_video_frame.shape[0]

        img_plot = Image.fromarray(plot_rgb)
        if img_plot.height != target_height:
            aspect = img_plot.width / img_plot.height
            img_plot = img_plot.resize((int(target_height * aspect), target_height), Image.Resampling.LANCZOS)
        plot_frame_final = np.array(img_plot)

        # -----------------------------------------------------------------
        # STEP C: Process Video Frame & Draw Dynamic Overlays
        # -----------------------------------------------------------------
        if raw_video_frame.ndim == 2:
            raw_video_frame = np.stack((raw_video_frame,)*3, axis=-1)
        elif raw_video_frame.shape[-1] == 4:
            raw_video_frame = raw_video_frame[..., :3]

        img_video = Image.fromarray(raw_video_frame)
        target_width = img_video.width
        target_height = img_video.height

        # 1. Zoom LHS video by 2x from video midpoint
        crop_w = target_width // 2
        crop_h = target_height // 2
        left = (target_width - crop_w) // 2
        top = (target_height - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h

        img_video = img_video.crop((left, top, right, bottom))
        img_video = img_video.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # 2. Setup scaled coordinates
        zoomed_x_throat = int((x_throat - left) * 2.0)
        zoomed_y_centerline = int((y_centerline - top) * 2.0)
        zoomed_px_per_mm = px_per_mm * 2.0

        draw = ImageDraw.Draw(img_video)

        # --- Draw Nozzle Profile Overlays (Dashed, Width = 1) ---
        profile_points_top = []
        profile_points_bottom = []
        for x_mm in np.linspace(0, 50, 200):
            # Calculate physical half-height (y_mm). get_area_ratio returns Area / ThroatArea.
            # Throat half height is 3.0mm, so y_mm = area_ratio * 3.0.
            y_mm = get_area_ratio(x_mm, aeat=Ae_At_exit, profile=nozzle_profile) * 3.0
            px_x = int(zoomed_x_throat + x_mm * zoomed_px_per_mm)
            px_y_top = int(zoomed_y_centerline - y_mm * zoomed_px_per_mm)
            px_y_bottom = int(zoomed_y_centerline + y_mm * zoomed_px_per_mm)
            profile_points_top.append((px_x, px_y_top))
            profile_points_bottom.append((px_x, px_y_bottom))
            
        dash_interval = 4  # Adjust this integer to make dashes longer or shorter
        for i in range(len(profile_points_top) - 1):
            if (i // dash_interval) % 2 == 0:
                draw.line([profile_points_top[i], profile_points_top[i+1]], fill=(0, 255, 255), width=1)
                draw.line([profile_points_bottom[i], profile_points_bottom[i+1]], fill=(0, 255, 255), width=1)
        # ------------------------------------
        
        origin_x = int(zoomed_x_throat + overlay_x_mm * zoomed_px_per_mm)
        origin_y = zoomed_y_centerline

        # Define 40mm vector line for rendering the calculated shock
        line_length_px = 18 * zoomed_px_per_mm 

        # 3. Dynamic Shock and Mach Calculations per Time-Step
        Pe_ambient = 1.01325
        area_ratio_eval = get_area_ratio(overlay_x_mm, profile=nozzle_profile)
        
        is_subsonic_regime = False
        if current_p2 <= Pe_ambient * 1.0001:
            is_subsonic_regime = True
            mach_at_eval = 0.0
        else:
            # Resolve the normal shock position and upstream Mach for this timestep.
            m1_ns, m2_ns, A_NS_ratio = solve_nozzle_shock(current_p2, Pe_ambient, 1.0, Ae_At_exit, gamma)

            # Draw the expected Normal Shock location if it is standing in the divergent section
            if 1.0001 < A_NS_ratio < (Ae_At_exit - 0.0001):
                def obj_x(x_val):
                    return get_area_ratio(x_val, aeat=Ae_At_exit, profile=nozzle_profile) - A_NS_ratio
                try:
                    # Root-finding to locate the corresponding axial distance (X mm) for the shock's area ratio
                    x_ns_mm = brentq(obj_x, 0.001, 50.0)
                    x_ns_px = int(zoomed_x_throat + x_ns_mm * zoomed_px_per_mm)
                    
                    # Convert Area Ratio back to physical vertical radius (y_mm)
                    y_ns_mm = A_NS_ratio * 3.0 
                    y_ns_px_offset = int(y_ns_mm * zoomed_px_per_mm)

                    # Draw yellow dot as throat centre
                    r = 4
                    draw.ellipse([(zoomed_x_throat-r, origin_y-r), (zoomed_x_throat+r, origin_y+r)], fill=(255, 0, 0))
                    
                    # Draw a vertical green line spanning the top and bottom nozzle boundary
                    draw.line([(x_ns_px, origin_y - y_ns_px_offset), (x_ns_px, origin_y + y_ns_px_offset)], fill=(0, 255, 0), width=2)
                except ValueError:
                    pass # Shock is out of bounds or cannot be mapped to the profile

            # Check if overlay location lies downstream of the standing normal shock
            if area_ratio_eval > A_NS_ratio and A_NS_ratio < (Ae_At_exit - 0.0001):
                is_subsonic_regime = True
                mach_at_eval = 0.0
            else:
                mach_at_eval = m1_ns

        # Draw structural oblique shock lines only if downstream state is supersonic
        if not is_subsonic_regime:
            shock_angle_deg, is_detached = solve_shock_angle(mach_at_eval, wedge_half_angle_deg)
            
            # Use theoretically calculated shock angle
            angle_rad = math.radians(shock_angle_deg)
            dx = line_length_px * math.cos(angle_rad)
            dy = line_length_px * math.sin(angle_rad)

            end_x = int(origin_x + dx)
            end_y1 = int(origin_y - dy)
            end_y2 = int(origin_y + dy)

            # Draw expected Shock Location (Dashed if Detached, Solid if Attached)
            if is_detached:
                draw_dashed_line(draw, (origin_x, origin_y), (end_x, end_y1), fill=(255, 255, 0), width=1)
                draw_dashed_line(draw, (origin_x, origin_y), (end_x, end_y2), fill=(255, 255, 0), width=1)
            else:
                pass

        raw_video_frame_final = np.array(img_video)

        # -----------------------------------------------------------------
        # STEP D: Combine Side-by-Side
        # -----------------------------------------------------------------
        combined_frame = np.hstack((raw_video_frame_final, plot_frame_final))
        
        h, w, c = combined_frame.shape
        if w % 2 != 0: 
            combined_frame = combined_frame[:, :w-1, :]
            
        combined_master_frames.append(combined_frame)
        output_frame_count += 1

    plt.close(fig)

    if combined_master_frames:
        print(f"Stacking {len(combined_master_frames)} master array frames...")
        output_block = np.stack(combined_master_frames, axis=0)
        
        print(f"Compiling output file: {output_path}...")
        iio.imwrite(output_path, output_block, fps=output_fps, plugin="FFMPEG")
        print("Success! Process complete.")
    else:
        print("Error: No frames could be synchronized. Verify your time vectors.")

# =====================================================================
# Execution Configurations & Calibration Inputs
# =====================================================================
if __name__ == "__main__":
    
    CSV_INPUT   = "data_20260602_151426.csv"
    VIDEO_INPUT = "video_20260602_151426.h264"
    MP4_OUTPUT  = "combined_animation_20260602_151426.mp4"
    
    # Timeline Sync Parameters
    CAMERA_NATIVE_FPS = 24    
    DESIRED_SPEEDUP   = 10    
    VIDEO_LAG_SECONDS = -2.0  
    OUTPUT_VIDEO_FPS  = 24    

    # =================================================================
    # USER MANUAL CALIBRATION VARIABLES
    # =================================================================
    SCALE_PX_PER_MM_BASE = 4.8    # 1x unzoomed scale factor (Pixels per mm)
    THROAT_X_PX          = 235    # Horizontal pixel index of throat
    CENTERLINE_Y_PX      = 228    # Vertical pixel index of nozzle centerline 247 for 180243
    
    OVERLAY_X_MM         = 18.0   # Distance downstream from throat (mm)
    WEDGE_HALF_ANGLE_DEG = 20.0   # Physical wedge angle forcing the oblique shock
    
    NOZZLE_PROFILE       = 'old'  # Options: 'old' or 'new'
    # =================================================================

    generate_synchronized_mp4(
        csv_path=CSV_INPUT,
        video_path=VIDEO_INPUT,
        output_path=MP4_OUTPUT,
        camera_fps=CAMERA_NATIVE_FPS,
        speedup=DESIRED_SPEEDUP,
        video_lag=VIDEO_LAG_SECONDS,
        output_fps=OUTPUT_VIDEO_FPS,
        px_per_mm=SCALE_PX_PER_MM_BASE,
        x_throat=THROAT_X_PX,
        y_centerline=CENTERLINE_Y_PX,
        overlay_x_mm=OVERLAY_X_MM,
        wedge_half_angle_deg=WEDGE_HALF_ANGLE_DEG,
        nozzle_profile=NOZZLE_PROFILE
    )