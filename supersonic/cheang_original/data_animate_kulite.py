import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline

# =====================================================================
# NOZZLE GEOMETRY DEFINITIONS
# =====================================================================
# Profile 1: GEN_M2p74 (Formerly "new")
X_PTS_M2p74 = np.array([0., 6.4, 12.8, 19.2, 25.6, 32., 38.4, 44.8, 51.2, 57.6, 64.])
Y_PTS_M2p74 = np.array([6., 6.502, 8.4451, 10.4906, 12.3916, 14.359, 16.1275, 17.4878, 18.7439, 19.5813, 20.])
nozzle_spline_M2p74 = CubicSpline(X_PTS_M2p74, Y_PTS_M2p74, bc_type=((1, 0.0), (1, 0.0)))

# Profile 2: GEN_M2p4
X_PTS_M2p4 = np.array([ 0., 6.4, 12.8, 19.2, 25.6, 32., 38.4, 44.8, 51.2, 57.6, 64. ])
Y_PTS_M2p4 = np.array([ 6., 6.3048, 7.4845, 8.7264, 9.8806, 11.0751, 12.1489, 12.9747, 13.7374, 14.2458, 14.5 ])
nozzle_spline_M2p4 = CubicSpline(X_PTS_M2p4, Y_PTS_M2p4, bc_type=((1, 0.0), (1, 0.0)))

def get_area_ratio(x, profile_type='GEN_M2p74', aeat_old=3.3, xmax_old=64):
    """Unified function to return Area Ratio (A/At) based on the chosen profile."""
    if x < 0:
        return 1 + (x**2)/4.5
    if profile_type == 'MLN':
        return np.tanh(2*x/xmax_old)/np.tanh(2)*(aeat_old-1) + 1
    elif profile_type == 'GEN_M2p74':
        return float(nozzle_spline_M2p74(x)) / Y_PTS_M2p74[0]
    elif profile_type == 'GEN_M2p4':
        return float(nozzle_spline_M2p4(x)) / Y_PTS_M2p4[0]

# =====================================================================
# 1. NOZZLE & SHOCK ISENTROPIC FLUID DYNAMICS MATH
# =====================================================================
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

def get_theoretical_pressure(x_eval, P0, Pe, profile_type='GEN_M2p74'):
    gamma = 1.4
    At_val = 1.0
    
    Ae_val = get_area_ratio(64.0 if profile_type in ['GEN_M2p74', 'GEN_M2p4'] else 64.0, profile_type)
    y_eval = np.array([get_area_ratio(x, profile_type) for x in x_eval])
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

def get_shock_location(P0, Pe, profile_type='GEN_M2p74'):
    """Calculates exactly where the theoretical normal shock is located (x, y_eval)."""
    gamma = 1.4
    if P0 <= Pe * 1.0001: return None, None
    
    aeat = get_area_ratio(64.0 if profile_type in ['GEN_M2p74', 'GEN_M2p4'] else 64, profile_type)
    limit = p_isen(solve_mach(aeat/1.0, False, gamma), gamma, -1)
    if (Pe/P0) > limit: return None, None
    
    try:
        m1, m2, A_NS_ratio = solve_nozzle_shock(P0, Pe, 1.0, aeat, gamma)
        if A_NS_ratio >= aeat - 1e-5: return None, None
        
        # Solve for x from the diverging section profile mathematically
        if profile_type == 'MLN':
            val = (A_NS_ratio - 1) / (aeat - 1) * math.tanh(2)
            if abs(val) >= 0.9999: return None, None
            x_shock = (32.0 / 2) * math.atanh(val)
        elif profile_type == 'GEN_M2p74':
            x_shock = brentq(lambda x: get_area_ratio(x, 'GEN_M2p74') - A_NS_ratio, 0.0, 64.0)
        elif profile_type == 'GEN_M2p4':
            x_shock = brentq(lambda x: get_area_ratio(x, 'GEN_M2p4') - A_NS_ratio, 0.0, 64.0)
            
        y_shock = A_NS_ratio
        return x_shock, y_shock
    except:
        return None, None

# =====================================================================
# 2. MASTER SYNCHRONIZATION AND RENDER ENGINE
# =====================================================================
def verify_profile(profile_type):
    """Plots the spline curve for user verification before video rendering."""
    print(f"Generating Nozzle Profile Verification Plot for {profile_type}...")
    x_dense = np.linspace(0, 64, 500)
    
    if profile_type == 'GEN_M2p74':
        x_pts, y_pts, spline = X_PTS_M2p74, Y_PTS_M2p74, nozzle_spline_M2p74
    elif profile_type == 'GEN_M2p4':
        x_pts, y_pts, spline = X_PTS_M2p4, Y_PTS_M2p4, nozzle_spline_M2p4
    elif profile_type == 'MLN':
        print("Analytical profile MLN selected. Skipping spline verification.")
        return
        
    y_dense = spline(x_dense)

    plt.figure(figsize=(10, 5))
    plt.plot(x_dense, y_dense, 'b-', linewidth=2, label=f'Clamped Cubic Spline ({profile_type})')
    plt.plot(x_pts, y_pts, 'ro', markersize=8, label='Input Data Points')
    plt.title(f"Nozzle Geometry Spline Verification: {profile_type}")
    plt.xlabel("X Coordinate (mm)")
    plt.ylabel("Y Coordinate / Height (mm)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Non-blocking show to allow script to prompt for input
    plt.show(block=False)
    
    val = input("\nDoes this profile look correct? Enter Y to proceed or N to terminate: ").strip().upper()
    plt.close()
    
    if val != 'Y':
        print("Terminating script to allow adjustments.")
        sys.exit(0)
    print("Verification successful. Moving to render sequence...")

def generate_synchronized_mp4(csv_path, video_path, output_path, camera_fps=24, speedup=20, video_lag=-2.0, output_fps=24,
                              px_per_mm=12.0, x_throat=280, y_bottom=300, throat_height_mm=6.0, profile_type='GEN_M2p74'):
    
    # Shift internal structural camera tracking left by 6.5mm mapped in pixels for GEN_M2p4
    if profile_type == 'GEN_M2p4':
        x_throat -= int((6.0) * px_per_mm)
        y_bottom -= int(0.5 * px_per_mm)
        kulite_x = [6, 12, 18, 24]
    else:
        kulite_x = [0, 6, 12, 18]

    # IMPORTANT: Force Matplotlib headless mode ONLY AFTER the user has verified the plot
    matplotlib.use('Agg')
    
    print("Loading data files into memory...")
    df = pd.read_csv(csv_path)
    df = df.sort_values(by='time').reset_index(drop=True)
    csv_max_time = df['time'].max()

    try:
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_medium = ImageFont.truetype("arial.ttf", 18)
        has_ttf = True
    except IOError:
        try:
            font_large = ImageFont.truetype("DejaVuSans.ttf", 24)
            font_medium = ImageFont.truetype("DejaVuSans.ttf", 18)
            has_ttf = True
        except IOError:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            has_ttf = False

    def draw_scaled_text(draw_obj, position, text, fill, font, scale=2):
        x, y = position
        if has_ttf:
            draw_obj.text((x, y), text, fill=fill, font=font)
        else:
            for dx in range(scale):
                for dy in range(scale):
                    draw_obj.text((x + dx, y + dy), text, fill=fill, font=font)

    xmax = 64.0 if profile_type in ['GEN_M2p74', 'GEN_M2p4'] else 63.8
    xc = np.linspace(-3.3, 0, 30)
    xd = np.linspace(0.1, xmax, 100)
    x_eval = np.concatenate([xc, xd])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    ax.set_xlim(-3, 20 if profile_type != 'GEN_M2p4' else 26)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('Distance from Throat (mm)', fontsize=11)
    ax.set_ylabel('Pressure Ratio (p/p0)', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    line_th, = ax.plot([], [], label='Theoretical Profile (1D ISEN + MOC)', color='green', linewidth=2)
    scatter_meas = ax.scatter([], [], color='crimson', s=60, zorder=5, label='Kulite Measurements (pax1-4)')
    title_text = ax.set_title('', fontsize=12)
    ax.legend(loc='upper right')
    plt.tight_layout()

    dt_per_frame = speedup / output_fps 
    combined_master_frames = []
    output_frame_count = 0
    BAR_SCALE = 140 

    print(f"Streaming H.264 and synthesizing matching timelines at {speedup}x speedup...")
    
    for video_frame_idx, raw_video_frame in enumerate(iio.imiter(video_path, plugin="pyav")):
        
        source_video_time = video_frame_idx / camera_fps
        t_real_corresponding = source_video_time - video_lag
        target_t_real = output_frame_count * dt_per_frame
        
        if target_t_real > csv_max_time:
            print("Reached the end of telemetry data timeframe.")
            break

        if t_real_corresponding < target_t_real: continue
            
        while target_t_real < t_real_corresponding and target_t_real <= csv_max_time:
            if t_real_corresponding - target_t_real > dt_per_frame:
                target_t_real = output_frame_count * dt_per_frame
                output_frame_count += 1
                continue
            break

        idx_csv = np.searchsorted(df['time'], target_t_real)
        idx_csv = min(idx_csv, len(df) - 1)
        
        P0 = df['p2'].iloc[idx_csv]
        Pe = 1.01600
        pax_vals = [df['pax1'].iloc[idx_csv], df['pax2'].iloc[idx_csv], 
                    df['pax3'].iloc[idx_csv], df['pax4'].iloc[idx_csv]]
        
        P_th = get_theoretical_pressure(x_eval, P0, Pe, profile_type=profile_type)
        P_th_ratio = P_th / P0
        
        pax_ratios = [p / P0 for p in pax_vals]
        P0_pb_ratio = P0 / Pe
        
        line_th.set_data(x_eval, P_th_ratio)
        scatter_meas.set_offsets(np.c_[kulite_x, pax_ratios])
        title_text.set_text(f'Telemetry Time: {target_t_real:.2f}s | P0/pb: {P0_pb_ratio:.2f}')
        
        fig.canvas.draw()
        plot_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        plot_rgb = plot_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4, ))[..., :3]

        target_height = raw_video_frame.shape[0]
        target_width = raw_video_frame.shape[1]

        img_plot = Image.fromarray(plot_rgb)
        if img_plot.height != target_height:
            aspect = img_plot.width / img_plot.height
            img_plot = img_plot.resize((int(target_height * aspect), target_height), Image.Resampling.LANCZOS)
        plot_frame_final = np.array(img_plot)

        if raw_video_frame.ndim == 2:
            raw_video_frame = np.stack((raw_video_frame,)*3, axis=-1)
        elif raw_video_frame.shape[-1] == 4:
            raw_video_frame = raw_video_frame[..., :3]
            
        raw_video_frame = np.flipud(raw_video_frame)
        y_bottom_flipped = target_height - y_bottom
            
        img_video = Image.fromarray(raw_video_frame)
        
        crop_w = target_width // 2
        crop_h = target_height // 2
        base_left = x_throat - crop_w // 2
        base_top = y_bottom_flipped - crop_h // 2
        
        px_per_mm_base = px_per_mm / 2.0 
        y_offset = 5
        shift_x = int(9.0 * px_per_mm_base)
        shift_y = int((6.0 - y_offset) * px_per_mm_base) + int(2.0 * px_per_mm_base)
        
        # Move crop box 10mm right in unscaled pixels (5mm in base space) for GEN_M2p4
        if profile_type == 'GEN_M2p4':
            shift_x += int(12 * px_per_mm_base)
        
        left = max(0, min(base_left + shift_x, target_width - crop_w))
        top = max(0, min(base_top + shift_y, target_height - crop_h))
        right = left + crop_w
        bottom = top + crop_h
        
        img_video = img_video.crop((left, top, right, bottom))
        img_video = img_video.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        zoomed_x_throat = int((x_throat - (base_left + int(9.0 * px_per_mm_base))) * 2.0)
        
        # Move plot structure anchor up by exactly 2mm in video scale (- 2.0 * px_per_mm)
        zoomed_y_bottom = int((y_bottom_flipped - (base_top + int((6.0 - 0*y_offset) * px_per_mm_base))) * 2.0) - int(2.0 * px_per_mm) - int(2.0 * px_per_mm)
        
        # Apply the layout shifts for GEN_M2p4: +6.5mm baseline layout adjustment, then -10mm global left translation
        if profile_type == 'GEN_M2p4':
            zoomed_x_throat += int((5.0 - 10.0) * px_per_mm)
            zoomed_y_bottom += int((1.0) * px_per_mm)

        draw = ImageDraw.Draw(img_video)
        
        bg_pad_left = int(25 * px_per_mm)
        bg_pad_right = int(25 * px_per_mm)
        bg_pad_top = int(1.2 * BAR_SCALE)
        bg_pad_bottom = 60  
        
        draw.rectangle(
            [(zoomed_x_throat - bg_pad_left, zoomed_y_bottom - bg_pad_top),
             (zoomed_x_throat + bg_pad_right, zoomed_y_bottom + bg_pad_bottom)],
            fill=(0, 0, 0)
        )
        
        # Position axis 6.5mm to the left of the profile origin instead of 10mm to be closer
        scale_x = int(zoomed_x_throat - 6.5 * px_per_mm)
        if scale_x > 10:
            max_scale_p = 1.05  
            draw.line([(scale_x, zoomed_y_bottom), (scale_x, int(zoomed_y_bottom - max_scale_p * BAR_SCALE))], fill=(200, 200, 200), width=1)
            draw_scaled_text(draw, (scale_x - 55, zoomed_y_bottom + 5), "p/p0", fill=(200, 200, 200), font=font_large)
            
            for tick_p in np.arange(0.0, 1.05, 0.1):
                tick_y = int(zoomed_y_bottom - tick_p * BAR_SCALE)
                is_major = math.isclose(tick_p % 0.2, 0.0, abs_tol=1e-5) or math.isclose(tick_p, 1.0, abs_tol=1e-5)
                tick_len = 6 if is_major else 3
                draw.line([(scale_x - tick_len, tick_y), (scale_x, tick_y)], fill=(200, 200, 200), width=1)
                
                max_grid_x = int(zoomed_x_throat + 24 * px_per_mm + 10) if profile_type == 'GEN_M2p4' else int(zoomed_x_throat + 18 * px_per_mm + 10)
                grid_color = (255, 255, 255) if is_major else (120, 120, 120)
                for gx in range(scale_x, max_grid_x, 6):
                    draw.line([(gx, tick_y), (gx + 3, tick_y)], fill=grid_color, width=1)
                
                if is_major:
                    draw_scaled_text(draw, (scale_x - 45, tick_y - 10), f"{tick_p:.1f}", fill=(200, 200, 200), font=font_medium)
        
        for loc_mm, p_ratio in zip(kulite_x, pax_ratios):
            x_pos = int(zoomed_x_throat + loc_mm * px_per_mm)
            bar_len = max(1, int(p_ratio * BAR_SCALE)) 
            draw.line([(x_pos, zoomed_y_bottom), (x_pos, zoomed_y_bottom - bar_len)], fill=(255, 50, 50), width=3)
            
            r = 4
            draw.ellipse([(x_pos-r, zoomed_y_bottom-r), (x_pos+r, zoomed_y_bottom+r)], fill=(255, 0, 0))
            draw.line([(x_pos, zoomed_y_bottom), (x_pos, zoomed_y_bottom + 5)], fill=(200, 200, 200), width=1)
            draw_scaled_text(draw, (x_pos - 8, zoomed_y_bottom + 8), f"{loc_mm}", fill=(230, 230, 230), font=font_large)

        draw_scaled_text(draw, (int(zoomed_x_throat - 3 * px_per_mm), zoomed_y_bottom + 34), "Axial Distance (mm)", fill=(230, 230, 230), font=font_large)

        th_pts = []
        nozzle_pts = []
        y_eval_full = np.array([get_area_ratio(x, profile_type) for x in x_eval])
        
        for x_val, p_rat, y_val in zip(x_eval, P_th_ratio, y_eval_full):
            px = int(zoomed_x_throat + x_val * px_per_mm)
            py_th = int(zoomed_y_bottom - p_rat * BAR_SCALE)
            py_noz = int(zoomed_y_bottom + y_val * throat_height_mm * px_per_mm)
            th_pts.append((px, py_th))
            nozzle_pts.append((px, py_noz))
            
        for i in range(len(th_pts)-1):
            draw.line([th_pts[i], th_pts[i+1]], fill=(0, 255, 0), width=2)

        # --- Draw Nozzle Profile Overlay (Dotted, Width = 1) shifted DOWN 2mm (+ 2.0 * px_per_mm) ---
        profile_points_bottom = []
        for x_mm in np.linspace(0, xmax, 200):
            y_mm = get_area_ratio(x_mm, profile_type) * throat_height_mm
            px_x = int(zoomed_x_throat + x_mm * px_per_mm)
            px_y_bottom = int(zoomed_y_bottom + y_mm * px_per_mm + px_per_mm * y_offset) + int(2.0 * px_per_mm)
            profile_points_bottom.append((px_x, px_y_bottom))
            
        dash_interval = 4
        for i in range(len(profile_points_bottom) - 1):
            if (i // dash_interval) % 2 == 0:
                draw.line([profile_points_bottom[i], profile_points_bottom[i+1]], fill=(0, 255, 255), width=1)
        # ------------------------------------

        # --- Standing Normal Shock Line overlay shifted DOWN 2mm (+ 2.0 * px_per_mm) ---
        x_shock, y_shock = get_shock_location(P0, Pe, profile_type)
        if x_shock is not None:
            shock_px = int(zoomed_x_throat + x_shock * px_per_mm)
            # The flat top profile is located at y_mm = 0 relative to the nozzle channel base
            shock_py_top_shifted = int(zoomed_y_bottom + px_per_mm * y_offset) + int(2.0 * px_per_mm)
            # Compute real height mapping to get the lower curved profile coordinate boundary at x_shock
            y_shock_mm = get_area_ratio(x_shock, profile_type) * throat_height_mm
            shock_py_bottom_shifted = int(zoomed_y_bottom + y_shock_mm * px_per_mm + px_per_mm * y_offset) + int(2.0 * px_per_mm)
            
            # Render normal shock spanning perfectly between top flat profile wall and bottom curved profile wall
            draw.line([(shock_px, shock_py_top_shifted), (shock_px, shock_py_bottom_shifted)], fill=(0, 255, 0), width=3)

        video_frame_final = np.array(img_video)

        combined_frame = np.hstack((video_frame_final, plot_frame_final))
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
    
    # TOGGLE BETWEEN PROFILES HERE ('GEN_M2p74', 'MLN', or 'GEN_M2p4')
    NOZZLE_PROFILE_TYPE = 'GEN_M2p4'

    # Run the verification step FIRST
    #verify_profile(NOZZLE_PROFILE_TYPE)
    
    CSV_INPUT   = "data_20260602_171128.csv"
    VIDEO_INPUT = "video_20260602_171128.h264"
    MP4_OUTPUT  = "combined_animation_20260602_171128.mp4"
    
    CAMERA_NATIVE_FPS = 24    
    DESIRED_SPEEDUP   = 10    
    VIDEO_LAG_SECONDS = -2.0  
    OUTPUT_VIDEO_FPS  = 24    

    SCALE_PX_PER_MM = 10      
    THROAT_X_PX     = 285     
    BOTTOM_Y_PX     = 300     
    THROAT_HEIGHT_MM = 6.0    

    generate_synchronized_mp4(
        csv_path=CSV_INPUT,
        video_path=VIDEO_INPUT,
        output_path=MP4_OUTPUT,
        camera_fps=CAMERA_NATIVE_FPS,
        speedup=DESIRED_SPEEDUP,
        video_lag=VIDEO_LAG_SECONDS,
        output_fps=OUTPUT_VIDEO_FPS,
        px_per_mm=SCALE_PX_PER_MM,
        x_throat=THROAT_X_PX,
        y_bottom=BOTTOM_Y_PX,
        throat_height_mm=THROAT_HEIGHT_MM,
        profile_type=NOZZLE_PROFILE_TYPE
    )