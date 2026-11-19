import sys
import math
import time
import threading
from collections import deque

# Sensor & Hardware Imports
# Each import is optional: anything missing falls back to sim_hardware.py.
# Run with --sim to force every subsystem to simulation (even on the Pi).
import sim_hardware
import nozzle_theory

SIM_ALL = "--sim" in sys.argv
if SIM_ALL:
    sys.argv.remove("--sim")

HAVE_ADC = HAVE_I2C = HAVE_CAM = False
if not SIM_ALL:
    try:
        import Adafruit_ADS1x15
        HAVE_ADC = True
    except Exception as e:
        print(f"[hw] ADS1115 library unavailable: {e}")
    try:
        import adafruit_mcp9600
        import board
        import busio
        HAVE_I2C = True
    except Exception as e:
        print(f"[hw] I2C/MCP9600 libraries unavailable: {e}")
    try:
        from picamera import PiCamera
        from picamera.array import PiRGBArray
        HAVE_CAM = True
    except Exception as e:
        print(f"[hw] picamera unavailable: {e}")

# Math & UI Imports
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui

# TCP for labVIEW connection
import socket, struct

# CSV Export & Dynamic Filename Imports
import atexit
import csv
from datetime import datetime
import json
import os

# Calibration of Kulite sensors (bar, added to each Pax reading). These are defaults:
# edit live in the sidebar "Kulite calibration" box; changes are saved to SETTINGS_FILE.
PAX_CALIBRATION = [0.23, 0.23, 0.23, 0.23]

# -----------------------
# Video & Overlay Settings (edit here)
# -----------------------
CAMERA_RESOLUTION = (640, 480)   # capture size from the Pi camera
CAMERA_FRAMERATE = 24
# Fixed crop in camera pixels (x, y, w, h), applied before display. None = full frame.
# Usually leave None and zoom/pan live instead (scroll or pinch to zoom, drag the video to pan;
# the view is saved). (0, 120, 640, 240) reproduces the old fixed view: the middle 240 rows.
VIDEO_CROP: "tuple[int, int, int, int] | None" = None
VIDEO_MAX_ZOOM = 8.0
VIDEO_FLIP_V = True              # show the camera image upside down: flat tap wall on top, pressure plots above it
CAMERA_CONTOUR_ABOVE = True      # in the raw camera image, the contoured wall is above the flat (tap) wall
# None = resizable live by dragging the splitter bar above the video.
# (w, h) = fixed size in screen pixels, e.g. (960, 360).
VIDEO_FIXED_SIZE = None
VIDEO_INITIAL_HEIGHT = 300       # starting height when resizable
VIDEO_MAX_SCALE = None           # e.g. 2.0 caps upscaling; None = no cap

PAX_OVERLAY = True
# Pressure bars drawn over the video, one per tap. Taps are equally spaced horizontally.
# Positions are fractions of the cropped frame: (0,0) top-left, (1,1) bottom-right.
# Tune live with the "Pax overlay" controls in the sidebar (saved automatically to SETTINGS_FILE).
PAX1_POS = (0.20, 0.50)          # Pax1 tap location (bar baseline = zero pressure)
PAX_SPACING = 0.20               # gap between neighbouring taps, fraction of frame width
PAX_BAR_SCALE = 0.10             # bar length per 1 bar of pressure, fraction of frame height
PAX_BAR_WIDTH = 3                # px
PAX_SHOW_VALUES = True           # small number at the end of each bar
PAX_DOT_R = 3.5                  # px, solid dot at the end of each bar
PAX_RING_R = 7                   # px, thin ring around the dot
PAX_STEP = 0.005                 # change per click / arrow key (Shift = x5)
OVERLAY_FONT_PT = 9
OVERLAY_COLOR = "#00E5FF"
# Values tuned in the sidebar (overlay position/spacing/scale, Kulite calibration) are saved here
# and reloaded on start, overriding the defaults in this file. Delete the file to go back to defaults.
# The simulated camera gets its own file so tuning in --sim never disturbs the real alignment.
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "prc_settings.json" if HAVE_CAM else "prc_settings_sim.json")

# Nozzle outline over the video, positioned from the Pax alignment (Pax1 tap + tap spacing set the
# position and mm scale), so it doubles as an alignment check. The theoretical shock is marked with
# small triangles just outside the walls, leaving the flow itself unobstructed.
NOZZLE_OUTLINE = True
CONTOUR_BELOW_ON_SCREEN = (CAMERA_CONTOUR_ABOVE == VIDEO_FLIP_V)   # derived; don't edit
NOZZLE_OUTLINE_COLOR = "#FFFFFF"
SHOCK_MARKER_COLOR = "#FF1744"

# Theory reference: yellow curve of wall pressure along the nozzle for fully supersonic isentropic
# flow at the measured P0. Taps agree while the flow is shock-free there, then depart one by one
# as the shock moves upstream past them.
PAX_SHOW_ISENTROPIC = True
PAX_ABSOLUTE = True              # True: Pax values are absolute bar (as the original analysis assumed); False: gauge
ISEN_COLOR = "#FFEA00"

# Nozzle geometry (shared by the shock solver, nozzle plot and theory reference).
# Profiles and tap positions are in nozzle_theory.py. GEN_M2p74: A/A* = 3.33, Pax1 at the throat, taps 6 mm apart.
NOZZLE_PROFILE = "GEN_M2p74"
NOZZLE_WIDTH_M = 0.008           # constant channel width (design spreadsheet)
# Which plenum transducer is the nozzle stagnation pressure P0: "p1" = upstream reservoir.
# (The original code used "p2" here; P2 is the downstream plenum.)
P0_CHANNEL = "p1"
P_ATM_PA = 101625.0

# --- Load values saved from the sidebar on a previous run ---
def load_saved_settings():
    try:
        with open(SETTINGS_FILE) as f:
            cfg = json.load(f)
        print(f"Loaded saved settings from {SETTINGS_FILE}")
        return cfg
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Ignoring unreadable settings file ({e})")
        return {}

SAVED = load_saved_settings()
try:
    if "pax_calibration" in SAVED:
        PAX_CALIBRATION = [float(v) for v in SAVED["pax_calibration"]][:4]
except Exception as e:
    print(f"Ignoring saved calibration ({e})")

def save_settings():
    """Write all sidebar-tuned values. Called whenever one changes, and on quit."""
    cfg = {"pax_calibration": [round(v, 4) for v in PAX_CALIBRATION]}
    if "video_widget" in globals():
        cfg.update(video_widget.settings_dict())
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"Could not save settings: {e}")

# -----------------------
# Sensors, Camera and TCP Setup
# -----------------------
simulated = []  # names of subsystems running on fake data (shown in GUI banner)

def init_device(name, make_real, make_fake, available=True):
    """Try the real device; on any failure, record it as simulated and return the fake."""
    if available:
        try:
            return make_real()
        except Exception as e:
            print(f"[hw] {name} init failed: {e}")
    simulated.append(name)
    return make_fake()

adc = init_device("ADC", lambda: Adafruit_ADS1x15.ADS1115(address=0x48, busnum=1),
                  sim_hardware.FakeADC, HAVE_ADC)
GAIN = 2/3

i2c = None
if HAVE_I2C:
    try:
        i2c = busio.I2C(board.SCL, board.SDA, frequency=50000)
    except Exception as e:
        print(f"[hw] I2C bus init failed: {e}")
thermocouple1 = init_device("TC1", lambda: adafruit_mcp9600.MCP9600(i2c, address=0x67),
                            lambda: sim_hardware.FakeThermocouple(20.0), i2c is not None)
thermocouple2 = init_device("TC2", lambda: adafruit_mcp9600.MCP9600(i2c, address=0x60),
                            lambda: sim_hardware.FakeThermocouple(19.5), i2c is not None)

if not HAVE_CAM:
    simulated.append("CAM")

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(5)  # fail after 5 seconds instead of hanging indefinitely

tcp = 0 # no need to change this
recording = 0  # 2 = data & video, 1 = data only, 0 = nothing

# Track high-precision unrounded variables to prevent dM/dt division-by-zero errors
last_raw_time = None
last_raw_mass = None


if not SIM_ALL:
    print("Starting TCP conection")
    try:
        s.connect(('10.32.240.206', 2021)) #10.231.241.117
        print("TCP connection established")
        tcp = 1
    except socket.timeout:
        print("ERROR: Connection timed out — is LabVIEW listening on port 2021?")
    except ConnectionRefusedError:
        print("ERROR: Connection refused — port 2021 is not open on the host")
    except OSError as e:
        print(f"ERROR: TCP connection failed ({e}) — not on the lab network?")
if tcp == 0:
    simulated.append("TCP")

if simulated:
    print(f"*** SIMULATED: {', '.join(simulated)} ***")

# -----------------------
# Config & Data Buffers
# -----------------------
window_duration = 20  
sample_interval = 0.02
buffer_size = int(window_duration / sample_interval) + 1000 

# Data dictionary contains bounded deques optimized for fast UI updates
data_lock = threading.Lock()  # guards data & csv_history: one sample is appended across all keys atomically
data = {
    'time': deque(maxlen=buffer_size),
    'p1': deque(maxlen=buffer_size), 'p2': deque(maxlen=buffer_size),
    't1': deque(maxlen=buffer_size), 't2': deque(maxlen=buffer_size),
    'm': deque(maxlen=buffer_size), 'mdot': deque(maxlen=buffer_size),
    'M1': deque(maxlen=buffer_size), 'M2': deque(maxlen=buffer_size),
    'A_NS': deque(maxlen=buffer_size),
    'pax1': deque(maxlen=buffer_size), 'pax2': deque(maxlen=buffer_size),
    'pax3': deque(maxlen=buffer_size), 'pax4': deque(maxlen=buffer_size),
}

# Unbounded storage repository to safely capture full run history for CSV export
csv_history = {
    'time': [], 'p1': [], 'p2': [], 't1': [], 't2': [], 'm': [], 'mdot': [],
    'M1': [], 'M2': [], 'A_NS': [], 'pax1': [], 'pax2': [], 'pax3': [], 'pax4': []
}

start_time = time.time()
# Generate localized date-time strings for parallel CSV and video files
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
sim_tag = "_SIM" if simulated else ""  # so simulated runs can't be mistaken for real data
csv_filename = f"data_{timestamp_str}{sim_tag}.csv"
video_filename = f"video_{timestamp_str}{sim_tag}.h264"

# --- Export Routine on Close ---
def save_data_to_csv():
    if recording < 1:
        print("Recording mode disabled (0) — skipping CSV telemetry export.")
        return
        
    print(f"Saving complete telemetry history to {csv_filename}...")
    try:
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = list(csv_history.keys())
            writer.writerow(headers)
            
            # Unpack the complete historical lists matching lengths
            rows = zip(*[csv_history[key] for key in headers])
            writer.writerows(rows)
        print(f"Data successfully saved! Total samples written: {len(csv_history['time'])}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

atexit.register(save_data_to_csv)

# -----------------------
# Filters & Physics Logic
# -----------------------
class BiquadLPF:
    def __init__(self, fs, fc):
        w0 = 2 * math.pi * fc / fs
        alpha = math.sin(w0) / 1.414
        cos_w0 = math.cos(w0)
        a0 = 1 + alpha
        self.b0, self.b1, self.b2 = ((1-cos_w0)/2)/a0, (1-cos_w0)/a0, ((1-cos_w0)/2)/a0
        self.a1, self.a2 = (-2*cos_w0)/a0, (1-alpha)/a0
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0
        self.initialized = False

    def seed(self, val):
        self.x1 = self.x2 = self.y1 = self.y2 = val
        self.initialized = True

    def update(self, x):
        if not self.initialized: self.seed(x)
        y = (self.b0*x + self.b1*self.x1 + self.b2*self.x2 - self.a1*self.y1 - self.a2*self.y2)
        self.x2, self.x1, self.y2, self.y1 = self.x1, x, self.y1, y
        return y

class AdaptiveFilter:
    def __init__(self, alpha_min=0.02, alpha_max=0.5, sensitivity=0.1):
        self.prev_y = 0.0
        self.alpha_min, self.alpha_max, self.sensitivity = alpha_min, alpha_max, sensitivity
        self.initialized = False

    def update(self, x):
        if not self.initialized:
            self.prev_y = x
            self.initialized = True
            return x
        diff = abs(x - self.prev_y)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (1 - math.exp(-self.sensitivity * diff))
        y = alpha * x + (1 - alpha) * self.prev_y
        self.prev_y = y
        return y

filtpt1, filtpt2 = BiquadLPF(30, 1.5), BiquadLPF(30, 1.5)
filttc1, filttc2 = BiquadLPF(30, 1.0), BiquadLPF(30, 1.0)
filtmdot = BiquadLPF(30, 1.0)
filtmdot_adaptive = AdaptiveFilter(alpha_min=0.01, alpha_max=0.04, sensitivity=0.5)

NOZZLE = nozzle_theory.Nozzle(NOZZLE_PROFILE)
NOZZLE_AT = NOZZLE.h_throat / 1000 * NOZZLE_WIDTH_M   # m^2 (48 mm^2)
NOZZLE_AE = NOZZLE_AT * NOZZLE.ae_at                   # m^2 (160 mm^2 for GEN_M2p74)
solve_nozzle_shock = nozzle_theory.solve_nozzle_shock

def tap_theory_bar(p0_abs_bar):
    """Theoretical static pressure at each Pax tap, in the same units as the Pax readings."""
    p_atm = P_ATM_PA / 1e5
    p = nozzle_theory.wall_pressure(NOZZLE, NOZZLE.taps_mm, p0_abs_bar, p_atm)
    return p if PAX_ABSOLUTE else [v - p_atm for v in p]

# In --sim, fake Pax readings follow the same theory (minus the calibration that update() adds back),
# so the bars should sit on the theory ticks.
sim_hardware.PAX_MODEL = lambda p0_abs_bar: [v - c for v, c in zip(tap_theory_bar(p0_abs_bar), PAX_CALIBRATION)]
# Fake camera draws this nozzle with the shock at its theoretical position
sim_hardware.NOZZLE = NOZZLE
sim_hardware.SHOCK_X = lambda p0_abs_bar: nozzle_theory.shock_x(NOZZLE, p0_abs_bar, P_ATM_PA / 1e5)
if not HAVE_CAM:  # default overlay lines up with the fake camera's nozzle out of the box
    _cx, _cy, _cw, _ch = VIDEO_CROP or (0, 0, *CAMERA_RESOLUTION)
    _tx, _ty = sim_hardware.CAM_THROAT_PX
    _pitch_px = (NOZZLE.taps_mm[1] - NOZZLE.taps_mm[0]) * sim_hardware.CAM_PX_PER_MM
    _fy = (_ty - _cy) / _ch
    PAX1_POS = ((_tx + NOZZLE.taps_mm[0] * sim_hardware.CAM_PX_PER_MM - _cx) / _cw,
                1 - _fy if VIDEO_FLIP_V else _fy)
    PAX_SPACING = _pitch_px / _cw

# A failed sensor read holds the last good value instead of killing the data thread.
_last_good = {}
_read_failing = set()
def safe_read(name, fn, default):
    try:
        v = fn()
        _last_good[name] = v
        if name in _read_failing:
            print(f"[hw] {name} reads recovered")
            _read_failing.discard(name)
        return v
    except Exception as e:
        if name not in _read_failing:
            print(f"[hw] {name} read failed ({e}); holding last value")
            _read_failing.add(name)
        return _last_good.get(name, default)

def update_data():
    global last_raw_time, last_raw_mass
    Pe_atm, V, At, Ae = P_ATM_PA, 0.3, NOZZLE_AT, NOZZLE_AE
    while True:
        raw_now = time.time() - start_time  

        # --- TCP Reading & Robust Parsing ---
        if tcp == 1:
            try:
                raw_bytes = s.recv(1024)
                str_data = raw_bytes.decode('utf-8').strip().replace('\n', ',').split(',')
                floats = []
                for x in reversed(str_data):
                    try:
                        floats.append(float(x))
                        if len(floats) == 4: break
                    except ValueError:
                        continue
                if len(floats) == 4:
                    tcp_pressures = list(reversed(floats))
                else:
                    tcp_pressures = [data['pax1'][-1], data['pax2'][-1], data['pax3'][-1], data['pax4'][-1]] if data['time'] else [0.0, 0.0, 0.0, 0.0]
            except Exception:
                tcp_pressures = [data['pax1'][-1], data['pax2'][-1], data['pax3'][-1], data['pax4'][-1]] if data['time'] else [0.0, 0.0, 0.0, 0.0]
        else:
            tcp_pressures = sim_hardware.fake_pax()

        # --- RPI based DaQ ---
        raw_p1 = safe_read("P1", lambda: (adc.read_adc(0, gain=GAIN)/32767*6.144*50-25)*0.0689*0.75, 0.0)
        raw_p2 = safe_read("P2", lambda: (adc.read_adc(1, gain=GAIN)/32767*6.144*50-25)*0.0689*0.75 + 0.05, 0.0)

        p1, p2 = filtpt1.update(raw_p1), filtpt2.update(raw_p2)
        t1 = filttc1.update(safe_read("TC1", lambda: thermocouple1.temperature, 20.0))
        t2 = filttc2.update(safe_read("TC2", lambda: thermocouple2.temperature, 20.0))
        
        p01, p02 = (p1 * 100000) + Pe_atm, (p2 * 100000) + Pe_atm
        m_curr = 1000 * V * (p01 / (287 * (t1 + 273.15)))

        # --- LabVIEW based DaQ --

        pax1, pax2, pax3, pax4 = tcp_pressures[0], tcp_pressures[1], tcp_pressures[2], tcp_pressures[3]
        
        mdot_val = 0
        if last_raw_time is not None:
            dt = raw_now - last_raw_time
            if dt > 0 and raw_now > 2:
                mdot_val = filtmdot.update(filtmdot_adaptive.update((m_curr - last_raw_mass)/dt))
        
        P0 = p01 if P0_CHANNEL == "p1" else p02
        m1, m2, A_NS = solve_nozzle_shock(P0, Pe_atm, At, Ae)
        
        last_raw_time = raw_now
        last_raw_mass = m_curr
        
        # Raw value collection array (Rounding removed completely - MODIFIED)
        raw_values = [raw_now, p01/100000, p02/100000, t1, t2, m_curr, mdot_val, m1, m2, A_NS, pax1, pax2, pax3, pax4]
        
        # Stream raw full-precision metrics directly into both deques and history logs (MODIFIED)
        with data_lock:
            for k, v in zip(data.keys(), raw_values):
                data[k].append(v)
                csv_history[k].append(v)
            
        time.sleep(0.01)

threading.Thread(target=update_data, daemon=True).start()

# -----------------------
# Camera Thread (HORIZONTAL CROP APPLIED)
# -----------------------
class VideoThread(QtCore.QThread):
    change_pixmap_signal = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self._run_flag = True

    def _emit(self, img, crop):
        h, w, ch = img.shape
        # Copy only the cropped region (less data handed to the GUI thread)
        qimg = QtGui.QImage(img.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        qimg = qimg.copy(crop) if crop else qimg.copy()
        if VIDEO_FLIP_V:
            qimg = qimg.mirrored(False, True)
        self.change_pixmap_signal.emit(qimg)

    def _run_fake(self):
        cam = sim_hardware.FakeCamera(*CAMERA_RESOLUTION)
        crop = QtCore.QRect(*VIDEO_CROP) if VIDEO_CROP else None
        while self._run_flag:
            self._emit(cam.frame(), crop)
            self.msleep(int(1000 / CAMERA_FRAMERATE))

    def run(self):
        if not HAVE_CAM:
            self._run_fake()
            return
        try:
            with PiCamera() as camera:
                camera.resolution = CAMERA_RESOLUTION
                camera.framerate = CAMERA_FRAMERATE
                camera.exposure_compensation = -20

                if recording == 2:
                    camera.start_recording(self.filename)

                raw_capture = PiRGBArray(camera, size=CAMERA_RESOLUTION)
                crop = QtCore.QRect(*VIDEO_CROP) if VIDEO_CROP else None
                time.sleep(0.5)

                for frame in camera.capture_continuous(raw_capture, format="rgb", use_video_port=True):
                    if not self._run_flag:
                        break

                    self._emit(frame.array, crop)
                    raw_capture.truncate(0)
                
                if recording == 2:
                    camera.stop_recording()
        except Exception as e:
            print(f"Camera Thread Error: {e}")

    def stop(self):
        self._run_flag = False
        self.wait()

# -----------------------
# Video display widget: scales frame to fit, draws pax overlay
# -----------------------
class VideoWidget(QtWidgets.QWidget):
    overlay_changed = QtCore.pyqtSignal()  # position/spacing/scale changed (refreshes sidebar readout)

    def __init__(self):
        super().__init__()
        self._img = None
        self._pax = None
        self._p0 = None        # stagnation pressure (abs bar) for the theory curve
        self._shock_x = None   # theoretical shock position, mm from throat (None = not in nozzle)
        self._noz_x = np.linspace(-3.0, NOZZLE.x_exit, 80)          # outline sample points (mm)
        self._noz_h = [NOZZLE.height(x) for x in self._noz_x]
        # Fully supersonic isentropic p/p0 along the nozzle: geometry is fixed, so computed once
        self._isen_x = np.linspace(0.0, NOZZLE.x_exit, 120)
        self._isen_ratio = [nozzle_theory.p_isen(nozzle_theory.solve_mach(NOZZLE.area_ratio(x), True))
                            for x in self._isen_x]
        self._target = None    # where the frame was last drawn (for mouse hit-testing)
        self._drag = None      # (mode, start mouse pos, start value(s)) while dragging
        self._font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)  # monospace on any OS
        self._font.setPointSize(OVERLAY_FONT_PT)
        self._font.setBold(True)
        self._color = QtGui.QColor(OVERLAY_COLOR)
        self._shadow = QtGui.QColor(0, 0, 0, 200)  # dark outline keeps lines visible on any background
        # Live-tunable overlay geometry (sidebar controls change these)
        self.pax1 = list(PAX1_POS)
        self.spacing = PAX_SPACING
        self.bar_scale = PAX_BAR_SCALE
        # View: zoom factor (1 = whole frame fits) and the image point (fractions) at the widget centre
        self.zoom = 1.0
        self.center = [0.5, 0.5]
        self._load_settings()
        self._save_timer = QtCore.QTimer(self)  # zoom/pan saves once things settle, not on every event
        self._save_timer.setSingleShot(True); self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(save_settings)
        self.setMouseTracking(True)  # hover cursor over draggable bars
        self.setMinimumSize(160, 60)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)  # skip background erase

    def set_frame(self, img):
        self._img = img
        self.update()  # repaint once per camera frame

    def set_pax(self, vals, p0_abs=None, shock_x=None):
        self._pax, self._p0, self._shock_x = vals, p0_abs, shock_x  # drawn on the next camera frame

    # --- View: zoom & pan ---
    def _view_rect(self):
        """Screen rectangle the whole (cropped) frame maps to at the current zoom/pan."""
        iw, ih = self._img.width(), self._img.height()
        W, H = self.width(), self.height()
        s = min(W / iw, H / ih)
        if VIDEO_MAX_SCALE: s = min(s, VIDEO_MAX_SCALE)
        tw, th = iw * s * self.zoom, ih * s * self.zoom
        # Centre the frame along an axis where it's smaller than the widget; otherwise don't pan past its edge
        for i, (size, view) in enumerate(((tw, W), (th, H))):
            half = view / (2 * size)
            self.center[i] = 0.5 if half >= 0.5 else min(max(self.center[i], half), 1 - half)
        return QtCore.QRectF(W / 2 - self.center[0] * tw, H / 2 - self.center[1] * th, tw, th)

    def zoom_at(self, factor, pos=None):
        """Zoom by factor, keeping the image point under pos (default: widget centre) fixed."""
        if self._img is None:
            return
        t = self._view_rect()
        pos = pos if pos is not None else QtCore.QPointF(self.width() / 2, self.height() / 2)
        fx, fy = (pos.x() - t.x()) / t.width(), (pos.y() - t.y()) / t.height()
        old = self.zoom
        self.zoom = min(max(self.zoom * factor, 1.0), VIDEO_MAX_ZOOM)
        k = self.zoom / old
        self.center = [fx - (pos.x() - self.width() / 2) / (t.width() * k),
                       fy - (pos.y() - self.height() / 2) / (t.height() * k)]
        self.update()
        self._save_timer.start()

    def reset_view(self):
        self.zoom, self.center = 1.0, [0.5, 0.5]
        self.update()
        self._save_timer.start()

    def wheelEvent(self, e):
        dy = e.angleDelta().y()
        if dy:
            self.zoom_at(1.0015 ** dy, e.position())
        e.accept()

    def event(self, e):
        # Trackpad pinch (macOS)
        if e.type() == QtCore.QEvent.NativeGesture and e.gestureType() == QtCore.Qt.ZoomNativeGesture:
            self.zoom_at(1.0 + e.value(), e.localPos())
            return True
        return super().event(e)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtCore.Qt.black)
        if self._img is None:
            p.end(); return

        target = self._view_rect()
        # Draw only the visible part of the frame (cheap even when zoomed right in)
        visible = target.intersected(QtCore.QRectF(self.rect()))
        sx = self._img.width() / target.width()
        sy = self._img.height() / target.height()
        src = QtCore.QRectF((visible.x() - target.x()) * sx, (visible.y() - target.y()) * sy,
                            visible.width() * sx, visible.height() * sy)
        p.drawImage(visible, self._img, src)  # no smooth transform hint = fast scaling
        self._target = target

        if PAX_OVERLAY and NOZZLE_OUTLINE:
            self._draw_nozzle(p, target)
        if PAX_OVERLAY and self._pax is not None:
            self._draw_pax_bars(p, target, self._pax)
            self._draw_theory(p, target)
        if PAX_OVERLAY and NOZZLE_OUTLINE:
            self._draw_shock_markers(p, target)   # last, so bars/curve never hide them
        p.end()

    # --- Overlay drawing ---
    def _px_per_mm(self, target):
        return self.spacing * target.width() / (NOZZLE.taps_mm[1] - NOZZLE.taps_mm[0])

    def _mm_to_screen(self, target, x_mm, h_mm):
        """Nozzle coords (mm from throat along the flat wall, mm off the wall) -> screen point.
        Scale comes from the tap spacing; Pax1 sits at NOZZLE.taps_mm[0]."""
        ppm = self._px_per_mm(target)
        x0 = target.x() + self.pax1[0] * target.width() - NOZZLE.taps_mm[0] * ppm
        y0 = target.y() + self.pax1[1] * target.height()
        toward_contour = 1 if CONTOUR_BELOW_ON_SCREEN else -1
        return QtCore.QPointF(x0 + x_mm * ppm, y0 + toward_contour * h_mm * ppm)

    def _draw_nozzle(self, p, target):
        if self.spacing <= 0:
            return
        contour = QtGui.QPolygonF([self._mm_to_screen(target, x, h) for x, h in zip(self._noz_x, self._noz_h)])
        flat = (self._mm_to_screen(target, self._noz_x[0], 0), self._mm_to_screen(target, self._noz_x[-1], 0))
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        wall = QtGui.QColor(NOZZLE_OUTLINE_COLOR); wall.setAlpha(170)
        for color, w in ((self._shadow, 3), (wall, 1)):
            p.setPen(QtGui.QPen(color, w))
            p.drawPolyline(contour)
            p.drawLine(*flat)

    def _draw_shock_markers(self, p, target):
        QPF = QtCore.QPointF
        if self._shock_x is None or self.spacing <= 0:
            return
        # Triangles just outside each wall, pointing at the shock position
        xs = self._shock_x
        wall_c = self._mm_to_screen(target, xs, NOZZLE.height(xs))
        wall_f = self._mm_to_screen(target, xs, 0)
        out = 1 if wall_c.y() > wall_f.y() else -1     # screen-y direction away from flow at contoured wall
        gap, size = 3, 9
        p.setPen(QtGui.QPen(self._shadow, 1))
        p.setBrush(QtGui.QColor(SHOCK_MARKER_COLOR))
        for pt, d in ((wall_c, out), (wall_f, -out)):
            tip = QPF(pt.x(), pt.y() + d * gap)
            p.drawPolygon(QtGui.QPolygonF([tip, QPF(pt.x() - size / 2, tip.y() + d * size),
                                           QPF(pt.x() + size / 2, tip.y() + d * size)]))
        p.setBrush(QtCore.Qt.NoBrush)

    def _geometry(self, target, n):
        """Screen x of each tap, the zero-pressure baseline y, and pixels per bar."""
        tw, th = target.width(), target.height()
        xs = [target.x() + (self.pax1[0] + i * self.spacing) * tw for i in range(n)]
        return xs, target.y() + self.pax1[1] * th, self.bar_scale * th

    @staticmethod
    def _pen(color, width, style=QtCore.Qt.SolidLine):
        pn = QtGui.QPen(color, width, style)
        pn.setCapStyle(QtCore.Qt.FlatCap)
        return pn

    def _draw_pax_bars(self, p, target, pax):
        QPF = QtCore.QPointF
        xs, y0, px_per_bar = self._geometry(target, len(pax))
        tops = [QPF(x, y0 - v * px_per_bar) for x, v in zip(xs, pax)]

        p.setRenderHint(QtGui.QPainter.Antialiasing)
        # Each element drawn twice: dark wider stroke underneath, bright stroke on top
        for color, extra in ((self._shadow, 2), (self._color, 0)):
            p.setPen(self._pen(color, 1 + extra))
            p.drawLine(QPF(xs[0] - 10, y0), QPF(xs[-1] + 10, y0))     # baseline
            p.setPen(self._pen(color, PAX_BAR_WIDTH + extra))
            for x, top in zip(xs, tops):
                p.drawLine(QPF(x, y0), top)                           # bar
            # End marker: solid dot with a thin ring around it
            p.setPen(self._pen(color, 1.5 + extra))
            p.setBrush(QtCore.Qt.NoBrush)
            for top in tops:
                p.drawEllipse(top, PAX_RING_R, PAX_RING_R)
            p.setPen(self._pen(color, extra) if extra else QtCore.Qt.NoPen)
            p.setBrush(color)
            for top in tops:
                p.drawEllipse(top, PAX_DOT_R, PAX_DOT_R)
            p.setBrush(QtCore.Qt.NoBrush)

        if PAX_SHOW_VALUES:
            p.setFont(self._font)
            fm = p.fontMetrics()
            for v, top in zip(pax, tops):
                txt = f"{v:.2f}"
                tx = top.x() - fm.horizontalAdvance(txt) / 2
                gap = PAX_RING_R + 3
                ty = top.y() - gap if v >= 0 else top.y() + fm.ascent() + gap  # beyond the end marker
                p.setPen(self._shadow); p.drawText(QPF(tx + 1, ty + 1), txt)
                p.setPen(self._color);  p.drawText(QPF(tx, ty), txt)

    def _draw_theory(self, p, target):
        """Yellow curve: wall pressure for fully supersonic isentropic flow at the measured P0.
        Bars on it = isentropic; bars leaving it = the shock has reached that tap."""
        if not PAX_SHOW_ISENTROPIC or self._p0 is None or self.spacing <= 0:
            return
        QPF = QtCore.QPointF
        _, y0, px_per_bar = self._geometry(target, 1)
        offset = 0.0 if PAX_ABSOLUTE else P_ATM_PA / 1e5
        pts = [QPF(self._mm_to_screen(target, x, 0).x(), y0 - (self._p0 * r - offset) * px_per_bar)
               for x, r in zip(self._isen_x, self._isen_ratio)]
        isen = QtGui.QColor(ISEN_COLOR)
        for color, w in ((self._shadow, 4), (isen, 2)):
            p.setPen(self._pen(color, w))
            p.drawPolyline(QtGui.QPolygonF(pts))
        p.setFont(self._font)
        end = pts[-1]
        p.setPen(self._shadow); p.drawText(QPF(end.x() + 5, end.y() + 4), "isentropic")
        p.setPen(isen);         p.drawText(QPF(end.x() + 4, end.y() + 3), "isentropic")

    def nudge(self, dx=0.0, dy=0.0, dspacing=0.0, scale_mult=1.0):
        self.pax1[0] += dx
        self.pax1[1] += dy
        self.spacing = max(0.0, self.spacing + dspacing)
        self.bar_scale *= scale_mult
        self.update()
        self.overlay_changed.emit()
        save_settings()

    # --- Settings persistence (file is read once at startup into SAVED) ---
    def _load_settings(self):
        try:
            if "pax1_pos" in SAVED:
                self.pax1 = [float(v) for v in SAVED["pax1_pos"]]
                self.spacing = float(SAVED["spacing"])
                self.bar_scale = float(SAVED["bar_scale"])
            if "zoom" in SAVED:
                self.zoom = float(SAVED["zoom"])
                self.center = [float(v) for v in SAVED["view_center"]]
        except Exception as e:
            print(f"Ignoring unreadable overlay settings ({e})")

    def settings_dict(self):
        return {"pax1_pos": [round(v, 4) for v in self.pax1],
                "spacing": round(self.spacing, 4),
                "bar_scale": round(self.bar_scale, 4),
                "zoom": round(self.zoom, 4),
                "view_center": [round(v, 4) for v in self.center]}

    # --- Mouse: drag Pax1/baseline to move bars, last bar for spacing, anywhere else to pan ---
    def _hit(self, pos):
        if self._target is None or self._pax is None:
            return None
        xs, y0, px_per_bar = self._geometry(self._target, len(self._pax))
        tops = [y0 - v * px_per_bar for v in self._pax]
        def near_bar(i):
            return abs(pos.x() - xs[i]) < 12 and min(y0, tops[i]) - 12 <= pos.y() <= max(y0, tops[i]) + 12
        if len(xs) > 1 and near_bar(len(xs) - 1):
            return "spacing"
        if any(near_bar(i) for i in range(len(xs) - 1)):
            return "move"
        if xs[0] - 10 <= pos.x() <= xs[-1] + 10 and abs(pos.y() - y0) < 8:
            return "move"  # anywhere on the baseline
        return None

    def mousePressEvent(self, e):
        mode = self._hit(e.pos()) if e.button() == QtCore.Qt.LeftButton else None
        if mode:
            self._drag = (mode, e.pos(), list(self.pax1), self.spacing)
            self.setCursor(QtCore.Qt.ClosedHandCursor if mode == "move" else QtCore.Qt.SizeHorCursor)
        elif self._target is not None:
            self._drag = ("pan", e.pos(), list(self.center), None)
            self.setCursor(QtCore.Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._drag is None:  # hover feedback only
            mode = self._hit(e.pos())
            if mode == "move": self.setCursor(QtCore.Qt.OpenHandCursor)
            elif mode == "spacing": self.setCursor(QtCore.Qt.SizeHorCursor)
            else: self.unsetCursor()
            return
        if self._target is None:
            return
        mode, start, v0, spacing_0 = self._drag
        dx = (e.pos().x() - start.x()) / self._target.width()
        dy = (e.pos().y() - start.y()) / self._target.height()
        if mode == "pan":
            self.center = [v0[0] - dx, v0[1] - dy]
        elif mode == "move":
            self.pax1 = [v0[0] + dx, v0[1] + dy]
            self.overlay_changed.emit()
        elif self._pax is not None:
            self.spacing = max(0.0, spacing_0 + dx / (len(self._pax) - 1))
            self.overlay_changed.emit()
        self.update()

    def mouseReleaseEvent(self, e):
        if self._drag is not None:
            self._drag = None
            self.unsetCursor()
            save_settings()

def make_overlay_controls(vw):
    """Sidebar box with buttons to align the pax bars to the taps in the video."""
    box = QtWidgets.QGroupBox("Pax overlay")
    box.setStyleSheet(
        "QGroupBox { color: #00E5FF; font-weight: bold; border: 1px solid #333; margin-top: 8px; padding-top: 6px; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 6px; }"
        "QToolButton { background: #222; color: white; border: 1px solid #444; min-width: 26px; min-height: 22px; }"
        "QToolButton:pressed { background: #00E5FF; color: black; }"
        "QLabel { color: #BBB; }")
    g = QtWidgets.QGridLayout(box)
    readout = QtWidgets.QLabel()
    readout.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)  # select & copy into config
    readout.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))

    def refresh():
        readout.setText(f"PAX1_POS = ({vw.pax1[0]:.3f}, {vw.pax1[1]:.3f})\n"
                        f"PAX_SPACING = {vw.spacing:.3f}\n"
                        f"PAX_BAR_SCALE = {vw.bar_scale:.3f}")

    def btn(text, tip, **kw):
        b = QtWidgets.QToolButton(); b.setText(text); b.setToolTip(tip)
        b.setAutoRepeat(True)  # hold to keep moving
        b.clicked.connect(lambda: vw.nudge(**kw))
        return b

    s = PAX_STEP
    rows = [("Pax1",    [btn("◀", "Move left",  dx=-s), btn("▶", "Move right", dx=s),
                         btn("▲", "Move up",    dy=-s), btn("▼", "Move down",  dy=s)]),
            ("Spacing", [btn("−", "Closer together", dspacing=-s), btn("+", "Further apart", dspacing=s)]),
            ("Scale",   [btn("−", "Shorter bars", scale_mult=1/1.1), btn("+", "Taller bars", scale_mult=1.1)])]
    def view_btn(text, tip, fn):
        b = QtWidgets.QToolButton(); b.setText(text); b.setToolTip(tip)
        b.setAutoRepeat(text != "Fit")
        b.clicked.connect(fn)
        return b
    rows.append(("View", [view_btn("−", "Zoom out", lambda: vw.zoom_at(1 / 1.2)),
                          view_btn("+", "Zoom in", lambda: vw.zoom_at(1.2)),
                          view_btn("Fit", "Show the whole frame", vw.reset_view)]))
    for r, (name, buttons) in enumerate(rows):
        g.addWidget(QtWidgets.QLabel(name), r, 0)
        for c, b in enumerate(buttons):
            g.addWidget(b, r, c + 1)
    g.addWidget(readout, len(rows), 0, 1, 5)
    hint = QtWidgets.QLabel("Drag Pax1/baseline to move, last bar for spacing.\n"
                            "Scroll/pinch to zoom, drag the video to pan.\n"
                            "Keys: arrows move (Shift ×5), [ ] spacing, − = scale")
    hint.setStyleSheet("color: #777; font-size: 10px;")
    g.addWidget(hint, len(rows) + 1, 0, 1, 5)

    # Keyboard shortcuts work wherever focus is in the window
    def key(seq, **kw):
        sc = QtWidgets.QShortcut(QtGui.QKeySequence(seq), box)
        sc.setContext(QtCore.Qt.ApplicationShortcut)
        sc.activated.connect(lambda: vw.nudge(**kw))
    for mult, mod in ((1, ""), (5, "Shift+")):
        key(mod + "Left", dx=-s * mult); key(mod + "Right", dx=s * mult)
        key(mod + "Up", dy=-s * mult);   key(mod + "Down", dy=s * mult)
    key("[", dspacing=-s); key("]", dspacing=s)
    key("-", scale_mult=1/1.1); key("=", scale_mult=1.1); key("+", scale_mult=1.1)

    vw.overlay_changed.connect(refresh)  # buttons, keys and dragging all update the readout
    refresh()
    return box

# -----------------------
# GUI Setup
# -----------------------
app = QtWidgets.QApplication(sys.argv)
pg.setConfigOption('background', '#050505')
main_widget = QtWidgets.QWidget()
layout = QtWidgets.QGridLayout(main_widget)
main_widget.resize(1400, 1000)

win = pg.GraphicsLayoutWidget()
video_widget = VideoWidget()

# Vertical splitter: drag the bar between plots and video to resize the feed live
splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
splitter.addWidget(win)
splitter.addWidget(video_widget)
if VIDEO_FIXED_SIZE:
    video_widget.setFixedSize(*VIDEO_FIXED_SIZE)
else:
    splitter.setSizes([1000 - VIDEO_INITIAL_HEIGHT, VIDEO_INITIAL_HEIGHT])
layout.addWidget(splitter, 0, 0)
layout.setColumnStretch(0, 4)

def create_lcd(label_text, color):
    container = QtWidgets.QWidget(); vbox = QtWidgets.QVBoxLayout(container)
    vbox.setContentsMargins(2, 4, 2, 4)
    label = QtWidgets.QLabel(label_text); label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 13px;")
    label.setAlignment(QtCore.Qt.AlignCenter)
    lcd = QtWidgets.QLCDNumber(); lcd.setDigitCount(6); lcd.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
    lcd.setMinimumHeight(45); lcd.setStyleSheet(f"border: 1px solid {color}; color: {color}; background-color: black;")
    vbox.addWidget(label); vbox.addWidget(lcd)
    return container, lcd

def make_calibration_controls():
    """Sidebar box with one spin box per Kulite offset; edits apply live and are saved."""
    box = QtWidgets.QGroupBox("Kulite calibration (bar offset)")
    box.setStyleSheet(
        "QGroupBox { color: #00B0FF; font-weight: bold; border: 1px solid #333; margin-top: 8px; padding-top: 6px; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 6px; }"
        "QLabel { color: #BBB; }"
        "QDoubleSpinBox { background: #222; color: white; border: 1px solid #444; padding: 2px; }")
    g = QtWidgets.QGridLayout(box)
    for i in range(4):
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(-5.0, 5.0); sb.setDecimals(3); sb.setSingleStep(0.01)
        sb.setValue(PAX_CALIBRATION[i])
        sb.setKeyboardTracking(False)  # apply when Enter pressed / focus leaves, not on every keystroke
        def on_change(v, i=i):
            PAX_CALIBRATION[i] = v
            save_settings()
        sb.valueChanged.connect(on_change)
        g.addWidget(QtWidgets.QLabel(f"Pax{i+1}"), i // 2, (i % 2) * 2)
        g.addWidget(sb, i // 2, (i % 2) * 2 + 1)
    return box

# Sidebar content scrolls; the Quit button stays pinned below it
sidebar = QtWidgets.QWidget(); sidebar.setStyleSheet("background-color: #0A0A0A;")
right_panel = QtWidgets.QVBoxLayout(sidebar)
sidebar_scroll = QtWidgets.QScrollArea()
sidebar_scroll.setWidget(sidebar)
sidebar_scroll.setWidgetResizable(True)
sidebar_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
sidebar_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
right_col = QtWidgets.QWidget(); right_col.setStyleSheet("background-color: #0A0A0A;")
right_col_layout = QtWidgets.QVBoxLayout(right_col)
right_col_layout.setContentsMargins(0, 0, 0, 0)
right_col_layout.addWidget(sidebar_scroll)
layout.addWidget(right_col, 0, 1)
if simulated:
    sim_banner = QtWidgets.QLabel("SIMULATED: " + ", ".join(simulated))
    sim_banner.setAlignment(QtCore.Qt.AlignCenter)
    sim_banner.setWordWrap(True)
    sim_banner.setStyleSheet("background-color: #B71C1C; color: white; font-weight: bold; font-size: 14px; padding: 6px;")
    right_panel.addWidget(sim_banner)
c_p1, l_p1 = create_lcd("P1 (bar)", "#00E5FF"); c_p2, l_p2 = create_lcd("P2 (bar)", "#FF3D00")
c_t1, l_t1 = create_lcd("T1 (°C)", "#00E676"); c_t2, l_t2 = create_lcd("T2 (°C)", "#FFEA00")
c_mass, l_mass = create_lcd("Mass (kg)", "#BA68C8"); c_md, l_md = create_lcd("Mdot (g/s)", "#FF1744")
c_m1, l_m1 = create_lcd("M1 (Shock)", "#D1C4E9"); c_m2, l_m2 = create_lcd("M2 (Post)", "#B39DDB")
c_pax1, l_pax1 = create_lcd("Pax1 (bar)", "#00B0FF"); c_pax2, l_pax2 = create_lcd("Pax2 (bar)", "#00B0FF")
c_pax3, l_pax3 = create_lcd("Pax3 (bar)", "#00B0FF"); c_pax4, l_pax4 = create_lcd("Pax4 (bar)", "#00B0FF")

grid_panel = QtWidgets.QGridLayout()
for i, c in enumerate([c_p1, c_p2, c_t1, c_t2, c_mass, c_md, c_m1, c_m2, c_pax1, c_pax2, c_pax3, c_pax4]): grid_panel.addWidget(c, i // 2, i % 2)
right_panel.addLayout(grid_panel)
right_panel.addWidget(make_calibration_controls())
if PAX_OVERLAY:
    right_panel.addWidget(make_overlay_controls(video_widget))
right_panel.addStretch()
# Size the scroll area to the content width so only vertical scrolling is ever needed
sidebar_scroll.setMinimumWidth(sidebar.sizeHint().width() + sidebar_scroll.verticalScrollBar().sizeHint().width())

quit_btn = QtWidgets.QPushButton("Quit")
quit_btn.setToolTip("Stop camera, save data and exit (Ctrl+Q / ⌘Q)")
quit_btn.setStyleSheet("QPushButton { background: #B71C1C; color: white; font-weight: bold; font-size: 14px; padding: 8px; border: none; }"
                       "QPushButton:pressed { background: #7F0000; }")
quit_btn.clicked.connect(app.quit)
right_col_layout.addWidget(quit_btn)
quit_sc = QtWidgets.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.Quit), main_widget)
quit_sc.setContext(QtCore.Qt.ApplicationShortcut)
quit_sc.activated.connect(app.quit)

# Plots
p_p = win.addPlot(title="Pressure (bar)"); cur_p1, cur_p2 = p_p.plot(pen='#00E5FF'), p_p.plot(pen='#FF3D00')
win.nextRow()
p_mdot = win.addPlot(title="Mass Flow Rate (g/s)"); cur_mdot = p_mdot.plot(pen='#FF1744')

vid_thread = VideoThread(video_filename)
vid_thread.change_pixmap_signal.connect(video_widget.set_frame)
vid_thread.start()

last_jump_time = 0
def update():
    global last_jump_time
    # Snapshot under the lock so every series has the same length (data thread appends concurrently)
    with data_lock:
        if not data['time']: return
        snap = {k: np.array(v) for k, v in data.items()}
    t_arr = snap['time']; t_latest = t_arr[-1]
    if t_latest - last_jump_time >= 10:
        new_min = max(0, (int(t_latest / 10) * 10) - 5)
        for p in [p_p, p_mdot]: p.setXRange(new_min, new_min + 20, padding=0)
        last_jump_time = t_latest

    cur_p1.setData(t_arr, snap['p1']); cur_p2.setData(t_arr, snap['p2'])
    cur_mdot.setData(t_arr, snap['mdot'])
    
    pax_cal = [snap[f'pax{i+1}'][-1] + PAX_CALIBRATION[i] for i in range(4)]
    # Theoretical tap pressures and shock position from the measured stagnation pressure (absolute bar)
    ans, m1_val = snap['A_NS'][-1], snap['M1'][-1]
    ar_shock = ans / NOZZLE_AT
    x_shock = None
    if m1_val > 1.01 and 1.001 < ar_shock < NOZZLE.ae_at - 1e-4:
        x_shock = NOZZLE.x_at_area_ratio(ar_shock)
    video_widget.set_pax(pax_cal, snap[P0_CHANNEL][-1], x_shock)

    for l, d in zip([l_p1, l_p2, l_t1, l_t2, l_mass, l_md, l_m1, l_m2, l_pax1, l_pax2, l_pax3, l_pax4],
                    [snap['p1'][-1], snap['p2'][-1], snap['t1'][-1], snap['t2'][-1], snap['m'][-1]/1000, snap['mdot'][-1], snap['M1'][-1], snap['M2'][-1]] + pax_cal):
        l.display(f"{d:.2f}" if l != l_mass else f"{d:.3f}")

main_widget.show()
timer = QtCore.QTimer(); timer.timeout.connect(update); timer.start(50)
def shutdown():
    """Runs on Quit button, Ctrl+Q or window close. CSV export follows via atexit."""
    print("Shutting down...")
    save_settings()
    vid_thread.stop()   # stops camera loop and finishes any video recording
    if tcp == 1:
        try: s.close()
        except Exception: pass
app.aboutToQuit.connect(shutdown)
sys.exit(app.exec_())
