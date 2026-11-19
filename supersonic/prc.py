import sys
import math
import time
import threading
from collections import deque

# Sensor & Hardware Imports
# Each import is optional: anything missing falls back to sim_hardware.py.
# Run with --sim to force every subsystem to simulation (even on the Pi).
import sim_hardware

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
from scipy.optimize import brentq

# TCP for labVIEW connection
import socket, struct

# CSV Export & Dynamic Filename Imports
import atexit
import csv
from datetime import datetime

# Calibration of Kulite sensors
pax1_calibration=0.23
pax2_calibration=0.23
pax3_calibration=0.23
pax4_calibration=0.23

# -----------------------
# Video & Overlay Settings (edit here)
# -----------------------
CAMERA_RESOLUTION = (640, 480)   # capture size from the Pi camera
CAMERA_FRAMERATE = 24
# Crop in camera pixels (x, y, w, h). None = full frame.
# (0, 120, 640, 240) reproduces the old view: the middle 240 rows.
VIDEO_CROP = (0, 120, 640, 240)
# None = resizable live by dragging the splitter bar above the video.
# (w, h) = fixed size in screen pixels, e.g. (960, 360).
VIDEO_FIXED_SIZE = None
VIDEO_INITIAL_HEIGHT = 300       # starting height when resizable
VIDEO_MAX_SCALE = None           # e.g. 2.0 caps upscaling; None = no cap

PAX_OVERLAY = True
# Tap positions as fractions of the cropped frame: (0,0) top-left, (1,1) bottom-right.
PAX_POSITIONS = [(0.20, 0.50), (0.40, 0.50), (0.60, 0.50), (0.80, 0.50)]
PAX_LABELS = ["Pax1", "Pax2", "Pax3", "Pax4"]
OVERLAY_FONT_PT = 11
OVERLAY_COLOR = "#00E5FF"

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

def solve_nozzle_shock(P0, Pe, At, Ae, gamma=1.4):
    def am_rel(M, g):
        if M <= 0: return float('inf')
        return (1/M) * (2/(g+1) * (1 + (g-1)/2 * M**2))**((g+1)/(2*(g-1)))
    
    def p_isen(M, g, inv):
        return (1 + (g-1)/2 * M**2)**((inv)*g/(g-1))
    
    def solve_mach(area_ratio, is_supersonic=False):
        if area_ratio <= 1.0001: return 1.0
        try:
            if is_supersonic: return brentq(lambda M: am_rel(M, gamma) - area_ratio, 1.0, 10.0)
            else: return brentq(lambda M: am_rel(M, gamma) - area_ratio, 1e-6, 1.0)
        except: return 1.0

    def objective(M1, gamma_val):
        p1_p01 = p_isen(M1, gamma_val, -1)
        p2_p1 = 1 + (2*gamma_val/(gamma_val+1))*(M1**2-1)
        M2 = np.sqrt((1 + (gamma_val-1)/2 * M1**2) / (gamma_val*M1**2 - (gamma_val-1)/2))
        p02_p2 = p_isen(M2, gamma_val, 1)
        Ae_A2star = (Ae/At) * (1/am_rel(M1, gamma_val)) * am_rel(M2, gamma_val)
        Me = solve_mach(Ae_A2star, is_supersonic=False)
        pe_p0e = p_isen(Me, gamma_val, -1)
        return (pe_p0e * p02_p2 * p2_p1 * p1_p01) - (Pe/P0)

    if P0 <= Pe * 1.0001: return 0.0, 0.0, At
    
    limit = p_isen(solve_mach(Ae/At, False), gamma, -1)
    if (Pe/P0) > limit:
        try:
            me = math.sqrt(max(0, (2/(gamma-1))*((P0/Pe)**((gamma-1)/gamma)-1)))
            at_astar = am_rel(me, gamma) / (Ae/At)
            return solve_mach(at_astar, False), me, At
        except: return 0.0, 0.0, At
        
    try:
        m_sup = solve_mach(Ae/At, True)
        m1 = brentq(objective, 1.0, m_sup, args=(gamma,))
        m2 = np.sqrt((1 + (gamma-1)/2 * m1**2) / (gamma*m1**2 - (gamma-1)/2))
        return m1, m2, am_rel(m1, gamma) * At
    except:
        m_sup = solve_mach(Ae/At, True)
        return m_sup, m_sup, Ae

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
    Pe_atm, V, At, Ae = 101625.0, 0.3, 0.000048, 0.000160
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
        
        m1, m2, A_NS = solve_nozzle_shock(p02, Pe_atm, At, Ae)
        
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
    def __init__(self):
        super().__init__()
        self._img = None
        self._pax = None
        self._font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)  # monospace on any OS
        self._font.setPointSize(OVERLAY_FONT_PT)
        self._font.setBold(True)
        self._color = QtGui.QColor(OVERLAY_COLOR)
        self._bg = QtGui.QColor(0, 0, 0, 170)
        self.setMinimumSize(160, 60)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)  # skip background erase

    def set_frame(self, img):
        self._img = img
        self.update()  # repaint once per camera frame

    def set_pax(self, vals):
        self._pax = vals  # drawn on the next camera frame

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtCore.Qt.black)
        if self._img is None:
            p.end(); return

        iw, ih = self._img.width(), self._img.height()
        s = min(self.width() / iw, self.height() / ih)
        if VIDEO_MAX_SCALE: s = min(s, VIDEO_MAX_SCALE)
        tw, th = int(iw * s), int(ih * s)
        target = QtCore.QRect((self.width() - tw) // 2, (self.height() - th) // 2, tw, th)
        p.drawImage(target, self._img)  # no smooth transform hint = fast scaling

        if PAX_OVERLAY and self._pax is not None:
            p.setFont(self._font)
            fm = p.fontMetrics()
            for (fx, fy), name, v in zip(PAX_POSITIONS, PAX_LABELS, self._pax):
                x = target.x() + int(fx * tw)
                y = target.y() + int(fy * th)
                txt = f"{name} {v:.2f}"
                r = fm.boundingRect(txt).adjusted(-4, -2, 4, 2)
                r.moveCenter(QtCore.QPoint(x, y - r.height()))  # label sits above tap
                p.fillRect(r, self._bg)
                p.setPen(self._color)
                p.drawText(r, QtCore.Qt.AlignCenter, txt)
                p.drawLine(x, r.bottom(), x, y)         # leader line
                p.fillRect(x - 2, y - 2, 5, 5, self._color)  # tap marker
        p.end()

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

sidebar = QtWidgets.QWidget(); sidebar.setStyleSheet("background-color: #0A0A0A;")
right_panel = QtWidgets.QVBoxLayout(sidebar); layout.addWidget(sidebar, 0, 1)
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
right_panel.addStretch()

# Plots
p_p = win.addPlot(title="Pressure (bar)"); cur_p1, cur_p2 = p_p.plot(pen='#00E5FF'), p_p.plot(pen='#FF3D00')
win.nextRow()
p_mdot = win.addPlot(title="Mass Flow Rate (g/s)"); cur_mdot = p_mdot.plot(pen='#FF1744')
win.nextRow()
p_nozzle = win.addPlot(title="Shock Location"); p_nozzle.setAspectLocked(True)

xmax_noz, ht_noz, At_noz, Ae_noz = 0.0319, 0.006, 0.000048, 0.000160
x_noz = np.linspace(-0.015, xmax_noz + 0.005, 150)
ar_total = np.where(x_noz < 0, 1 + np.abs(x_noz) * 40, np.tanh(2 * np.maximum(0, x_noz) / xmax_noz) / np.tanh(2) * ((Ae_noz/At_noz) - 1) + 1)
p_nozzle.plot(x_noz, (ht_noz/2)*ar_total, pen='w'); p_nozzle.plot(x_noz, -(ht_noz/2)*ar_total, pen='w')
shock_line = p_nozzle.plot(pen=pg.mkPen('r', width=3))

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
    
    pax_cal = [snap['pax1'][-1]+pax1_calibration, snap['pax2'][-1]+pax2_calibration,
               snap['pax3'][-1]+pax3_calibration, snap['pax4'][-1]+pax4_calibration]
    video_widget.set_pax(pax_cal)

    for l, d in zip([l_p1, l_p2, l_t1, l_t2, l_mass, l_md, l_m1, l_m2, l_pax1, l_pax2, l_pax3, l_pax4],
                    [snap['p1'][-1], snap['p2'][-1], snap['t1'][-1], snap['t2'][-1], snap['m'][-1]/1000, snap['mdot'][-1], snap['M1'][-1], snap['M2'][-1]] + pax_cal):
        l.display(f"{d:.2f}" if l != l_mass else f"{d:.3f}")

    ans, m1_val = snap['A_NS'][-1], snap['M1'][-1]
    if m1_val > 1.01 and At_noz * 1.001 < ans <= Ae_noz * 1.01:
        x_s = (xmax_noz/2) * math.atanh(max(0, min(0.9999, ((ans/At_noz)-1)/((Ae_noz/At_noz)-1)*math.tanh(2))))
        y_lim = (ht_noz/2) * (ans/At_noz)
        shock_line.setData([x_s, x_s], [-y_lim, y_lim])
    else: shock_line.setData([], [])

main_widget.show()
timer = QtCore.QTimer(); timer.timeout.connect(update); timer.start(50)
app.aboutToQuit.connect(vid_thread.stop)
sys.exit(app.exec_())
