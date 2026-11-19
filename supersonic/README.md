# Supersonic nozzle rig: live display (`prc.py`)

## Dev machine (no hardware)

```bash
cd supersonic
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate    # once per terminal; then plain `python` uses the venv
python prc.py --sim
```

`--sim` simulates every device and skips the 5 s attempt to connect to LabVIEW.
Without `--sim`, any device that can't be found is simulated automatically.
A red **SIMULATED** banner lists what's fake, and saved files get a `_SIM` suffix.
The fake devices are in `sim_hardware.py`.

## Raspberry Pi

```bash
sudo apt install python3-pyqt5          # no pip wheel for PyQt5 on the Pi
cd supersonic
python3 -m venv --system-site-packages .venv   # lets the venv see apt's PyQt5
.venv/bin/pip install -r requirements.txt
.venv/bin/python prc.py
```

Enable I2C (`sudo raspi-config` → Interface Options) for the ADS1115 and MCP9600s.
`picamera` only works with the legacy camera stack (Raspberry Pi OS Bullseye or
earlier with legacy camera enabled). On Bookworm the camera falls back to simulation.
