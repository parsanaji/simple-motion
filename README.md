<img src="assets/icon.svg" width="64" height="64" alt="">

# lezur

Created in [T3 Code](https://t3.codes).

Webcam motion detection (OpenCV MOG2) that lights a Govee H6147 LED strip over Bluetooth LE.

## Setup (once)

macOS:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Windows (PowerShell or cmd):

```bat
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

The venv is required on a Homebrew Mac: Homebrew Python blocks a global
`pip install` (PEP 668), and there is no `python` command, only `python3`.

## Run

Replace `.venv/bin/python` with `.venv\Scripts\python` on Windows.

```sh
.venv/bin/python motion_detect.py --list-cameras   # find the camera index first
.venv/bin/python motion_detect.py                  # motion detector + Govee strip
.venv/bin/python motion_detect.py --no-govee       # camera only
.venv/bin/python govee_h6147.py --variant rgb-02   # strip test: red 5 s, then off
```

Press `q` in the video window to quit. Settings (camera index, minimum contour
area, cooldown, Govee device name, hold time) are in the CONFIG block at the
top of `motion_detect.py`.

## Tests

```sh
.venv/bin/pip install pytest
.venv/bin/python -m pytest -q
```
