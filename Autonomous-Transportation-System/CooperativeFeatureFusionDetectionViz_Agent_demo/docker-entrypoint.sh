#!/usr/bin/env bash
set -e

# xvfb-run -a -s "-screen 0 1920x1080x24 +extension GLX +render -noreset" python main.py

xvfb-run -a python main.py
# xvfb-run -a python tests/open3d_test.py
