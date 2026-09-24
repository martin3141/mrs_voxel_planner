import os

# Run Qt without a display, so the viewer tests are headless (e.g. in CI) and don't open
# windows on the desktop. Forced rather than defaulted: desktops often set QT_QPA_PLATFORM
# (e.g. to wayland), and a tiling window manager would then resize the test windows.
os.environ["QT_QPA_PLATFORM"] = "offscreen"
