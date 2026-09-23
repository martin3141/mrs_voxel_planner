"""Entry point: python -m mrs_voxel_planner [T1.nii.gz] [--voxel voxel.json|.rda|.nii.gz]"""

import argparse
import sys

from PyQt6 import QtWidgets

from .viewer import MainWindow
from .volume import make_phantom


def main(argv=None):
    parser = argparse.ArgumentParser(description="MRS voxel planner")
    parser.add_argument("t1", nargs="?", help="T1-weighted NIfTI image")
    parser.add_argument("--voxel", help="voxel to load (JSON, Siemens .rda, or NIfTI-MRS)")
    parser.add_argument("--phantom", action="store_true", help="use a synthetic head phantom")
    args = parser.parse_args(argv)

    app = QtWidgets.QApplication(sys.argv[:1])
    win = MainWindow()
    if args.t1:
        win.open_t1(args.t1)
    elif args.phantom:
        win.set_volume(make_phantom())
    if args.voxel:
        win.load_voxel(args.voxel)
    win.resize(1500, 720)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
