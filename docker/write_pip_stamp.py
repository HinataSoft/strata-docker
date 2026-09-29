#!/usr/bin/env python3
"""Write the stamp pip_install() checks, so setup.py's "Python packages" step is a no-op in the image.

pip_install() keeps a list of what it has installed in <sys.prefix>/.strata-pip.json and installs only the
names missing from it.  The image installs the same packages at build time, so the stamp just records them.

cmake and ninja are listed although they are NOT installed: PY_PACKAGES names them for the compile path, and
nothing in the runtime image compiles.  Listing them keeps setup.py from pulling ~80 MB of wheels it will
never use.
"""
import json
import sys
from pathlib import Path

# setup.py's PY_PACKAGES, plus the CUDA wheels it installs for a downloaded engine
INSTALLED = [
    "numpy", "jinja2", "regex", "pyyaml", "tqdm", "requests", "pillow", "psutil",
    "cmake", "ninja",                                    # recorded, deliberately not installed
    "nvidia-cublas==13.0.2.14", "nvidia-cuda-runtime==13.0.96",
]

stamp = Path(sys.prefix) / ".strata-pip.json"
stamp.write_text(json.dumps(sorted(INSTALLED), indent=0), encoding="utf-8")
print(f"wrote {stamp}")
