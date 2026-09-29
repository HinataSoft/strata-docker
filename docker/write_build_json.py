#!/usr/bin/env python3
"""Write engine/BUILD.json for the binaries the builder stage just compiled.

The file is the one build_engine() writes, so setup.py adopts these binaries instead of downloading a
ready-made engine (there is no Linux asset) or compiling one (the runtime stage has no compiler):

  source="local"   get_prebuilt() returns the folder as-is, whatever its version
  src/vision_src   setup.py's own source fingerprints; while they match the tree, update_installed_engine()
                   leaves the engine alone rather than trying to rebuild it
  cuda_dirs=[]     falsy, so the generated config falls back to cuda_lib_dirs() - NVIDIA's pip wheels

    python3 docker/write_build_json.py "120"        # or "86;120"
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import setup as S  # noqa: E402


def main() -> int:
    archs_arg = sys.argv[1] if len(sys.argv) > 1 else "120"
    archs = [int(a.split("-")[0]) for a in archs_arg.replace(",", ";").split(";") if a.strip()]
    if not archs:
        print("no CUDA architectures given", file=sys.stderr)
        return 1

    meta = {
        "source": "local",
        "version": S.source_version(),
        "archs": archs,
        "vision": "gpu",
        "cuda_dirs": [],
        "src": S.source_hash(S.ENGINE_SOURCES),
        "vision_src": S.source_hash(S.VISION_SOURCES),
    }
    out = ROOT / "engine" / "BUILD.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
