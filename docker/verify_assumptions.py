#!/usr/bin/env python3
"""Assert everything this image assumes about the Strata tree, at BUILD time.

The container drives setup.py and the engine through names that are not a public interface: module-level
constants, CLI flag spellings, the layout of the generated config.  Strata and this repository move
independently, so an upgrade can break them.  Two of those breakages are silent at run time - init would
finish and the server would then look for a config that is not there, or add flags the engine ignores - so
every assumption is checked here instead, where a failure stops the build.

    python3 docker/verify_assumptions.py --llama-commit <sha>
    python3 docker/verify_assumptions.py --root ./src --no-engine     # against a checkout, before building

Each check names the file that depends on it, so a failure says what to fix, not just what broke.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Resolved by the host driver through nvidia-container-toolkit, never present at build time, so ldd
# reporting them missing here means nothing.
DRIVER_LIBS = ("libcuda.so", "libnvidia-")

failures: list[str] = []
checks = 0


def check(ok: bool, what: str, needed_by: str, detail: str = "") -> bool:
    global checks
    checks += 1
    if ok:
        print(f"  [ok] {what}")
        return True
    failures.append(f"{what}\n         needed by: {needed_by}" + (f"\n         {detail}" if detail else ""))
    print(f"  [X]  {what}   <- {needed_by}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-commit", default="")
    ap.add_argument("--root", default=None,
                    help="the Strata tree (default: the parent of docker/, which is what the image has)")
    ap.add_argument("--no-engine", action="store_true",
                    help="skip the engine/BUILD.json checks: for running against a plain checkout")
    a = ap.parse_args()

    global ROOT
    ROOT = Path(a.root).resolve() if a.root else Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))

    print(f"Verifying this image's assumptions about the Strata tree at {ROOT}\n")

    # ---- setup.py is importable and still exposes what the build and the entrypoint read ----------
    try:
        import setup as S
    except Exception as e:
        print(f"  [X]  setup.py cannot be imported: {e}")
        return 1
    print("  [ok] setup.py imports")

    for name in ("ENGINE_SOURCES", "VISION_SOURCES", "PY_PACKAGES", "FAMILIES", "MODELS", "LLAMA_CPP_COMMIT"):
        check(hasattr(S, name), f"setup.{name} exists", "docker/write_build_json.py, docker/write_pip_stamp.py")
    for name in ("source_hash", "source_version", "pip_install"):
        check(callable(getattr(S, name, None)), f"setup.{name}() exists", "docker/write_build_json.py")
    check(callable(getattr(S, "cuda_lib_dirs", None)), "setup.cuda_lib_dirs() exists",
          "the shared-library check below, and the generated config's lib_dirs")

    # ---- the engine we compiled matches the tree we shipped --------------------------------------
    meta_path = ROOT / "engine" / "BUILD.json"
    if a.no_engine:
        print("  [--] engine/BUILD.json checks skipped (--no-engine)")
    elif check(meta_path.is_file(), "engine/BUILD.json exists", "setup.py get_prebuilt()"):
        meta = json.loads(meta_path.read_text())
        check(meta.get("source") == "local", 'BUILD.json source == "local"',
              "setup.py get_prebuilt(): anything else re-downloads a Linux engine that is not published")
        check(meta.get("src") == S.source_hash(S.ENGINE_SOURCES), "BUILD.json src matches the shipped sources",
              "setup.py update_installed_engine(): a mismatch makes it try to recompile, which fails here",
              "the runtime stage must carry the same CMakeLists.txt, src/, include/, third_party/ggml/ as the builder")
        check(meta.get("vision_src") == S.source_hash(S.VISION_SOURCES), "BUILD.json vision_src matches tools/vision",
              "setup.py update_installed_engine()")
        check(not meta.get("cuda_dirs"), "BUILD.json cuda_dirs is empty",
              "setup.py: a non-empty value would point the config at a CUDA toolkit this image does not have")

    if not a.no_engine:
        for exe in ("strata", "strata-vision"):
            p = ROOT / "engine" / exe
            check(p.is_file() and os.access(p, os.X_OK), f"engine/{exe} is present and executable",
                  "docker/entrypoint.sh")

        # Every shared library must resolve with the search path the server will give the binaries, or the
        # process dies at exec before printing anything.  server.py hides the vision encoder's stderr in the
        # engine log, so at run time that shows up only as an empty "the vision encoder did not start:".
        # This is how libnccl.so.2 - linked by ggml-cuda, not shipped by NVIDIA's pip wheels - got missed.
        lib_dirs = S.cuda_lib_dirs()
        check(bool(lib_dirs), "cuda_lib_dirs() finds NVIDIA's pip wheels",
              "the generated config's lib_dirs, hence LD_LIBRARY_PATH for the engine and the encoder")
        env = dict(os.environ)
        if lib_dirs:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                lib_dirs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
        if shutil.which("ldd") is None:
            print("  [--] ldd is not available - shared library check skipped")
        else:
            for exe in ("strata", "strata-vision"):
                p = ROOT / "engine" / exe
                if not p.is_file():
                    continue
                r = subprocess.run(["ldd", str(p)], capture_output=True, text=True, env=env, timeout=120)
                missing = [ln.strip() for ln in r.stdout.splitlines()
                           if "not found" in ln and not any(d in ln for d in DRIVER_LIBS)]
                check(not missing, f"engine/{exe}: every shared library resolves",
                      "the binary would die at exec, before it can report anything",
                      "; ".join(missing))

    # the pip stamp only exists in the image's venv
    if not a.no_engine:
        check((Path(sys.prefix) / ".strata-pip.json").is_file(),
              "the pip stamp is at <sys.prefix>/.strata-pip.json",
              "docker/write_pip_stamp.py; setup.py pip_install()")

    # ---- the pinned llama.cpp is the one setup.py expects ----------------------------------------
    if a.llama_commit:
        check(S.LLAMA_CPP_COMMIT == a.llama_commit, "LLAMA_CPP_COMMIT matches the build arg",
              "Dockerfile: source_hash() seeds on this commit, so a mismatch invalidates BUILD.json",
              f"setup.py says {S.LLAMA_CPP_COMMIT}, the Dockerfile passed {a.llama_commit}")

    # ---- the setup.py CLI flags the entrypoint passes --------------------------------------------
    help_text = subprocess.run([sys.executable, str(ROOT / "setup.py"), "--help"],
                               capture_output=True, text=True, timeout=120).stdout
    for flag in ("--family", "--model", "--context", "--kv", "--vision", "--data-dir", "--gguf-dir",
                 "--host", "--port", "--gpu", "--yes", "--no-start", "--calibrate",
                 "--experimental-speed-projection"):
        check(flag in help_text, f"setup.py accepts {flag}", "docker/entrypoint.sh")

    # ---- the engine flags the config carries -----------------------------------------------------
    gen = ROOT / "src" / "program" / "generate.cpp"
    if check(gen.is_file(), "src/program/generate.cpp is present", "this check"):
        src = gen.read_text(encoding="utf-8", errors="replace")
        for flag in ("--prompt-cache", "--prompt-cache-every", "--kv-resident", "--max-context", "--serve"):
            check(f'"{flag}"' in src, f"the engine parses {flag}",
                  "docker/patch_config.py" if flag.startswith("--prompt-cache") else "setup.py's generated config")

    # ---- the server CLI the entrypoint calls -----------------------------------------------------
    server = ROOT / "serve" / "server.py"
    if check(server.is_file(), "serve/server.py is present", "docker/entrypoint.sh"):
        src = server.read_text(encoding="utf-8", errors="replace")
        for flag in ("--engine", "--config", "--host", "--port", "--gpu"):
            check(f'"{flag}"' in src, f"server.py accepts {flag}", "docker/entrypoint.sh")
        # patch_config redirects cfg["log"]; the server must still be the thing that reads it
        check('cfg["log"]' in src or "cfg.get('log')" in src or 'cfg.get("log")' in src,
              "server.py reads the config's log path", "docker/patch_config.py")

    # ---- the config file name the entrypoint resolves --------------------------------------------
    for fam, data in S.FAMILIES.items():
        check("tag" in data, f'FAMILIES["{fam}"] has a "tag"', "docker/entrypoint.sh builds the config name from it")
    model = os.environ.get("STRATA_MODEL", "IQ3_S")
    check(model in S.MODELS, f"MODELS has {model}", "docker/entrypoint.sh, .env")

    print()
    if failures:
        print(f"{len(failures)} of {checks} assumptions FAILED:\n")
        for f in failures:
            print(f"  *  {f}\n")
        print("The Strata tree has moved away from what this image drives.  Fix the files named above, or\n"
              "pin the strata build context to a revision that still matches.")
        return 1
    print(f"all {checks} assumptions hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
