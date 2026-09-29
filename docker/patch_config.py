#!/usr/bin/env python3
"""Move the config setup.py wrote onto the config volume, add what setup.py cannot write, and record which
config is the active one.

    python3 docker/patch_config.py <source dir> <config dir>

WHY THE CONFIG MOVES.  setup.py writes strata-<tag>.json next to setup.py, inside the image.  It has to live
on the config volume instead: the server keeps <config>.shared-settings.json beside it (server.py derives
that path from the config's own), which fails on a read-only mount, and the file must survive the container
being recreated.

WHY THE NAME IS NOT COMPUTED TWICE.  The name comes from FAMILIES[family]["tag"] + model, lowercased.  Having
the entrypoint rebuild that string from environment variables meant two places had to agree with setup.py;
when they did not, init succeeded and serve then failed to find a config that was right there.  Instead the
file is found by glob and its path written to <config dir>/active, which is the only thing serve reads.

WHAT IS ADDED.  Two engine settings setup.py has no option for, whose defaults are sized for a 64 GB desktop:

  --prompt-cache N        conversation checkpoints kept between requests (engine default 6, ~118 MB of RAM
                          each).  An agent resends the whole conversation every turn; a checkpoint is what
                          lets the engine resume instead of reading a 262K prompt again.  The chain's root
                          (the end of the system prompt) is pinned, the rest rotates by least recent use.
  --prompt-cache-every N  also checkpoint every N freshly read prompt tokens (engine default 16384).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def set_flag(args: list, flag: str, value: str) -> None:
    """args with `flag value` set: replaced where present, appended otherwise."""
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            args[i + 1] = value
            return
        del args[i:]
    args.extend([flag, value])


def find_config(src_dir: Path) -> Path | None:
    """The config setup.py just wrote: the newest strata-*.json that is not a sidecar of one."""
    cands = [p for p in src_dir.glob("strata-*.json") if not p.name.endswith(".shared-settings.json")]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    src_dir, cfg_dir = Path(sys.argv[1]), Path(sys.argv[2])

    src = find_config(src_dir)
    if src is None:
        print(f"patch_config: no strata-*.json in {src_dir} - did setup.py finish?", file=sys.stderr)
        return 1

    dst = cfg_dir / src.name
    log = cfg_dir / (src.stem + ".log")

    cfg = json.loads(src.read_text(encoding="utf-8-sig"))
    args = cfg.setdefault("args", [])
    set_flag(args, "--prompt-cache", os.environ.get("STRATA_PROMPT_CACHE", "24"))
    set_flag(args, "--prompt-cache-every", os.environ.get("STRATA_PROMPT_CACHE_EVERY", "8192"))
    cfg["log"] = str(log)

    cfg_dir.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    # the pointer serve reads - one line, so the entrypoint needs no JSON parser
    (cfg_dir / "active").write_text(str(dst) + "\n", encoding="utf-8")

    val = lambda f, d="-": args[args.index(f) + 1] if f in args and args.index(f) + 1 < len(args) else d  # noqa: E731
    print(f"  config       {dst}")
    print(f"  active       {cfg_dir / 'active'}")
    print(f"  model        {cfg.get('model_name')}")
    print(f"  context      {val('--max-context')} tokens, KV {val('--kv', 'fp16')}"
          + (f", streaming {val('--kv-resident')} cells resident" if "--kv-resident" in args else ""))
    print(f"  prompt cache {val('--prompt-cache')} checkpoints, every {val('--prompt-cache-every')} tokens")
    print(f"  vision       {'on' if cfg.get('vision') else 'off'}")
    print(f"  log          {cfg['log']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
