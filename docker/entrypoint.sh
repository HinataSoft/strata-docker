#!/bin/sh
# Strata container entrypoint.  The command decides what runs; there is no auto-detection, so a plain
# `docker run` never starts a download and the model volume can stay mounted read-only.
#
#   init        one time: check the machine, download the model, build the pack and the MTP draft layer,
#               write the run config.  Needs /data mounted READ-WRITE.
#   serve       (default) run the server against the config init wrote.  /data may be read-only.
#   calibrate   measure this machine's engine settings (5-10 min) and save them into the config.
#   preflight   check that the binaries and model files can actually start, with errors visible
#   <anything>  executed as given (e.g. `bash`, `python chat.py`).
#
# Which config is active is NOT recomputed here - init records it in $STRATA_CONFIG_DIR/active and serve
# reads that one line.  See docker/patch_config.py.
set -eu

: "${STRATA_DATA:=/data}"
: "${STRATA_CONFIG_DIR:=/config}"
: "${STRATA_FAMILY:=qwen}"
: "${STRATA_MODEL:=IQ3_S}"
: "${STRATA_CONTEXT:=262144}"
: "${STRATA_KV:=int8}"
: "${STRATA_VISION:=gpu}"
: "${STRATA_HOST:=0.0.0.0}"
: "${STRATA_PORT:=8080}"
: "${STRATA_PROMPT_CACHE:=24}"
: "${STRATA_PROMPT_CACHE_EVERY:=8192}"

ROOT=/opt/strata
PYTHON=/opt/venv/bin/python
ACTIVE="$STRATA_CONFIG_DIR/active"

# settings.json (the data folder and the saved calibration) belongs on the config volume, not in the image
export XDG_CONFIG_HOME="$STRATA_CONFIG_DIR"
export PYTHONUNBUFFERED=1

die() { printf '\n  [X]  %s\n' "$1" >&2; shift; for l in "$@"; do printf '       %s\n' "$l" >&2; done; exit 1; }
note() { printf '  %s\n' "$1"; }

check_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || die \
        "nvidia-smi is not in the container" \
        "Run with the NVIDIA runtime (--gpus all, or deploy.resources in compose) and make sure" \
        "NVIDIA_DRIVER_CAPABILITIES includes 'utility' - without it there is no nvidia-smi and setup refuses."
}

# The engine registers the whole expert arena (50 GB for IQ3_S) with cudaHostRegister and falls back to
# mlock.  Both are charged against RLIMIT_MEMLOCK, whose Docker default is far too small.  Failure is NOT
# fatal - the engine swallows it - so the only symptom is host-to-device copies running at a fraction of
# their speed, which is a miserable thing to diagnose after the fact.
check_memlock() {
    lim=$(ulimit -l 2>/dev/null || echo unlimited)
    [ "$lim" = "unlimited" ] && return 0
    case "$lim" in ''|*[!0-9]*) return 0 ;; esac
    if [ "$lim" -lt 33554432 ]; then          # ulimit -l is in KB; the arena wants tens of GB
        printf '\n  [!]  RLIMIT_MEMLOCK is %s KB - the engine cannot pin its expert arena, so every\n' "$lim" >&2
        printf '       copy to the GPU will be slow.  Add:  --ulimit memlock=-1   (compose: ulimits.memlock: -1)\n\n' >&2
    fi
}

# /proc/meminfo inside a container reports the HOST's RAM, so setup.py cannot see a -m limit: the model is
# OOM-killed halfway through loading instead of being refused up front.
check_cgroup_memory() {
    for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
        [ -r "$f" ] || continue
        v=$(cat "$f" 2>/dev/null || echo max)
        [ "$v" = "max" ] && return 0
        case "$v" in ''|*[!0-9]*) return 0 ;; esac
        if [ "$v" -lt 1000000000000000 ] && [ "$v" -lt 68719476736 ]; then
            printf '\n  [!]  the container memory limit is %s GB, below what this model needs.\n' \
                   "$((v / 1073741824))" >&2
            printf '       setup.py reads /proc/meminfo and sees the host RAM, so it will not warn you.\n\n' >&2
        fi
        return 0
    done
}

# The API key, from STRATA_API_KEY_FILE (a docker secret, preferred) or STRATA_API_KEY.  Empty = no key,
# which is the default: the server then serves /v1/* to anyone who can reach the port.
resolve_api_key() {
    if [ -n "${STRATA_API_KEY_FILE:-}" ]; then
        [ -r "$STRATA_API_KEY_FILE" ] || die \
            "STRATA_API_KEY_FILE is set but $STRATA_API_KEY_FILE cannot be read" \
            "With docker secrets the file appears at /run/secrets/<name>."
        head -n 1 "$STRATA_API_KEY_FILE" | tr -d '\r\n'
        return 0
    fi
    printf '%s' "${STRATA_API_KEY:-}"
}

resolve_config() {
    if [ -n "${STRATA_CONFIG:-}" ]; then
        printf '%s' "$STRATA_CONFIG"
        return 0
    fi
    [ -f "$ACTIVE" ] || return 1
    head -n 1 "$ACTIVE"
}

cmd_init() {
    check_gpu
    [ -w "$STRATA_DATA" ] || die \
        "$STRATA_DATA is not writable" \
        "init downloads the model there; mount it read-write for this step and read-only afterwards." \
        "  docker compose --profile init run --rm strata-init"
    [ -w "$STRATA_CONFIG_DIR" ] || die "$STRATA_CONFIG_DIR is not writable"

    note "Strata init: $STRATA_FAMILY $STRATA_MODEL, $STRATA_CONTEXT ctx, vision=$STRATA_VISION"
    note "data: $STRATA_DATA    config: $STRATA_CONFIG_DIR"
    check_cgroup_memory

    set -- --yes --no-start \
        --family "$STRATA_FAMILY" \
        --model "$STRATA_MODEL" \
        --context "$STRATA_CONTEXT" \
        --kv "$STRATA_KV" \
        --vision "$STRATA_VISION" \
        --experimental-speed-projection off \
        --data-dir "$STRATA_DATA" \
        --host "$STRATA_HOST" \
        --port "$STRATA_PORT"
    [ -n "${STRATA_GPU:-}" ] && set -- "$@" --gpu "$STRATA_GPU"
    [ -n "${STRATA_GGUF_DIR:-}" ] && set -- "$@" --gguf-dir "$STRATA_GGUF_DIR"
    # written into the config, so the running server needs no key on its command line (where the host's
    # `ps` would show it).  serve can still override it per start.
    # `|| exit` matters: resolve_api_key runs in a subshell, so its die() would otherwise only end that
    # subshell and init would carry on with no key - silently unauthenticated.
    api_key=$(resolve_api_key) || exit 1
    if [ -n "$api_key" ]; then
        set -- "$@" --api-key "$api_key"
        note "API key: required (stored in the config)"
    else
        note "API key: none - anyone who can reach the port can use the model"
    fi

    cd "$ROOT"
    "$PYTHON" setup.py "$@"

    printf '\n=== Writing the run config ===\n'
    "$PYTHON" "$ROOT/docker/patch_config.py" "$ROOT" "$STRATA_CONFIG_DIR"
    printf '\nInit done.  %s can go back to read-only; start the server with:\n' "$STRATA_DATA"
    printf '  docker compose up -d strata\n'
}

cmd_serve() {
    check_gpu
    cfg=$(resolve_config) || die \
        "no active config in $STRATA_CONFIG_DIR" \
        "Run the one-time init first:" \
        "  docker compose --profile init run --rm strata-init" \
        "(it needs $STRATA_DATA mounted read-write).  Set STRATA_CONFIG to choose a config by hand."
    [ -f "$cfg" ] || die "$ACTIVE points at $cfg, which does not exist" \
        "Re-run init, or set STRATA_CONFIG to the config you want."

    [ -w "$STRATA_CONFIG_DIR" ] || printf \
        '  [!]  %s is read-only: the chat page cannot save its shared settings.\n' "$STRATA_CONFIG_DIR" >&2
    check_memlock
    check_cgroup_memory

    cd "$ROOT"
    set -- serve/server.py --engine strata --config "$cfg" --host "$STRATA_HOST" --port "$STRATA_PORT"
    [ -n "${STRATA_GPU:-}" ] && set -- "$@" --gpu "$STRATA_GPU"
    # Only when the environment sets one: it overrides whatever init wrote into the config, which is how a
    # key gets rotated without re-running init.  Otherwise the config's own key (if any) applies.
    api_key=$(resolve_api_key) || exit 1
    [ -n "$api_key" ] && set -- "$@" --api-key "$api_key"
    note "Starting Strata on $STRATA_HOST:$STRATA_PORT"
    note "config: $cfg"
    note "Loading the experts into RAM takes 1-3 minutes; the health check allows for it."
    exec "$PYTHON" "$@"
}

cmd_calibrate() {
    check_gpu
    cfg=$(resolve_config) || die "no active config - run init first"
    [ -f "$cfg" ] || die "$cfg does not exist - run init first"
    [ -w "$STRATA_CONFIG_DIR" ] || die "$STRATA_CONFIG_DIR must be writable to save the calibration"
    check_memlock

    # setup.py --calibrate works on the configs next to setup.py, so the file visits the image and comes back
    cd "$ROOT"
    name=$(basename "$cfg")
    cp "$cfg" "$ROOT/$name"
    "$PYTHON" setup.py --calibrate --no-start --data-dir "$STRATA_DATA"
    cp "$ROOT/$name" "$cfg"
    note "calibration saved to $cfg and to $XDG_CONFIG_HOME/strata/settings.json"
}

cmd_preflight() {
    cfg=$(resolve_config) || die "no active config - run init first"
    check_memlock
    cd "$ROOT"
    exec "$PYTHON" "$ROOT/docker/preflight.py" "$cfg"
}

case "${1:-serve}" in
    init)      shift; cmd_init "$@" ;;
    serve)     shift; cmd_serve "$@" ;;
    calibrate) shift; cmd_calibrate "$@" ;;
    preflight) shift; cmd_preflight "$@" ;;
    *)         exec "$@" ;;
esac
