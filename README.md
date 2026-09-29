# strata-docker

A Docker harness for [Strata](https://github.com/Niko1221/Strata) — Qwen3.8-Flash-Next running on one NVIDIA
GPU plus system RAM, serving an OpenAI- and Anthropic-compatible API.

The Strata sources are **not part of this repository** and you do not need to clone them. The build fetches
them itself from a pinned tag, as a named build context, so nothing here is copied into the Strata tree and
nothing there knows about this one.

```bash
git clone <this repository> && cd strata-docker
cp .env.example .env          # required; review STRATA_BIND, STRATA_GPU_DEVICE, STRATA_DATA_DIR
docker compose build
docker compose --profile init run --rm strata-init   # once: ~92 GB download, pack + MTP
docker compose up -d strata
```

`.env` is where the Strata version is pinned, and it is the only place — compose has no fallback, so it
refuses to do anything until the file exists.

Reference deployment: RTX 5060 Ti 16 GB, Threadripper 1920X (no AVX-512), 128 GB DDR4, IQ3_S at 262K context,
vision on, LAN-only with no API key.

---

## 1. Requirements

### On the host

| | |
|---|---|
| GPU | NVIDIA RTX 30/40/50, compute capability 8.0+. 12 GB VRAM or more (8 GB works, slowly) |
| Driver | **580 or newer** (CUDA 13.0). `nvidia-smi` tells you |
| RAM | Depends on the model: 48 GB for Q2_0/IQ2_XS, 60 GB for IQ3_XXS, **62 GB for IQ3_S** |
| Disk | ~92 GB for IQ3_S with vision, on NVMe. The engine reads experts from the GGUF while it runs |
| CPU | x86-64 with AVX2. AVX-512 is a little faster but not required |
| Software | Docker with BuildKit, Docker Compose v2.17+, and **nvidia-container-toolkit** |

Check the three things that actually block you:

```bash
nvidia-smi                                    # driver >= 580, and your cards
docker compose version                        # v2.17+ for additional_contexts
docker run --rm --gpus all ubuntu nvidia-smi  # the toolkit is wired up
```

If the last one fails, install
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and restart the Docker daemon.

Nothing else is needed. The Strata sources are fetched during the build.

---

## 2. Configure

```bash
cp .env.example .env
```

Open `.env` and set at least these three:

```bash
# Which card. `nvidia-smi -L` prints UUIDs, which survive a reboot reordering the PCI bus.
STRATA_GPU_DEVICE=GPU-1a2b3c4d-...

# Where the ~92 GB goes. Put it on the NVMe.
STRATA_DATA_DIR=/srv/strata/data

# Which host address the port is published on. There is no API key, so never 0.0.0.0
# on a machine with a public interface.
STRATA_BIND=192.168.1.10
```

Everything else in the file already has a working value, including `STRATA_SRC`, which pins the Strata
version. The full table is in §7.

The file is not optional: compose deliberately has no built-in fallback for `STRATA_SRC`, so without `.env`
every command stops with

```
required variable STRATA_SRC is missing a value: set STRATA_SRC in .env (cp .env.example .env)
```

That way the pinned version lives in exactly one place and cannot drift from a second copy in compose.

---

## 3. Build the image

```bash
docker compose build
```

This compiles the Strata engine and the vision encoder from source. **Expect 20–40 minutes** the first time;
later builds reuse the layer cache unless the sources change.

No GPU is needed for the build — the target architecture is a build argument, not something probed from the
machine.

Two build arguments are worth knowing about:

- **`CUDA_ARCHITECTURES`** (default `120` = RTX 50). Use `86` for RTX 30, `89` for RTX 40, or `"86;120"` to
  cover two cards. Each extra architecture costs build time and image size.
- **`STRATA_PORTABLE`** (default `ON`). `OFF` builds ggml with `-march=native`: slightly better code for the
  CPU you build on, but the image then runs **only** on that CPU — anywhere else the engine dies with SIGILL.
  Set it `OFF` only if you build on the machine that will run it and the image never travels.

The last build step runs a smoke test that asserts everything this harness assumes about the Strata tree. If
it fails, the build stops and each failure names the file to fix — see §9.

---

## 4. Install the model (once)

```bash
docker compose --profile init run --rm strata-init
```

This is the only step that writes to the data volume. It:

1. checks the GPU, driver, RAM, CPU and free disk
2. downloads the model (~84 GB for IQ3_S, resumable — rerun it if the connection drops)
3. downloads the vision encoder (~0.9 GB) if `STRATA_VISION=gpu`
4. prepares the pack (seconds — without AVX-512 this is the *native* pack, which reads experts straight from
   the GGUF instead of writing a 40 GB copy)
5. fetches and packs the MTP draft layer (~5 GB, roughly doubles output speed)
6. writes the run config to `/config` and records its path in `/config/active`

**Expect a few hours**, nearly all of it download. Everything is marked done as it completes, so an
interrupted run picks up where it stopped.

`init` is a separate command on purpose: a plain `docker run` never downloads anything, which is what lets
the data volume be mounted read-only afterwards.

---

## 5. Run it

```bash
docker compose up -d strata
docker compose logs -f strata
```

The server loads 50 GB of experts into RAM and pins part of it for the GPU. **The first 1–3 minutes the
machine may feel sluggish** — that is normal, and the health check allows 15 minutes before it complains.

Check it is up:

```bash
curl -s http://192.168.1.10:8080/health
curl -s http://192.168.1.10:8080/v1/models
docker compose ps                                   # STATUS should reach "healthy"
```

Ask it something:

```bash
curl http://192.168.1.10:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"strata","messages":[{"role":"user","content":"Write a haiku about GPUs."}],"max_tokens":256}'
```

The chat page, a live monitor (tokens/s, VRAM, PCIe traffic, context in use) and the connection details are
at `http://192.168.1.10:8080/`.

### Connecting clients

| API | Endpoint |
|---|---|
| OpenAI Chat Completions (streaming, tools) | `POST /v1/chat/completions` |
| Anthropic Messages (streaming, tools) | `POST /v1/messages` |
| Model list / health | `GET /v1/models`, `GET /health` |
| What it is doing right now | `GET /status` |
| Everything the monitor shows | `GET /metrics` |

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.168.1.10:8080/v1", api_key="none")
print(client.chat.completions.create(
    model="strata", messages=[{"role": "user", "content": "Hello!"}]).choices[0].message.content)
```

Any API key string is accepted — none is configured. Thinking effort is per request:
`"reasoning_effort": "none" | "low" | "medium" | "high"` on the OpenAI side, `"output_config": {"effort": ...}`
on the Anthropic side. The default is `high`.

---

## 6. Day to day

```bash
docker compose up -d strata            # start
docker compose stop strata             # stop
docker compose restart strata          # restart
docker compose logs -f strata          # follow the log
docker compose ps                      # health
```

The engine's own stderr goes to `<config volume>/strata-<tag>.log`.

### Optional: tune for this machine

```bash
docker compose --profile calibrate run --rm strata-calibrate
```

Measures three settings that depend on the machine more than on the model — the PCIe share, the speculative
decoding floor, and the CPU worker count — and saves the ones that beat the defaults by more than 3%. Takes
5–10 minutes with the machine busy. Worth doing once, especially on an older CPU where the defaults (measured
on a Ryzen 5 7600) are least likely to fit.

### A shell in the container

```bash
docker compose run --rm strata bash
docker compose run --rm strata python chat.py      # terminal chat
```

Any command that is not `init`, `serve` or `calibrate` is executed as given.

---

## 7. Configuration

All of it lives in `.env`. Changes to `STRATA_*` runtime variables need only a restart; changes to
`CUDA_ARCHITECTURES`, `STRATA_PORTABLE` or `STRATA_SRC` need a rebuild.

The "value" column is what `.env.example` ships. Compose supplies a fallback for every variable below
*except* `STRATA_SRC`, which is required — see §2.

### Sources

| Variable | Value in `.env.example` | |
|---|---|---|
| `STRATA_SRC` | `https://github.com/Niko1221/Strata.git#v0.1.21` | **Required.** Where Strata comes from: a git URL (fetched during the build) or a local path such as `./src`. **Pin a tag or commit, never a branch** — a branch changes the build without changing anything you can see |

### Build

| Variable | Default | |
|---|---|---|
| `CUDA_ARCHITECTURES` | `120` | `86` = RTX 30, `89` = RTX 40, `120` = RTX 50. `"86;120"` for both |
| `STRATA_PORTABLE` | `ON` | `OFF` = `-march=native`, ties the image to one CPU |

### Model

| Variable | Default | |
|---|---|---|
| `STRATA_FAMILY` | `qwen` | `qwen`, `swift` (thinks shorter), `coder` (half the experts, ~32 GB RAM) |
| `STRATA_MODEL` | `IQ3_S` | `Q2_0`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`. See the table below |
| `STRATA_CONTEXT` | `262144` | `8192` … `262144` |
| `STRATA_KV` | `int8` | `int8`, or `q4_0` for half the KV memory and slightly less precision |
| `STRATA_VISION` | `gpu` | `gpu`, `cpu`, `none` |

| Model | Download | Experts in RAM | RAM needed | Quality |
|---|---:|---:|---:|---|
| `Q2_0` | 66 GB | 34.0 GB | 48 GB | fastest |
| `IQ2_XS` | 68 GB | 35.5 GB | 48 GB | a little better |
| `IQ3_XXS` | 76 GB | 42.9 GB | 60 GB | better |
| `IQ3_S` | 84 GB | 50.3 GB | 62 GB | matches the full BF16 model |

### Conversation cache

| Variable | Default | |
|---|---|---|
| `STRATA_PROMPT_CACHE` | `24` | Checkpoints kept between requests, ~118 MB of RAM each (engine default: 6) |
| `STRATA_PROMPT_CACHE_EVERY` | `8192` | Also checkpoint every N freshly read prompt tokens (engine default: 16384) |

### Paths and network

| Variable | Default | |
|---|---|---|
| `STRATA_DATA_DIR` | `./strata-data` | The model volume, ~92 GB |
| `STRATA_CONFIG_VOLUME` | `./strata-config` | Config, calibration and log — a few kB |
| `STRATA_PORT` | `8080` | |
| `STRATA_BIND` | `127.0.0.1` | The **host** address the port is published on |
| `STRATA_GPU_DEVICE` | `0` | An index or a `GPU-<uuid>` from `nvidia-smi -L` |
| `STRATA_CONFIG` | — | Override which config `serve` uses, instead of `/config/active` |

---

## 8. Updating

### A new Strata version

Bump the pin in `.env` and rebuild:

```bash
STRATA_SRC=https://github.com/Niko1221/Strata.git#v0.1.22

docker compose build
docker compose up -d strata
```

The build runs the smoke test, so a harness that no longer matches the new Strata **fails the build** rather
than producing a broken image. Each failure names the file that depends on the broken assumption; fix it in
`docker/`, or go back to the previous pin.

The model files are untouched — they live on the data volume and `init` does not need to run again unless the
model itself changes.

To check compatibility without waiting for a build, clone the version you are considering and run the smoke
test against it directly:

```bash
git clone --depth 1 -b v0.1.22 https://github.com/Niko1221/Strata.git /tmp/strata-next
python3 docker/verify_assumptions.py --root /tmp/strata-next --no-engine
```

### Building a modified Strata

Only needed if you change Strata itself. Clone it anywhere and point `STRATA_SRC` at the path:

```bash
git clone https://github.com/Niko1221/Strata.git src
# in .env:
STRATA_SRC=./src
```

`src/` is gitignored here, so it stays its own repository with its own history. The build then picks up
whatever is in the working tree — no commit or push needed between rebuilds, which is the reason to use a
path rather than a URL.

### Changing model, context or cache settings

Editing `STRATA_PROMPT_CACHE` or `STRATA_PROMPT_CACHE_EVERY` alone still needs `init` to rerun, because they
are written into the config — but it is fast, since nothing gets downloaded again:

```bash
docker compose --profile init run --rm strata-init      # with /data writable
docker compose up -d strata
```

Changing `STRATA_MODEL` downloads the new model and keeps the old one. Delete the unwanted
`<data>/models/<tag>/` and `<data>/packs/<tag>/` by hand.

---

## 9. Troubleshooting

**`nvidia-smi is not in the container`** — the NVIDIA runtime is not wired up, or
`NVIDIA_DRIVER_CAPABILITIES` is missing `utility`. Test with
`docker run --rm --gpus all ubuntu nvidia-smi`.

**`RLIMIT_MEMLOCK is … KB` warning** — `--ulimit memlock=-1` is not taking effect. Compose sets it; if you are
running `docker run` by hand, add it. Without it the engine cannot pin its 50 GB expert arena, the failure is
*silent*, and every copy to the GPU crawls. This is the single most likely cause of "it works but it is slow".

**The container is OOM-killed while loading** — a container memory limit is set. `setup.py` reads
`/proc/meminfo`, which reports the *host's* RAM, so it cannot see the limit and will not warn you. The
entrypoint checks cgroups and warns, but the fix is to remove the limit.

**`Illegal instruction` / SIGILL at startup** — the image was built with `STRATA_PORTABLE=OFF` on a different
CPU than it runs on. Rebuild with `STRATA_PORTABLE=ON`.

**`no active config in /config`** — `init` has not run, or it ran against a different config volume. Run
`docker compose --profile init run --rm strata-init`.

**`/data is not writable` during init** — init needs the data volume read-write. The `strata-init` service
mounts it that way; you get this when running `init` through the `strata` service instead.

**The build stops on `verify_assumptions.py`** — the Strata tree has moved away from what this harness drives.
Read the named failures; they point at the file in `docker/` to update, or tell you to stay on the old pin.

**Slow prompt processing on long contexts** — expected on the first request of a conversation. A 262K prompt
takes minutes to read. Subsequent turns should be fast: that is what the conversation cache is for. If every
turn is slow, check `STRATA_PROMPT_CACHE` and whether the client is changing the start of the prompt (a
changing timestamp in the system prompt defeats the cache entirely).

**`not enough free disk space`** — IQ3_S needs ~92 GB free. `STRATA_DATA_DIR` must point at a volume that has
it.

---

## 10. How it works

### Why the engine is compiled

As of v0.1.21 the upstream release still publishes **`strata-windows-x64.zip` only** — no Linux asset (checked against the GitHub releases API). `setup.py`'s
`get_prebuilt()` gets a 404 and falls through to `build_engine()`, which would `apt-get` an 8–10 GB CUDA
toolkit inside the container at first run.

This harness is unaffected either way: `engine/BUILD.json` marks the compiled binaries as a local build, so
`get_prebuilt()` adopts them and never looks at the release. If a Linux asset does appear upstream, the only
cost of compiling anyway is build time.

The builder stage does that work instead, then writes `engine/BUILD.json` with `source: "local"` and
`setup.py`'s own source fingerprints. `setup.py` then adopts the binaries and skips the download, the compile
and the build-tool install entirely. The runtime stage carries no compiler and no CUDA toolkit — just the two
binaries and NVIDIA's CUDA libraries from pip.

### Why `init` is a separate command

The image's `CMD` is `serve`. A plain `docker run` starts the server and never downloads anything, so the
~92 GB data volume can be mounted `:ro` in normal operation. Verified: the engine's only file writes are behind
its `--dump-*` debug flags, which this harness never sets.

The config cannot live on that read-only volume, though — `server.py` derives `<config>.shared-settings.json`
from the config's own path and writes it there. Hence the separate, tiny `/config` volume, which also holds
the calibration and the log.

### Why the config path is recorded, not computed

The config name comes from `FAMILIES[family]["tag"] + model`, lowercased. Rebuilding that string in the
entrypoint meant two places had to agree with `setup.py`, and when they did not, `init` succeeded and `serve`
then failed to find a config that was sitting right there. Instead `patch_config.py` finds the file by glob
and writes its path to `/config/active`, which is the only thing `serve` reads.

### The smoke test

This harness drives `setup.py` and the engine through names that are not a public interface: module-level
constants, CLI flag spellings, the shape of the generated config. Strata and this repository move
independently, and two of those breakages are silent at run time.

So `docker/verify_assumptions.py` asserts all of them in a `RUN` layer at build time — 41 checks covering
`setup.py`'s constants and CLI, the engine's flag parsing in `generate.cpp`, `server.py`'s CLI, and the
fingerprints in `BUILD.json`. Every check names the file that depends on it.

### Layout

```
Dockerfile              builder (CUDA devel) -> runtime (ubuntu + venv)
docker-compose.yml      strata-init (profile) · strata · strata-calibrate (profile)
.env.example            every setting
strata.dockerignore     filters the Strata context — BuildKit looks for it here, not in the checkout
docker/
  entrypoint.sh         init | serve | calibrate | <anything>
  write_build_json.py   makes setup.py adopt the compiled engine
  write_pip_stamp.py    makes setup.py skip its pip step
  patch_config.py       moves the config to /config, adds the cache flags, records the active one
  verify_assumptions.py the build-time smoke test
src/                    optional local Strata checkout — gitignored, its own repository
```

---

## 11. Hardware notes

**Context is cheaper than it looks.** Only 12 of Strata's 48 layers are full attention; the other 36 are GDN
with a constant-size state. The KV cache at 262K is ~3.6 GB in RAM, of which `--kv-resident 32768` keeps about
450 MB in VRAM — a constant, whatever the context length. Doubling the context from 128K to 262K costs no VRAM
at all, so it does not push experts out of the cache.

**The conversation cache matters more than it looks** for agent work. A checkpoint is ~118 MB of running state
(the GDN recurrences, the PLE history, the QSA indexer tails) and lets a request resume instead of re-reading
a long prompt — the difference between seconds and minutes per turn. The chain's root, the end of the system
prompt, is pinned; the rest rotates by least recent use. The engine's default of 6 is sized for a 64 GB
desktop, so `.env` raises it to 24 (about 2.8 GB of RAM).

Note that the cache holds one branch of history at a time, because the KV cache is a single arena. Alternating
between two long conversations means re-reading; the shared system prompt always survives.

**Without AVX-512** (Zen 1/2, older Intel) `setup.py` picks the *native* pack for every quant, which skips the
one-time ~40 GB Q2_0 expert conversion and saves that much disk. The canonical Q2_0 pack is AVX-512-only and
the engine refuses to start on it — so never bake a pack built on an AVX-512 machine into a non-AVX-512
deployment.

**Do not use `--cpuset-cpus`.** The engine pins worker threads with `pthread_setaffinity_np` to CPU numbers it
reads from `/sys/devices/system/cpu/*/topology`. Inside a cpuset those pins land outside the allowed mask.

**A second GPU probably will not help.** Since v0.1.21 Strata can split layers across cards (`--gpus 0,1`),
and each card then caches experts for its own layers — roughly twice the experts on two cards. But
`docs/MULTI_GPU.md` lists what it does not yet work with, and three of them matter here:

- **no `--vision`** — images are off in a split
- **no KV streaming (`--kv-resident`)** — which is what makes a 262K context cheap in VRAM
- **no mid-prompt checkpoints (`--prompt-cache-every`)** — only the turn-boundary ones survive

For a vision-enabled 262K deployment the split is therefore not usable, and this harness pins a single card
(`STRATA_GPU_DEVICE`) on purpose. It is worth revisiting for a text-only, shorter-context setup: to try it,
set `"gpu": [0, 1]` and `"layer_split": "auto"` in the config on the config volume and restart. Every card
needs compute capability 8.0+, and `CUDA_ARCHITECTURES` must then cover both (e.g. `"86;120"`).

**On multi-die CPUs** (Threadripper, EPYC) set memory interleaving to UMA/Distributed rather than NPS2. The
expert arena is one large mapping read by pinned worker threads; with NUMA nodes exposed, half of them read
across the interconnect. Also make sure every memory channel is populated — bandwidth matters more here than
clock speed.
