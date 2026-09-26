# kev-mapping-experiment justfile

KEV_PORT := env_var_or_default("KEV_PORT", "8009")
# Kev sizes that fit a 32 GB Mac. 27b has no Mac path (LongLeaf a100-gpu).
KEV_SIZES := "0.8b 4b 9b"

_default:
    @just --list

# Install the experiment env (python 3.12)
[group('setup')]
install:
    uv sync --group dev

# Clone Kev (if needed) and build its serving env (torch + MLX backend)
[group('setup')]
kev-install:
    #!/usr/bin/env bash
    set -euo pipefail
    [ -d kev ] || git clone --depth 1 https://github.com/jaredpalmer/kev.git kev
    cd kev && uv sync --extra serve

# Serve one Kev size on KEV_PORT (foreground) through scripts/serve_kev.py, which caps MLX's buffer
# cache (plain kev.serve grows without bound on Metal). First run downloads adapter + Qwen3.5 base.
# Stop omlx-served models before serving 9b: ~22 GB resident.
[group('kev')]
serve SIZE="4b":
    cd kev && KEV_DTYPE=bf16 uv run --extra serve python ../scripts/serve_kev.py --run jaredpalmer/kev-{{SIZE}} --port {{KEV_PORT}}

# Serve JevK5 (Apache-2.0, Qwen3.5-4B + LoRA, letters A-P readout: k<=15 candidates) on KEV_PORT.
# Env: `just jevk5-install`. MODEL: alibiserikbay/JevK5 or alibiserikbay/JevK5-2B.
[group('kev')]
serve-jevk5 MODEL="alibiserikbay/JevK5":
    jevk5/.venv/bin/jevk5-serve --model {{MODEL}} --port {{KEV_PORT}}

[group('setup')]
jevk5-install:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p jevk5 && cd jevk5 && uv venv --python 3.12 -q
    uv pip install "jevk5[fast] @ git+https://github.com/allebee/jevk5@v0.2.0"

# Show which checkpoint/backend the running server has loaded
[group('kev')]
models:
    curl -s localhost:{{KEV_PORT}}/v1/models | python3 -m json.tool | head -40

# Build gold set, Mondo/SNOMED term tables, candidate retrieval, and tiered eval items
[group('experiment')]
build *ARGS:
    uv run kevmap build {{ARGS}}

# Run every eval item against the served model (resumable). MODEL names the results dir.
[group('experiment')]
run MODEL *ARGS:
    uv run kevmap run --model {{MODEL}} --port {{KEV_PORT}} {{ARGS}}

# Score every results/<model>/responses.jsonl and write metrics + the comparison table
[group('experiment')]
eval *ARGS:
    uv run kevmap eval {{ARGS}}

# Emit results/<model>/mappings.sssom.tsv from a run
[group('experiment')]
sssom MODEL:
    uv run kevmap sssom --model {{MODEL}}

[group('dev')]
test:
    uv run pytest -q

[group('dev')]
lint:
    uv run ruff check src tests
