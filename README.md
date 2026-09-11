# sgl-tui

A btop-style terminal dashboard for [SGLang](https://github.com/sgl-project/sglang) inference servers.
Live gauges, braille-style sparklines, KV-cache pool breakdown, latency percentiles — all in a dark neon terminal UI.

Built with [Textual](https://textual.textualize.io/). No server-side agents — it just scrapes the standard Prometheus `/metrics` endpoint.

## Requirements

The SGLang server must be launched with `--enable-metrics`:

```bash
python -m sglang.launch_server --model-path <model> --port 30000 --enable-metrics
```

## Install

> `sglang-tui` requires Python ≥ 3.10. With uv, the right interpreter is fetched automatically; pip/pipx use your active Python.

```bash
# ① uv (recommended — installs into ~/.local/bin)
uv tool install sglang-tui                          # from PyPI (once published)
uv tool install git+https://github.com/gybcb/sglang_tui.git
uv tool install --editable .                        # from a local clone (dev mode)
uv tool update-shell                                # if ~/.local/bin isn't on PATH

# ② pipx
pipx install sglang-tui

# ③ pip (system / user site)
pip install sglang-tui                              # or: pip install . from a clone

# one-off run without installing
uvx sglang-tui --url http://localhost:30000
```

Then run it from any directory:

```bash
sglang-tui --url http://localhost:30000
```

Developing in the repo? `uv sync && uv run sglang-tui …` works without any install.

## Options

| Flag        | Default                   | Description                        |
|-------------|---------------------------|------------------------------------|
| `--url`     | `http://localhost:30000`  | SGLang server base URL             |
| `--refresh` | `1.0`                     | Poll interval in seconds           |

## Keys

| Key          | Action                     |
|--------------|----------------------------|
| `q`          | Quit                       |
| `+` / `-`    | Refresh interval ±0.5s     |
| `h`          | Sparkline window 30s / 90s |
| `?` / `F1`   | Help overlay               |

## Panels

| Panel        | Source metrics |
|--------------|----------------|
| **REQUESTS** | `sglang:num_running_reqs`, `num_queue_reqs`, `num_grammar_queue_reqs`, `num_retracted_reqs`, request/abort rate |
| **THROUGHPUT** | `sglang:gen_throughput` (server-reported) + `realtime_tokens_total{mode=decode/prefill_compute}` sliding-window rates |
| **KV CACHE** | `token_usage`, `kv_used/evictable/available_tokens` stacked bar, `cache_hit_rate`, hicache L2 host pool |
| **LATENCY** | p50/p99 from `time_to_first_token_seconds`, `e2e_request_latency_seconds`, `inter_token_latency_seconds`, `queue_time_seconds` histograms (last-30s window) |
| **MEMORY** | `weight/kv_cache/graph_memory_usage_gb`, `utilization`, CUDA-graph state |
| **SPEC DECODE** | `spec_accept_rate`, `spec_accept_length`, speculative steps |

Rates use a sliding ~30s baseline because SGLang flushes counters in bursts (its log interval);
quantiles are computed from per-interval bucket deltas accumulated over the last 30s.

## Develop

```bash
uv run pytest            # parser/differ unit tests (incl. a real metrics fixture)
uv run sglang-tui --url http://localhost:30000
```

## Release (maintainers)

```bash
uv build                 # → dist/*.whl + *.tar.gz
uv publish               # first time: create an API token at pypi.org → Account → API tokens
```

Push a version tag to trigger the bundled `.github/workflows/publish.yml`, or publish manually as above.
Check the name is free first: `pip index versions sglang-tui` (rename to `sgl-tui` if taken).

## macOS LAN note

macOS 15+'s **Local Network privacy** gate blocks non-exempt binaries (uv/Homebrew/python.org
Pythons) from opening LAN sockets — with `Errno 65: No route to host`. The permission is attributed
per *app*: your terminal app (Terminal/iTerm/…) gets the prompt, while processes reparented to an
ad-hoc-signed daemon (e.g. tmux-like session managers such as herdr) can never get one.

**sgl-tui handles this automatically**: if the Python HTTP fetch is blocked, it transparently falls
back to the system `curl` (Apple's own binary, exempt from the gate). Nothing to configure. This
holds for system installs too — Local Network permission is attributed to the terminal app that
*launches* `sglang-tui`, not to the installed command itself.

To grant the permission anyway (native path, no curl):
1. Run `sglang-tui` once from your terminal app.
2. *System Settings → Privacy & Security → Local Network* → enable your terminal app.
3. Fully quit the terminal (**Cmd+Q**, not just close the window) and rerun.
4. Missing from the list? `tccutil reset LocalNetwork`, then repeat 1–3.

Verify: `uv run python -c "import socket;s=socket.socket();s.settimeout(3);s.connect(('<server-ip>',8002));print('LAN OK')"`
