# 09-28: README - 480p (848x480), aldern i galleriet, packcache + viktbudget i Configuration/Memory.
#   python tools/patch_readme_480p.py
import io

P = r"C:\PulseCore\PulseX\zimage_dit\README.md"
s = io.open(P, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"


def rep(old, new):
    global s
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    if s.count(old) != 1:
        raise SystemExit("ankare %d ggr: %r" % (s.count(old), old[:70]))
    s = s.replace(old, new)


MOOSE = ("a large bull moose walking on a snowy forest path at night, snow-covered spruce trees, "
         "crescent moon in a starry sky, warm light from a cabin far behind the trees, cinematic")

rep("Typical times on an X Plus: 512 + upscale ~19 s cold and 10–12 s in the REPL; native 1024 ~40 s.",
    "Typical times on an X Plus: 512 + upscale ~19 s cold and 10–12 s in the REPL; wide 480p\n"
    "(848 × 480) ~17 s cold and ~16 s in the REPL; native 1024 ~40 s.")

rep("""The NPU is not bit-deterministic, so the same command gives a nearly — not exactly — identical
image.""", """**Wide 480p** — 848 × 480, the frame size many of the newer video models work in:

![A moose on a snowy forest path at night, 848 x 480](docs/examples/moose-480p.jpg)

`zimage make "%s" -Size 480p -Seed 7`

The NPU is not bit-deterministic, so the same command gives a nearly — not exactly — identical
image.""" % MOOSE)

rep("""| native 1024 — most detail, for keepers | `zimage make "a lighthouse at dusk" -Size 1024` |
""", """| native 1024 — most detail, for keepers | `zimage make "a lighthouse at dusk" -Size 1024` |
| wide 480p (848 × 480), no upscaler | `zimage make "a lighthouse at dusk" -Size 480p` |
| any width × height | `zimage make "a lighthouse at dusk" -Size 1280x720` |
""")

rep("""`-Enrich` appends words the model responds to""", """**Sizes.** `512` and `1024` are square; `480p` is 848 × 480 and `720p` is 1280 × 720; any
`WxH` works when both sides are divisible by 16 (the decoder works in 8 × 8 pixel blocks and the
transformer in 2 × 2 of those), from 256 to 2048. That is why the 854 × 480 of video is 848 × 480
here. Only 512 goes through the x4 upscaler, which is compiled for exactly 512 × 512. Cost follows
the pixel count: 848 × 480 is 1.6× the pixels of 512 × 512 and 40 % of 1024 × 1024.

`-Enrich` appends words the model responds to""")

rep("""| `a red fox in the snow --size 1024` | the same, long form |
""", """| `a red fox in the snow --size 1024` | the same, long form |
| `a red fox in the snow --480p` | wide 848 × 480 from now on (`--size 1280x720` for any W×H divisible by 16) |
""")

rep("""| `defaults.size` | `512` | `512` or `1024` |""",
    """| `defaults.size` | `512` | `512`, `1024`, `"480p"` (848 × 480), `"720p"` or `"WxH"` with both sides divisible by 16 |""")
rep("""| `defaults.purge_on_start` / `purge_on_exit` | `standby` |""",
    """| `defaults.npu_weight_budget_mb` | `0` | NPU memory for the image model's weights: `0` streams them through two slots (~0.2 GB), `-1` keeps all 3.3 GB on the NPU. See Memory |
| `defaults.pack_cache` | `true` | store the weights once in the NPU's own tiled layout next to the model (`.hexpack`, 3.5 GB) and just copy them from then on |
| `defaults.purge_on_start` / `purge_on_exit` | `standby` |""")

rep("""| Diffusion transformer weights | 3.3 GB |
| its graph buffers (512 / 1024) | 0.15 / 0.6 GB |""",
    """| Diffusion transformer weights | 0.2 GB by default (streamed, see below); 3.3 GB with `npu_weight_budget_mb: -1` |
| its graph buffers (512 / 480p / 1024) | 0.15 / 0.24 / 0.6 GB |""")

rep("""To give PulseX room, the REPL can empty""", """**The image model's weights stream.** The transformer has 34 blocks of ~97 MB. By default only
two block-sized slots live on the NPU; each step copies the next block's weights into the free slot
while the NPU computes on the other. The source is `z-image-turbo-q4_0.hexpack`, a copy of the
weights already in the NPU's tiled layout — written next to the model the first time (~5 s,
3.5 GB) and rebuilt by itself when the model file changes. It is read through Windows' file cache,
which shows as *Cached*, not *Committed*, and which Windows can take back at any time. Measured on
an X Plus, 4 steps, identical images:

| | all weights on the NPU | streamed (default) |
|---|---|---|
| 512 × 512 | 9.7 s, +3.75 GB committed | 8.8 s, +0.64 GB |
| 1024 × 1024 | 36.4 s, +4.6 GB | 35.6 s, +1.9 GB |

`npu_weight_budget_mb` sits anywhere between: blocks stay on the NPU, in the order they run, while
they fit under the budget, and the rest stream.

To give PulseX room, the REPL can empty""")

rep("""| `zimage_stream.cpp` | `zimage-dit-stream.exe`, the DiT on ggml-hexagon (built by the ggml-hexagon tree) |""",
    """| `zimage_stream.cpp` | `zimage-dit-stream.exe`, the DiT on ggml-hexagon (built by the ggml-hexagon tree): weight budget, streaming slots, `.hexpack` cache |""")

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("OK")
