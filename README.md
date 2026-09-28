# PulseX — text to image on the Snapdragon NPU

[![Latest release](https://img.shields.io/github/v/release/anvandaren-matrix/pulsex-zimage)](https://github.com/anvandaren-matrix/pulsex-zimage/releases/latest)
[![License: MIT](https://img.shields.io/github/license/anvandaren-matrix/pulsex-zimage)](LICENSE)

PulseX turns a sentence into a picture with **Z-Image-Turbo**, running the diffusion
transformer on the Hexagon NPU of a Snapdragon X laptop (Windows on ARM) through the PulseX
fork of ggml-hexagon. The command is `zimage`; the window and the UI say PulseX.

| Stage | Model | Runs on |
|---|---|---|
| Text encoder | Qwen3-4B, GGUF Q4_0 (`hidden_states[-2]`) | CPU from the file cache (default) or NPU |
| Diffusion transformer | Z-Image-Turbo, GGUF Q4_0, 4 steps | NPU (HTP0) |
| Decoder | taef1 (C++/NEON) or the full Flux VAE | CPU |
| Upscaler (512 → 1024) | Real-ESRGAN x4, QNN context via ONNX Runtime | NPU |

Typical times on an X Plus: 512 + upscale ~19 s cold and 10–12 s in the REPL; native 1024 ~40 s.

## Examples

All five made on the NPU at native 1024 × 1024 — the prompt is exactly what was typed. Two of
them are in Swedish; `-Enrich` adds the English words the model needs (see *Making images*).

| | |
|---|---|
| ![Rapids under the midnight sun](docs/examples/rapids.jpg) | ![A moose under the northern lights](docs/examples/moose.jpg) |
| `zimage make "forsarna i kalixälven under midnattssolen" -Size 1024 -Enrich -Seed 11` | `zimage make "en älg i en snöig granskog under norrskenet" -Size 1024 -Enrich -Seed 11` |
| ![A woman by a window](docs/examples/woman.jpg) | ![A cat on a windowsill](docs/examples/cat.jpg) |
| `zimage make "portrait of a young woman with freckles by a window, soft window light" -Size 1024 -Enrich -Seed 4242` | `zimage make "a tabby cat on a windowsill, winter light, snow outside" -Size 1024 -Enrich -Seed 4242` |
| ![An old man with a hat](docs/examples/man-with-hat.jpg) | |
| `zimage make "portrait of an old man with a felt hat and a grey beard, warm evening light" -Size 1024 -Enrich -Seed 11` | |

The NPU is not bit-deterministic, so the same command gives a nearly — not exactly — identical
image.

## Setup

**Hardware.** Tested on Snapdragon X Plus (X1P42100, Hexagon NPU v73). Snapdragon X Elite and
the other X1 chips have the same NPU and should work. Snapdragon X2 is untested: its NPU skel
(`libggml-htp-v81.so`) is built and picked automatically, but the kernels were tuned on v73 and
the Real-ESRGAN QNN context is compiled for X1 and has to be recompiled for X2.

**Requirements:** a Snapdragon X laptop (Hexagon NPU v73 or newer) on Windows 11 ARM64, the
[Microsoft Visual C++ Redistributable for ARM64](https://aka.ms/vs/17/release/vc_redist.arm64.exe)
and Python 3.12 for ARM64 with `pip install -r requirements.txt`.

**Download.** Clone the repository, or take the zip from the [latest release](https://github.com/anvandaren-matrix/pulsex-zimage/releases/latest) — both contain everything in this folder, binaries included.

**Prebuilt binaries.** The release repository ships them in `bin/` (llama-server,
zimage-dit-stream, ggml, taef1_decode and the DSP skel `libggml-htp-v7x.so` + catalog), plus
`zimage.exe` next to the scripts. To build them yourself: the PulseX fork of ggml-hexagon
(`bld_wos.bat`; `zimage_stream.cpp` builds there as `examples/zimage_dit`), `bld_cli.bat` for
`zimage.exe`, and `taef1_decode.cpp` from the PulseX tree.

> **The DSP skel is self-signed.** Windows only loads it on a machine set up to accept
> self-signed NPU modules (a developer setup). On a stock machine the NPU part is refused; that
> needs a Microsoft-attestation-signed skel, which this project does not have yet.

**Models** (not included — see License). Put them on a fast SSD (an SD card made weight loading
23× slower) in one folder:

| File | What / where from |
|---|---|
| `z-image-turbo-q4_0.gguf` | Tongyi-MAI/Z-Image-Turbo, converted with `tools/convert_zimage_to_gguf_q4.py` in the ggml-hexagon fork |
| `qwen3-4b-zimage-q4_0.gguf` | the Qwen3-4B text encoder of Z-Image-Turbo, converted and quantized to Q4_0 with llama.cpp |
| `taef1/diffusion_pytorch_model.safetensors` | madebyollin/taef1 |
| `vae/ae.safetensors` | optional, the full Flux VAE (only for `-Vae full`) |
| Real-ESRGAN x4 QNN context (`esrgan_x4.bin` + `.wrap.onnx`) | Qualcomm AI Hub, Real-ESRGAN-x4plus, compiled for the device |

**Configure.** Copy `zimage.example.json` to `zimage.json` and set the paths (`zimage.json` is
local and not checked in). `zimage config` lists every path with `[ok]` or `[MISSING]`.

## Making images

```
zimage make "a lighthouse at dusk"
```
512 × 512, upscaled to 1024 × 1024. Saved as `zimage_0001.jpg`, then `_0002` … in the current
folder — an earlier image is never overwritten.

| What | Example |
|---|---|
| plain 512, fastest — for trying prompts | `zimage make "a lighthouse at dusk" -NoUp` |
| force the upscaler on (if the config has it off) | `zimage make "a lighthouse at dusk" -Up` |
| native 1024 — most detail, for keepers | `zimage make "a lighthouse at dusk" -Size 1024` |
| another image for the same text | `zimage make "a lighthouse at dusk" -Seed 7` |
| exact file name (.jpg or .png) | `zimage make "a lighthouse at dusk" -Out lighthouse.png` |
| JPEG quality | `zimage make "a lighthouse at dusk" -Quality 90` |
| add material words to the prompt | `zimage make "portrait of an old fisherman" -Enrich` |
| see where the time goes | `zimage make "a lighthouse at dusk" -Timings` |

`-Enrich` appends words the model responds to (from `cues.json`), for example
`portrait` → "skin pores, natural skin texture". The prompt that is actually sent is printed.
It adds detail but can change the composition.

**Swedish prompts.** The model gets the scene and mood from Swedish but drops specific nouns:
"en älg i en snöig granskog under norrskenet" gave a snowy forest without the moose or the
aurora. With `-Enrich` / `--enrich`, known Swedish words also add the English noun, first, before the
material words — then the moose and the aurora appear. The table covers the north (`älg`,
`norrsken`, `granskog`, `snöig`, `stuga`, `fors`/`forsarna`, `midnattssol`, `Kalixälven` …),
cafés (`kafé`, `fika`, `ljusslinga`, `pappersstjärna` …), people, sea and town, with common
inflections. English prompts remain the most reliable.

The café words cover all three kinds of café: the old timber house (`kakelugn`, `förstukvist`,
`spetsgardin`, `emaljskylt`, `julstjärna`, `thonetstol`, `blommig tapet`, `vaxduk` …), the
small-town café (`konditori`, `prinsesstårta`, `semla`, `kardemummabulle`, `porslin` …) and the
rock'n'roll diner (`hamburgerbar`, `jukebox`, `neonskylt`, `LP-skivor`, `jänkare` …).

A prompt that only lists things tends to become a close-up still life. Say the view first:
`interiörbild av …` (wide interior shot), `vidvinkel` / `helbild` (wide-angle) or `närbild`
(close-up).

```
zimage make "interiörbild av ett gammalt konditori med kakelugn, spetsgardiner och prinsesstårta på porslin" -Enrich
```

```
zimage make "ett gammalt kafé med ljusslingor och en pappersstjärna i fönstret" -Enrich
```

## Text on the picture

The model cannot spell reliably, so text is drawn afterwards, correctly spelled (å, ä, ö work).

| What | Example |
|---|---|
| a title | `zimage make "a cozy café interior" -Title "Open House"` |
| title and a smaller line under it | `zimage make "a cozy café interior" -Title "Open House" -Subtitle "Saturday 10-14"` |
| at the top instead of the bottom | `zimage make "a cozy café interior" -Title "Open House" -TextPos top` |
| on an existing photo (no new image) | `zimage text photo.jpg -Title "Open House" -Subtitle "Saturday 10-14"` |

`zimage text` writes `photo_text.jpg` next to the original unless `-Out` is given.

Everything combines — native 1024, material words and text in one go:

```
zimage make "a cozy café interior" -Size 1024 -Enrich -Title "Open House" -Subtitle "Saturday 10-14"
```

The same in the REPL (the order of the options does not matter; the text ends at the next `--`):

```
a cozy café interior --1024 --enrich --title Open House --sub Saturday 10-14
```

`--1024` and `--enrich` stay in effect for the following images; the text is for this image only.

## Interactive mode (REPL)

```
zimage repl
```
The models stay loaded, so from the second image on you only pay for the generation.
Type a prompt and press Enter. Options go at the end of the line:

| Line | What it does |
|---|---|
| `a red fox in the snow` | an image with the current settings |
| `a red fox in the snow --1024` | native 1024 from now on (`--512` goes back) |
| `a red fox in the snow --size 1024` | the same, long form |
| `a red fox in the snow --noup` | plain 512 from now on (`--up` turns the upscaler back on) |
| `a red fox in the snow --seed 42` | restart the seed sequence at 42 (each image adds 1) |
| `a red fox in the snow --enrich` | material words on from now on (`--noenrich` turns them off) |
| `a red fox in the snow --title Winter Sale --sub Friday 6 pm` | text on this image only |
| `a red fox in the snow --title Winter Sale --top` | text at the top, this image only |
| `--1024` | change a setting without making an image |

Up/Down recalls earlier lines. `Ctrl+D`, `quit` or closing the window ends the session and frees
the NPU memory. After 60 idle seconds the image model is unloaded on its own (see Memory).

## Other commands

| Command | What it does |
|---|---|
| `zimage serve` | start the text encoder and leave it running (`make` then skips the start-up) |
| `zimage make "..." -Keep` | leave the encoder running after this image |
| `zimage stop` | stop everything, free NPU memory, drop the model files from the cache |
| `zimage status` | what is running, and how much memory is cached |
| `zimage config` | the config file, every path with `[ok]`/`[MISSING]`, the defaults |
| `zimage version` | the version |
| `zimage help` | all commands and options |

Rarely needed: `-Steps N` (the model is trained for 4), `-Vae full` (the reference decoder,
slower), `-EncoderOn npu` (see Memory), `-NoServer` (no warm encoder, slower fallback),
`-Pause` (wait for Enter before closing — for shortcuts), `-Cascade` (1024 via a 512 pass,
experimental, with `-Tail N`).

## Configuration (`zimage.json`)

Every key is documented in `zimage.example.json` (`_help`). Command-line options win over it.

| Key | Default | Meaning |
|---|---|---|
| `defaults.size` | `512` | `512` or `1024` |
| `defaults.upscale` | `true` | 512 images go through Real-ESRGAN x4 to `upscale_to` |
| `defaults.upscale_detail` | `0.5` | 0..1: share of the Real-ESRGAN result; the rest is a plain Lanczos enlargement. 1 is sharpest but can look plastic, 0 is softest. |
| `defaults.shift` | `1.0` | sigma shift; 1.0 (linear [1, .75, .5, .25]) gave more skin and fur detail than 3.0 on every test prompt |
| `defaults.encoder` | `cpu` | `cpu` or `npu`, see Memory |
| `defaults.enrich` | `false` | material words on by default |
| `defaults.unload_after_s` | `60` | REPL: unload the image model after this many idle seconds (0 = never) |
| `defaults.purge_on_start` / `purge_on_exit` | `standby` | empty Windows' standby cache when the REPL starts / on stop and exit: `standby`, `full` (also trim every process's working set) or `off` |

## Memory

Everything the NPU holds is `rpcmem`, which Windows counts as **committed** memory. Measured on
an X Plus (16 GB):

| Part | Committed |
|---|---|
| Diffusion transformer weights | 3.3 GB |
| its graph buffers (512 / 1024) | 0.15 / 0.6 GB |
| Text encoder on the NPU | 2.8 GB |
| Text encoder on the CPU (`--no-repack`, mmap) | 0.5 GB |
| Real-ESRGAN session | 0.9 GB — released after every upscale, reopened behind the next DiT run |
| taef1 decoder at 1024 | 0.8 GB, only while decoding |

That is why the encoder defaults to the CPU, and why the REPL unloads the image model after an
idle minute (the next image then takes a few seconds longer).

To give PulseX room, the REPL can empty Windows' whole standby cache ("Cached" in Task Manager)
when it starts and when it closes. That call needs administrator rights, so it only happens when
PulseX runs as administrator — set the shortcut to *Properties > Shortcut > Advanced > Run as
administrator*. Otherwise PulseX drops only its own model files from the cache.

Mapping the weights from the file does not help: the Windows FastRPC driver can import a
read-only file view into the DSP (`remote_register_buf_attr2` with
`FASTRPC_ATTR_IMPORT_BUFFER | FASTRPC_ATTR_READ_ONLY`, then `rpcmem_to_fd` and `fastrpc_mmap` —
the path QNN uses), and the DSP reads it correctly, but the driver charges the full size to
committed memory while it is mapped (measured +254–258 MB for a 256 MB view). Anything the NPU can
see costs committed memory; only a model that stays on the CPU side can live purely in the file
cache.

## Files

| File | Role |
|---|---|
| `zimage.ps1` | the CLI: commands, help, encoder-server lifecycle, cache cleanup |
| `zimage.py` | the engine driver: encoder → DiT → decoder → upscaler, REPL, text overlay |
| `zimage_stream.cpp` | `zimage-dit-stream.exe`, the DiT on ggml-hexagon (built by the ggml-hexagon tree) |
| `zimage_launch.c`, `zimage.rc`, `zimage.ico`, `bld_cli.bat` | the `zimage.exe` launcher (also `--drop-cache` and `--purge`) |
| `enrich.py`, `cues.json` | prompt enrichment |
| `zimage.example.json` | configuration template |
| `tools/` | measurement and diagnostic scripts (memory sampler, profilers, probes) |
| other `*.py` / `*.cpp` | development references and golden-data checks from the port |

## License

The PulseX code in this repository is released under the [MIT License](LICENSE).

The models are **not** included and keep their own licenses — check them before you use or
share images commercially: Z-Image-Turbo (Tongyi-MAI), Qwen3-4B (Qwen), taef1 (madebyollin),
the Flux VAE (Black Forest Labs) and Real-ESRGAN (Xintao Wang et al.). ggml-hexagon and
llama.cpp are MIT-licensed; ONNX Runtime is MIT-licensed; the Qualcomm QNN/QAIRT runtime is
under Qualcomm's own license.
