# -*- coding: utf-8 -*-
"""zimage - gor en bild ur en prompt, helt pa NPU:n (utom VAE-avkodningen).

    python zimage.py "a photo of a cat" -o katt.jpg
    python zimage.py "a cat on a beach" -o katt.jpg --size 1024 --seed 7

Kedjan: prompt -> Qwen3-4B (cap_feats = hidden_states[-2]) -> Z-Image-Turbo DiT
-> VAE -> JPEG. Encoder och DiT kor pa HTP0 i Q4_0; VAE:n kor i torch pa CPU
eftersom NPU-VAE:n inte racker till pa 1024 (den vill allokera 2,4 GB i en buffert).
"""
import argparse, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np

# --- sokvagar och forval: zimage.json (ZIMAGE_CONFIG pekar om) ---------------
# zimage.json ar ENDA kallan for sokvagar (kopiera zimage.example.json). Relativa sokvagar tolkas
# mot json-filens katalog, {models} = paths.models. Forvalen nedan galler for nycklar filen saknar.
HERE = Path(__file__).resolve().parent
_DEFAULTS = {"out": "zimage.jpg", "size": 512, "steps": 4, "shift": 1.0, "seed": 1234, "quality": 95,
             "vae": "taef1", "encoder": "cpu", "upscale": True, "upscale_to": 1024, "enrich": False,
             "unload_after_s": 60, "purge_on_start": "standby", "purge_on_exit": "standby",
             "upscale_detail": 0.5}
_PATH_KEYS = ("models", "dit", "encoder", "tokenizer", "vae_full", "taef1", "bin", "esrgan_x4", "qnn_runtime")


def load_config():
    p = Path(os.environ.get("ZIMAGE_CONFIG", str(HERE / "zimage.json")))
    if not p.exists():
        sys.exit("PulseX: no config file at %s - copy zimage.example.json to zimage.json and set the paths."
                 % p)
    user = json.loads(p.read_text(encoding="utf-8"))
    missing = [k for k in _PATH_KEYS if k not in user.get("paths", {})]
    if missing:
        sys.exit("PulseX: %s lacks paths.%s (see zimage.example.json)" % (p, ", paths.".join(missing)))
    defaults = dict(_DEFAULTS)
    defaults.update(user.get("defaults", {}))
    server = {"port": 8099}
    server.update(user.get("server", {}))
    base = p.parent

    def res(v):
        q = Path(v)
        return q if q.is_absolute() else (base / q).resolve()

    models = res(user["paths"]["models"])
    paths = {k: (models if k == "models" else res(str(v).replace("{models}", str(models))))
             for k, v in user["paths"].items()}
    return paths, defaults, server, p


PATHS, DEFAULTS, SERVER, CONFIG_FILE = load_config()
from enrich import enrich as enrich_prompt          # gamla ZImages materialordstabell (cues.json)
# Sigma-shift for DiT-motorn (zimage_stream.cpp laser ZI_SHIFT; utan den kor den 3,0).
# 1,0 = gamla ZImages linjara schema - mer hud/pals-detalj, matt 09-27. Miljon vinner.
os.environ.setdefault("ZI_SHIFT", "%g" % float(DEFAULTS.get("shift", 1.0)))
MODELS = PATHS["models"]
BIN    = PATHS["bin"]
DIT    = PATHS["dit"]
ENC    = PATHS["encoder"]
TOK    = PATHS["tokenizer"]
VAE    = PATHS["vae_full"]
TAEF1X = PATHS["taef1"]
ESRGAN = PATHS["esrgan_x4"]
QNN_RT = PATHS["qnn_runtime"]

# Z-Image: VAE skalar 8x, patch 2, 16 latentkanaler, cap_feats ar 2560-dim.
INCH, PATCH, VAEF, CAPD = 16, 2, 8, 2560
SCALE, SHIFT, GN_G, EPS = 0.3611, 0.1159, 32, 1e-6
QWEN_PENULT_LAYER = 35          # Qwen3-4B har 36 lager; hidden_states[-2] = ingang till det sista
ENC_ON = "NPU" if os.environ.get("ZI_ENCODER_ON", str(DEFAULTS.get("encoder", "cpu"))) == "npu" else "CPU"   # for etiketten


def log(msg, t0=None):
    print(f"[zimage] {msg}" + (f"  ({time.perf_counter()-t0:.1f} s)" if t0 else ""), flush=True)


SERVER_URL = os.environ.get("ZIMAGE_SERVER", "http://127.0.0.1:%d" % int(SERVER["port"]))


def _server_post(ep, payload, timeout=180):
    import json as _json, urllib.request, urllib.error
    req = urllib.request.Request(SERVER_URL + ep, data=_json.dumps(payload).encode("utf8"),
                                 headers={"Content-Type": "application/json"})
    try:
        return _json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except (urllib.error.URLError, OSError):
        return None


def _server_template(prompt):
    """Servern applicerar chattmallen. VERIFIERAT teckenidentisk med HF:s
    apply_chat_template(..., enable_thinking=True) - sa vi slipper importera transformers,
    vilket ensamt kostade 11 s per bild."""
    r = _server_post("/apply-template", {"messages": [{"role": "user", "content": prompt}]}, timeout=30)
    if r is None:
        return None
    return r.get("prompt") if isinstance(r, dict) else r


def _server_cap_feats(text, verbose):
    """Fraga en varm llama-server. Returnerar None om ingen svarar.
    ⚠ Servern strippar INTE avslutande radslut (det gor bara `llama-embedding -f`), sa
    mallen skickas ORORD - en extra radbrytning skulle bli token 271 ("

") i stallet
    for 198 ("
") och ge en annan sista vektor."""
    import json as _json, urllib.request, urllib.error
    try:
        req = urllib.request.Request(SERVER_URL + "/embeddings",
                                     data=_json.dumps({"input": text}).encode("utf8"),
                                     headers={"Content-Type": "application/json"})
        r = _json.loads(urllib.request.urlopen(req, timeout=180).read())
    except (urllib.error.URLError, OSError):
        return None
    emb = r[0]["embedding"] if isinstance(r, list) else r["data"][0]["embedding"]
    a = np.array(emb, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(-1, CAPD)
    if verbose:
        log(f"cap_feats {a.shape} std={a.std():.2f} (server)")
    return a


def cap_feats(prompt, workdir, verbose):
    """Prompt -> [Scap, 2560] float32. Varm server om den finns, annars llama-embedding."""
    text = _server_template(prompt)
    if text is not None:
        a = _server_cap_feats(text, verbose)
        if a is not None:
            return a

    # Ingen server: mallen maste byggas lokalt, och da behovs tokenizern.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOK))
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=True)
    # TVA fallor i llama-embedding, bada tysta:
    #   * '-f' strippar en avslutande radbrytning - mallens sista token AR en '\n',
    #     sa vi skriver en extra. Utan den blir det 12 tokens i stallet for 13.
    #   * split_lines delar prompten pa '\n' och kor varje rad som EGEN prompt, vilket
    #     gav 10 tokens. --embd-separator satter nagot som inte forekommer.
    pf = workdir / "prompt.txt"
    pf.write_text(text + "\n", encoding="utf8", newline="")

    env = {**os.environ,
           "ADSP_LIBRARY_PATH": str(BIN),          # utan denna: 0x80000406 + segfault
           "QWEN_EMBD_LAYER": str(QWEN_PENULT_LAYER)}
    cmd = [str(BIN / "llama-embedding.exe"), "-m", str(ENC), "-f", str(pf),
           "--pooling", "none", "--embd-normalize", "-1",
           "--embd-output-format", "array", "--embd-separator", "<#nosplit#>",
           "-c", "512"]
    # Samma val som servern (zimage.ps1 -EncoderOn): cpu laser vikterna ur filcachen (+0,5 GB
    # commit), npu kopierar dem till rpcmem (+2,8 GB).
    if os.environ.get("ZI_ENCODER_ON", str(DEFAULTS.get("encoder", "cpu"))) == "npu":
        cmd += ["-ngl", "99", "--device", "HTP0"]
    else:
        cmd += ["-ngl", "0", "--device", "none", "--no-repack"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit("encodern misslyckades:\n" + r.stderr[-1500:])
    a = np.array(json.loads(r.stdout), dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(-1, CAPD)
    if verbose:
        log(f"cap_feats {a.shape} std={a.std():.2f}")
    return a


def build_inputs(cap, px, seed, workdir):
    """Skriver allt harnessen laser. cap_ids ar 1..Scap; img_ids0 ar Scap+1."""
    lh = px // VAEF
    ht = wt = lh // PATCH
    nimg, scap = ht * wt, cap.shape[0]

    cap.astype(np.float32).tofile(workdir / "cap_raw.bin")
    np.arange(1, scap + 1, dtype=np.int32).tofile(workdir / "cap_ids0.bin")
    for k in (1, 2):
        np.zeros(scap, np.int32).tofile(workdir / f"cap_ids{k}.bin")

    ids = np.zeros((nimg, 3), np.int32)
    for y in range(ht):
        ids[y*wt:(y+1)*wt, 0] = scap + 1
        ids[y*wt:(y+1)*wt, 1] = y
        ids[y*wt:(y+1)*wt, 2] = np.arange(wt)
    for k in range(3):
        ids[:, k].astype(np.int32).tofile(workdir / f"img_ids{k}.bin")

    (workdir / "meta.txt").write_text(f"{ht} {wt} {nimg} {scap} {nimg+scap}\n")
    rng = np.random.default_rng(seed)
    rng.standard_normal((INCH, lh, lh)).astype(np.float32).tofile(workdir / "lat_init.f32")
    np.zeros(nimg * INCH * PATCH * PATCH, np.float32).tofile(workdir / "img_raw.bin")  # rakans om
    np.array([1.0], np.float32).tofile(workdir / "t.bin")
    return nimg + scap


LAT_READY = "[resident] klar -> lat_out.f32"


def start_dit(workdir, steps, verbose):
    """Kor DiT-motorn och returnerar (lat, finish) SA FORT lat_out.f32 ar skriven.

    Efter latenten river motorn ned: ~1 s for att frigora de residenta vikterna och stanga
    DSP-sessionen (matt med fasklockorna). Det ar NPU/drivrutinsarbete och VAE:n ar CPU, sa
    anroparen avkodar under tiden och kallar finish() efterat. finish() vantar in processen,
    skriver ZI_ENGINE_LOG, kollar rc och visar budgeten; den ar idempotent.
    Motorn flushar stdout efter markorraden - utan det kommer raden forst vid exit."""
    import threading
    env = {**os.environ, "ADSP_LIBRARY_PATH": str(BIN),
           "ZI_FLASH": "1", "ZI_DUMP": "0",
           # PD-dump om DSP-processen dor (0x72); kostar inget annars (ggml-hexagon patch_0x72_diagnose.py)
           "GGML_HEXAGON_PD_DUMP": os.environ.get("GGML_HEXAGON_PD_DUMP", "1"), "ZI_STEPS": str(steps)}
    p = subprocess.Popen([str(BIN / "zimage-dit-stream.exe"), str(DIT), str(workdir), "HTP0"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    out, err, ready = [], [], threading.Event()

    # Bada roren maste tommas hela tiden: ett fullt stderr-ror blockerar motorn.
    def pump(stream, buf, watch):
        for line in stream:
            buf.append(line)
            if watch and line.startswith(LAT_READY):
                ready.set()
    pumps = [threading.Thread(target=pump, args=(p.stdout, out, True), daemon=True),
             threading.Thread(target=pump, args=(p.stderr, err, False), daemon=True)]
    for t in pumps:
        t.start()

    done = []

    def finish():
        if done:
            return
        done.append(True)
        rc = p.wait()
        for t in pumps:
            t.join()
        so, se = "".join(out), "".join(err)
        # ZI_ENGINE_LOG=<path>: keep the engine's full output (e.g. GGML_HEXAGON_PROFILE lines,
        # thousands per image). Off by default so the CLI stays quiet. Written before the rc
        # check so a failing run leaves its log too.
        log_path = os.environ.get("ZI_ENGINE_LOG")
        if log_path:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(so)
                f.write(se)
        if rc != 0:
            sys.exit("DiT misslyckades (rc %d):\n" % rc + (so[-800:] + se[-800:]))
        if verbose:
            for line in se.split("\n"):
                if "[budget]" in line:
                    log(line.split("[budget]")[1].strip())

    while not ready.wait(0.05):
        if p.poll() is not None:
            break
    if not ready.is_set():
        finish()            # slutade utan markor: rc-fel (avbryter) - eller en motor utan flush
    return np.fromfile(workdir / "lat_out.f32", dtype=np.float32), finish


def run_dit(workdir, steps, verbose):
    lat, finish = start_dit(workdir, steps, verbose)
    finish()
    return lat


def taef1_decode(lat, tmp):
    """AutoencoderTiny (madebyollin/taef1) via C++/NEON-avkodaren.

    Gamla motorns FORVAL, och skalet till att dess VAE tog ~1 s dar var tar 34: vi
    kor referensavkodaren dar den korde den lilla destillerade. Matt dar:
    taef1 i torch 0,95 s, denna 343 ms (2,65x), cos 1,000000000 mot torch och
    cos 0,996 mot den riktiga Flux-VAE:n.
    Matt 09-27 (exe:n ensam): 1190 ms @512 / 5140 @1024 -> efter tre bit-identiska
    karnandringar 634 / 2551 ms. Se taef1_decode.cpp och patch_conv*.py.

    Storleken ar RUNTIME i avkodaren (LH/LW harleds ur latentfilens storlek,
    OUT = L*8), sa 64x64 -> 512x512 gar utan omexport.

    KONTRAKT: taef1 har EGEN intern latentskalning och tar darfor den RAA
    diffusionslatenten - INTE lat/SCALE + SHIFT som Flux-AE:n vill ha.
    """
    H = int(round((lat.size // INCH) ** 0.5))
    fi = Path(tmp) / "taef1_lat.f32"
    fo = Path(tmp) / "taef1_out.rgb"
    lat.reshape(INCH, H, H).astype(np.float32).tofile(str(fi))
    # vikterna ligger i {models}/taef1/ (09-28: forut hardkodat i exe:n - fungerade bara har)
    env = dict(os.environ)
    env.setdefault("PULSE_TAEF1_WEIGHTS", str(MODELS / "taef1" / "diffusion_pytorch_model.safetensors"))
    r = subprocess.run([str(TAEF1X), str(fi), str(fo)], capture_output=True, text=True, env=env)
    if r.returncode != 0 or not fo.exists():
        raise RuntimeError("taef1_decode: rc=%d %s" % (r.returncode, r.stderr[-400:]))
    px = H * VAEF
    return np.fromfile(str(fo), dtype=np.uint8).reshape(px, px, 3)


def decode(lat, tmp):
    """VAE-vag. taef1 ar forval (som i gamla motorn); ZI_VAE=full ger referensen."""
    if os.environ.get("ZI_VAE", "taef1").lower() == "full":
        return vae_decode(lat)
    return taef1_decode(lat, tmp)


_VAE_W = None


def vae_weights():
    """Laddas EN gang per process - i serve-laget ar det skillnaden mellan 10 s och 0."""
    global _VAE_W
    if _VAE_W is not None:
        return _VAE_W
    import struct, torch
    with open(VAE, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n)); base = 8 + n
        W = {}
        for k, m in hdr.items():
            if k == "__metadata__":
                continue
            o0, o1 = m["data_offsets"]; fh.seek(base + o0)
            dt = {"F32": np.float32, "F16": np.float16}[m["dtype"]]
            W[k] = torch.from_numpy(np.frombuffer(fh.read(o1-o0), dtype=dt)
                                    .reshape(m["shape"]).astype(np.float32).copy())
    # channels_last pa alla 4D-weights: faltningar pa CPU blir 24 % snabbare (37,5 -> 28,4 s
    # for en 512px-bild). Ren layoutandring - utdatan ar oforandrad.
    for k, v in W.items():
        if v.dim() == 4:
            W[k] = v.to(memory_format=torch.channels_last)
    _VAE_W = W
    return W


def vae_decode(lat):
    """Z-Image-VAE i torch. Samma matte som decode_latent.py."""
    import torch
    import torch.nn.functional as F
    W = vae_weights()

    def conv(x, p, k=3):
        return F.conv2d(x.to(memory_format=torch.channels_last), W[p+".weight"], W[p+".bias"], padding=k//2)
    def gn(x, p):        return F.group_norm(x, GN_G, W[p+".weight"], W[p+".bias"], EPS)

    def resnet(x, p):
        h = conv(F.silu(gn(x, p+".norm1")), p+".conv1", 3)
        h = conv(F.silu(gn(h, p+".norm2")), p+".conv2", 3)
        s = x if (p+".nin_shortcut.weight") not in W else conv(x, p+".nin_shortcut", 1)
        return s + h

    def attn(x, p):
        B, C, H, Wd = x.shape; h = gn(x, p+".norm")
        q = conv(h, p+".q", 1).reshape(B, C, H*Wd).permute(0, 2, 1)
        k = conv(h, p+".k", 1).reshape(B, C, H*Wd)
        v = conv(h, p+".v", 1).reshape(B, C, H*Wd).permute(0, 2, 1)
        a = torch.softmax((q @ k) * (C ** -0.5), -1)
        return x + conv((a @ v).permute(0, 2, 1).reshape(B, C, H, Wd), p+".proj_out", 1)

    H = int(round((lat.size // INCH) ** 0.5))
    z = (torch.from_numpy(lat.reshape(1, INCH, H, H)) / SCALE + SHIFT).to(memory_format=torch.channels_last)
    with torch.no_grad():
        h = conv(z, "decoder.conv_in", 3)
        h = resnet(h, "decoder.mid.block_1"); h = attn(h, "decoder.mid.attn_1")
        h = resnet(h, "decoder.mid.block_2")
        for lvl in (3, 2, 1, 0):
            for b in range(3):
                h = resnet(h, f"decoder.up.{lvl}.block.{b}")
            if lvl != 0:
                h = conv(F.interpolate(h, scale_factor=2, mode="nearest"),
                         f"decoder.up.{lvl}.upsample.conv", 3)
        img = conv(F.silu(gn(h, "decoder.norm_out")), "decoder.conv_out", 3)
    return ((img[0].clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 0).numpy()


class DitServer:
    """Haller zimage-dit-stream.exe vid liv (ZI_SERVE=1) sa modell, weights och
    HTP-sessionen betalas EN gang. Harnessen genererar argv-katalogen direkt och
    ber sedan om nasta katalog pa stdin; '[serve] klar' pa stderr ar klarsignalen.
    Dess stdout maste dranneras eller skickas till DEVNULL, annars fyller den roret
    och processen laser sig."""

    MARK = "[serve] klar"

    def __init__(self, first_dir, steps, verbose):
        self.verbose = verbose
        env = {**os.environ, "ADSP_LIBRARY_PATH": str(BIN),
               "ZI_FLASH": "1", "ZI_DUMP": "0",
           # PD-dump om DSP-processen dor (0x72); kostar inget annars (ggml-hexagon patch_0x72_diagnose.py)
           "GGML_HEXAGON_PD_DUMP": os.environ.get("GGML_HEXAGON_PD_DUMP", "1"), "ZI_STEPS": str(steps), "ZI_SERVE": "1"}
        self.p = subprocess.Popen(
            [str(BIN / "zimage-dit-stream.exe"), str(DIT), str(first_dir), "HTP0"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, env=env, bufsize=1)
        self._wait()

    def _wait(self):
        for line in self.p.stderr:
            if self.verbose and "[budget]" in line:
                log(line.split("[budget]")[1].strip())
            if self.MARK in line:
                return
        raise RuntimeError("DiT-servern dog")

    def render(self, workdir):
        self.p.stdin.write(str(workdir) + "\n")
        self.p.stdin.flush()
        self._wait()

    def close(self):
        try:
            self.p.stdin.write("quit\n"); self.p.stdin.flush()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


def release_encoder():
    """Stanger encodern nar den gjort sitt.

    Matt under en 1024-korning: llama-server holl **1,9 GB PRIVAT minne i 110 s** fast
    den anvands i 0,9. DiT:ns egna 5,1 GB ar daremot nastan helt minnesmappad GGUF
    (privat bara 228 MB) - filbackade sidor som OS:et kan aterta, alltsa ingen verklig
    kostnad. Det ar encodern som ar vard att slappa, och bara i engangslaget:
    i serve/repl ar hela poangen att den ar kvar.
    """
    if os.environ.get("ZI_ENCODER_TRANSIENT") != "1":
        return
    try:
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                       capture_output=True, timeout=10)
    except Exception:
        pass


def one_image(prompt, px, seed, steps, out, quality, verbose, tmp, dit=None):
    """Gor en bild. Med `dit` satt ateranvands en levande DiT-server."""
    t = time.perf_counter(); cap = cap_feats(prompt, tmp, verbose); log("encoder", t)
    release_encoder()
    su = build_inputs(cap, px, seed, tmp)
    t = time.perf_counter()
    if dit is None:
        lat = run_dit(tmp, steps, verbose)
    else:
        dit.render(tmp)
        lat = np.fromfile(tmp / "lat_out.f32", dtype=np.float32)
    log(f"DiT {px}px, {steps} steps, {su} tokens", t)
    if not np.isfinite(lat).all():
        raise RuntimeError("the latent contains NaN")
    t = time.perf_counter(); im = decode(lat, tmp); log("VAE", t)
    from PIL import Image
    out = Path(out); img = Image.fromarray(im)
    img.save(out, quality=quality) if out.suffix.lower() in (".jpg", ".jpeg") else img.save(out)
    return out, im.shape


# Colours, matching zimage.ps1: cyan headings, green prompt, yellow flags.
# Windows does NOT interpret ANSI unless the console mode has
# ENABLE_VIRTUAL_TERMINAL_PROCESSING set - without this the codes print as literal
# text ('<-[96m'). Falls back to no colour if it cannot be enabled.
def _enable_ansi():
    if os.name != 'nt':
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not k.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        return bool(k.SetConsoleMode(h, mode.value | 0x0004))
    except Exception:
        return False


if _enable_ansi():
    C_GREEN, C_CYAN, C_YELLOW, C_DIM, C_OFF = (
        '[92m', '[96m', '[93m', '[90m', '[0m')
else:
    C_GREEN = C_CYAN = C_YELLOW = C_DIM = C_OFF = ''


_ESR = None
import threading as _threading
_ESR_LOCK = _threading.Lock()   # skapas vid import: en lat skapelse kan rasa mellan tradarna


def esrgan_release():
    """REPL: let go of the ESRGAN session right after it is used. It holds ~0.9 GB COMMITTED
    (the 2048x2048 activations) and would sit there between images; esrgan_prewarm() reopens it
    behind the next DiT run (~1 s vs a DiT of 8+ s), so this costs no waiting time (09-28)."""
    global _ESR
    with _ESR_LOCK:
        _ESR = None
    import gc
    gc.collect()


def esrgan_prewarm():
    """Open the ESRGAN session in a background thread while the DiT runs (make -Upscale).
    upscale_x4() then finds it ready; the lock makes sure it is only created once."""
    import threading
    th = threading.Thread(target=esrgan_session, daemon=True)
    th.start()
    return th


def esrgan_session():
    """Real-ESRGAN x4 as a QNN context via ONNX Runtime's QNN EP - the upscaler for
    512-px images. Opened once per process (~2.3 s); in the REPL it stays warm."""
    with _ESR_LOCK:
        return _esrgan_open()


def _esrgan_open():
    global _ESR
    if _ESR is None:
        import onnxruntime as ort
        try:
            ort.disable_telemetry_events()
        except Exception:
            pass
        for d in (QNN_RT, Path(ort.__file__).parent / "capi"):
            if d.is_dir():
                os.add_dll_directory(str(d))
        so = ort.SessionOptions()
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        _ESR = ort.InferenceSession(str(ESRGAN), so, providers=["QNNExecutionProvider"],
                                    # burst: utan den kor ESRGAN pa NPU:ns laga klocka nar inget annat
                                    # rostar upp den - matt 09-27: inferens 3,41 -> 1,10 s, session
                                    # 2,16 -> 0,97 s. (I repl rostade DiT-motorn redan, darfor 1,2 s dar.)
                                    provider_options=[{"backend_path": str(QNN_RT / "QnnHtp.dll"),
                                                       "htp_performance_mode": "burst"}])
    return _ESR


def upscale_x4(im, target):
    """512 x 512 RGB (uint8) -> Real-ESRGAN x4 -> 2048 -> Lanczos -> target. Measured 09-27:
    1.1 s on the NPU in burst mode, colours faithful (per-channel slope 0.998-1.010 vs Lanczos),
    more detail. A w8a16 build of the same net was SLOWER (1.33 s) and shifted blue by 2.4 %."""
    from PIL import Image
    if im.shape[0] != 512 or im.shape[1] != 512:
        raise ValueError("the x4 upscaler takes a 512 x 512 image, got %dx%d" % (im.shape[1], im.shape[0]))
    x = np.ascontiguousarray((im.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
    y = esrgan_session().run(None, {"image": x})[0]
    big = np.rint(np.clip(y[0].transpose(1, 2, 0), 0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.fromarray(big, "RGB")
    if target != img.size[0]:
        img = img.resize((target, target), Image.LANCZOS)
    # upscale_detail: ESRGAN's share. Pure ESRGAN paints fur and skin with hard, plastic-looking
    # edges; mixing in a plain Lanczos enlargement of the same 512 image softens that while most
    # of the detail stays (09-28, same tiger: Laplace variance 688 at 1.0, 310 at 0.5, 156 at 0).
    d = float(DEFAULTS.get("upscale_detail", 0.5))
    if d >= 1.0:
        return np.asarray(img)
    soft = np.asarray(Image.fromarray(im, "RGB").resize((target, target), Image.LANCZOS), dtype=np.float32)
    mix = d * np.asarray(img, dtype=np.float32) + (1.0 - d) * soft
    return np.clip(np.rint(mix), 0, 255).astype(np.uint8)


class Steps:
    """Numbered progress lines: what runs, where, and how long it took."""

    def __init__(self, n):
        self.i, self.n = 0, n

    def done(self, what, t0, note=""):
        self.i += 1
        dt = time.perf_counter() - t0
        print("  %s[%d/%d]%s %-40s %s%5.1f s%s%s" % (C_DIM, self.i, self.n, C_OFF, what, C_GREEN, dt, C_OFF,
                                                     ("   " + C_DIM + note + C_OFF) if note else ""), flush=True)


# --- text on the image ------------------------------------------------------------------------
# ** VARFOR TEXTEN LAGGS PA EFTERAT och inte skrivs av modellen (matt 2026-08-30 i gamla ZImage):
#    ett KORT ord i motivet fungerar ("VALKOMMEN" pa en skylt), men en affisch med rubrik + brodtext +
#    datum blev nonsens ("NORTYERN MPAIET", "39-84202") - modellen ritar TEXTUR SOM SER UT SOM TEXT.
#    Kompositering ger ratt stavning, skarpa kanter, och texten gar att ANDRA utan ny generering.
FONTS = [r"C:/Windows/Fonts/seguibl.ttf",    # Segoe UI Black
         r"C:/Windows/Fonts/segoeuib.ttf",   # Segoe UI Bold
         r"C:/Windows/Fonts/arialbd.ttf"]


def _font(px):
    from PIL import ImageFont
    for f in FONTS:
        try:
            return ImageFont.truetype(f, px)
        except Exception:
            pass
    return ImageFont.load_default()


def add_text(img, title, subtitle, pos="bottom"):
    """Title + subtitle on a PIL image, over a soft dark scrim so white text reads even on snow.
    Lines that do not fit shrink until they do (a long title used to be cut at the edge)."""
    from PIL import ImageDraw
    if not (title or subtitle):
        return img
    img = img.convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")
    marg = int(W * 0.055)
    width = W - 2 * marg

    def fit(text, start_px, min_px):
        f, px = _font(start_px), start_px
        while px > min_px:
            try:
                b = f.getbbox(text); w = b[2] - b[0]
            except Exception:
                w = len(text) * px * 0.55
            if w <= width:
                break
            px = int(px * 0.94); f = _font(px)
        return f, px

    ft, t_px = fit(title, max(18, int(W * 0.075)), max(12, int(W * 0.030))) if title else (None, 0)
    fs, s_px = fit(subtitle, max(12, int(W * 0.030)), max(9, int(W * 0.018))) if subtitle else (None, 0)
    gap = int(t_px * 0.28) if title else 0
    height = (t_px if title else 0) + (gap if title and subtitle else 0) + (s_px if subtitle else 0)
    band = height + 2 * marg
    y0 = 0 if pos == "top" else H - band
    # En rad i taget: gamla ZImage ritade 48 steg, och mot en jamn yta (sno) syntes trappstegen som ranader.
    for k in range(band):
        t = k / max(1, band - 1)
        a = int(165 * ((1 - t) if pos == "top" else t) ** 1.35)
        d.line([(0, y0 + k), (W, y0 + k)], fill=(8, 12, 20, a))
    y = y0 + marg
    if title:
        d.text((marg, y), title, font=ft, fill=(255, 255, 255, 255),
               stroke_width=max(1, t_px // 26), stroke_fill=(0, 0, 0, 190))
        y += t_px + (gap if subtitle else 0)
    if subtitle:
        d.text((marg, y), subtitle, font=fs, fill=(232, 238, 246, 255),
               stroke_width=max(1, s_px // 22), stroke_fill=(0, 0, 0, 170))
    return img


def save_image(im, out, quality, title=None, subtitle=None, pos="bottom"):
    from PIL import Image
    out = Path(out)
    img = Image.fromarray(im) if not hasattr(im, "save") else im
    if title or subtitle:
        img = add_text(img, title, subtitle, pos)
    if out.suffix.lower() in (".jpg", ".jpeg"):
        # subsampling 0 = full colour resolution: text edges stay sharp (4:2:0 smears red/white text)
        img.save(out, quality=quality, subsampling=0 if (title or subtitle) else -1)
    else:
        img.save(out)
    return out


def next_free(template):
    """<stem>_NNNN<suffix> with NNNN one higher than the highest that already exists next to it.
    Never overwrites an earlier image, also not one from an earlier session."""
    t = Path(template)
    rx = re.compile(re.escape(t.stem) + r"_(\d{4,})" + re.escape(t.suffix) + r"$", re.IGNORECASE)
    folder = t.parent if str(t.parent) not in ("", ".") else Path.cwd()
    hi = 0
    if folder.is_dir():
        for f in folder.iterdir():
            m = rx.match(f.name)
            if m:
                hi = max(hi, int(m.group(1)))
    return t.with_name("%s_%04d%s" % (t.stem, hi + 1, t.suffix))


def describe(px, steps, seed, up, up_to):
    size = "%d x %d" % (px, px)
    return ("  " + C_CYAN + "PulseX" + C_OFF + "  " + size + "  |  " + str(steps) + " steps  |  seed " + str(seed)
            + (("  |  upscale x4 -> " + str(up_to)) if up else ""))


VAE_NAME = {"taef1": "taef1, CPU", "full": "full VAE, CPU"}

# --- line input for the REPL -------------------------------------------------------------------
# Windows' console only hands Ctrl+D to input() AFTER Enter (as "\x04"), so "Ctrl+D quits" did not
# work. This reader takes the keys itself: Ctrl+D / Ctrl+Z quit at once, Ctrl+C too, Up/Down recall
# earlier lines, Left/Right/Home/End/Del edit, Esc clears. Long lines that wrap are redrawn correctly.
# Not a console (piped input, tests): plain input().
_HISTORY = []


def commit_mb():
    """System committed memory in MB (GetPerformanceInfo), -1 if unknown."""
    try:
        import ctypes
        class _PI(ctypes.Structure):
            _fields_ = ([("cb", ctypes.c_ulong)]
                        + [(k, ctypes.c_size_t) for k in ("CommitTotal", "CommitLimit", "CommitPeak",
                           "PhysicalTotal", "PhysicalAvailable", "SystemCache", "KernelTotal",
                           "KernelPaged", "KernelNonpaged", "PageSize")]
                        + [(k, ctypes.c_ulong) for k in ("HandleCount", "ProcessCount", "ThreadCount")])
        p = _PI(); p.cb = ctypes.sizeof(p)
        ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb)
        return p.CommitTotal * p.PageSize / 2**20
    except Exception:
        return -1.0


def unload_models(dit):
    """ComfyUI-style unload between images: close the DiT server (its NPU buffers are ~4.9 GB
    COMMITTED), drop the ESRGAN session (~0.9 GB), then drop their file pages from Windows'
    standby cache. The next image reloads them. Returns (committed MB freed, cache MB dropped)."""
    global _ESR
    c0 = commit_mb()
    if dit is not None:
        dit.close()
    with _ESR_LOCK:
        _ESR = None
    import gc
    gc.collect()
    dropped = 0
    exe = HERE / "zimage.exe"
    files = [str(DIT)] + [f for f in [str(ESRGAN).replace(".wrap.onnx", "")] if Path(f).exists()]
    if exe.exists():
        try:
            r = subprocess.run([str(exe), "--drop-cache"] + files, capture_output=True, text=True, timeout=60)
            for l in r.stdout.split(chr(10)):
                if l.startswith("dropped "):
                    dropped = int(l.split()[1])
        except Exception:
            pass
    c1 = commit_mb()
    return (c0 - c1 if c0 >= 0 and c1 >= 0 else 0.0), dropped


def read_line(prompt, prompt_width, idle=None):
    """idle = (seconds, callback): after that long without a key the callback runs once and
    may return a line to show above the prompt (the REPL uses it to unload the models)."""
    if os.name != "nt" or not sys.stdin.isatty():
        return input(prompt)
    import msvcrt, shutil
    cols = max(20, shutil.get_terminal_size((100, 30)).columns)
    buf, pos, hi = [], 0, len(_HISTORY)
    state = {"row": 0}

    def draw():
        if state["row"]:
            sys.stdout.write("\x1b[%dA" % state["row"])
        sys.stdout.write("\r" + prompt + "".join(buf) + "\x1b[J")
        end = prompt_width + len(buf)
        cur = prompt_width + pos
        end_row, cur_row = end // cols, cur // cols
        if end_row > cur_row:
            sys.stdout.write("\x1b[%dA" % (end_row - cur_row))
        sys.stdout.write("\x1b[%dG" % (cur % cols + 1))
        state["row"] = cur_row
        sys.stdout.flush()

    draw()
    t_idle = time.monotonic()
    while True:
        if idle is not None:
            while not msvcrt.kbhit():
                if idle[0] > 0 and time.monotonic() - t_idle >= idle[0]:
                    msg = idle[1]()
                    idle = None
                    if msg:
                        if state["row"]:
                            sys.stdout.write("\x1b[%dA" % state["row"])
                        sys.stdout.write("\r\x1b[J" + msg + "\n")
                        state["row"] = 0
                        draw()
                    break
                time.sleep(0.02)
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            pos = len(buf); draw(); sys.stdout.write("\n"); sys.stdout.flush()
            line = "".join(buf)
            if line.strip() and (not _HISTORY or _HISTORY[-1] != line):
                _HISTORY.append(line)
            return line
        if ch in ("\x04", "\x1a"):                    # Ctrl+D, Ctrl+Z
            sys.stdout.write("\n"); sys.stdout.flush(); raise EOFError
        if ch == "\x03":                              # Ctrl+C
            sys.stdout.write("\n"); sys.stdout.flush(); raise KeyboardInterrupt
        if ch in ("\x00", "\xe0"):                    # arrows and friends
            k = msvcrt.getwch()
            if k == "K" and pos > 0: pos -= 1
            elif k == "M" and pos < len(buf): pos += 1
            elif k == "G": pos = 0
            elif k == "O": pos = len(buf)
            elif k == "S" and pos < len(buf): del buf[pos]
            elif k == "H" and hi > 0:
                hi -= 1; buf = list(_HISTORY[hi]); pos = len(buf)
            elif k == "P":
                if hi < len(_HISTORY) - 1:
                    hi += 1; buf = list(_HISTORY[hi])
                else:
                    hi = len(_HISTORY); buf = []
                pos = len(buf)
            draw(); continue
        if ch == "\x08":                              # Backspace
            if pos > 0:
                del buf[pos - 1]; pos -= 1
            draw(); continue
        if ch == "\x1b":                              # Esc
            buf, pos = [], 0; draw(); continue
        if ch < " ":
            continue
        buf.insert(pos, ch); pos += 1
        draw()


def _take_text(words, k):
    """The words after --title/--sub, up to the next --option."""
    out = []
    while k < len(words) and not words[k].startswith("--"):
        out.append(words[k]); k += 1
    return " ".join(out), k


def serve(a, tmp):
    """REPL: model, weights and NPU session stay warm between images."""
    Y, G, D, O = C_YELLOW, C_GREEN, C_DIM, C_OFF
    print()
    print("  " + C_CYAN + "PulseX - interactive mode" + O)
    print("  Everything stays loaded between images, so from the second image on you only")
    print("  pay for the generation itself.")
    print()
    print("  " + G + "type a prompt" + O + " and press Enter        " + D + "a lighthouse at dusk" + O)
    print("  add options at the end of the line    " + D + "a lighthouse at dusk --size 1024" + O)
    print()
    print("  " + D + "these stay in effect until you change them:" + O)
    print("    " + Y + "--size 512|1024" + O + "   resolution, or just " + Y + "--512" + O + " / " + Y + "--1024"
          + O + " (now " + str(a.size) + ")")
    print("    " + Y + "--up" + O + " / " + Y + "--noup" + O + "     512 + Real-ESRGAN x4 -> " + str(a.upscale_to)
          + " on/off (now " + ("on" if a.upscale else "off") + ")")
    print("    " + Y + "--seed N" + O + "          start seed (each image adds 1)")
    print("    " + Y + "--enrich" + O + " / " + Y + "--noenrich" + O + " add material words (skin pores, wet sand ...) (now "
          + ("on" if a.enrich else "off") + ")")
    print("  " + D + "these apply to this one image:" + O)
    print("    " + Y + "--title" + O + " words     text on the picture    " + D + "--title Summer Sale" + O)
    print("    " + Y + "--sub" + O + " words       smaller line under it  " + D + "--sub Friday 6 pm" + O)
    print("    " + Y + "--top" + O + "             put the text at the top (bottom is default)")
    print()
    # Ctrl+D / :q avslutar, men star inte har (09-27: "underforstatt, som i GenieX").
    print("  Up/Down recalls earlier lines.  Images: " + D + os.getcwd() + O + " as "
          + Path(a.out).stem + "_NNNN" + Path(a.out).suffix + " (never overwritten)")
    print()
    if a.vae == "full":
        vae_weights()                  # betala torch-laddningen nu, inte pa forsta prompten
    dit, n, px, up, seed0 = None, 0, a.size, a.upscale, a.seed
    enr = a.enrich
    unload_s = int(DEFAULTS.get("unload_after_s", 60) or 0)
    if unload_s > 0:
        print("  " + D + "After %d s without a prompt the image model is unloaded and its memory given back;"
              % unload_s + O)
        print("  " + D + "the next image then takes a few seconds longer." + O)
        print()

    def _idle():
        nonlocal dit
        freed, dropped = unload_models(dit)
        dit = None
        return (D + "  (idle %d s: image model unloaded - %.1f GB memory and %.1f GB cache given back)"
                % (unload_s, freed / 1024.0, dropped / 1024.0) + O)

    try:
        while True:
            try:
                busy = dit is not None or _ESR is not None
                line = read_line(G + "> " + O, 2, (unload_s, _idle) if (busy and unload_s > 0) else None)
                if "\x04" in line:                     # piped input can still carry it
                    raise EOFError
                line = line.strip()
            except (EOFError, KeyboardInterrupt):
                print(D + "  bye - freeing NPU memory and cache..." + O); break
            if line in (":q", ":quit", "quit", "exit"):
                break
            if not line:
                continue
            words, prompt = line.split(), []
            title = subtitle = None
            tpos = "bottom"
            k = 0
            while k < len(words):
                w = words[k]
                if w in ("--512", "--1024"):                   # kortform (09-28: "--1024" foll igenom)
                    px = int(w[2:]); k += 1; continue
                if w == "--size" and k + 1 < len(words) and words[k + 1] in ("512", "1024"):
                    px = int(words[k + 1]); k += 2; continue
                if w == "--size":
                    # Tyst svald ValueError har lat "--size N" se ut att fungera medan
                    # upplosningen stod kvar - anvandaren fick fel bild utan att veta om det.
                    print("  " + Y + "--size" + O + " takes 512 or 1024 - keeping " + str(px) + ".")
                    k += 2; continue
                if w == "--seed" and k + 1 < len(words) and words[k + 1].lstrip("-").isdigit():
                    seed0 = int(words[k + 1]) - n; k += 2; continue
                if w in ("--up", "--upscale"):
                    up = True; k += 1; continue
                if w in ("--noup", "--no-upscale"):
                    up = False; k += 1; continue
                if w == "--title":
                    title, k = _take_text(words, k + 1); continue
                if w in ("--sub", "--subtitle"):
                    subtitle, k = _take_text(words, k + 1); continue
                if w == "--top":
                    tpos = "top"; k += 1; continue
                if w == "--enrich":
                    enr = True; k += 1; continue
                if w in ("--noenrich", "--no-enrich"):
                    enr = False; k += 1; continue
                if w == "--bottom":
                    tpos = "bottom"; k += 1; continue
                if w.startswith("--"):
                    print("  " + D + "unknown option " + w + " - ignored" + O); k += 1; continue
                prompt.append(w); k += 1
            if not prompt:
                print("  " + D + "(settings updated: size " + str(px) + ", upscale " + ("on" if up else "off")
                      + " - now type a prompt)" + O)
                continue
            text = " ".join(prompt)
            if enr:
                text, added = enrich_prompt(text)
                if added:
                    print("  " + D + "prompt sent: " + text + O)
            do_up = up and px == 512
            n += 1
            out = next_free(a.out)             # nasta lediga - skriver aldrig over en tidigare bild
            st = Steps(4 if do_up else 3)
            t_img = time.perf_counter()
            try:
                t = time.perf_counter(); cap = cap_feats(text, tmp, a.verbose)
                st.done("encode the prompt  (Qwen3-4B, %s)" % ENC_ON, t)
                su = build_inputs(cap, px, seed0 + n, tmp)
                if do_up and _ESR is None:
                    esrgan_prewarm()                          # sessionsstarten doljs bakom DiT:n
                t = time.perf_counter()
                if dit is None:
                    dit = DitServer(tmp, a.steps, a.verbose)      # forsta bilden gors av starten
                else:
                    dit.render(tmp)
                lat = np.fromfile(tmp / "lat_out.f32", dtype=np.float32)
                st.done("generate the image (Z-Image DiT, NPU)", t, "%d tokens, %d steps" % (su, a.steps))
                if not np.isfinite(lat).all():
                    print("  the latent contains NaN - skipping this one"); continue
                t = time.perf_counter(); im = decode(lat, tmp)
                st.done("decode to pixels   (%s)" % VAE_NAME.get(a.vae, a.vae), t)
                if do_up:
                    t = time.perf_counter(); im = upscale_x4(im, a.upscale_to)
                    st.done("upscale x4         (Real-ESRGAN, NPU)", t, "512 -> 2048 -> %d" % a.upscale_to)
                    esrgan_release()                          # ~0,9 GB tillbaka; ateroppnas bakom nasta DiT
                out = save_image(im, out, a.quality, title, subtitle, tpos)
                print("  " + G + "saved" + O + " %s  (%d x %d, seed %d)  in %.1f s" % (out, im.shape[1], im.shape[0], seed0 + n,
                                                                                    time.perf_counter() - t_img))
            except Exception as e:
                print(f"  failed: {e}")
    finally:
        if dit: dit.close()


def text_only(a):
    """zimage text: put a title/subtitle on an EXISTING image - no generation."""
    from PIL import Image
    src = Path(a.add_text)
    if not src.exists():
        sys.exit("PulseX: no such image: %s" % src)
    if not (a.title or a.subtitle):
        sys.exit("PulseX: give -Title and/or -Subtitle for the text")
    out = Path(a.out) if a.out_given else src.with_name(src.stem + "_text" + (src.suffix or ".jpg"))
    img = Image.open(src)
    out = save_image(img.convert("RGB"), out, a.quality, a.title, a.subtitle, a.text_pos)
    print("  " + C_GREEN + "saved" + C_OFF + " %s  (%d x %d)" % (out, img.size[0], img.size[1]))


def main():
    ap = argparse.ArgumentParser(prog="PulseX", description="Make an image from a prompt on the NPU.")
    ap.add_argument("prompt", nargs="?", default="", help="the prompt (empty with --serve)")
    ap.add_argument("-o", "--out", default=None, help="output file (.jpg or .png)")
    ap.add_argument("--size", type=int, default=int(DEFAULTS["size"]), choices=(512, 1024),
                    help="512 is ~3.5x cheaper than 1024 and good for trying prompts")
    ap.add_argument("--steps", type=int, default=int(DEFAULTS["steps"]), help="Z-Image-Turbo is distilled for 4")
    ap.add_argument("--seed", type=int, default=int(DEFAULTS["seed"]))
    ap.add_argument("-q", "--quality", type=int, default=int(DEFAULTS["quality"]), help="JPEG quality")
    ap.add_argument("--upscale", action="store_true", default=bool(DEFAULTS["upscale"]),
                    help="at 512: Real-ESRGAN x4 afterwards")
    ap.add_argument("--no-upscale", dest="upscale", action="store_false", help="keep the 512 image as it is")
    ap.add_argument("--enrich", dest="enrich", action="store_true", default=bool(DEFAULTS.get("enrich", False)),
                    help="add material words to the prompt (cues.json)")
    ap.add_argument("--no-enrich", dest="enrich", action="store_false", help=argparse.SUPPRESS)
    ap.add_argument("--upscale-to", type=int, default=int(DEFAULTS["upscale_to"]), choices=(1024, 2048),
                    help=argparse.SUPPRESS)   # 2048 fungerar men ar dold tills vidare (09-27)
    ap.add_argument("--title", default=None, help="text on the picture")
    ap.add_argument("--subtitle", default=None, help="smaller line under the title")
    ap.add_argument("--text-pos", default="bottom", choices=("top", "bottom"))
    ap.add_argument("--add-text", metavar="IMAGE", default=None, help="put the text on an existing image")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--keep", metavar="DIR", help="keep the intermediate files here")
    ap.add_argument("--serve", action="store_true", help="interactive mode (REPL)")
    a = ap.parse_args()
    a.out_given = a.out is not None
    if a.out is None:
        a.out = DEFAULTS["out"]
    if a.add_text:
        text_only(a)
        return
    a.vae = os.environ.get("ZI_VAE", DEFAULTS["vae"]).lower()
    os.environ["ZI_VAE"] = a.vae

    need = [DIT, ENC, TOK, BIN, TAEF1X if a.vae == "taef1" else VAE]
    if a.upscale:
        need += [ESRGAN, QNN_RT]
    missing = [p for p in need if not p.exists()]
    if missing:
        sys.exit("PulseX: missing files (see %s):\n  %s" % (CONFIG_FILE, "\n  ".join(str(p) for p in missing)))

    t_all = time.perf_counter()
    tmp = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="zimage_"))
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        if a.serve:
            serve(a, tmp)
            return
        do_up = a.upscale and a.size == 512
        print(describe(a.size, a.steps, a.seed, do_up, a.upscale_to))
        st = Steps(4 if do_up else 3)
        if a.enrich:
            a.prompt, added = enrich_prompt(a.prompt)
            if added:
                print("  " + C_DIM + "prompt sent: " + a.prompt + C_OFF)
        t = time.perf_counter(); cap = cap_feats(a.prompt, tmp, a.verbose)
        st.done("encode the prompt  (Qwen3-4B, %s)" % ENC_ON, t)
        release_encoder()
        su = build_inputs(cap, a.size, a.seed, tmp)
        if do_up:
            esrgan_prewarm()          # ~1 s sessionsstart doljs bakom DiT:n
        t = time.perf_counter(); lat, dit_finish = start_dit(tmp, a.steps, a.verbose)
        st.done("generate the image (Z-Image DiT, NPU)", t, "%d tokens, %d steps" % (su, a.steps))
        if not np.isfinite(lat).all():
            dit_finish()
            sys.exit("the latent contains NaN - aborting")
        # VAE:n (CPU) avkodar medan motorn river ned sina NPU-buffertar.
        t = time.perf_counter(); im = decode(lat, tmp)
        st.done("decode to pixels   (%s)" % VAE_NAME.get(a.vae, a.vae), t)
        t = time.perf_counter(); dit_finish()
        if a.verbose:
            log("DiT-nedstangning, vantan efter VAE:n", t)
        if do_up:
            # Efter dit_finish: motorns NPU-session ar stangd innan ESRGAN oppnar sin.
            t = time.perf_counter(); im = upscale_x4(im, a.upscale_to)
            st.done("upscale x4         (Real-ESRGAN, NPU)", t, "512 -> 2048 -> %d" % a.upscale_to)
        # Utan -Out: nasta lediga zimage_NNNN.jpg. Med -Out: exakt den filen (uttryckligt val).
        out = save_image(im, a.out if a.out_given else next_free(a.out), a.quality, a.title, a.subtitle, a.text_pos)
        print("  " + C_GREEN + "saved" + C_OFF + " %s  (%d x %d)  in %.1f s" % (out, im.shape[1], im.shape[0],
                                                                            time.perf_counter() - t_all))
    finally:
        if not a.keep:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
