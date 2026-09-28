# Matsond for encoderns kallstart: startar llama-server exakt som zimage.ps1 (samma env och
# argument) plus extra argument fran kommandoraden, pollar /health var 20 ms, och gor sedan
# samma tva anrop som zimage.py (apply-template + embeddings). Skriver tider och sparar
# embeddingen sa att varianter kan jamforas pa UTDATA.
#
#   python tools/enc_start_probe.py <namn> [extra llama-server-argument ...]
# Utdata: D:\prov\2026-09-27_enc\<namn>.err (serverlogg), <namn>.f32 (cap_feats), en rad pa stdout.
import hashlib, json, os, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path

import numpy as np

BIN = Path(r"C:\PulseCore\PulseX\ggml-hexagon\build-wos\bin")
ENC = Path(r"C:\PulseCore\models\zImage\qwen3-4b-zimage-q4_0.gguf")
OUT = Path(r"D:\prov\2026-09-27_enc")
PORT = 8099
URL = "http://127.0.0.1:%d" % PORT
PROMPT = os.environ.get("PROBE_PROMPT", "a photo of a red fox in autumn forest")

name, extra = sys.argv[1], sys.argv[2:]
OUT.mkdir(parents=True, exist_ok=True)


def post(ep, payload, timeout=60):
    req = urllib.request.Request(URL + ep, data=json.dumps(payload).encode("utf8"),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def healthy():
    try:
        urllib.request.urlopen(URL + "/health", timeout=1).read()
        return True
    except (urllib.error.URLError, OSError):
        return False


assert not healthy(), "en server lyssnar redan pa porten"
env = dict(os.environ, ADSP_LIBRARY_PATH=str(BIN), QWEN_EMBD_LAYER="35")
err = open(OUT / (name + ".err"), "w")
t0 = time.perf_counter()
p = subprocess.Popen([str(BIN / "llama-server.exe"), "-m", str(ENC), "--embeddings", "--pooling", "none",
                      "--embd-normalize", "-1", "-ngl", "99", "-c", "512", "--host", "127.0.0.1",
                      "--port", str(PORT)] + extra,
                     env=env, stdout=subprocess.DEVNULL, stderr=err)
try:
    while not healthy():
        if p.poll() is not None:
            raise SystemExit("servern dog, rc=%s - se %s.err" % (p.returncode, name))
        if time.perf_counter() - t0 > 180:
            raise SystemExit("ingen /health efter 180 s")
        time.sleep(0.02)
    t_ready = time.perf_counter() - t0

    t1 = time.perf_counter()
    text = post("/apply-template", {"messages": [{"role": "user", "content": PROMPT}]})["prompt"]
    t_tpl = time.perf_counter() - t1
    t1 = time.perf_counter()
    r = post("/embeddings", {"input": text})
    t_emb = time.perf_counter() - t1
    t1 = time.perf_counter()
    r2 = post("/embeddings", {"input": text})
    t_emb2 = time.perf_counter() - t1
finally:
    p.kill()
    p.wait()
    err.close()

emb = r[0]["embedding"] if isinstance(r, list) else r["data"][0]["embedding"]
a = np.asarray(emb, dtype=np.float32)
a.tofile(OUT / (name + ".f32"))
print("%-14s klar %.2f s | mall %.3f | embeddings 1:a %.3f  2:a %.3f s | summa till cap_feats %.2f s | %s md5 %s"
      % (name, t_ready, t_tpl, t_emb, t_emb2, t_ready + t_tpl + t_emb, a.shape,
         hashlib.md5(a.tobytes()).hexdigest()[:6]))
