# Belastar den varma encodern (llama-server pa 8099) med embeddinganrop i en slinga, sa att en
# ANDRA NPU-session tavlar om VTCM/HMX medan DiT-motorn kor. Test for 0x72-hypotesen:
# DSP:ns vtcm_acquire() gor abort() om VTCM inte kan fas inom 10 s.
#   python tools/hammer_encoder.py <sekunder> <loggfil>
import json, sys, time, urllib.request

secs, logf = float(sys.argv[1]), sys.argv[2]
URL = "http://127.0.0.1:8099/embeddings"
text = ("<|im_start|>user\n" + "a very detailed photograph of a red fox standing in an autumn forest, " * 12 +
        "<|im_end|>\n<|im_start|>assistant\n")
n = err = 0
t_end = time.time() + secs
with open(logf, "w") as f:
    while time.time() < t_end:
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(URL, data=json.dumps({"input": text}).encode(), headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=60).read()
            n += 1
            f.write("%.3f ok %.3f s\n" % (time.time(), time.perf_counter() - t0))
        except Exception as e:
            err += 1
            f.write("%.3f FEL %s\n" % (time.time(), e))
            time.sleep(0.2)
        f.flush()
    f.write("klart: %d anrop, %d fel\n" % (n, err))
