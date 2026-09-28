# 09-28 ("kor encodern pa NPU:n igen, vag 2"): encodern som KORTLIVAD NPU-process
# (zimage-encode.exe). REPL: startas vid forsta tangenten (laddningen ~1,4 s doljs bakom skrivandet),
# Enter kodar (0,86 s), processen frigor modellen och avslutar sjalv (ren DSP-stangning). Committed
# +2,8 GB bara medan den lever, sedan +-0 (matt, tools/encode_probe.py; cos 1,00000 mot gamla encodern).
# Forval encoder = npu; cpu (varm llama-server ur filcachen) finns kvar som val.
import io, json
PY = r"C:\PulseCore\PulseX\zimage_dit\zimage.py"
PS = r"C:\PulseCore\PulseX\zimage_dit\zimage.ps1"
JS = [r"C:\PulseCore\PulseX\zimage_dit\zimage.json", r"C:\PulseCore\PulseX\zimage_dit\zimage.example.json"]

s = io.open(PY, encoding="utf-8", newline="").read()


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt:
        raise SystemExit("py: ankare %d ggr: %r" % (s.count(old), old[:70]))
    s = s.replace(old, new)


# 1) klassen, direkt efter ENC_ON-raden
i = s.index("ENC_ON = ")
j = s.index("\n", i) + 1
s = s[:j] + r'''
# --- the text encoder as a short-lived NPU process (encoder = npu) -----------------------------
# Everything the NPU sees is pinned and counts as committed memory, so the encoder must not sit on
# the NPU between prompts. zimage-encode.exe loads Qwen3-4B (from the file cache), answers READY,
# encodes ONE templated prompt to cap_feats (hidden_states[-2]), frees the model and exits by itself
# - a clean DSP shutdown. Measured 09-28: ready 1.4 s, encode 0.86 s, +2.8 GB committed only while
# it lives, cos 1.00000 against the llama-server encoder.
ENCODE_EXE = BIN / "zimage-encode.exe"
CHAT_TEMPLATE = ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n")   # == /apply-template (checked)


class NpuEncoder:
    def __init__(self):
        import threading
        env = {**os.environ, "ADSP_LIBRARY_PATH": str(BIN)}
        self.p = subprocess.Popen([str(ENCODE_EXE), str(ENC), "HTP0", str(QWEN_PENULT_LAYER)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, encoding="utf-8", env=env,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.ready, self.err = threading.Event(), None
        threading.Thread(target=self._wait_ready, daemon=True).start()

    def _wait_ready(self):
        line = self.p.stdout.readline().strip()
        if line != "READY":
            self.err = line or "the encoder process ended"
        self.ready.set()

    def encode(self, prompt, workdir):
        self.ready.wait(180)
        if self.err:
            self.close()
            raise RuntimeError("text encoder: " + self.err)
        tf, of = Path(workdir) / "encode_in.txt", Path(workdir) / "encode_out.f32"
        tf.write_bytes((CHAT_TEMPLATE[0] + prompt + CHAT_TEMPLATE[1]).encode("utf-8"))
        self.p.stdin.write("%s\t%s\n" % (tf, of))
        self.p.stdin.flush()
        ans = self.p.stdout.readline().strip()
        self.close()                       # EOF: the process frees the model and exits by itself
        if not ans.startswith("DONE"):
            raise RuntimeError("text encoder: " + (ans or "no answer"))
        return np.fromfile(of, dtype=np.float32).reshape(-1, CAPD)

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass


_NPU_ENC = None


def npu_encoder_prestart():
    """REPL: called on the first key of a new prompt - the model loads while you type."""
    global _NPU_ENC
    if ENC_ON == "NPU" and _NPU_ENC is None and ENCODE_EXE.exists():
        _NPU_ENC = NpuEncoder()


def npu_encoder_release():
    global _NPU_ENC
    if _NPU_ENC is not None:
        _NPU_ENC.close()
        _NPU_ENC = None

''' + s[j:]

# 2) cap_feats: NPU-vagen forst
rep('''def cap_feats(prompt, workdir, verbose):
    """Prompt -> [Scap, 2560] float32. Varm server om den finns, annars llama-embedding."""
''', '''def cap_feats(prompt, workdir, verbose):
    """Prompt -> [Scap, 2560] float32. encoder=npu: kortlivad NPU-process (forstartad i repl).
    Annars varm server om den finns, annars llama-embedding."""
    global _NPU_ENC
    if ENC_ON == "NPU" and ENCODE_EXE.exists():
        enc = _NPU_ENC if _NPU_ENC is not None else NpuEncoder()
        _NPU_ENC = None
        a = enc.encode(prompt, workdir)
        if verbose:
            log(f"cap_feats {a.shape} std={a.std():.2f} (npu)")
        return a
''')

# 3) read_line: on_key vid forsta tangenten
rep("def read_line(prompt, prompt_width, idle=None):\n", "def read_line(prompt, prompt_width, idle=None, on_key=None):\n")
rep('''        return input(prompt)
''', '''        if on_key is not None:
            on_key()
        return input(prompt)
''')
rep('''        ch = msvcrt.getwch()
        if ch in ("\\r", "\\n"):''', '''        ch = msvcrt.getwch()
        if on_key is not None:                      # forsta tangenten: NPU-encodern laddar medan du skriver
            on_key()
            on_key = None
        if ch in ("\\r", "\\n"):''')

# 4) serve: forstart, idle-slapp, avslut
rep('''        freed, dropped = unload_models(dit)
''', '''        npu_encoder_release()
        freed, dropped = unload_models(dit)
''')
rep('''                busy = dit is not None or _ESR is not None
                line = read_line(G + "> " + O, 2, (unload_s, _idle) if (busy and unload_s > 0) else None)''',
    '''                busy = dit is not None or _ESR is not None or _NPU_ENC is not None
                line = read_line(G + "> " + O, 2, (unload_s, _idle) if (busy and unload_s > 0) else None,
                                 on_key=npu_encoder_prestart)''')
rep('''    finally:
        if dit: dit.close()
''', '''    finally:
        npu_encoder_release()
        if dit: dit.close()
''')
io.open(PY, "w", encoding="utf-8", newline="").write(s)

# 5) ps1: ingen llama-server i npu-lage
p = io.open(PS, encoding="ascii", newline="").read()
nl = "\r\n" if "\r\n" in p else "\n"


def rp(old, new):
    global p
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    if p.count(old) != 1:
        raise SystemExit("ps1: ankare %d ggr: %r" % (p.count(old), old[:70]))
    p = p.replace(old, new)


rp("""function Start-Encoder {
    if (Test-Encoder) { return }""", """function Start-Encoder {
    # npu: zimage.py starts zimage-encode.exe per prompt (nothing may sit on the NPU in between)
    if ($EncoderOn -eq 'npu') { return }
    if (Test-Encoder) { return }""")
rp("""    'serve'  { Start-Encoder; W "  the text encoder is listening on 127.0.0.1:$Port" Green }""",
   """    'serve'  {
        if ($EncoderOn -eq 'npu') { W '  the NPU text encoder starts per prompt - nothing to keep running (use -EncoderOn cpu for a warm server).' DarkGray; exit 0 }
        Start-Encoder; W "  the text encoder is listening on 127.0.0.1:$Port" Green
    }""")
rp("""        @('-EncoderOn cpu|npu', 'text encoder: cpu uses ~2.3 GB less memory, npu is ~1 s faster', "now $EncoderOn"),""",
   """        @('-EncoderOn npu|cpu', 'text encoder: npu loads per prompt while you type; cpu stays warm', "now $EncoderOn"),""")
io.open(PS, "w", encoding="ascii", newline="").write(p)

for J in JS:
    j = json.load(open(J, encoding="utf-8"))
    j["defaults"]["encoder"] = "npu"
    if "_help" in j:
        j["_help"]["defaults.encoder"] = ("npu (default): the encoder is loaded on the NPU for each prompt (while you type in the "
                                        "REPL) and released right after, so it never sits in memory next to the image model; "
                                        "cpu: a warm llama-server that reads the weights from the file cache (~0.5 GB)")
    io.open(J, "w", encoding="utf-8", newline="\n").write(json.dumps(j, indent=2, ensure_ascii=False) + "\n")

s = io.open(PY, encoding="utf-8", newline="").read()
s = s.replace('"vae": "taef1", "encoder": "cpu",', '"vae": "taef1", "encoder": "npu",')
io.open(PY, "w", encoding="utf-8", newline="").write(s)
print("OK")
