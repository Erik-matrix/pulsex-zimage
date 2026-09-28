# Etiketten "(Qwen3-4B, NPU)" ljog sedan CPU blev forval. Den foljer nu ZI_ENCODER_ON
# (satt av zimage.ps1 fran -EncoderOn / zimage.json).
import io
P = r"C:\PulseCore\PulseX\zimage_dit\zimage.py"
s = io.open(P, encoding="utf-8", newline="").read()
old = 'st.done("encode the prompt  (Qwen3-4B, NPU)", t)'
assert s.count(old) == 2
s = s.replace(old, 'st.done("encode the prompt  (Qwen3-4B, %s)" % ENC_ON, t)')
anchor = "QWEN_PENULT_LAYER = 35"
assert s.count(anchor) == 1
i = s.index(anchor); j = s.index("\n", i) + 1
s = s[:j] + 'ENC_ON = "NPU" if os.environ.get("ZI_ENCODER_ON", "cpu") == "npu" else "CPU"   # for etiketten\n' + s[j:]
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("OK")
