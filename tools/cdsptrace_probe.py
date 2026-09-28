# Spara QNN:s FastRPC-/rpcmem-anrop (2026-09-27). Oppnar ESRGAN-kontexten via ORT QNN EP med
# disable_file_mapped_weights=<0|1>, kor en inferens och stanger - med cdsptrace.dll inkrokad i
# QnnHtp.dll. Loggen visar om vikterna nar DSP:n via en FILMAPPAD vy (MAPPED + filnamn) eller
# via rpcmem (PRIVATE), och med vilka anrop.
#   python tools/cdsptrace_probe.py 0|1 <logg>
import ctypes, os, sys, time
from pathlib import Path
import numpy as np

QNN = Path(r"C:\PulseCore\tools\qnn\2.46-runtime")
MODEL = r"C:\PulseCore\models\sdxl-qnn\esrgan_x4.bin.wrap.onnx"
flag, logp = sys.argv[1], sys.argv[2]
os.add_dll_directory(str(QNN))
tr = ctypes.CDLL(str(Path(__file__).resolve().parent / "cdsptrace" / "cdsptrace.dll"))
tr.trace_install.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
tr.trace_note.argtypes = [ctypes.c_char_p]
rc = tr.trace_install(logp, str(QNN / "QnnHtp.dll"))
assert rc == 0, rc
# 09-28: aven ORT:s QNN-provider (egen rpcmem-vag utanfor QnnHtp)
import onnxruntime as _ort_pre
tr.trace_patch_path.argtypes = [ctypes.c_wchar_p]
_prov = Path(_ort_pre.__file__).parent / "capi" / "onnxruntime_providers_qnn.dll"
print("provider patch rc", tr.trace_patch_path(str(_prov)))
print("ort core patch rc", tr.trace_patch_path(str(Path(_ort_pre.__file__).parent / "capi" / "onnxruntime.dll")))


def commit_mb():
    class _PI(ctypes.Structure):
        _fields_ = ([("cb", ctypes.c_ulong)]
                    + [(k, ctypes.c_size_t) for k in ("CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal",
                       "PhysicalAvailable", "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")]
                    + [(k, ctypes.c_ulong) for k in ("HandleCount", "ProcessCount", "ThreadCount")])
    p = _PI(); p.cb = ctypes.sizeof(p)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb)
    return p.CommitTotal * p.PageSize / 2**20


import onnxruntime as ort
so = ort.SessionOptions(); so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
c0 = commit_mb()
tr.trace_note(("SESSION START disable_file_mapped_weights=%s commit=%.0f" % (flag, c0)).encode())
s = ort.InferenceSession(MODEL, so, providers=["QNNExecutionProvider"],
                         provider_options=[{"backend_path": str(QNN / "QnnHtp.dll"), "htp_performance_mode": "burst",
                                            "disable_file_mapped_weights": flag}])
c1 = commit_mb()
tr.trace_note(("SESSION READY commit=%+.0f" % (c1 - c0)).encode())
s.run(None, {"image": np.random.default_rng(0).random((1, 3, 512, 512), dtype=np.float32)})
c2 = commit_mb()
tr.trace_note(("RUN DONE commit=%+.0f" % (c2 - c0)).encode())
del s
time.sleep(0.5)
tr.trace_note(("SESSION CLOSED commit=%+.0f" % (commit_mb() - c0)).encode())
print("flag=%s  commit: session %+.0f MB, after run %+.0f MB" % (flag, c1 - c0, c2 - c0))
