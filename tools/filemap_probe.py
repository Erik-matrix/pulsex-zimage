# Kontrollprov 09-27: kan den har drivrutinen ge DSP:n FILBACKADE sidor? ORT-QNN mappar
# kontextbinarens vikter fran filen som forval (disable_file_mapped_weights=0). Oppna samma
# ESRGAN-kontext (99 MB) med och utan och mat committed + processens privata minne.
#   python tools/filemap_probe.py 0|1
import os, sys, time, ctypes
from pathlib import Path
import numpy as np
import onnxruntime as ort
QNN = Path(r"C:\PulseCore\tools\qnn\2.46-runtime")
MODEL = r"C:\PulseCore\models\sdxl-qnn\esrgan_x4.bin.wrap.onnx"
os.add_dll_directory(str(QNN))

class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + \
               [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
class PERF(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong)] + [(n, ctypes.c_size_t) for n in ("CommitTotal", "CommitLimit", "CommitPeak",
                "PhysicalTotal", "PhysicalAvailable", "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged",
                "PageSize")] + [("HandleCount", ctypes.c_ulong), ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]
psapi = ctypes.WinDLL("psapi")
def snap():
    p = PMC(); p.cb = ctypes.sizeof(p)
    psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(p), p.cb)
    s = PERF(); s.cb = ctypes.sizeof(s); psapi.GetPerformanceInfo(ctypes.byref(s), s.cb)
    return p.PrivateUsage / 2**20, s.CommitTotal * s.PageSize / 2**20, s.SystemCache * s.PageSize / 2**20

flag = sys.argv[1]
so = ort.SessionOptions(); so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
x = np.random.default_rng(0).random((1, 3, 512, 512), dtype=np.float32)
time.sleep(1); a = snap()
s = ort.InferenceSession(MODEL, so, providers=["QNNExecutionProvider"],
                         provider_options=[{"backend_path": str(QNN / "QnnHtp.dll"), "htp_performance_mode": "burst",
                                            "disable_file_mapped_weights": flag}])
s.run(None, {"image": x}); time.sleep(1); b = snap()
print("disable_file_mapped_weights=%s: privat +%.0f MB, system commit +%.0f MB, cache +%.0f MB"
      % (flag, b[0] - a[0], b[1] - a[1], b[2] - a[2]))
del s
