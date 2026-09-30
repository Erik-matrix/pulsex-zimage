# sr_memory_1001.py — 2026-10-01: committed memory (private bytes) of the two upscaler sessions, opened and run once
# on the same 512 image. Each in its own process so nothing is shared.   python sr_memory_1001.py quicksrnet|esrgan
import ctypes, ctypes.wintypes as W, os, sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import zimage as Z


class PMC(ctypes.Structure):
    _fields_ = [("cb", W.DWORD), ("PageFaultCount", W.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]


def private_mb():
    c = PMC(); c.cb = ctypes.sizeof(c)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PrivateUsage / 2 ** 20


eng = sys.argv[1]
im = np.asarray(Image.open(r"D:\prov\2026-09-30_zimage_ab_bin\cat_512_v2_0.png").convert("RGB"))
Z.set_upscaler(eng)
m0 = private_mb(); c0 = Z.commit_mb()
(Z.qsr_session if eng == "quicksrnet" else Z.esrgan_session)()
m1 = private_mb(); c1 = Z.commit_mb()
Z.upscale_x4(im, 1024)
m2 = private_mb(); c2 = Z.commit_mb()
print("%s: private +%.0f / +%.0f MB, SYSTEM COMMIT +%.0f / +%.0f MB (session / after one upscale)" % (eng, m1 - m0, m2 - m0, c1 - c0, c2 - c0))
