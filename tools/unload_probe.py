# Prov 09-27: ComfyUI-slappet i repl utan konsol. En bild (DitServer + ESRGAN), mat committed och
# standby, unload_models(), mat igen, och gor en bild till (maste ladda om och fungera).
# Kraver en varm encoder pa porten (CPU).  python tools/unload_probe.py <utkatalog>
import sys, time, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import zimage as Z


def standby_mb():
    import subprocess
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "[int]((Get-Counter '\\Memory\\Standby Cache Normal Priority Bytes').CounterSamples[0].CookedValue/1MB)"],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return -1


out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
tmp = Path(tempfile.mkdtemp(prefix="zi_unload_"))
print("fore:        commit %6.0f MB  standby %6d MB" % (Z.commit_mb(), standby_mb()))


def one(dit, seed, name):
    cap = Z.cap_feats("a girl in jungle, close up", tmp, False)
    Z.build_inputs(cap, 512, seed, tmp)
    Z.esrgan_prewarm()
    t = time.perf_counter()
    if dit is None:
        dit = Z.DitServer(tmp, 4, False)
    else:
        dit.render(tmp)
    lat = np.fromfile(tmp / "lat_out.f32", dtype=np.float32)
    im = Z.decode(lat, tmp)
    im = Z.upscale_x4(im, 1024)
    Z.save_image(im, out / name, 95, None, None, "bottom")
    print("%s klar %.1f s" % (name, time.perf_counter() - t))
    return dit


dit = one(None, 1235, "u1.jpg")
time.sleep(1)
print("laddad:      commit %6.0f MB  standby %6d MB" % (Z.commit_mb(), standby_mb()))
freed, dropped = Z.unload_models(dit)
dit = None
time.sleep(1)
print("slappt:      commit %6.0f MB  standby %6d MB   (unload_models: %.0f MB commit, %d MB cache)"
      % (Z.commit_mb(), standby_mb(), freed, dropped))
dit = one(None, 1236, "u2.jpg")
print("omladdad:    commit %6.0f MB" % Z.commit_mb())
dit.close()
