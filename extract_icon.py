# -*- coding: utf-8 -*-
# Hamtar HELA ikongruppen (alla storlekar) ur en PE-fil och skriver en riktig .ico.
# ExtractAssociatedIcon ger bara EN storlek; en .ico ar en katalog over flera.
import struct, sys
import ctypes as C
from ctypes import wintypes as W

src, dst = sys.argv[1], sys.argv[2]
RT_ICON, RT_GROUP_ICON = 3, 14
k = C.WinDLL("kernel32", use_last_error=True)
k.LoadLibraryExW.restype = W.HMODULE
k.FindResourceW.restype = W.HANDLE
k.LoadResource.restype = W.HANDLE
k.LockResource.restype = C.c_void_p
k.SizeofResource.restype = W.DWORD
k.LoadLibraryExW.argtypes = [W.LPCWSTR, W.HANDLE, W.DWORD]
k.FindResourceW.argtypes  = [W.HMODULE, C.c_void_p, C.c_void_p]
k.LoadResource.argtypes   = [W.HMODULE, W.HANDLE]
k.LockResource.argtypes   = [W.HANDLE]
k.SizeofResource.argtypes = [W.HMODULE, W.HANDLE]
LOAD_LIBRARY_AS_DATAFILE = 0x2

h = k.LoadLibraryExW(src, None, LOAD_LIBRARY_AS_DATAFILE)
if not h: sys.exit("kan inte oppna " + src)

names = []
ENUMPROC = C.WINFUNCTYPE(W.BOOL, W.HMODULE, C.c_void_p, C.c_void_p, C.c_void_p)
k.EnumResourceNamesW.argtypes = [W.HMODULE, C.c_void_p, ENUMPROC, C.c_void_p]
def cb(mod, typ, name, lp):
    names.append(name); return True   # name ar LPCWSTR eller ett heltals-ID maskerat som pekare
k.EnumResourceNamesW(h, C.c_void_p(RT_GROUP_ICON), ENUMPROC(cb), None)
if not names: sys.exit("ingen ikongrupp i filen")

def res(typ, name):
    r = k.FindResourceW(h, C.c_void_p(name) if isinstance(name, int) else name, C.c_void_p(typ))
    if not r: return None
    sz = k.SizeofResource(h, r); p = k.LockResource(k.LoadResource(h, r))
    return C.string_at(p, sz)

grp = res(RT_GROUP_ICON, names[0])
count = struct.unpack_from("<H", grp, 4)[0]
entries, images, off = [], [], 6 + 14 * count
for i in range(count):
    w_, h_, cc, _r, planes, bits, size, ident = struct.unpack_from("<BBBBHHIH", grp, 6 + 14*i)
    img = res(RT_ICON, ident)
    entries.append((w_, h_, cc, planes, bits, len(img), off)); images.append(img); off += len(img)

out = bytearray(struct.pack("<HHH", 0, 1, count))
for (w_, h_, cc, planes, bits, size, o) in entries:
    out += struct.pack("<BBBBHHII", w_, h_, cc, 0, planes, bits, size, o)
for img in images: out += img
open(dst, "wb").write(bytes(out))
print(f"{count} storlekar -> {dst} ({len(out)} byte): " +
      ", ".join(f"{e[0] or 256}x{e[1] or 256}" for e in entries))
