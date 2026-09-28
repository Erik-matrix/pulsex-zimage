# zimage.exe --drop-cache mater sjalv hur mycket systemcache den slappte (GetPerformanceInfo,
# billig, inga rattigheter) och skriver "dropped <MB>" som sista rad. zimage.ps1 fragade forut
# Get-Counter fore och efter - 1,0-1,4 s STYCK - efter varje make, nar bilden redan var klar.
# Filen ar LF. Backslash skrivs som chr(92): heredocs har gjort om \n till riktiga radslut.
import io
P = r"C:\PulseCore\PulseX\zimage_dit\zimage_launch.c"
s = io.open(P, encoding="utf-8", newline="").read()
assert "GetPerformanceInfo" not in s
BS = chr(92)


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)


rep("#include <windows.h>\n", "#include <windows.h>\n#include <psapi.h>\n")

old_head = "int wmain(int argc, wchar_t ** argv) {\n"
new_fn = (
    "/* Systemcachen i MB (standby-listan + systemets arbetsmangd). Billig och utan rattigheter -\n"
    " * zimage.ps1 fragade forut Get-Counter fore och efter, 1,0-1,4 s STYCK, efter varje bild. */\n"
    "static long long system_cache_mb(void) {\n"
    "    PERFORMANCE_INFORMATION pi = { sizeof(pi) };\n"
    "    if (!GetPerformanceInfo(&pi, sizeof(pi))) { return -1; }\n"
    "    return (long long) pi.SystemCache * (long long) pi.PageSize / (1024 * 1024);\n"
    "}\n\n")
rep(old_head, new_fn + old_head)

old = ("        int n = 0;\n"
       "        for (int i = 2; i < argc; ++i) { n += drop_file_cache(argv[i]); }\n"
       "        wprintf(L\"[zimage] slappte cachen for %d fil(er)" + BS + "n\", n);\n")
new = ("        /* Sista raden ar maskinlasbar: \"dropped <MB>\" (-1 om matningen inte gick). */\n"
       "        const long long before = system_cache_mb();\n"
       "        int n = 0;\n"
       "        for (int i = 2; i < argc; ++i) { n += drop_file_cache(argv[i]); }\n"
       "        const long long after = system_cache_mb();\n"
       "        wprintf(L\"[zimage] slappte cachen for %d fil(er)" + BS + "n\", n);\n"
       "        wprintf(L\"dropped %lld" + BS + "n\", (before >= 0 && after >= 0) ? before - after : -1LL);\n")
rep(old, new)
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("OK")
