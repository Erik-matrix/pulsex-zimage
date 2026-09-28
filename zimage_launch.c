/*
 * zimage.exe - startare for zimage.ps1.
 *
 * Gor tre saker: hittar sin EGEN katalog (sa den fungerar oavsett var den anropas
 * ifran och var den lagts pa PATH), startar PowerShell med zimage.ps1 darifran, och
 * skickar vidare argumenten oforandrade. Exitkoden ar skriptets.
 *
 * Skalet att den finns: `zimage make "en katt"` i stallet for
 * `powershell -NoProfile -ExecutionPolicy Bypass -File C:\...\zimage.ps1 make "en katt"`,
 * plus att en exe kan bara ikonen. Motorn ar och forblir zimage.ps1 + zimage.py.
 *
 * Bygg: bld_cli.bat
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <stdio.h>
#include <wchar.h>

/* Citerar ett argument enligt Windows kommandoradsregler: backslash fore ett citattecken
 * maste dubblas, och bara da. Utan detta gar `-Out "D:\bild.jpg"` sonder pa slutslashen. */
static void append_quoted(wchar_t * dst, size_t cap, const wchar_t * arg) {
    size_t n = wcslen(dst);
    if (n + 3 >= cap) { return; }
    dst[n++] = L'"';
    for (const wchar_t * p = arg; *p && n + 3 < cap; ++p) {
        size_t slashes = 0;
        while (*p == L'\\') { slashes++; p++; }
        if (*p == L'\0') {
            for (size_t i = 0; i < slashes * 2 && n + 3 < cap; ++i) { dst[n++] = L'\\'; }
            break;
        }
        for (size_t i = 0; i < slashes * (*p == L'"' ? 2 : 1) && n + 3 < cap; ++i) { dst[n++] = L'\\'; }
        if (*p == L'"' && n + 3 < cap) { dst[n++] = L'\\'; }
        dst[n++] = *p;
    }
    dst[n++] = L'"';
    dst[n]   = L'\0';
}

/* Slapper en fils cachade sidor ur Windows standby-lista.
 *
 * En minnesmappad GGUF lamnar sina sidor kvar efter att processen dott - de rakas som
 * TILLGANGLIGA (de star i 'Cached', inte i 'In use'), men de syns i Task Manager och
 * behovs inte nar man ar klar. Att oppna filen med FILE_FLAG_NO_BUFFERING tvingar
 * cachehanteraren att tomma vyn for just den filen nar sista buffrade handtaget slapper.
 * Riktat mot VARA filer - ingen systemomfattande tomning, inga adminrattigheter.
 */
static int drop_file_cache(const wchar_t * path) {
    HANDLE h = CreateFileW(path, GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, NULL);
    if (h == INVALID_HANDLE_VALUE) { return 0; }
    CloseHandle(h);
    return 1;
}

/* Systemcachen i MB (standby-listan + systemets arbetsmangd). Billig och utan rattigheter -
 * zimage.ps1 fragade forut Get-Counter fore och efter, 1,0-1,4 s STYCK, efter varje bild. */
static long long system_cache_mb(void) {
    PERFORMANCE_INFORMATION pi = { sizeof(pi) };
    if (!GetPerformanceInfo(&pi, sizeof(pi))) { return -1; }
    return (long long) pi.SystemCache * (long long) pi.PageSize / (1024 * 1024);
}

/* Systemomfattande rensning, samma mekanism som PulseCore Memory / RAMMap / System Informer
 * (NtSetSystemInformation, SystemMemoryListInformation = 80). Kraver SeProfileSingleProcessPrivilege,
 * dvs en FORHOJD process - zimage.ps1 startar den via den schemalagda uppgiften "PulseX Purge"
 * (zimage purge-setup), sa ingen UAC-fraga vid varje stangning.
 *   standby : flush modifierade sidor + tom standby-listan  (Task Manager: "Cached" -> nara 0)
 *   full    : dessutom trimma ALLA processers arbetsmangd forst (som PulseCore Memory; hardare) */
typedef LONG (WINAPI * nt_set_sysinfo_t)(int, PVOID, ULONG);

static int enable_privilege(const wchar_t * name) {
    HANDLE tok = NULL;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) { return 0; }
    LUID luid;
    int  ok = 0;
    if (LookupPrivilegeValueW(NULL, name, &luid)) {
        TOKEN_PRIVILEGES tp = { 0 };
        tp.PrivilegeCount           = 1;
        tp.Privileges[0].Luid       = luid;
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
        AdjustTokenPrivileges(tok, FALSE, &tp, sizeof(tp), NULL, NULL);
        ok = (GetLastError() == ERROR_SUCCESS);   /* ERROR_NOT_ALL_ASSIGNED utan admin */
    }
    CloseHandle(tok);
    return ok;
}

static int purge_memory(int full) {
    enum { SystemMemoryListInformation = 80, MemoryEmptyWorkingSets = 2, MemoryFlushModifiedList = 3,
           MemoryPurgeStandbyList = 4 };
    HMODULE          nt  = GetModuleHandleW(L"ntdll.dll");
    nt_set_sysinfo_t set = nt ? (nt_set_sysinfo_t) GetProcAddress(nt, "NtSetSystemInformation") : NULL;
    if (!set) { return -1; }
    if (!enable_privilege(L"SeProfileSingleProcessPrivilege")) { return -2; }
    int cmd;
    if (full) { cmd = MemoryEmptyWorkingSets; set(SystemMemoryListInformation, &cmd, sizeof(cmd)); }
    cmd = MemoryFlushModifiedList; set(SystemMemoryListInformation, &cmd, sizeof(cmd));
    cmd = MemoryPurgeStandbyList;
    return set(SystemMemoryListInformation, &cmd, sizeof(cmd)) == 0 ? 0 : -3;
}

int wmain(int argc, wchar_t ** argv) {
    /* `zimage.exe --purge [full]` - kors forhojd av uppgiften "PulseX Purge". Sista raden ar
     * maskinlasbar: "purged <MB>" eller "purge needs-admin". Exitkod 0 / 2 (ej admin) / 1. */
    if (argc > 1 && wcscmp(argv[1], L"--purge") == 0) {
        const int       full   = (argc > 2 && wcscmp(argv[2], L"full") == 0);
        const long long before = system_cache_mb();
        const int       rc     = purge_memory(full);
        const long long after  = system_cache_mb();
        if (rc == -2) { wprintf(L"purge needs-admin\n"); return 2; }
        if (rc != 0)  { wprintf(L"purge failed %d\n", rc); return 1; }
        wprintf(L"purged %lld\n", (before >= 0 && after >= 0) ? before - after : -1LL);
        return 0;
    }
    /* `zimage.exe --drop-cache <fil>...` - anropas av zimage.ps1:s stop-verb. */
    if (argc > 1 && wcscmp(argv[1], L"--drop-cache") == 0) {
        /* Sista raden ar maskinlasbar: "dropped <MB>" (-1 om matningen inte gick). */
        const long long before = system_cache_mb();
        int n = 0;
        for (int i = 2; i < argc; ++i) { n += drop_file_cache(argv[i]); }
        const long long after = system_cache_mb();
        wprintf(L"[zimage] slappte cachen for %d fil(er)\n", n);
        wprintf(L"dropped %lld\n", (before >= 0 && after >= 0) ? before - after : -1LL);
        return 0;
    }

    wchar_t exe[MAX_PATH];
    DWORD   len = GetModuleFileNameW(NULL, exe, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) {
        fwprintf(stderr, L"zimage: hittar inte min egen sokvag\n");
        return 1;
    }
    wchar_t * slash = wcsrchr(exe, L'\\');
    if (slash) { *slash = L'\0'; }

    wchar_t script[MAX_PATH];
    _snwprintf_s(script, MAX_PATH, _TRUNCATE, L"%s\\zimage.ps1", exe);
    if (GetFileAttributesW(script) == INVALID_FILE_ATTRIBUTES) {
        fwprintf(stderr, L"zimage: hittar inte %s\n", script);
        return 1;
    }

    static wchar_t cmd[32768];
    cmd[0] = L'\0';
    wcscat_s(cmd, 32768, L"powershell.exe -NoProfile -ExecutionPolicy Bypass -File ");
    append_quoted(cmd, 32768, script);
    for (int i = 1; i < argc; ++i) {
        wcscat_s(cmd, 32768, L" ");
        append_quoted(cmd, 32768, argv[i]);
    }

    STARTUPINFOW        si = { sizeof(si) };
    PROCESS_INFORMATION pi = { 0 };
    if (!CreateProcessW(NULL, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        fwprintf(stderr, L"zimage: kunde inte starta PowerShell (fel %lu)\n", GetLastError());
        return 1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD rc = 1;
    GetExitCodeProcess(pi.hProcess, &rc);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return (int) rc;
}
