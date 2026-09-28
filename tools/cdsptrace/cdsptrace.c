/* cdsptrace.dll - loggar vilka FastRPC-/rpcmem-anrop QNN gor, i VAR EGEN process (2026-09-27).
 *
 * Fragan: QNN (ORT QNN EP, disable_file_mapped_weights=0) haller kontextbinarens vikter utanfor
 * committed-minnet. remote_mem_map(fd=-1) pa en filmappad vy ar AEE_EUNSUPPORTED pa den har
 * drivrutinen, sa QNN gor nagot annat. Vad?
 *
 * Metod: IAT-krok. trace_install() forladdar QnnHtp.dll (samma sokvag som ORT sedan anvander) och
 * byter GetProcAddress + LoadLibrary*-posterna i dess importtabell. Varje modul den sedan laddar
 * (stubben QnnHtpV73Stub.dll m.fl.) patchas likadant. Nar nagon slar upp en bevakad funktion i
 * libcdsprpc far den en loggande omslagsfunktion som anropar den riktiga. For varje pekare
 * loggas VirtualQuery: MAPPED (+ filnamn) / PRIVATE / IMAGE.
 * Anvands av tools/cdsptrace_probe.py. Ingen avkodning av nagons binar - bara anropsloggning.
 */
#include <windows.h>
#include <psapi.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static FILE *            g_log;
static CRITICAL_SECTION  g_cs;
static HMODULE           g_patched[64];
static int               g_npatched;

static void tlog(const char * fmt, ...) {
    if (!g_log) return;
    va_list ap;
    EnterCriticalSection(&g_cs);
    va_start(ap, fmt);
    vfprintf(g_log, fmt, ap);
    va_end(ap);
    fputc('\n', g_log);
    fflush(g_log);
    LeaveCriticalSection(&g_cs);
}

static void descr(const void * p, char * out, size_t n) {
    MEMORY_BASIC_INFORMATION mbi;
    if (!p || !VirtualQuery(p, &mbi, sizeof(mbi))) { snprintf(out, n, "-"); return; }
    const char * t = mbi.Type == MEM_MAPPED ? "MAPPED" : mbi.Type == MEM_PRIVATE ? "PRIVATE"
                   : mbi.Type == MEM_IMAGE ? "IMAGE" : "?";
    wchar_t fn[MAX_PATH] = L"";
    if (mbi.Type == MEM_MAPPED) K32GetMappedFileNameW(GetCurrentProcess(), (LPVOID) p, fn, MAX_PATH);
    snprintf(out, n, "%s base=%p off=0x%llx region=%zu prot=0x%lx file=%ls", t, mbi.AllocationBase,
             (unsigned long long) ((const char *) p - (const char *) mbi.AllocationBase), mbi.RegionSize,
             mbi.Protect, fn);
}

/* ---- omslag ------------------------------------------------------------------------------ */
typedef void * (*rpcmem_alloc_t)(int, uint32_t, int);
typedef void * (*rpcmem_alloc2_t)(int, uint32_t, size_t);
typedef void (*rpcmem_free_t)(void *);
typedef int (*rpcmem_to_fd_t)(void *);
typedef int (*fastrpc_mmap_t)(int, int, void *, int, size_t, int);
typedef int (*fastrpc_munmap_t)(int, int, void *, size_t);
typedef void (*reg_buf_t)(void *, int, int);
typedef void (*reg_buf_attr_t)(void *, int, int, int);
typedef void (*reg_buf_attr2_t)(void *, size_t, int, int);
typedef int (*reg_dma_t)(int, uint32_t);
typedef int (*reg_dma_attr_t)(int, uint32_t, uint32_t);
typedef void * (*reg_fd_t)(int, int);
typedef void * (*reg_fd2_t)(int, size_t);
typedef int (*mem_map_t)(int, int, int, uint64_t, size_t, uint64_t *);
typedef int (*mmap64_t)(int, uint32_t, uint64_t, int64_t, uint64_t *);

static rpcmem_alloc_t   r_rpcmem_alloc;
static rpcmem_alloc2_t  r_rpcmem_alloc2;
static rpcmem_free_t    r_rpcmem_free;
static rpcmem_to_fd_t   r_rpcmem_to_fd;
static fastrpc_mmap_t   r_fastrpc_mmap;
static fastrpc_munmap_t r_fastrpc_munmap;
static reg_buf_t        r_reg_buf;
static reg_buf_attr_t   r_reg_buf_attr;
static reg_buf_attr2_t  r_reg_buf_attr2;
static reg_dma_t        r_reg_dma;
static reg_dma_attr_t   r_reg_dma_attr;
static reg_fd_t         r_reg_fd;
static reg_fd2_t        r_reg_fd2;
static mem_map_t        r_mem_map;
static mmap64_t         r_mmap64;

static void * w_rpcmem_alloc(int h, uint32_t f, int sz) {
    void * p = r_rpcmem_alloc(h, f, sz);
    tlog("rpcmem_alloc heap=%d flags=0x%x size=%d -> %p", h, f, sz, p);
    return p;
}
static void * w_rpcmem_alloc2(int h, uint32_t f, size_t sz) {
    void * p = r_rpcmem_alloc2(h, f, sz);
    tlog("rpcmem_alloc2 heap=%d flags=0x%x size=%zu -> %p", h, f, sz, p);
    return p;
}
static void w_rpcmem_free(void * p) {
    tlog("rpcmem_free %p", p);
    r_rpcmem_free(p);
}
static int w_rpcmem_to_fd(void * p) {
    int fd = r_rpcmem_to_fd(p);
    char d[512]; descr(p, d, sizeof(d));
    tlog("rpcmem_to_fd %p -> %d  [%s]", p, fd, d);
    return fd;
}
static int w_fastrpc_mmap(int dom, int fd, void * a, int off, size_t len, int fl) {
    char d[512]; descr(a, d, sizeof(d));
    int rc = r_fastrpc_mmap(dom, fd, a, off, len, fl);
    tlog("fastrpc_mmap dom=%d fd=%d addr=%p off=%d len=%zu flags=%d -> 0x%x  [%s]", dom, fd, a, off, len, fl, rc, d);
    return rc;
}
static int w_fastrpc_munmap(int dom, int fd, void * a, size_t len) {
    int rc = r_fastrpc_munmap(dom, fd, a, len);
    tlog("fastrpc_munmap dom=%d fd=%d addr=%p len=%zu -> 0x%x", dom, fd, a, len, rc);
    return rc;
}
static void w_reg_buf(void * b, int sz, int fd) {
    char d[512]; descr(b, d, sizeof(d));
    tlog("remote_register_buf %p size=%d fd=%d  [%s]", b, sz, fd, d);
    r_reg_buf(b, sz, fd);
}
static void w_reg_buf_attr(void * b, int sz, int fd, int at) {
    char d[512]; descr(b, d, sizeof(d));
    tlog("remote_register_buf_attr %p size=%d fd=%d attr=0x%x  [%s]", b, sz, fd, at, d);
    r_reg_buf_attr(b, sz, fd, at);
}
static void w_reg_buf_attr2(void * b, size_t sz, int fd, int at) {
    char d[512]; descr(b, d, sizeof(d));
    tlog("remote_register_buf_attr2 %p size=%zu fd=%d attr=0x%x  [%s]", b, sz, fd, at, d);
    r_reg_buf_attr2(b, sz, fd, at);
}
static int w_reg_dma(int fd, uint32_t len) {
    int rc = r_reg_dma(fd, len);
    tlog("remote_register_dma_handle fd=%d len=%u -> 0x%x", fd, len, rc);
    return rc;
}
static int w_reg_dma_attr(int fd, uint32_t len, uint32_t at) {
    int rc = r_reg_dma_attr(fd, len, at);
    tlog("remote_register_dma_handle_attr fd=%d len=%u attr=0x%x -> 0x%x", fd, len, at, rc);
    return rc;
}
static void * w_reg_fd(int fd, int sz) {
    void * p = r_reg_fd(fd, sz);
    char d[512]; descr(p, d, sizeof(d));
    tlog("remote_register_fd fd=%d size=%d -> %p  [%s]", fd, sz, p, d);
    return p;
}
static void * w_reg_fd2(int fd, size_t sz) {
    void * p = r_reg_fd2(fd, sz);
    char d[512]; descr(p, d, sizeof(d));
    tlog("remote_register_fd2 fd=%d size=%zu -> %p  [%s]", fd, sz, p, d);
    return p;
}
static int w_mem_map(int dom, int fd, int fl, uint64_t va, size_t sz, uint64_t * rva) {
    char d[512]; descr((void *) (uintptr_t) va, d, sizeof(d));
    int rc = r_mem_map(dom, fd, fl, va, sz, rva);
    tlog("remote_mem_map dom=%d fd=%d flags=%d va=0x%llx size=%zu -> 0x%x rva=0x%llx  [%s]", dom, fd, fl,
         (unsigned long long) va, sz, rc, (unsigned long long) (rva ? *rva : 0), d);
    return rc;
}
static int w_mmap64(int fd, uint32_t fl, uint64_t va, int64_t sz, uint64_t * out) {
    char d[512]; descr((void *) (uintptr_t) va, d, sizeof(d));
    int rc = r_mmap64(fd, fl, va, sz, out);
    tlog("remote_mmap64 fd=%d flags=0x%x va=0x%llx size=%lld -> 0x%x out=0x%llx  [%s]", fd, fl,
         (unsigned long long) va, (long long) sz, rc, (unsigned long long) (out ? *out : 0), d);
    return rc;
}

struct watch { const char * name; void ** real; void * wrap; };
static struct watch W[] = {
    {"rpcmem_alloc", (void **) &r_rpcmem_alloc, (void *) w_rpcmem_alloc},
    {"rpcmem_alloc2", (void **) &r_rpcmem_alloc2, (void *) w_rpcmem_alloc2},
    {"rpcmem_free", (void **) &r_rpcmem_free, (void *) w_rpcmem_free},
    {"rpcmem_to_fd", (void **) &r_rpcmem_to_fd, (void *) w_rpcmem_to_fd},
    {"fastrpc_mmap", (void **) &r_fastrpc_mmap, (void *) w_fastrpc_mmap},
    {"fastrpc_munmap", (void **) &r_fastrpc_munmap, (void *) w_fastrpc_munmap},
    {"remote_register_buf", (void **) &r_reg_buf, (void *) w_reg_buf},
    {"remote_register_buf_attr", (void **) &r_reg_buf_attr, (void *) w_reg_buf_attr},
    {"remote_register_buf_attr2", (void **) &r_reg_buf_attr2, (void *) w_reg_buf_attr2},
    {"remote_register_dma_handle", (void **) &r_reg_dma, (void *) w_reg_dma},
    {"remote_register_dma_handle_attr", (void **) &r_reg_dma_attr, (void *) w_reg_dma_attr},
    {"remote_register_fd", (void **) &r_reg_fd, (void *) w_reg_fd},
    {"remote_register_fd2", (void **) &r_reg_fd2, (void *) w_reg_fd2},
    {"remote_mem_map", (void **) &r_mem_map, (void *) w_mem_map},
    {"remote_mmap64", (void **) &r_mmap64, (void *) w_mmap64},
};

/* ---- krokar -------------------------------------------------------------------------------- */
static void patch_module(HMODULE m);

/* ordinal -> exportnamn (09-28: uppslag via ordinal loggades inte alls forut) */
static const char * export_name_by_ordinal(HMODULE m, WORD ord) {
    BYTE * base = (BYTE *) m;
    IMAGE_NT_HEADERS * nt = (IMAGE_NT_HEADERS *) (base + ((IMAGE_DOS_HEADER *) base)->e_lfanew);
    IMAGE_DATA_DIRECTORY dd = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    if (!dd.VirtualAddress) return NULL;
    IMAGE_EXPORT_DIRECTORY * ed = (IMAGE_EXPORT_DIRECTORY *) (base + dd.VirtualAddress);
    DWORD * names = (DWORD *) (base + ed->AddressOfNames);
    WORD *  ords  = (WORD *) (base + ed->AddressOfNameOrdinals);
    for (DWORD i = 0; i < ed->NumberOfNames; i++) {
        if ((DWORD) ords[i] + ed->Base == ord) return (const char *) (base + names[i]);
    }
    return NULL;
}

static FARPROC WINAPI hk_GetProcAddress(HMODULE m, LPCSTR name) {
    FARPROC r = GetProcAddress(m, name);
    if (r && IS_INTRESOURCE(name)) {
        wchar_t mp[MAX_PATH] = L"";
        GetModuleFileNameW(m, mp, MAX_PATH);
        const char * en = export_name_by_ordinal(m, (WORD) (uintptr_t) name);
        tlog("GetProcAddress(ORDINAL %u = %s) in %ls", (unsigned) (uintptr_t) name, en ? en : "?", mp);
        if (!en) return r;
        name = en;
    }
    if (!r) return r;
    {
        wchar_t mp[MAX_PATH] = L"";
        GetModuleFileNameW(m, mp, MAX_PATH);
        if (wcsstr(mp, L"cdsprpc")) tlog("GetProcAddress(libcdsprpc, %s)", name);
    }
    for (size_t i = 0; i < sizeof(W) / sizeof(W[0]); i++) {
        if (strcmp(name, W[i].name) == 0) {
            *W[i].real = (void *) r;
            tlog("resolve %s", name);
            return (FARPROC) W[i].wrap;
        }
    }
    return r;
}
static HMODULE WINAPI hk_LoadLibraryExW(LPCWSTR n, HANDLE f, DWORD fl) {
    HMODULE h = LoadLibraryExW(n, f, fl);
    tlog("LoadLibraryExW %ls -> %p", n ? n : L"(null)", h);
    if (h) patch_module(h);
    return h;
}
static HMODULE WINAPI hk_LoadLibraryExA(LPCSTR n, HANDLE f, DWORD fl) {
    HMODULE h = LoadLibraryExA(n, f, fl);
    tlog("LoadLibraryExA %s -> %p", n ? n : "(null)", h);
    if (h) patch_module(h);
    return h;
}
static HMODULE WINAPI hk_LoadLibraryW(LPCWSTR n) {
    HMODULE h = LoadLibraryW(n);
    tlog("LoadLibraryW %ls -> %p", n ? n : L"(null)", h);
    if (h) patch_module(h);
    return h;
}
static HMODULE WINAPI hk_LoadLibraryA(LPCSTR n) {
    HMODULE h = LoadLibraryA(n);
    tlog("LoadLibraryA %s -> %p", n ? n : "(null)", h);
    if (h) patch_module(h);
    return h;
}


/* ---- fd-kedjan i QnnHtp: CreateFileMappingA / MapViewOfFile / DeviceIoControl / _open_osfhandle ---- */
static HANDLE WINAPI hk_CreateFileMappingA(HANDLE f, LPSECURITY_ATTRIBUTES sa, DWORD prot, DWORD hi, DWORD lo, LPCSTR nm) {
    HANDLE h = CreateFileMappingA(f, sa, prot, hi, lo, nm);
    tlog("CreateFileMappingA file=%p prot=0x%lx size=%lu:%lu name=%s -> %p", f, prot, hi, lo, nm ? nm : "-", h);
    return h;
}
static LPVOID WINAPI hk_MapViewOfFile(HANDLE m, DWORD acc, DWORD hi, DWORD lo, SIZE_T n) {
    LPVOID p = MapViewOfFile(m, acc, hi, lo, n);
    tlog("MapViewOfFile map=%p access=0x%lx off=%lu:%lu n=%zu -> %p", m, acc, hi, lo, n, p);
    return p;
}
static BOOL WINAPI hk_DeviceIoControl(HANDLE d, DWORD code, LPVOID in, DWORD nin, LPVOID out, DWORD nout, LPDWORD ret,
                                      LPOVERLAPPED ov) {
    char hx[3 * 48 + 1] = "";
    for (DWORD i = 0; in && i < nin && i < 48; i++) snprintf(hx + 3 * i, 4, "%02x ", ((unsigned char *) in)[i]);
    BOOL r = DeviceIoControl(d, code, in, nin, out, nout, ret, ov);
    char ho[3 * 32 + 1] = "";
    for (DWORD i = 0; r && out && ret && i < *ret && i < 32; i++) snprintf(ho + 3 * i, 4, "%02x ", ((unsigned char *) out)[i]);
    tlog("DeviceIoControl dev=%p code=0x%08lx in=%lu [%s] out=%lu -> %d ret=%lu [%s]", d, code, nin, hx, nout, r,
         ret ? *ret : 0, ho);
    return r;
}
typedef int (__cdecl * open_osf_t)(intptr_t, int);
static open_osf_t r_open_osf;
static int __cdecl hk_open_osfhandle(intptr_t h, int fl) {
    int fd = r_open_osf(h, fl);
    tlog("_open_osfhandle handle=%p flags=0x%x -> fd %d", (void *) h, fl, fd);
    return fd;
}

static int patch_iat(HMODULE m, const char * fn, void * nf) {
    BYTE * base = (BYTE *) m;
    IMAGE_DOS_HEADER * dos = (IMAGE_DOS_HEADER *) base;
    IMAGE_NT_HEADERS * nt  = (IMAGE_NT_HEADERS *) (base + dos->e_lfanew);
    IMAGE_DATA_DIRECTORY dd = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dd.VirtualAddress) return 0;
    int n = 0;
    for (IMAGE_IMPORT_DESCRIPTOR * d = (IMAGE_IMPORT_DESCRIPTOR *) (base + dd.VirtualAddress); d->Name; d++) {
        if (!d->OriginalFirstThunk) continue;
        IMAGE_THUNK_DATA * o = (IMAGE_THUNK_DATA *) (base + d->OriginalFirstThunk);
        IMAGE_THUNK_DATA * t = (IMAGE_THUNK_DATA *) (base + d->FirstThunk);
        for (; o->u1.AddressOfData; o++, t++) {
            if (IMAGE_SNAP_BY_ORDINAL(o->u1.Ordinal)) continue;
            IMAGE_IMPORT_BY_NAME * ibn = (IMAGE_IMPORT_BY_NAME *) (base + o->u1.AddressOfData);
            if (strcmp((const char *) ibn->Name, fn) != 0) continue;
            DWORD old;
            if (VirtualProtect(&t->u1.Function, sizeof(void *), PAGE_READWRITE, &old)) {
                t->u1.Function = (ULONG_PTR) nf;
                VirtualProtect(&t->u1.Function, sizeof(void *), old, &old);
                n++;
            }
        }
    }
    return n;
}

/* Statiska importer (stubben importerar libcdsprpc direkt): byt varje bevakad post i IAT:n. */
static int patch_iat_watch(HMODULE m) {
    BYTE * base = (BYTE *) m;
    IMAGE_DOS_HEADER * dos = (IMAGE_DOS_HEADER *) base;
    IMAGE_NT_HEADERS * nt  = (IMAGE_NT_HEADERS *) (base + dos->e_lfanew);
    IMAGE_DATA_DIRECTORY dd = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dd.VirtualAddress) return 0;
    int n = 0;
    for (IMAGE_IMPORT_DESCRIPTOR * d = (IMAGE_IMPORT_DESCRIPTOR *) (base + dd.VirtualAddress); d->Name; d++) {
        if (!d->OriginalFirstThunk) continue;
        IMAGE_THUNK_DATA * o = (IMAGE_THUNK_DATA *) (base + d->OriginalFirstThunk);
        IMAGE_THUNK_DATA * t = (IMAGE_THUNK_DATA *) (base + d->FirstThunk);
        for (; o->u1.AddressOfData; o++, t++) {
            if (IMAGE_SNAP_BY_ORDINAL(o->u1.Ordinal)) continue;
            const char * nm = (const char *) ((IMAGE_IMPORT_BY_NAME *) (base + o->u1.AddressOfData))->Name;
            for (size_t i = 0; i < sizeof(W) / sizeof(W[0]); i++) {
                if (strcmp(nm, W[i].name) != 0) continue;
                if (!*W[i].real) *W[i].real = (void *) t->u1.Function;
                DWORD old;
                if (VirtualProtect(&t->u1.Function, sizeof(void *), PAGE_READWRITE, &old)) {
                    t->u1.Function = (ULONG_PTR) W[i].wrap;
                    VirtualProtect(&t->u1.Function, sizeof(void *), old, &old);
                    tlog("iat-hook %s in %s", nm, (const char *) (base + d->Name));
                    n++;
                }
            }
        }
    }
    return n;
}

static void patch_module(HMODULE m) {
    EnterCriticalSection(&g_cs);
    for (int i = 0; i < g_npatched; i++) if (g_patched[i] == m) { LeaveCriticalSection(&g_cs); return; }
    if (g_npatched < 64) g_patched[g_npatched++] = m;
    LeaveCriticalSection(&g_cs);
    int n = patch_iat(m, "GetProcAddress", (void *) hk_GetProcAddress)
          + patch_iat(m, "LoadLibraryExW", (void *) hk_LoadLibraryExW)
          + patch_iat(m, "LoadLibraryExA", (void *) hk_LoadLibraryExA)
          + patch_iat(m, "LoadLibraryW", (void *) hk_LoadLibraryW)
          + patch_iat(m, "LoadLibraryA", (void *) hk_LoadLibraryA)
          + patch_iat_watch(m);
    {   /* fd-kedjan bara i QnnHtp.dll - overallt hangde processen (libcdsprpc:s egna ioctl-koer) */
        wchar_t q[MAX_PATH] = L"";
        GetModuleFileNameW(m, q, MAX_PATH);
        const wchar_t * b = wcsrchr(q, L'\\');
        if (b && _wcsicmp(b + 1, L"QnnHtp.dll") == 0) {
            n += patch_iat(m, "CreateFileMappingA", (void *) hk_CreateFileMappingA)
               + patch_iat(m, "MapViewOfFile", (void *) hk_MapViewOfFile)
               + patch_iat(m, "DeviceIoControl", (void *) hk_DeviceIoControl)
               + patch_iat(m, "_open_osfhandle", (void *) hk_open_osfhandle);
        }
    }
    wchar_t p[MAX_PATH] = L"";
    GetModuleFileNameW(m, p, MAX_PATH);
    tlog("patched %ls (%d IAT entries)", p, n);
}

__declspec(dllexport) int trace_install(const wchar_t * log_path, const wchar_t * qnn_htp) {
    InitializeCriticalSection(&g_cs);
    g_log = _wfopen(log_path, L"w");
    if (!g_log) return -1;
    r_open_osf = (open_osf_t) GetProcAddress(GetModuleHandleW(L"ucrtbase.dll"), "_open_osfhandle");
    HMODULE h = LoadLibraryExW(qnn_htp, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    tlog("preload %ls -> %p", qnn_htp, h);
    if (!h) return -2;
    patch_module(h);
    return 0;
}

/* Varden kan logga egna markorer (t.ex. commit fore/efter) i samma fil. */
__declspec(dllexport) void trace_note(const char * s) { tlog("NOTE %s", s); }

/* 09-28: kroka aven en modul som laddas UTANFOR QnnHtp-kedjan (ORT:s QNN-provider). */
__declspec(dllexport) int trace_patch_path(const wchar_t * path) {
    HMODULE h = LoadLibraryExW(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    tlog("preload %ls -> %p", path, h);
    if (!h) return -1;
    patch_module(h);
    return 0;
}
