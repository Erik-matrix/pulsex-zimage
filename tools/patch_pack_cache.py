# 09-28 ("Bygg cachefilen med forpackade vikter"): ZI_PACKCACHE=<fil>. Blockvikterna sparas
# EN gang i ggml-hexagons tegelpackade layout; sedan laddas de med memcpy i stallet for repack -
# bade vid start (residenta) och varje steg (ringen). Byggs automatiskt om filen saknas eller om
# nyckeln (GGUF-storlek + mtime + packtagg) inte stammer.   python tools/patch_pack_cache.py
import io

P = r"C:\PulseCore\PulseX\zimage_dit\zimage_stream.cpp"
s = io.open(P, encoding="utf-8", newline="").read()


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt:
        raise SystemExit("ankare %d ggr: %r" % (s.count(old), old[:80]))
    s = s.replace(old, new)


rep('''#include <atomic>
''', '''#include <sys/stat.h>
#include <cstddef>
#include <atomic>
''')

# 1) PackCache i StreamCtx + kallhjalpare
rep('''    std::map<std::string, bool> streamed;   // block som fatt nej av budgeten (beslutet ar stabilt)
};
''', '''    std::map<std::string, bool> streamed;   // block som fatt nej av budgeten (beslutet ar stabilt)
    // ZI_PACKCACHE: blockvikterna i tegelpackad form (se pack_cache_open). on = anvands.
    struct {
        bool   on = false;
        FILE * f  = nullptr;
        std::map<int64_t, std::pair<uint64_t, uint64_t>> ent;   // gguf-id -> (offset, bytes) i cachefilen
        bool (*get_packed)(const ggml_tensor *, void *, size_t) = nullptr;
        bool (*set_packed)(ggml_tensor *, const void *, size_t) = nullptr;
    } pk;
};

// Var en vikts bytes ligger: i packcachen (fardigpackad) eller i GGUF:en (ra, packas i tensor_set).
static bool weight_src(const StreamCtx & sc, int64_t id, size_t & off, size_t & sz) {
    if (sc.pk.on) {
        auto it = sc.pk.ent.find(id);
        if (it != sc.pk.ent.end()) { off = (size_t) it->second.first; sz = (size_t) it->second.second; return true; }
    }
    off = sc.data_base + gguf_get_tensor_offset(sc.gctx, id);
    sz  = gguf_get_tensor_size(sc.gctx, id);
    return false;
}
''')

# 2) prefetcher: las fran packcachen nar den ar pa (pf.f pekar da pa cachefilen)
rep('''            size_t off = sc->data_base + gguf_get_tensor_offset(sc->gctx, id);
            size_t sz  = gguf_get_tensor_size(sc->gctx, id);
            std::vector<uint8_t> b(sz);''', '''            size_t off, sz;
            weight_src(*sc, id, off, sz);
            std::vector<uint8_t> b(sz);''')

# 3) uppladdningen i run_block
rep('''            size_t off = sc.data_base + gguf_get_tensor_offset(sc.gctx, to_upload_id[i]);
            size_t sz = gguf_get_tensor_size(sc.gctx, to_upload_id[i]);
            const int64_t t_r0 = ggml_time_us();''', '''            size_t off, sz;
            const bool packed = weight_src(sc, to_upload_id[i], off, sz);
            FILE * src_f = packed ? sc.pk.f : sc.gguf_file;
            const int64_t t_r0 = ggml_time_us();''')
rep('''                _fseeki64(sc.gguf_file, (long long) off, SEEK_SET);
                size_t got = fread(stage.data(), 1, sz, sc.gguf_file);
                if (got != sz) { fprintf(stderr, "short read for %s\\n", to_upload[i]->name); exit(1); }''',
    '''                _fseeki64(src_f, (long long) off, SEEK_SET);
                size_t got = fread(stage.data(), 1, sz, src_f);
                if (got != sz) { fprintf(stderr, "short read for %s\\n", to_upload[i]->name); exit(1); }''')
rep('''            ggml_backend_tensor_set(to_upload[i], stage.data(), 0, sz);
            g_times.w_read += t_r1 - t_r0;''', '''            if (packed) {
                // fardigpackad: bara en kopia in i NPU-bufferten (+ samma flaggor som set_tensor satter)
                if (!sc.pk.set_packed(to_upload[i], stage.data(), sz)) {
                    fprintf(stderr, "packcache: %s passar inte bufferten (%zu B)\\n", to_upload[i]->name, sz); exit(1);
                }
            } else {
                ggml_backend_tensor_set(to_upload[i], stage.data(), 0, sz);
            }
            g_times.w_read += t_r1 - t_r0;''')

# 4) bygg/oppna cachen - fore prefetch-traden, som ska lasa ur den
rep('''    // Prefetch-traden far ETT EGET FILE*: delad filposition mellan tva tradar ar en
    // tystnande datatavling (_fseeki64 + fread ar inte atomiska tillsammans).
    if (ZI_PREFETCH) {
        sc.pf.f = fopen(gguf_path.c_str(), "rb");''', '''    if (const char * pc = getenv("ZI_PACKCACHE")) {
        if (*pc) pack_cache_open(sc, gguf_path, pc);
    }
    g_ph.lap("start: packcache");

    // Prefetch-traden far ETT EGET FILE*: delad filposition mellan tva tradar ar en
    // tystnande datatavling (_fseeki64 + fread ar inte atomiska tillsammans).
    if (ZI_PREFETCH) {
        sc.pf.f = fopen(sc.pk.on ? getenv("ZI_PACKCACHE") : gguf_path.c_str(), "rb");''')
rep('''    fclose(sc.gguf_file);
    gguf_free(sc.gctx);''', '''    if (sc.pk.f) fclose(sc.pk.f);
    fclose(sc.gguf_file);
    gguf_free(sc.gctx);''')

# 5) pack_cache_open fore run_block
PC = r'''
// ---- packcachen (ZI_PACKCACHE) ---------------------------------------------------------------
// Fil: 64 B huvud {magic "PXRP0001", tagg[32], gguf-storlek u64, gguf-mtime u64, n u32, 4 B},
// sedan n indexposter a 152 B {namn[96], typ u32, 0 u32, ne[4] i64, offset u64, bytes u64}, sedan
// data (4096-justerad per tensor). En post = en blocktensor exakt som den ligger i en hexagon-
// WEIGHTS-buffert efter set_tensor (tegelpackad for Q4_0, ra for f32).
struct PackHdr { char magic[8]; char tag[32]; uint64_t gsize, gmtime; uint32_t n, pad; };
struct PackEnt { char name[96]; uint32_t type, pad; int64_t ne[4]; uint64_t off, size; };
static_assert(sizeof(PackHdr) == 64 && sizeof(PackEnt) == 152, "packcache layout");

static bool is_block_tensor(const char * nm) {
    return !strncmp(nm, "layers.", 7) || !strncmp(nm, "noise_refiner.", 14) || !strncmp(nm, "context_refiner.", 16);
}

static void pack_cache_open(StreamCtx & sc, const std::string & gguf_path, const std::string & path) {
    const int64_t t0 = ggml_time_us();
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(sc.backend));
    auto tag_fn = (const char * (*)(void)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_pack_tag");
    sc.pk.get_packed = (bool (*)(const ggml_tensor *, void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_get_tensor_packed");
    sc.pk.set_packed = (bool (*)(ggml_tensor *, const void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_set_tensor_packed");
    if (!tag_fn || !sc.pk.get_packed || !sc.pk.set_packed) {
        fprintf(stderr, "[pack] backenden saknar packad-API - packcachen av\n");
        return;
    }
    struct _stat64 st;
    if (_stat64(gguf_path.c_str(), &st) != 0) return;
    PackHdr want{};
    memcpy(want.magic, "PXRP0001", 8);
    strncpy(want.tag, tag_fn(), sizeof(want.tag) - 1);
    want.gsize = (uint64_t) st.st_size;
    want.gmtime = (uint64_t) st.st_mtime;

    // alla blocktensorer, i GGUF-ordning, med sin packade storlek
    ggml_backend_buffer_type_t bt = ggml_backend_get_default_buffer_type(sc.backend);
    std::vector<int64_t> ids;
    std::vector<PackEnt> ents;
    {
        ggml_init_params p = { ggml_tensor_overhead() * 2, nullptr, true };
        const int64_t n = gguf_get_n_tensors(sc.gctx);
        for (int64_t id = 0; id < n; id++) {
            const char * nm = gguf_get_tensor_name(sc.gctx, id);
            if (!is_block_tensor(nm)) continue;
            if (strlen(nm) >= sizeof(PackEnt::name)) { fprintf(stderr, "[pack] for langt namn %s\n", nm); return; }
            ggml_context * c = ggml_init(p);
            ggml_tensor * t = ggml_new_tensor(c, gguf_get_tensor_type(sc.gctx, id), GGML_MAX_DIMS, gguf_get_tensor_ne(sc.gctx, id));
            PackEnt e{};
            strcpy(e.name, nm);
            e.type = (uint32_t) t->type;
            for (int k = 0; k < 4; k++) e.ne[k] = t->ne[k];
            e.size = ggml_backend_buft_get_alloc_size(bt, t);
            ggml_free(c);
            ids.push_back(id);
            ents.push_back(e);
        }
    }
    const uint64_t data0 = GGML_PAD(sizeof(PackHdr) + ents.size() * sizeof(PackEnt), 4096);
    {
        uint64_t off = data0;
        for (auto & e : ents) { e.off = off; off += GGML_PAD(e.size, 4096); }
    }

    // finns en giltig fil? (samma nyckel, samma poster)
    bool ok = false;
    if (FILE * f = fopen(path.c_str(), "rb")) {
        PackHdr h{};
        std::vector<PackEnt> got(ents.size());
        ok = fread(&h, sizeof h, 1, f) == 1 && !memcmp(&h, &want, offsetof(PackHdr, n)) && h.n == ents.size() &&
             fread(got.data(), sizeof(PackEnt), got.size(), f) == got.size() &&
             !memcmp(got.data(), ents.data(), got.size() * sizeof(PackEnt));
        fclose(f);
        if (!ok) fprintf(stderr, "[pack] %s ar inaktuell (annan modell, fil eller packlayout) - bygger om\n", path.c_str());
    }

    if (!ok) {
        // Bygg: varje tensor packas EN gang i en ateranvand WEIGHTS-buffert och sparas som den
        // blev. Skrivs till .tmp och doptes forst nar allt ar pa disk - en avbruten byggnad
        // lamnar aldrig en halv cache som ser giltig ut.
        const std::string tmp = path + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        if (!f) { fprintf(stderr, "[pack] kan inte skriva %s - packcachen av\n", tmp.c_str()); return; }
        uint64_t maxsz = 0;
        for (auto & e : ents) maxsz = e.size > maxsz ? e.size : maxsz;
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(bt, (size_t) maxsz);
        if (!b) { fclose(f); remove(tmp.c_str()); fprintf(stderr, "[pack] ingen byggbuffert\n"); return; }
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        PackHdr h = want;
        h.n = (uint32_t) ents.size();
        fwrite(&h, sizeof h, 1, f);
        fwrite(ents.data(), sizeof(PackEnt), ents.size(), f);
        std::vector<uint8_t> raw, packed;
        bool good = true;
        uint64_t total = 0;
        for (size_t i = 0; i < ents.size() && good; i++) {
            ggml_init_params p = { ggml_tensor_overhead() * 2, nullptr, true };
            ggml_context * c = ggml_init(p);
            ggml_tensor * t = ggml_new_tensor(c, (ggml_type) ents[i].type, GGML_MAX_DIMS, ents[i].ne);
            ggml_tallocr ta = ggml_tallocr_new(b);
            good = ggml_tallocr_alloc(&ta, t) == GGML_STATUS_SUCCESS;
            const size_t rsz = gguf_get_tensor_size(sc.gctx, ids[i]);
            raw.resize(rsz);
            _fseeki64(sc.gguf_file, (long long) (sc.data_base + gguf_get_tensor_offset(sc.gctx, ids[i])), SEEK_SET);
            good = good && fread(raw.data(), 1, rsz, sc.gguf_file) == rsz;
            if (good) ggml_backend_tensor_set(t, raw.data(), 0, rsz);   // tegelpackningen
            packed.assign((size_t) GGML_PAD(ents[i].size, 4096), 0);
            good = good && sc.pk.get_packed(t, packed.data(), (size_t) ents[i].size);
            _fseeki64(f, (long long) ents[i].off, SEEK_SET);
            good = good && fwrite(packed.data(), 1, packed.size(), f) == packed.size();
            total += ents[i].size;
            ggml_free(c);
        }
        ggml_backend_buffer_free(b);
        good = fclose(f) == 0 && good;
        if (good) remove(path.c_str());
        if (!good || rename(tmp.c_str(), path.c_str()) != 0) {
            remove(tmp.c_str());
            fprintf(stderr, "[pack] bygget misslyckades - packcachen av\n");
            return;
        }
        fprintf(stderr, "[pack] byggd: %zu tensorer, %.0f MB, %.1f s -> %s\n", ents.size(), total / 1048576.0,
                (ggml_time_us() - t0) * 1e-6, path.c_str());
    }

    sc.pk.f = fopen(path.c_str(), "rb");
    if (!sc.pk.f) return;
    for (size_t i = 0; i < ents.size(); i++) sc.pk.ent[ids[i]] = { ents[i].off, ents[i].size };
    sc.pk.on = true;
    fprintf(stderr, "[pack] packcache pa: %zu tensorer (%.2f s)\n", ents.size(), (ggml_time_us() - t0) * 1e-6);
}

'''
rep('''// runs one block: loads its weights from disk, computes, returns the new x (host-side).''',
    PC + '''// runs one block: loads its weights from disk, computes, returns the new x (host-side).''')

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("OK")
