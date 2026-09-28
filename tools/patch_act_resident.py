# Residualstrommen stannar pa NPU:n mellan blocken (ZI_ACT_RESIDENT, forval 1).
#
# Forut: fore VARJE block konverterades x till f16 pa varden och laddades upp (31,6 MB), efter
# blocket lastes 63 MB tillbaka - 130 ganger per 1024-bild ("in 2,0 | ut 0,3 s" i budgeten).
# Nu: en persistent HTP-buffert med tva platser (ping-pong). Blockets indata binds till plats A,
# dess sista nod (x_out, en vanlig ggml_add) till plats B med ggml_backend_tensor_alloc - gallocr
# hoppar over tensorer som redan har data, sa ingen extra kopia behovs. Uppladdning bara i
# kedjans forsta block (f32), nedlasning bara i dess sista (eller alltid med ZI_STATS/ZI_DUMP).
# ggml_backend_graph_compute ar synkron (async + synchronize), sa per-anropsbufferten kan
# fortfarande frias direkt efter.
# Numeriskt = ZI_F16_IO=0-vagen (ingen f16-avrundning vid blockgranserna) - den ar facit.
# Filen ar LF.
import io

P = r"C:\PulseCore\PulseX\zimage_dit\zimage_stream.cpp"
s = io.open(P, encoding="utf-8", newline="").read()
assert "\r\n" not in s and "ZI_ACT_RESIDENT" not in s


def rep(old, new, count=1):
    global s
    if s.count(old) != count:
        raise SystemExit("ankare %d ggr (vill %d): %r" % (s.count(old), count, old[:90]))
    s = s.replace(old, new)


# 1. StreamCtx: bufferten och vilken plats som galler
rep("""    ggml_gallocr_t galloc = nullptr;
};
""", """    ggml_gallocr_t galloc = nullptr;
    // ZI_ACT_RESIDENT: residualstrommen pa NPU:n mellan blocken, tva platser a act_slot byte.
    // act_cur = platsen som haller senaste blockets utdata (-1 = varden ar auktoritativ).
    ggml_backend_buffer_t act_buf = nullptr;
    size_t  act_slot = 0;
    int     act_cur  = -1;
    int64_t act_S    = 0;
};
""")

# 2. signaturen: in_dev (ta indata fran platsen) och out_host (las tillbaka)
rep("""                                     const std::string & next_pfx = std::string(), bool next_mod = false) {
    const bool want_stage_dbg = want_stage_dbg_in && ZI_STATS;""",
"""                                     const std::string & next_pfx = std::string(), bool next_mod = false,
                                     bool in_dev = false, bool out_host = true) {
    const bool want_stage_dbg = want_stage_dbg_in && ZI_STATS;""")

rep("""    static const int ZI_F16_IO = []{ const char * e = getenv("ZI_F16_IO"); return e ? atoi(e) : 1; }();
""", """    static const int ZI_F16_IO = []{ const char * e = getenv("ZI_F16_IO"); return e ? atoi(e) : 1; }();
    // Residualstrommen stannar pa NPU:n mellan blocken, se StreamCtx::act_buf. Da ar ZI_F16_IO
    // meningslos (ingen transport kvar) och x ar f32 hela vagen. ZI_ACT_RESIDENT=0 = gamla vagen.
    static const int ZI_ACT_RESIDENT = []{ const char * e = getenv("ZI_ACT_RESIDENT"); return e ? atoi(e) : 1; }();
    const bool res = ZI_ACT_RESIDENT != 0;
    if (!res) { in_dev = false; out_host = true; }
    if (in_dev && (sc.act_cur < 0 || sc.act_S != S)) {
        fprintf(stderr, "%s: in_dev utan giltig plats (cur %d, S %lld mot %lld)\\n", pfx.c_str(), sc.act_cur, (long long) sc.act_S, (long long) S);
        exit(1);
    }
    const bool f16_io = ZI_F16_IO && !res;
""")

rep("""    ggml_tensor * x_t = ggml_new_tensor_2d(ctx, ZI_F16_IO ? GGML_TYPE_F16 : GGML_TYPE_F32, DIM, S);""",
    """    ggml_tensor * x_t = ggml_new_tensor_2d(ctx, f16_io ? GGML_TYPE_F16 : GGML_TYPE_F32, DIM, S);""")

# 3. bind x_t till platsen FORE alloc_ctx_tensors (som da hoppar over den)
rep("""    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, sc.backend);
    if (!buf) { fprintf(stderr, "failed to allocate block buffer for %s\\n", pfx.c_str()); exit(1); }""",
"""    if (res) {
        const size_t need = GGML_PAD((size_t) DIM * S * sizeof(float), 4096);
        if (!in_dev && need > sc.act_slot) {          // bara nar varden laddar upp: inget att bevara
            if (sc.act_buf) ggml_backend_buffer_free(sc.act_buf);
            sc.act_buf = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(sc.backend), 2 * need);
            if (!sc.act_buf) { fprintf(stderr, "failed to allocate activation buffer (%zu B)\\n", 2 * need); exit(1); }
            sc.act_slot = need;
        }
        if (!in_dev) sc.act_cur = 0;                  // varden skriver plats 0 nedan
        char * base = (char *) ggml_backend_buffer_get_base(sc.act_buf);
        if (ggml_backend_tensor_alloc(sc.act_buf, x_t, base + (size_t) sc.act_cur * sc.act_slot) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: kunde inte binda x_t till aktiveringsplatsen\\n", pfx.c_str()); exit(1);
        }
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, sc.backend);
    if (!buf) { fprintf(stderr, "failed to allocate block buffer for %s\\n", pfx.c_str()); exit(1); }""")

# 4. uppladdningen
rep("""    if (ZI_F16_IO) {
        sc.in_buf.resize(x_in.size());""",
"""    if (in_dev) {
        // indata ligger redan i platsen sc.act_cur - forra blockets utdata
    } else if (f16_io) {
        sc.in_buf.resize(x_in.size());""")

rep("""    ggml_tensor * x_g = ZI_F16_IO ? ggml_cast(ctx, x_t, GGML_TYPE_F32) : x_t;""",
    """    ggml_tensor * x_g = f16_io ? ggml_cast(ctx, x_t, GGML_TYPE_F32) : x_t;""")

# 5. bind x_out till den andra platsen fore gallocr
rep("""    ggml_set_output(x_out);
    if (m_dbg) ggml_set_output(m_dbg);""",
"""    ggml_set_output(x_out);
    if (res) {
        if (x_out->view_src || !ggml_is_contiguous(x_out) || x_out->type != GGML_TYPE_F32 ||
            x_out->ne[0] != DIM || x_out->ne[1] != S) {
            fprintf(stderr, "%s: x_out kan inte bindas till aktiveringsplatsen\\n", pfx.c_str()); exit(1);
        }
        char * base = (char *) ggml_backend_buffer_get_base(sc.act_buf);
        if (ggml_backend_tensor_alloc(sc.act_buf, x_out, base + (size_t) (1 - sc.act_cur) * sc.act_slot) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: kunde inte binda x_out till aktiveringsplatsen\\n", pfx.c_str()); exit(1);
        }
    }
    if (m_dbg) ggml_set_output(m_dbg);""")

# 6. efter compute: platsen byter, nedlasning bara om den behovs
rep("""    ggml_backend_tensor_get(x_out, sc.out_buf.data(), 0, sc.out_buf.size() * sizeof(float));
    g_times.io_down += ggml_time_us() - t_d0;""",
"""    if (res) { sc.act_cur = 1 - sc.act_cur; sc.act_S = S; }
    if (out_host) ggml_backend_tensor_get(x_out, sc.out_buf.data(), 0, sc.out_buf.size() * sizeof(float));
    g_times.io_down += ggml_time_us() - t_d0;""")

# 7. frigor bufferten fore backenden
rep("""    if (sc.galloc) ggml_gallocr_free(sc.galloc);
""", """    if (sc.galloc) ggml_gallocr_free(sc.galloc);
    if (sc.act_buf) ggml_backend_buffer_free(sc.act_buf);
""")

# 8. anropsplatserna. Indata fran platsen fran och med andra blocket i en kedja; nedlasning i
#    sista blocket (varden behover x/cap/uni dar) eller alltid med ZI_STATS/ZI_DUMP.
rep("""        cap.swap(run_block(sc, "context_refiner." + std::to_string(r), false, cap, Scap, cap_ids0, cap_ids1, cap_ids2, {},
                           false, r + 1 < NR ? "context_refiner." + std::to_string(r + 1) : std::string(), false));""",
"""        cap.swap(run_block(sc, "context_refiner." + std::to_string(r), false, cap, Scap, cap_ids0, cap_ids1, cap_ids2, {},
                           false, r + 1 < NR ? "context_refiner." + std::to_string(r + 1) : std::string(), false,
                           r > 0, r + 1 == NR || ZI_STATS || ZI_DUMP));""")
rep("""        x.swap(run_block(sc, "noise_refiner." + std::to_string(r), true, x, Nimg, img_ids0, img_ids1, img_ids2, adaln, r == 0,
                         r + 1 < NR ? "noise_refiner." + std::to_string(r + 1) : std::string("layers.0"), true));""",
"""        x.swap(run_block(sc, "noise_refiner." + std::to_string(r), true, x, Nimg, img_ids0, img_ids1, img_ids2, adaln, r == 0,
                         r + 1 < NR ? "noise_refiner." + std::to_string(r + 1) : std::string("layers.0"), true,
                         r > 0, r + 1 == NR || ZI_STATS || ZI_DUMP));""")
rep("""        uni.swap(run_block(sc, "layers." + std::to_string(l), true, uni, Su, uni_ids0, uni_ids1, uni_ids2, adaln, l == 0,
                           l + 1 < NL ? "layers." + std::to_string(l + 1) : std::string(), true));""",
"""        uni.swap(run_block(sc, "layers." + std::to_string(l), true, uni, Su, uni_ids0, uni_ids1, uni_ids2, adaln, l == 0,
                           l + 1 < NL ? "layers." + std::to_string(l + 1) : std::string(), true,
                           l > 0, l + 1 == NL || ZI_STATS || ZI_DUMP));""")

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("OK")
