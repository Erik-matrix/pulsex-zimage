# run_block: de sma indatatensorerna (i0, i1, i2, i_all, ff, adaln_t) i EN persistent buffert.
#
# Forut: en NY per-anropsbuffert per block (ggml_backend_alloc_ctx_tensors pa ctx) = en ny fd = en ny
# DSP-mappning, 130 ganger per bild, i en tabell med 64 platser (HTP_MAX_MMAPS). Var ~30:e block tog
# platserna slut och prep_op_bufs slappte ALLA 32-bitars mappningar (aven vikternas). Det ar
# omsattningen runt huvudmisstanken for 0x72 (mmap efter defrag, se ggml-hexagon
# tools_local/patch_0x72_diagnose.py). Nu: sc.io_buf allokeras en gang (vaxer om S vaxer) och
# tensorerna binds dit med ggml_backend_tensor_alloc - samma monster som act_buf. Med x_t ocksa bunden
# (ZI_ACT_RESIDENT) har ctx inget kvar att allokera, sa per-anropsbufferten forsvinner helt.
# Samma data, samma uppladdningar => bit-identiskt. Filen ar LF.
import io
P = r"C:\PulseCore\PulseX\zimage_dit\zimage_stream.cpp"
s = io.open(P, encoding="utf-8", newline="").read()
assert "\r\n" not in s and "io_buf" not in s


def rep(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        raise SystemExit("ankare %d ggr (vill %d): %r" % (n, count, old[:90]))
    s = s.replace(old, new)


rep("""    int64_t act_S    = 0;
};
""", """    int64_t act_S    = 0;
    // De sma indatatensorerna (ids, i_all, ff, adaln) - persistenta i stallet for en ny buffert per
    // block. Se patch_io_buf_persistent.py.
    ggml_backend_buffer_t io_buf = nullptr;
    size_t  io_cap   = 0;
};
""")

rep("""    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, sc.backend);
    if (!buf) { fprintf(stderr, "failed to allocate block buffer for %s\\n", pfx.c_str()); exit(1); }
    if (!ZI_RESIDENT) {
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
""", """    // Med residenta aktiveringar OCH vikter binds aven de sma indatatensorerna till en persistent
    // buffert: ingen ny fd/DSP-mappning per block (64 platser, 130 block per bild).
    const bool io_persist = res && ZI_RESIDENT;
    if (io_persist) {
        ggml_tensor * small[6] = { i0, i1, i2, i_all, ff, adaln_t };
        ggml_backend_buffer_type_t bt = ggml_backend_get_default_buffer_type(sc.backend);
        const size_t al = ggml_backend_buft_get_alignment(bt);
        size_t off[6];
        size_t need = 0;
        for (int k = 0; k < 6; k++) {
            off[k] = need;
            if (small[k]) need += GGML_PAD(ggml_nbytes(small[k]), al);
        }
        if (need > sc.io_cap) {
            if (sc.io_buf) ggml_backend_buffer_free(sc.io_buf);
            sc.io_buf = ggml_backend_buft_alloc_buffer(bt, need);
            if (!sc.io_buf) { fprintf(stderr, "failed to allocate io buffer (%zu B)\\n", need); exit(1); }
            sc.io_cap = need;
        }
        char * base = (char *) ggml_backend_buffer_get_base(sc.io_buf);
        for (int k = 0; k < 6; k++) {
            if (small[k] && ggml_backend_tensor_alloc(sc.io_buf, small[k], base + off[k]) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: kunde inte binda indatatensor %d\\n", pfx.c_str(), k); exit(1);
            }
        }
    }
    ggml_backend_buffer_t buf = nullptr;
    if (!io_persist) {
        buf = ggml_backend_alloc_ctx_tensors(ctx, sc.backend);
        if (!buf) { fprintf(stderr, "failed to allocate block buffer for %s\\n", pfx.c_str()); exit(1); }
        if (!ZI_RESIDENT) {
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
    }
""")

rep("""    const int64_t t_f0 = ggml_time_us();
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);""", """    const int64_t t_f0 = ggml_time_us();
    if (buf) ggml_backend_buffer_free(buf);
    ggml_free(ctx);""")

rep("""    if (sc.galloc) ggml_gallocr_free(sc.galloc);
    if (sc.act_buf) ggml_backend_buffer_free(sc.act_buf);
""", """    if (sc.galloc) ggml_gallocr_free(sc.galloc);
    if (sc.act_buf) ggml_backend_buffer_free(sc.act_buf);
    if (sc.io_buf) ggml_backend_buffer_free(sc.io_buf);
""")
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("OK")
