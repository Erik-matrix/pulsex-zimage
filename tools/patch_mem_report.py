# 09-28: var sitter DiT:ns ~4,9 GB committed? Vikterna ar 3,47 GB (GGUF Q4_0) - resten okant.
# Summera varje NPU-buffert per slag och skriv en [budget] minne-rad vid slutet.
import io
P = r"C:\PulseCore\PulseX\zimage_dit\zimage_stream.cpp"
s = io.open(P, encoding="utf-8", newline="").read()


def rep(old, new):
    global s
    if s.count(old) != 1:
        raise SystemExit("ankare %d ggr: %r" % (s.count(old), old[:60]))
    s = s.replace(old, new)


rep("struct StreamCtx {\n", "static size_t g_mem_w = 0, g_mem_act = 0, g_mem_io = 0;   // NPU-buffertar per slag (minnesrapport)\n\nstruct StreamCtx {\n")
rep("""        ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
""", """        ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        g_mem_w += ggml_backend_buffer_get_size(buf_w);
""")
rep("""            if (!sc.act_buf) { fprintf(stderr, "failed to allocate activation buffer (%zu B)\\n", 2 * need); exit(1); }
""", """            if (!sc.act_buf) { fprintf(stderr, "failed to allocate activation buffer (%zu B)\\n", 2 * need); exit(1); }
            g_mem_act = ggml_backend_buffer_get_size(sc.act_buf);
""")
rep("""            if (!sc.io_buf) { fprintf(stderr, "failed to allocate io buffer (%zu B)\\n", need); exit(1); }
""", """            if (!sc.io_buf) { fprintf(stderr, "failed to allocate io buffer (%zu B)\\n", need); exit(1); }
            g_mem_io = ggml_backend_buffer_get_size(sc.io_buf);
""")
rep("""    g_ph.lap("skriv lat_out");
    {
        const double k = 1e-6;
""", """    g_ph.lap("skriv lat_out");
    fprintf(stderr, "[budget] minne: vikter %.0f MB | graf (gallocr) %.0f MB | aktivering %.0f MB | io %.0f MB\\n",
            g_mem_w / 1048576.0, sc.galloc ? ggml_gallocr_get_buffer_size(sc.galloc, 0) / 1048576.0 : 0.0,
            g_mem_act / 1048576.0, g_mem_io / 1048576.0);
    {
        const double k = 1e-6;
""")
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("OK")
