# 09-28 ("Krymp encoderns spik sa den inte ligger ovanpa DiT:n"): i serverlaget holl DiT:n
# ~0,9 GB committed mellan bilderna (ringplatser, gallocr, aktiverings-/io-buffertar, vardbuffertar)
# - och encodern (NPU, ~2,3 GB) laddas just mellan bilderna. ZI_SERVE_RELEASE (forval 1): slapp
# allt det efter varje bild; det allokeras latt igen vid nasta. Residenta vikter (tak -1) behalls.
#   python tools/patch_serve_release.py
import io

P = r"C:\PulseCore\PulseX\zimage_dit\zimage_stream.cpp"
s = io.open(P, encoding="utf-8", newline="").read()

old = '''    g_ph.lap("budgetutskrift");
    if (!ZI_SERVE) break;
'''
new = '''    g_ph.lap("budgetutskrift");
    if (!ZI_SERVE) break;
    // Serverlaget: slapp allt som inte ar residenta vikter medan vi vantar pa nasta bild - dar laddas
    // text-encodern pa NPU:n, och den ska inte ligga ovanpa oss. Allt allokeras latt igen.
    static const int ZI_SERVE_RELEASE = []{ const char * e = getenv("ZI_SERVE_RELEASE"); return e ? atoi(e) : 1; }();
    if (ZI_SERVE_RELEASE) {
        ggml_backend_synchronize(sc.backend);
        for (int k = 0; k < 4; k++) {
            if (sc.ring_buf[k]) { ggml_backend_buffer_free(sc.ring_buf[k]); sc.ring_buf[k] = nullptr; }
            if (sc.ring_ctx[k]) { ggml_free(sc.ring_ctx[k]); sc.ring_ctx[k] = nullptr; }
            sc.ring_owner[k].clear();
            sc.ring_rec[k] = false;
        }
        sc.ring_cap = 0; sc.ring_next = 0; g_mem_ring = 0;
        if (sc.galloc) { ggml_gallocr_free(sc.galloc); sc.galloc = nullptr; }
        if (sc.act_buf) { ggml_backend_buffer_free(sc.act_buf); sc.act_buf = nullptr; }
        sc.act_slot = 0; sc.act_cur = -1; sc.act_S = 0; g_mem_act = 0;
        if (sc.io_buf) { ggml_backend_buffer_free(sc.io_buf); sc.io_buf = nullptr; }
        sc.io_cap = 0; g_mem_io = 0;
        std::vector<float>().swap(sc.out_buf);
        std::vector<ggml_fp16_t>().swap(sc.in_buf);
        {
            std::lock_guard<std::mutex> lk(sc.pf.mu);
            if (sc.pf.inflight.empty()) { sc.pf.buf.clear(); sc.pf.have.clear(); }
        }
        g_ph.lap("serve: slapp buffertar");
    }
'''
if s.count(old) != 1:
    raise SystemExit("ankare %d ggr" % s.count(old))
s = s.replace(old, new)
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("OK")
