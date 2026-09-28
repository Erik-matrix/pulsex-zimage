# Fasklockor i zimage_stream.cpp:s main() - for att attribuera tiden UTANFOR blockloopen
# (motorprocessens vagg 58,6 s mot "block totalt" 42,1 s vid 1024 px = 16,5 s omatt).
#
# g_ph.lap(namn) bokfor tiden sedan forra lap -> fonstren ar DISJUNKTA och tacker hela main().
# Lapparna som innehaller run_block-loopar har "(block)" i namnet; rapporten drar av
# g_times.total fran dem sa att "mellan blocken" syns separat. g_ph.sub() ar "varav"-poster
# inne i ett fonster (raknas inte dubbelt i summan).
# Tiden FORE main (DLL-laddning, statisk init) och EFTER return (atexit, DSP-nedstangning)
# syns inte har: den ar processens vagg minus "main totalt".
#
# Filen ar LF. Ankare ar hela rader; varje ankare maste finnas exakt en gang.
import io

P = r"C:\PulseCore\PulseX\zimage_dit\zimage_stream.cpp"
s = io.open(P, encoding="utf-8", newline="").read()
assert "\r\n" not in s
assert "g_ph" not in s
L = s.split("\n")


def find(line, start=0):
    idx = [i for i in range(start, len(L)) if L[i] == line]
    if len(idx) != 1:
        raise SystemExit("ankare %d ggr: %r" % (len(idx), line))
    return idx[0]


def after(line, new):
    i = find(line)
    L[i + 1:i + 1] = new if isinstance(new, list) else [new]


def before(line, new):
    i = find(line)
    L[i:i] = new if isinstance(new, list) else [new]


def lap(name, ind="    "):
    return ind + 'g_ph.lap("%s");' % name


CLOCK = r"""// Fasklockor for main() utanfor blockloopen. lap() bokfor tiden sedan forra lap (disjunkta
// fonster over hela main), sub() ar en varav-post inne i ett fonster.
struct PhaseClock {
    // ggml_time_us() delar med en frekvens som satts forst i ggml_time_init() (INIT_ONCE):
    // en statisk initierare hade delat med noll. start() ar forsta satsen i main.
    int64_t t0 = 0, t = 0;
    void start() { ggml_time_init(); t0 = t = ggml_time_us(); }
    std::vector<std::pair<std::string, int64_t>> acc, subs;
    static void put(std::vector<std::pair<std::string, int64_t>> & v, const char * n, int64_t us) {
        for (auto & p : v) if (p.first == n) { p.second += us; return; }
        v.emplace_back(n, us);
    }
    void lap(const char * n) { const int64_t now = ggml_time_us(); put(acc, n, now - t); t = now; }
    void sub(const char * n, int64_t us) { put(subs, n, us); }
};
static PhaseClock g_ph;
""".split("\n")
before("int main(int argc, char ** argv) {", CLOCK)
after("int main(int argc, char ** argv) {", "    g_ph.start();")

after(r'    printf("[stream] final_layer backend: %s\n", ggml_backend_dev_description(final_dev));',
      lap("start: backend_load_all + 2 dev_init"))
i = find("        if (sc.pf.f) sc.pf.th = std::thread(prefetch_worker, &sc);")
assert L[i + 1] == "    }"
L[i + 2:i + 2] = [lap("start: gguf-index + prefetch-trad")]

# debugdumpen i embed() (x_embedder): extra 63 MB readback + skrivning till D:
i = find('        if (w_name.rfind("x_embedder", 0) == 0) {')
L[i:i] = ["        const int64_t t_dd = ggml_time_us();"]
j = next(k for k in range(i, len(L)) if L[k] == "            fclose(f);")
assert L[j + 1] == "        }"
L[j + 2:j + 2] = ['        g_ph.sub("x_embedder: debugdump (readback + D:)", ggml_time_us() - t_dd);']

before('    std::vector<float> cap = embed("cap_embedder.1.weight", "cap_embedder.1.bias", cap_raw, CAPD, DIM, Scap);',
       lap("bild: indata + schema"))
after('    dump("got_cap_after_embed.f32", cap);', lap("bild: cap_embedder"))
after(r'    printf("[stream] cap done\n");', lap("bild: context_refiner (block)"))

i = [k for k in range(len(L)) if L[k].startswith('    if (ZI_DUMP) { FILE * fa = fopen((dir + "\\\\got_adaln.f32")')]
assert len(i) == 1, i
L[i[0] + 1:i[0] + 1] = [lap("steg: adaln + patchify")]
after('    dump("got_x_after_embed.f32", x);', lap("steg: x_embedder"))
after(r'    printf("[stream] x done\n");', lap("steg: noise_refiner (block)"))
before("    std::vector<float> uni_prev;", lap("steg: uni-sammanfogning"))
before("    // ---- final_layer: adaLN scale (LayerNorm, not RMSNorm) + linear -> unpatchify ----",
       lap("steg: layers (block)"))
before("        auto up_fl = [&](ggml_tensor * t, const std::string & name) {", lap("final: ctx + buffert", "        "))
after('        up_fl(lin_b, "final_layer.linear.bias");', lap("final: vikter fran GGUF", "        "))
after("        ggml_backend_tensor_set(uni_t, uni.data(), 0, uni.size() * sizeof(float));",
      lap("final: indata upp", "        "))
after("        ggml_backend_graph_compute(final_backend, gf);", lap("final: gallocr + compute", "        "))
before("        // unpatchify: outs[c, hi*PATCH+phi, wi*PATCH+pwi] = outtok[(hi*Wt+wi), (phi*PATCH+pwi)*INCH+c]",
       lap("final: ut + fri", "        "))
before("    // FlowMatchEuler-steg: x += (sigma[i+1]-sigma[i]) * (-dit).  Samma matte som",
       lap("steg: unpatchify + statistik"))
before("    }  // ---- slut pa residens-loopen ----", lap("steg: schemauppdatering"))

i = find("        const double k = 1e-6;")
assert L[i - 1] == "    {"
L[i - 1:i - 1] = [lap("skriv lat_out")]
before("    if (!ZI_SERVE) break;", lap("budgetutskrift"))
after("    if (sc.galloc) ggml_gallocr_free(sc.galloc);", lap("slut: fri residenta vikter"))
after("    if (sc.pf.f) { fclose(sc.pf.f); sc.pf.f = nullptr; }", lap("slut: prefetch-join"))

REPORT = r"""    g_ph.lap("slut: gguf + backend_free");
    {
        const double k = 1e-6;
        int64_t sum = 0, blk = 0;
        for (auto & p : g_ph.acc) {
            fprintf(stderr, "[budget] fas: %-40s %6.2f s\n", p.first.c_str(), p.second * k);
            sum += p.second;
            if (p.first.find("(block)") != std::string::npos) blk += p.second;
        }
        for (auto & p : g_ph.subs) fprintf(stderr, "[budget] fas:   varav %-34s %6.2f s\n", p.first.c_str(), p.second * k);
        fprintf(stderr, "[budget] fas: main totalt %.2f s | block (run_block) %.2f | mellan blocken %.2f | utanfor blocken %.2f s\n",
                sum * k, g_times.total * k, (blk - g_times.total) * k, (sum - blk) * k);
    }""".split("\n")
i = find("    ggml_backend_free(final_backend);")
assert L[i + 1] == "    return 0;"
L[i + 1:i + 1] = REPORT

out = "\n".join(L)
io.open(P, "w", encoding="utf-8", newline="").write(out)
print("OK", out.count("g_ph.lap("), "lap")
