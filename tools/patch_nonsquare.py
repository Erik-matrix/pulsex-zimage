# 09-28 ("480p, 854x480 ... skruva NPU:n till 842x480?"): icke-kvadratiska latenter i DiT-motorn
# och taef1-avkodaren. Sidorna maste vara delbara med 16 (VAE x8, patch x2) -> 848x480.
#   python tools/patch_nonsquare.py
import io

BS = chr(92)


def patch(path, pairs):
    s = io.open(path, encoding="utf-8", newline="").read()
    if "const int LW = Wt * PATCH" in s or "int hintH = 0" in s:
        print("redan patchad:", path)
        return
    nl = "\r\n" if "\r\n" in s else "\n"      # taef1_decode.cpp ar CRLF
    pairs = [(a.replace("\n", nl), b.replace("\n", nl)) for a, b in pairs]
    for a, b in pairs:
        if s.count(a) != 1:
            raise SystemExit("%s: ankare %d ggr: %r" % (path, s.count(a), a[:70]))
        s = s.replace(a, b)
    io.open(path, "w", encoding="utf-8", newline="").write(s)


patch(r"C:\PulseCore\PulseX\zimage_dit\zimage_stream.cpp", [
    ("    const int LH = Ht * PATCH;\n",
     "    const int LH = Ht * PATCH;\n    const int LW = Wt * PATCH;                 // 09-28: icke-kvadratiskt (t.ex. 848x480)\n"),
    ("rå latent [INCH,LH,LH]", "rå latent [INCH,LH,LW]"),
    ("if (xlat.size() != (size_t) INCH * LH * LH) {", "if (xlat.size() != (size_t) INCH * LH * LW) {"),
    ("xlat.size(), INCH * LH * LH);", "xlat.size(), INCH * LH * LW);"),
    ("n_steps, LH, LH, sig[0]", "n_steps, LH, LW, sig[0]"),
    ("// patchify: [INCH,LH,LH]", "// patchify: [INCH,LH,LW]"),
    ("lat[(size_t) c * LH * LH + (hi * PATCH + phi) * LH + (wi * PATCH + pwi)];",
     "lat[(size_t) c * LH * LW + (hi * PATCH + phi) * LW + (wi * PATCH + pwi)];"),
])

# taef1: --hw HxW anger latentens form (annars som forut: kvadrat ur filstorleken)
patch(r"C:\PulseCore\src\vae_engine\taef1_decode.cpp", [
    ('printf("bruk: taef1_decode.exe <latent.f32> <out.rgb> [out_f32] [--dump <dir>]' + BS + 'n");',
     'printf("bruk: taef1_decode.exe <latent.f32> <out.rgb> [out_f32] [--dump <dir>] [--hw HxW]' + BS + 'n");'),
    ("    const char* f32Path = nullptr;\n",
     "    const char* f32Path = nullptr;\n    int hintH = 0, hintW = 0;   // 09-28: --hw HxW for icke-kvadratiska latenter (848x480 -> 60x106)\n"),
    ('        if (!strcmp(argv[i], "--dump") && i + 1 < argc) { g_dump = argv[++i]; }\n',
     '        if (!strcmp(argv[i], "--dump") && i + 1 < argc) { g_dump = argv[++i]; }\n'
     '        else if (!strcmp(argv[i], "--hw") && i + 1 < argc) { sscanf(argv[++i], "%dx%d", &hintH, &hintW); }\n'),
    ("""        long hw = floats / LC;
        int side = (int)lrint(sqrt((double)hw));
        if ((long)side * side != hw) {
            printf("[vae] FEL: %ld element per kanal ar inte kvadratiskt""" + BS + """n", hw); return 1;
        }
        LH = LW = side; OUTH = LH * 8; OUTW = LW * 8;""",
     """        long hw = floats / LC;
        if (hintH > 0 && hintW > 0) {
            if ((long)hintH * hintW != hw) {
                printf("[vae] FEL: --hw %dx%d passar inte %ld element per kanal""" + BS + """n", hintH, hintW, hw); return 1;
            }
            LH = hintH; LW = hintW;
        } else {
            int side = (int)lrint(sqrt((double)hw));
            if ((long)side * side != hw) {
                printf("[vae] FEL: %ld element per kanal ar inte kvadratiskt (ange --hw HxW)""" + BS + """n", hw); return 1;
            }
            LH = LW = side;
        }
        OUTH = LH * 8; OUTW = LW * 8;"""),
])
print("OK")
