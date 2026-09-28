# 09-28: svenska nycklar i berikningen + budget 50 tokens.
# Budgeten 22 kom fran gamla motorns 32-tokenstak (32 - 8 mall - 2 marginal); var motor har inget
# sadant tak, sa langa prompter fick forut NOLL tillagg. Svenska ord pekar pa samma fragment som
# motsvarande engelska nyckel (fragmenten ar pa engelska - det ar dem modellen svarar pa).
# Bojningsformer listas uttryckligen: ordgransmatchningen tillater bara ett efterstallt 's'.
import io, json
P = r"C:\PulseCore\PulseX\zimage_dit\cues.json"
c = json.load(open(P, encoding="utf-8"))
en = {k: v for k, v in c["cues"]}
SV = {
    # djur och natur (norrland)
    "älg": "moose", "älgen": "moose", "älgar": "moose", "björn": "bear", "björnen": "bear",
    "hackspett": "woodpecker", "kantarell": "chanterelle", "kantareller": "chanterelle",
    "svamp": "mushroom", "svampar": "mushroom", "norrsken": "aurora", "norrskenet": "aurora",
    "bäck": "stream", "bäcken": "stream", "björk": "birch", "björken": "birch", "björkar": "birch",
    "gran": "spruce", "granen": "spruce", "granar": "spruce", "tallen": "pine",
    "stuga": "cabin", "stugan": "cabin", "stugor": "cabin", "timmerstuga": "cabin",
    "lappland": "lapland", "nordisk": "nordic", "nordiskt": "nordic", "snöstorm": "blizzard",
    "skog": "forest", "skogen": "forest", "träd": "tree", "fjäll": "mountain", "fjället": "mountain",
    "berg": "mountain", "berget": "mountain", "sjö": "lake", "sjön": "lake", "älv": "river",
    "älven": "river", "flod": "river", "hav": "sea", "havet": "sea", "strand": "beach",
    "stranden": "beach", "snö": "snow", "snön": "snow", "regn": "rain", "dimma": "mist",
    "natt": "night", "natten": "night", "solnedgång": "sunset", "blomma": "flower",
    "blommor": "flower", "häst": "horse", "hästen": "horse", "uggla": "owl", "ugglan": "owl",
    "katten": "cat", "hunden": "dog",
    # älvarna (09-28: "forsarna, klassiskt tema") - gamla tabellens ÄLVARNA-sektion
    "fors": "rapids", "forsen": "rapids", "forsar": "rapids", "forsarna": "rapids",
    "vattenfall": "waterfall", "vattenfallet": "waterfall", "midnattssol": "midnight sun",
    "midnattssolen": "midnight sun", "älvstrand": "riverbank", "älvstranden": "riverbank",
    "stenblock": "boulder", "kanjon": "canyon", "fiske": "fishing", "flugfiske": "fishing",
    # fler norrlandsord (09-28) - befintliga engelska nycklar
    "havsis": "sea ice", "havsisen": "sea ice", "skärgård": "archipelago", "skärgården": "archipelago",
    "islossning": "thaw", "töväder": "thaw", "vass": "reeds", "vassen": "reeds", "sälg": "willow",
    "kaj": "quay", "kajen": "quay", "blockhus": "log cabin", "timmerhus": "log cabin",
    "isväg": "ice road", "isvägen": "ice road", "isbana": "ice track", "skoter": "snowmobile",
    "skotern": "snowmobile", "snöskoter": "snowmobile", "skridskor": "skating", "skridskoåkning": "skating",
    "ishall": "ice rink", "isrink": "ice rink", "skidbacke": "ski slope", "slalombacke": "ski slope",
    "strålkastare": "floodlight", "snöskulptur": "snow sculpture", "julbelysning": "christmas lights",
    "polarnatt": "polar night", "mörkertid": "polar night", "stadshus": "town hall", "rådhus": "town hall",
    "snövall": "snowbank", "snövallar": "snowbank", "plogvall": "snowbank", "rönn": "rowan",
    "rönnen": "rowan", "rönnbär": "rowan", "båthus": "boathouse", "sjöbod": "boathouse",
    "sjöbodar": "boathouse", "småbåtshamn": "marina", "gästhamn": "marina", "brygga": "jetty",
    "bryggan": "jetty", "pir": "pier", "piren": "pier", "ångbåt": "steamboat", "hjulångare": "steamboat",
    "kvarn": "mill", "sågverk": "mill", "fabrik": "factory", "vindkraftverk": "wind turbine",
    "skymning": "dusk", "skymningen": "dusk", "blåtimmen": "blue hour", "vimplar": "bunting",
    "kyrktorn": "church spire", "torg": "town square", "torget": "town square", "stationen": "station", "järnvägsstation": "station", "isen": "ice",
    "frosten": "frost", "rimfrost": "frost", "örn": "eagle", "örnen": "eagle", "fjällbjörk": "birch",
    "tjärn": "lake", "kalfjäll": "mountain", "kalfjället": "mountain", "fjällen": "mountain", "eld": "fire", "brasa": "fire",
    "lägereld": "fire",
    # människor
    "kvinna": "woman", "kvinnan": "woman", "mannen": "man", "flicka": "girl", "flickan": "girl",
    "tjej": "girl", "pojke": "boy", "pojken": "boy", "kille": "boy", "porträtt": "portrait",
    "ansikte": "face", "ansiktet": "face",
    # stad och hamn
    "bron": "bridge", "hamn": "harbour", "hamnen": "harbour", "båt": "boat",
    "båten": "boat", "bil": "car", "bilen": "car", "stad": "city", "staden": "city", "gata": "street",
    "gatan": "street", "bröd": "bread",
    # kaféer (sponsorns lokaler)
    "kafé": "cafe", "kafét": "cafe", "kaféet": "cafe", "kafe": "cafe", "café": "cafe",
    "kaffe": "coffee", "bulle": "bun", "bullar": "bun", "kanelbulle": "bun", "våffla": "waffle",
    "våfflor": "waffle", "ljusslinga": "string lights", "ljusslingor": "string lights",
    "pappersstjärna": "paper star", "adventsstjärna": "paper star", "emaljskylt": "enamel sign",
    "emaljskyltar": "enamel sign", "spets": "lace", "kyrka": "church",
    "kyrkan": "church", "kapell": "chapel", "stearinljus": "candle", "fåtölj": "armchair",
    "bordslampa": "table lamp", "krukväxt": "houseplant", "krukväxter": "houseplant",
    "skyltfönster": "shop window", "gardin": "curtain", "gardiner": "curtain", "kudde": "cushion",
    "kuddar": "cushion", "vinylskiva": "vinyl record", "vinylskivor": "vinyl record",
    "amerikanare": "classic car", "raggarbil": "classic car", "veteranbil": "classic car",
    # fler kaféord (09-28: "fler svenska ord för kaféerna")
    "kaféer": "cafe", "kaféerna": "cafe", "fik": "cafe", "fiket": "cafe", "konditori": "cafe",
    "konditoriet": "cafe", "serveringen": "servering", "kaffekopp": "coffee", "kaffekoppar": "coffee",
    "kopp": "coffee", "koppar": "coffee", "espresso": "coffee", "cappuccino": "coffee", "latte": "coffee",
    "fikat": "fika", "fikabröd": "fika", "kaka": "pastry", "kakor": "pastry", "bakelse": "pastry",
    "bakelser": "pastry", "wienerbröd": "pastry", "tårta": "pastry",
    "pepparkakor": "pastry",  "kardemummabulle": "bun",
    "ljusslingan": "string lights", "julstjärna": "paper star", "emaljskylten": "enamel sign",
    "reklamskylt": "enamel sign", "verandan": "veranda", "förstukvist": "porch", "farstukvist": "porch",
    "altan": "porch", "altanen": "porch", 
    "virkad": "lace", "virkat": "lace", "virkad duk": "lace", "femtiotal": "retro", "femtiotalet": "retro",
    "sextiotal": "retro", "loppis": "vintage", "loppisfynd": "vintage", "begagnade": "vintage",
    "gammaldags": "vintage", "patina": "vintage", "ljusstake": "candle", "levande ljus": "candle",
    "thonetstol": "thonet", "thonetstolar": "thonet", "böjträstol": "thonet", "böjträstolar": "thonet",
    "fåtöljen": "armchair", "fåtöljer": "armchair", "länstol": "armchair", "lampskärm": "lampshade",
    "lampskärmar": "lampshade", "golvlampa": "lampshade", "taklampa": "pendant lamp",
    "taklampor": "pendant lamp", "pendellampa": "pendant lamp", "industrilampa": "pendant lamp",
    "industrilampor": "pendant lamp", "blomkruka": "houseplant", "pelargon": "houseplant",
    "pelargoner": "houseplant", "skyltfönstret": "shop window", "draperi": "curtain",
    "draperier": "curtain", "dyna": "cushion", "dynor": "cushion",
    # dinern
    "dinern": "diner", "hamburgerbar": "diner", "gatukök": "diner", "rockabilly": "diner",
    "jukeboxen": "jukebox", "lp": "records", "lp-skiva": "records", "lp-skivor": "records",
    "skivor": "records", "skivomslag": "records", "neonskylt": "neon", "neonskyltar": "neon",
    "neonljus": "neon", "krom": "chrome", "förkromad": "chrome", "jänkare": "classic car",
    "jänkarbil": "classic car", "amerikanska bilar": "classic car", "hotrod": "hot rod",
    "trumset": "drum kit", "trummor": "drum kit", "gitarr": "guitar", "gitarren": "guitar",
    "väggmålning": "mural", "tegel": "brick", "tegelvägg": "brick",
}
# Sammansattningar och adjektiv (09-28: "lagg till granskog och snoig"): engelsk fras forst,
# sedan fragmenten fran varje ingaende engelsk nyckel.
COMPOUND = {
    "granskog": ("spruce forest", ["spruce", "forest"]), "granskogen": ("spruce forest", ["spruce", "forest"]),
    "tallskog": ("pine forest", ["pine", "forest"]), "tallskogen": ("pine forest", ["pine", "forest"]),
    "björkskog": ("birch forest", ["birch", "forest"]), "björkskogen": ("birch forest", ["birch", "forest"]),
    "snöig": ("snowy", ["snow"]), "snöigt": ("snowy", ["snow"]), "snöiga": ("snowy", ["snow"]),
    "snötäckt": ("snow-covered", ["snow"]), "snötäckta": ("snow-covered", ["snow"]),
    "kalixälven": ("wild northern river with rapids", ["river", "rapids"]),
    "råneälven": ("wild northern river with rapids", ["river", "rapids"]),
    "luleälven": ("wide northern river", ["river", "rapids"]),
    "piteälven": ("wild northern river with rapids", ["river", "rapids"]),
}
NEW = {
    "kakelugn": ("tiled stove", "glazed tiles, rounded corners, warm corner glow"),
    "kakelugnen": ("tiled stove", "glazed tiles, rounded corners, warm corner glow"),
    "vedspis": ("cast-iron wood stove", "warm firelight, soot-dark iron"),
    "trägolv": ("wooden floorboards", "worn boards, visible grain, soft sheen"),
    "plankgolv": ("wooden floorboards", "worn boards, visible grain, soft sheen"),
    "blommig tapet": ("floral wallpaper", "faded printed pattern, slight paper texture"),
    "tapet": ("patterned wallpaper", "faded printed pattern"),
    "tapeten": ("patterned wallpaper", "faded printed pattern"),
    "vaxduk": ("oilcloth tablecloth", "glossy printed pattern, slightly creased"),
    "rutig duk": ("checked tablecloth", "woven check pattern, soft folds"),
    "bokhylla": ("bookshelf", "worn book spines, uneven rows"),
    "bokhyllan": ("bookshelf", "worn book spines, uneven rows"),
    "kaffepanna": ("enamel coffee pot", "chipped enamel, steam from the spout"),
    "porslin": ("porcelain", "thin rim, glaze sheen, floral decor"),
    # norrlandsmotiv som saknades helt i tabellen (09-28)
    "ren": ("reindeer", "thick winter coat, antlers, breath vapour"),
    "renar": ("reindeer", "thick winter coat, antlers, breath vapour"),
    "renen": ("reindeer", "thick winter coat, antlers, breath vapour"),
    "renhjord": ("reindeer herd", "thick winter coats, antlers, breath vapour, trampled snow"),
    "räv": ("red fox", "dense red fur, bushy white-tipped tail"),
    "räven": ("red fox", "dense red fur, bushy white-tipped tail"),
    "varg": ("grey wolf", "thick grey fur, amber eyes"),
    "vargen": ("grey wolf", "thick grey fur, amber eyes"),
    "lodjur": ("lynx", "tufted ears, spotted fur, large paws"),
    "järv": ("wolverine", "dense dark brown fur, pale side stripe"),
    "tjäder": ("capercaillie", "dark iridescent feathers, fanned tail"),
    "ripa": ("ptarmigan", "white winter plumage, feathered feet"),
    "ripor": ("ptarmigans", "white winter plumage, feathered feet"),
    "myr": ("northern bog", "cotton grass, reflective pools, low shrubs"),
    "myren": ("northern bog", "cotton grass, reflective pools, low shrubs"),
    "hjortron": ("cloudberries", "amber-orange berries, bog leaves"),
    "lingon": ("lingonberries", "glossy red berries, small dark green leaves"),
    "blåbär": ("blueberries", "dusty blue bloom on the berries"),
    "bastu": ("sauna", "dark timber walls, steam, wood-fired stove"),
    "vedtrave": ("woodpile", "split birch logs, bark texture"),
    "vedstapel": ("woodpile", "split birch logs, bark texture"),
    "kåta": ("Sami lavvu tent", "birch poles, smoke hole, reindeer hides"),
    "kåtan": ("Sami lavvu tent", "birch poles, smoke hole, reindeer hides"),
    "kåsa": ("carved wooden kuksa cup", "birch burl grain, worn rim"),
    "sparkstötting": ("kick sled", "steel runners, wooden seat"),
    "hundspann": ("dog sled team", "huskies in harness, breath vapour, snow spray"),
    "gruva": ("iron ore mine", "rusted headframe, ore dust, industrial scale"),
    "gruvan": ("iron ore mine", "rusted headframe, ore dust, industrial scale"),
    "midsommar": ("Swedish midsummer", "maypole, flower wreaths, bright night sky"),
    "höst": ("autumn", "golden birch leaves, crisp air, low sun"),
    "hösten": ("autumn", "golden birch leaves, crisp air, low sun"),
    "vinter": ("winter", "snow, cold blue light"),
    "vintern": ("winter", "snow, cold blue light"),
    "sommarnatt": ("light summer night", "bright night sky, soft low light"),
    "hatt": ("hat", "felt brim, worn hatband"),
    "hatten": ("hat", "felt brim, worn hatband"),
    "mössa": ("knitted wool hat", "chunky knit texture"),
    # vyord (09-28): en uppraknings-prompt blev stilleben - bildutsnittet maste sagas
    "interiör": ("wide interior shot", "whole room visible, depth, window light"),
    "interiörbild": ("wide interior shot", "whole room visible, depth, window light"),
    "vidvinkel": ("wide-angle view", "whole scene visible, depth"),
    "helbild": ("wide-angle view", "whole scene visible, depth"),
    "närbild": ("close-up", "shallow depth of field"),
    # 09-28: "tårtan har grädde på sidorna, inte virkade mormors-mönster"
    "prinsesstårta": ("a Swedish princess cake on a table",
                      "smooth green marzipan dome, whipped cream on the sides, one slice cut out showing "
                      "sponge layers and cream, pink marzipan rose on top"),
    # platsbundna spetsord: den generiska "fine crochet texture" lade modellen PA tartan
    "spetsgardin": ("lace curtains in the window", "sheer lace, soft backlight"),
    "spetsgardiner": ("lace curtains in the window", "sheer lace, soft backlight"),
    "spetsduk": ("a lace tablecloth", "white lace on the table edge"),
    "semla": ("semla cream bun", "whipped cream, almond paste, powdered sugar lid"),
    "semlor": ("semla cream buns", "whipped cream, almond paste, powdered sugar lid"),
}
missing = sorted({v for v in SV.values() if v not in en} | {k for _, ks in COMPOUND.values() for k in ks if k not in en})
if missing:
    raise SystemExit("saknade engelska nycklar: %s" % missing)
# Bygg om de svenska raderna fran grunden (skriptet ska ga att kora om).
# Substantiven ligger i en EGEN tabell som enrich() lagger FORST: nar budgeten tog slut foll
# annars motivet bort ("alg" ar kortast och sorteras sist) medan materialorden fick plats.
sv_keys = set(SV) | set(COMPOUND) | set(NEW)
# Svenska nycklar som ocksa ar engelska ord slar till i ENGELSKA prompter ("a tall man" -> tallbarr,
# "a cat that is sleeping" -> is). De ar bortplockade; vakten hindrar att de kommer tillbaka.
ENGLISH_CLASH = {"is", "tall", "spark", "bro", "station", "man", "hat", "barn", "bad", "gift", "rock", "tar", "far"}
clash = sorted(k for k in sv_keys if k in ENGLISH_CLASH and k != "man")
if clash:
    raise SystemExit("svenska nycklar som ocksa ar engelska ord: %s" % clash)
old_sv = set(c.get("nouns", {}))           # forra korningens svenska rader (aven bortplockade)
PURGED = {"is", "tall", "spark", "bro"}      # svenska krockord som tidigare lagts in (09-28)
c["cues"] = [r for r in c["cues"] if r[0] not in sv_keys and r[0] not in old_sv and r[0] not in PURGED]
en = {k: v for k, v in c["cues"]}
nouns = {}
added = 0
for sv, key in SV.items():
    c["cues"].append([sv, en[key]]); nouns[sv] = key; added += 1
for sv, (phrase, keys) in COMPOUND.items():
    frags = []
    for k in keys:
        for f in en[k].split(","):
            f = f.strip()
            if f and f not in frags:
                frags.append(f)
    c["cues"].append([sv, ", ".join(frags)]); nouns[sv] = phrase; added += 1
for sv, (phrase, frags) in NEW.items():
    c["cues"].append([sv, frags]); nouns[sv] = phrase; added += 1
# Scenord for kaféerna: ett ensamt "cafe" forlorade mot ett konkret foremal (prinsesstartan blev
# ett stilleben, matt 09-28) - "cafe interior" sager att det ar ett RUM.
for k in ("kafé", "kafét", "kaféet", "kafe", "café", "kaféer", "kaféerna", "fik", "fiket", "konditori",
          "konditoriet", "servering", "serveringen"):
    if k in nouns:
        nouns[k] = "cafe interior"
c["nouns"] = nouns
c["budget_tokens"] = 70   # 09-28: 50 -> 70 sa prinsesstartan far plats
c["_about"] = c["_about"].rstrip() + (" Swedish keys point at the same (English) fragments as their English"
                                      " counterparts; inflected forms are listed explicitly.") \
    if "Swedish keys" not in c["_about"] else c["_about"]
io.open(P, "w", encoding="utf-8", newline="\n").write(json.dumps(c, indent=1, ensure_ascii=False) + "\n")
print("svenska nycklar tillagda:", added, "| budget", c["budget_tokens"], "| totalt", len(c["cues"]))
