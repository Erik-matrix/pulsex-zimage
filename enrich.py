"""Prompt enrichment: append material words that Z-Image-Turbo responds to.

Z-Image-Turbo is distilled to CFG=1: "photorealistic" does nothing, MATERIAL words do
("skin pores, catchlight in the eyes"). The table lives in cues.json (tools/extract_cues.py).

Rules kept from the old app, each one measured there:
  * whole words only ("cat" must not fire on "cathedral"); a trailing 's' is allowed
  * most specific key first (key length); 'front' cues before the rest
  * fragments are added while they FIT the budget, never cut mid-phrase - a long list of
    texture words made the old engine zoom in on surfaces and lose the composition
  * the enrichment is never hidden: the caller prints what is actually sent
"""
import json
from pathlib import Path

_T = None


def _table():
    global _T
    if _T is None:
        _T = json.loads((Path(__file__).with_name("cues.json")).read_text(encoding="utf-8"))
    return _T


def _word_at(low, i, step):
    """the word that begins at / ends before position i (step +1 / -1), after spaces; '' at a punctuation mark or the end"""
    while 0 <= i < len(low) and low[i] == " ":
        i += step
    j = i
    while 0 <= j < len(low) and (low[j].isalpha() or low[j] in "_'"):
        j += step
    return low[i:j] if step > 0 else low[j + 1:i + 1]


def _hasword(low, w, guard=None):
    """`guard` (cues.json "ambiguous"): the word only counts where it stands as a noun - followed by the end, a
    punctuation mark or one of guard["after"], and not preceded by one of guard["not_before"] ("ren skjorta",
    "skjortan är ren" are not reindeer)."""
    p = low.find(w)
    while p != -1:
        if not (p and (low[p - 1].isalpha() or low[p - 1] == "_")):
            e = p + len(w)
            if e < len(low) and low[e] == "s":
                e += 1
            if e >= len(low) or not low[e].isalpha():
                if guard is None:
                    return True
                nxt, prv = _word_at(low, e, 1), _word_at(low, p - 1, -1)
                if (nxt == "" or nxt in guard.get("after", ())) and prv not in guard.get("not_before", ()):
                    return True
        p = low.find(w, p + 1)
    return False


def _est_frag(s):
    w = p = 0
    inw = False
    for c in s:
        a = c.isalnum() or c == "'"
        if a and not inw:
            w += 1
            inw = True
        elif not a:
            inw = False
        if c in "-/":
            p += 1
    return w + p + 1


def _est_prompt(s):
    w = p = 0
    inw = False
    for c in s:
        a = c.isalnum()
        if a and not inw:
            w += 1
            inw = True
        elif not a:
            inw = False
        if c in "-/',.":
            p += 1
    return max(w + p, (len(s.encode("utf-8")) + 2) // 3)


def enrich(raw, table=None):
    """Returns (prompt_to_send, added_fragments). Unchanged prompt when nothing matches.
    `table`: another cue table with the same keys as cues.json (ltx_video merges in its video cues); default cues.json."""
    t = table if table is not None else _table()
    low = raw.lower()
    amb = t.get("ambiguous", {})               # words that are only sometimes the thing ("ren": reindeer / clean)
    hit = []
    for k, add in t["front"]:
        if _hasword(low, k, amb.get(k)):
            hit.append((len(k) + 1000, add))
    for k, add in t["cues"]:
        if _hasword(low, k, amb.get(k)):
            hit.append((len(k), add))
    for w in t["photo_words"]:
        if _hasword(low, w):
            hit.append((0, t["photo_add"]))
            break
    hit.sort(key=lambda h: -h[0])              # stable, like std::stable_sort
    add = []
    # Swedish words first add their English noun ("älg" -> "moose"): the model drops Swedish
    # nouns, and these must survive when the budget runs out (they are the subject).
    # Longest key first, and a key inside an already-matched longer key is skipped
    # ("blommig tapet" wins over "tapet"). The noun is skipped only if it is already a WORD
    # of the prompt - "neon" inside "neonskyltar" must still be added.
    # Nouns are added in the ORDER THE WORDS APPEAR in the prompt: the subject usually comes
    # first, and putting "princess cake" before "cafe" turned a café into a cake still life.
    matched = []
    for k in sorted(t.get("nouns", {}), key=len, reverse=True):
        if _hasword(low, k, amb.get(k)) and not any(k in m for m in matched):
            matched.append(k)
    for k in sorted(matched, key=lambda m: low.find(m)):
        noun = t["nouns"][k]
        if not _hasword(low, noun.lower()) and noun not in add:
            add.append(noun)
    for _, csv in hit:
        for frag in (f.strip() for f in csv.split(",")):
            if frag and frag.lower() not in low and frag not in add:
                add.append(frag)
    budget = int(t.get("budget_tokens", 22)) - _est_prompt(raw)
    out = raw.rstrip(" ,.")
    used = []
    for a in add:
        k = _est_frag(a)
        if k > budget:
            continue
        out += ", " + a
        budget -= k
        used.append(a)
    return (out if used else raw), used


if __name__ == "__main__":
    import sys
    p, u = enrich(" ".join(sys.argv[1:]))
    print(p)
