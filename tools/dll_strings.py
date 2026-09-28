import re, sys
pat = re.compile(sys.argv[2], re.I)
data = open(sys.argv[1], "rb").read()
seen = set()
for m in re.finditer(rb"[\x20-\x7e]{5,}", data):
    t = m.group().decode()
    if pat.search(t) and t not in seen:
        seen.add(t); print(t)
