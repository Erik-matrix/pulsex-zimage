# Foljd till patch_encoder_cpu.py: `-ArgumentList @(..) + $(..)` ar i kommandolage INTE ett
# uttryck - PowerShell skickar '+' som eget argument. Listan byggs i en variabel i stallet.
import io
PS = r"C:\PulseCore\PulseX\zimage_dit\zimage.ps1"
p = io.open(PS, encoding="ascii", newline="").read()
nl = "\r\n" if "\r\n" in p else "\n"
L = p.split(nl)
i = next(k for k, l in enumerate(L) if "Start-Process (Join-Path $Bin 'llama-server.exe')" in l)
assert L[i + 1].strip().startswith("-RedirectStandardOutput") and L[i + 2].strip().startswith("'-m', $Encoder")
j = next(k for k in range(i, i + 30) if L[k].strip() == "'--load-mode', 'mmap')")
body = L[i + 2:j + 1]
body[-1] = body[-1].replace("'--load-mode', 'mmap')", "'--load-mode', 'mmap')")
new = ["    $encArgs = @("] + body + [
    "    Start-Process (Join-Path $Bin 'llama-server.exe') -WindowStyle Hidden `",
    "        -RedirectStandardOutput \"$log.log\" -RedirectStandardError \"$log.err\" -ArgumentList $encArgs"]
L[i:j + 1] = new
io.open(PS, "w", encoding="ascii", newline="").write(nl.join(L))
print("OK")
