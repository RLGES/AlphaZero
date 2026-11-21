"""Write the public copies of the run-4 notes: no server address or admin commands.

python3 docs/run4/redact.py   # reads ../PROGRESS.md and ../RECOVERY_AND_RECIPE.md
"""

import re
from pathlib import Path

here = Path(__file__).parent
for name in ("PROGRESS.md", "RECOVERY_AND_RECIPE.md"):
    text = (here.parents[2] / name).read_text()
    # Drop server-admin sections (shared server, other services) entirely.
    text = re.sub(r"\*\*SixthSense on the server.*?(?=\n\*\*Each hour)", "", text, flags=re.S)
    text = re.sub(
        r"## Server \(Vultr.*?(?=\n## |\Z)",
        "## Server\n- CPU-only cloud server (8 vCPU, 15 GiB RAM); details kept off the repo.\n",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"root@\d+\.\d+\.\d+\.\d+|\d+\.\d+\.\d+\.\d+|\d+-\d+-\d+-\d+\.sslip\.io", "<server>", text
    )
    text = re.sub(r"ssh -i ~/\.ssh/\S+", "ssh", text)
    admin = re.compile(r"(?i)sixthsense|systemctl|docker compose|healthz|setu\b")
    lines = [line for line in text.splitlines() if not admin.search(line)]
    (here / name).write_text("\n".join(lines) + "\n")
    print(name, len(lines), "lines")
