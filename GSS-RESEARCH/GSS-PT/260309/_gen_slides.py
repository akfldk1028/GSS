"""Generate NotebookLM slide-deck from prompt file."""
import subprocess
import sys

prompt_file = r"C:\DK\GSS\GSS-RESEARCH\GSS-PT\260309\prompt_v4_algorithms_v2.txt"

with open(prompt_file, encoding="utf-8") as f:
    prompt = f.read().replace("\n", " ").replace("\r", "").strip()

# Set notebook context first
subprocess.run(["notebooklm", "use", "01581abe"], encoding="utf-8")

# Generate slide-deck
result = subprocess.run(
    ["notebooklm", "generate", "slide-deck", prompt, "--wait", "--language", "ko", "--retry", "10"],
    encoding="utf-8",
    capture_output=True,
    timeout=600,
)
print(result.stdout)
if result.stderr:
    print(result.stderr, file=sys.stderr)
