r"""Structural check of the LaTeX sources; it does not replace compiling.

    python paper/check_tex.py

Checks that braces and environments balance, every \cite has a bibliography
entry, every \ref has a \label, and every \input and \includegraphics target exists.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
tex = {p: re.sub(r"(?<!\\)%.*", "", p.read_text(encoding="utf-8")) for p in [ROOT / "main.tex", *sorted((ROOT / "sections").glob("*.tex"))]}
keys = set(re.findall(r"@\w+\{([^,]+),", (ROOT / "references.bib").read_text(encoding="utf-8")))
text = "\n".join(tex.values())
errors = []
for path, body in tex.items():
    plain = body.replace(r"\{", "").replace(r"\}", "")
    if plain.count("{") != plain.count("}"):
        errors.append(f"{path.name}: unbalanced braces")
    stack = []
    for kind, env in re.findall(r"\\(begin|end)\{(\w+\*?)\}", body):
        if kind == "begin":
            stack.append(env)
        elif not stack or stack.pop() != env:
            errors.append(f"{path.name}: end of {env} does not match")
    if stack:
        errors.append(f"{path.name}: unclosed {stack}")
for group in re.findall(r"\\cite\{([^}]+)\}", text):
    errors += [f"unknown citation {k}" for k in group.split(",") if k.strip() not in keys]
labels = set(re.findall(r"\\label\{([^}]+)\}", text))
errors += [f"unknown label {k}" for k in re.findall(r"\\ref\{([^}]+)\}", text) if k not in labels]
for target in re.findall(r"\\input\{([^}]+)\}", text):
    if not (ROOT / f"{target}.tex").exists():
        errors.append(f"missing input {target}")
for target in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
    if not list((ROOT / "figures").glob(f"{target}.*")):
        errors.append(f"missing figure {target}")
print("\n".join(sorted(set(errors))) or "ok")
sys.exit(1 if errors else 0)
