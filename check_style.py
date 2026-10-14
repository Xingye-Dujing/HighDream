"""Guard: no pure black and no pure blue anywhere in the product's own UI.

Vendored third-party bundles (MathJax, Font Awesome) and the omm-planner
reference file are skipped -- they are not ours to restyle.

    python check_style.py        # exits 1 on any violation

Escape hatch: a declaration that paints ink as a thin glyph rather than as a
panel may carry a trailing `style-guard: allow-glyph` comment, which is
visible in review.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent

# third-party / reference files that must not be edited
SKIP_PARTS = {"mathjax", "fontawesome", "omm-planner.html"}

BANNED = [
    (r"#000\b", "pure black hex"),
    (r"#000000\b", "pure black hex"),
    (r"\bblack\b", "the `black` keyword"),
    (r"background(?:-color)?\s*:\s*rgb\(\s*0\s*,\s*0\s*,\s*0\s*\)", "pure black rgb"),
    (r"#00f\b", "pure blue hex"),
    (r"#0000ff\b", "pure blue hex"),
    (r"#1976d2\b", "legacy brand blue"),
    (r"#5b89cd\b", "legacy brand blue"),
    (r"#3498db\b", "legacy spinner blue"),
    (r"#2196f3\b", "material blue"),
    (r"#3b7a70\b", "help-doc teal"),
    (r"#952d8b\b", "latex purple"),
    (r"#9c27b0\b", "autocomplete purple"),
    (r"#ff9800\b", "busy orange"),
    (r"#ef6c00\b", "truncation orange"),
    (r"#e65100\b", "warning orange"),
    (r"#4caf50\b", "success green"),
    (r"#2e7d32\b", "done green"),
    (r"#f44336\b", "error red"),
    (r"#c62828\b", "error red"),
]

DARK_FILL = re.compile(
    r"background(?:-color)?\s*:\s*(?:var\(--(?:text|jp-brand-color0|active-bg)\)"
    r"|#1a1a1a|#e8e8e8|#333\b|#222\b)",
    re.I,
)

EXEMPT = "style-guard: allow-glyph"


def strip_comments(src):
    """Blank out comment bodies but keep every newline, so line numbers in the
    result still line up with the original file."""
    def blank(m):
        return re.sub(r"[^\n]", " ", m.group(0))
    src = re.sub(r"/\*.*?\*/", blank, src, flags=re.S)
    src = re.sub(r"^[ \t]*//.*$", blank, src, flags=re.M)
    src = re.sub(r"(?<![:'\"])//[^\n\"`]*$", blank, src, flags=re.M)
    return src


def targets():
    for sub in ("static", "templates"):
        for pat in ("*.css", "*.js", "*.html"):
            for p in (ROOT / sub).rglob(pat):
                if any(part in SKIP_PARTS for part in p.parts) or p.name in SKIP_PARTS:
                    continue
                yield p


problems, checked = [], 0
for p in targets():
    raw = p.read_text(encoding="utf-8")
    lines = raw.splitlines()
    code = strip_comments(raw).splitlines()
    checked += 1
    for i, line in enumerate(code):
        if len(line) > 4000:      # minified vendor leftovers
            continue
        original = lines[i] if i < len(lines) else ""
        if EXEMPT in original:
            continue
        for pat, why in BANNED:
            if re.search(pat, line, re.I):
                problems.append((p.relative_to(ROOT), i + 1, why, line.strip()[:88]))
        if DARK_FILL.search(line):
            problems.append((p.relative_to(ROOT), i + 1, "dark fill", line.strip()[:88]))

# ---------------------------------------------------------------------------
# Invariant: text entry must look identical focused and unfocused.
# ---------------------------------------------------------------------------
TEXT_ENTRY = re.compile(
    r"(?:^|[\s,>])(?:input|textarea)(?=[\s,:>+~]|\[)"   # bare element
    r"|\.[a-z0-9-]*(?:input|textarea|expression|search)\b",  # a text field by class
    re.I,
)
FOCUS = re.compile(r":focus(?:-visible)?\b")
RESTYLE = re.compile(r"\b(?:border-color|background|background-color|box-shadow)\s*:",
                     re.I)

focus_problems = []
for p in targets():
    raw = p.read_text(encoding="utf-8")
    code = strip_comments(raw)
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", code):
        sel, body = m.group(1), m.group(2)
        if "@" in sel or not FOCUS.search(sel) or not TEXT_ENTRY.search(sel):
            continue
        if "checkbox" in sel or "radio" in sel:
            continue
        hit = RESTYLE.search(body)
        if hit:
            focus_problems.append(
                (p.relative_to(ROOT), " ".join(sel.split())[:60], hit.group(0).strip()))

if focus_problems:
    print("TEXT-ENTRY FOCUS GUARD FAILED -- a text field restyles itself on focus:")
    for f, sel, prop in focus_problems:
        print("  %-32s  %-46s %s" % (f, sel, prop))
    problems.append(("focus guard", 0, "", ""))

if problems:
    print()
    print("STYLE GUARD FAILED -- %d problem file(s):" % len(problems))
    for f, i, why, line in problems:
        if not why and not line:
            continue
        where = ("L%d" % i) if i else ""
        print("  %-32s %-5s %-24s %s" % (f, where, why, line))
    sys.exit(1)

print("style guard: clean -- %d files: no pure black, no pure blue, no dark fills,"
      % checked)
print("              and no text-entry control restyles itself on focus")
