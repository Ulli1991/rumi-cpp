"""Generate docs/CONFIG.md from the parameter table (src/core/params.cpp), and check the table against the code.

    python3 scripts/gen_config_md.py           writes docs/CONFIG.md
    python3 scripts/gen_config_md.py --check   fails if docs/CONFIG.md is stale, if config.txt holds a key that is
                                               not level user (or misses one), if a cfg.get_* key in src/ is not in
                                               the table (or a table key is read nowhere), or if the NBODY_* list
                                               differs from the getenv calls in src/

Pure text, no build needed. The notes below NOTES_MARK in docs/CONFIG.md are written by hand and kept.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(ROOT, "src", "core", "params.cpp")
OUT = os.path.join(ROOT, "docs", "CONFIG.md")
CONFIG_TXT = os.path.join(ROOT, "config.txt")
NOTES_MARK = "<!-- notes: written by hand below this line; gen_config_md.py keeps them -->"
TYPES = {"B": "bool", "I": "int", "D": "double", "S": "string"}
LEVELS = {"USER": "user", "EXPERT": "expert", "TEST": "test", "INTERNAL": "internal", "EXPERIMENT": "**experiment**"}
TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|([A-Za-z_][A-Za-z_0-9]*)|([{},])|(//[^\n]*)|(\s+)')


def _entries(src, decl):
    """Rows of the brace-initialised vector `decl` in `src`: lists of fields (strings and identifiers)."""
    body = src[src.index(decl) + len(decl):]
    body = body[:body.index("\n};")]
    rows, row, field, depth, pos = [], None, None, 0, 0
    while pos < len(body):
        m = TOKEN.match(body, pos)
        if not m:
            raise SystemExit(f"{TABLE}: cannot parse near '{body[pos:pos + 40]}'")
        pos = m.end()
        lit, ident, punct = m.group(1), m.group(2), m.group(3)
        if lit is not None:
            field = (field or "") + bytes(lit, "utf-8").decode("unicode_escape")   # adjacent literals join
        elif ident is not None:
            field = ident
        elif punct == "{":
            depth += 1
            row, field = [], None
        elif punct and punct in ",}":
            if field is not None and row is not None:
                row.append(field)
            field = None
            if punct == "}":
                depth -= 1
                rows.append(row)
                row = None
    return rows


def table():
    src = open(TABLE).read()
    params = [dict(zip(("key", "type", "default", "section", "level", "values", "doc"), r))
              for r in _entries(src, "const std::vector<ParamSpec> kParams = {")]
    env = [dict(zip(("name", "doc"), r)) for r in _entries(src, "const std::vector<DebugEnvSpec> kDebugEnv = {")]
    for p in params:
        if len(p) != 7:
            raise SystemExit(f"{TABLE}: malformed row {p}")
    return params, env


def _cell(s):
    return s.replace("|", "\\|")


def markdown(params, env, notes):
    n_user = sum(p["level"] == "USER" for p in params)
    md = ["# Config reference", "",
          f"All {len(params)} keys of the config file (`config.txt`, or `--config <file>`), generated from the "
          "parameter table in `src/core/params.cpp` by `scripts/gen_config_md.py`. Every key has one default, "
          "in that table; an unknown key in a config file is an error. Each run writes the effective value of "
          "every key, and whether it came from the file or the default, to `run_params.txt`.", "",
          "Levels:", "",
          f"- **user**: the choices of a run; these {n_user} keys are in `config.txt`.",
          "- **expert**: accuracy and output knobs; the default is the production value.",
          "- **test**: parameters of the few-body test problems.",
          "- **internal**: performance tuning thresholds; results do not depend on them (or only through the "
          "reduction order).",
          "- **experiment**: open or decided A/B switches; temporary, to be removed once decided.", ""]
    sections = []
    for p in params:
        if p["section"] not in sections:
            sections.append(p["section"])
    for sec in sections:
        md += [f"## {sec}", "", "| key | type | default | level | values | description |",
               "|---|---|---|---|---|---|"]
        for p in params:
            if p["section"] != sec:
                continue
            default = f"`{_cell(p['default'])}`" if p["default"] else "(empty)"
            md.append(f"| `{p['key']}` | {TYPES[p['type']]} | {default} | {LEVELS[p['level']]} | "
                      f"{_cell(p['values']).replace('<number>', 'a number')} | {_cell(p['doc'])} |")
        md.append("")
    md += ["## Debug environment variables", "",
           "`NBODY_*` switches read with getenv (diagnostics and A/B checks; not config keys). Unset: off / "
           "the stated default.", "", "| variable | effect |", "|---|---|"]
    md += [f"| `{e['name']}` | {_cell(e['doc'])} |" for e in env]
    md += ["", NOTES_MARK, ""]
    if notes:
        md += [notes, ""]
    return "\n".join(md)


def _src_files():
    for d, _, files in os.walk(os.path.join(ROOT, "src")):
        for f in files:
            if f.endswith((".cpp", ".h")):
                yield os.path.join(d, f)


def check(params, env, text):
    errors = []
    if not os.path.exists(OUT) or open(OUT).read() != text:
        errors.append("docs/CONFIG.md is stale: run python3 scripts/gen_config_md.py")
    keys = {p["key"]: p for p in params}
    cfg_keys = re.findall(r"^\s*([A-Za-z_0-9]+)\s*=", open(CONFIG_TXT).read(), re.M)
    for k in cfg_keys:
        if k not in keys:
            errors.append(f"config.txt: {k} is not in the parameter table")
        elif keys[k]["level"] != "USER":
            errors.append(f"config.txt: {k} is level {keys[k]['level'].lower()}, not user")
    errors += [f"config.txt: user key {p['key']} missing" for p in params
               if p["level"] == "USER" and p["key"] not in cfg_keys]
    read, getenv = set(), set()
    for f in _src_files():
        s = open(f).read()
        read |= set(re.findall(r'cfg\.get_(?:str|int|double|bool)\(\s*"([A-Z_0-9]+)"', s))
        read |= set(re.findall(r'Config::to_double\(\s*"([A-Z_0-9]+)"', s))
        read |= set(k for a, b in re.findall(r'cfg\.get_\w+\([^;]*\?\s*"([A-Z_0-9]+)"\s*:\s*"([A-Z_0-9]+)"', s)
                    for k in (a, b))
        getenv |= set(re.findall(r'getenv\("(NBODY_[A-Z_0-9]+)"\)', s))
    errors += [f"src reads {k}, which is not in the parameter table" for k in sorted(read - set(keys))]
    errors += [f"table key {k} is read nowhere in src (cfg.get_*)" for k in sorted(set(keys) - read)]
    listed = {e["name"] for e in env}
    errors += [f"getenv {k} is not in kDebugEnv" for k in sorted(getenv - listed)]
    errors += [f"kDebugEnv {k} is read nowhere in src" for k in sorted(listed - getenv)]
    return errors


def main():
    params, env = table()
    notes = ""
    if os.path.exists(OUT):
        old = open(OUT).read()
        if NOTES_MARK in old:
            notes = old.split(NOTES_MARK, 1)[1].strip("\n")
    text = markdown(params, env, notes)
    if "--check" in sys.argv:
        errors = check(params, env, text)
        for e in errors:
            print("FAIL:", e)
        print(f"{len(params)} parameters, {len(env)} debug variables: "
              f"{'OK' if not errors else str(len(errors)) + ' problem(s)'}")
        return 1 if errors else 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(text)
    print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(params)} parameters, {len(env)} debug variables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
