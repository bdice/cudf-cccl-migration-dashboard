#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""
Report libcudf's progress on two in-flight migrations, per function and per file.

1. rmm::device_buffer / rmm::device_uvector<T> -> cuda::device_buffer<T>
   (rapidsai/build-planning#321)
2. rmm::device_async_resource_ref -> cudf::memory_resources (NVIDIA/cudf#24268)

Per-function states (separately for each migration):

  unmigrated   mr:  takes rmm::device_async_resource_ref only
               buf: references rmm::device_buffer/device_uvector and nothing
                    from cuda::device_buffer
  partial      mr:  takes cudf::memory_resources but still calls
                    get_current_device_resource_ref(), passes its mr to a callee that
                    still takes rmm::device_async_resource_ref, or leaves a callee's
                    mr argument defaulted
               buf: still references rmm types but also uses cuda::device_buffer, or
                    only calls repo functions that return rmm types
  migrated     nothing left to do
  not_applicable  mr: no memory resource parameter; buf: touches no device buffers
  in_progress  unmigrated/partial on the base ref, but an open PR (merged with the
               base ref) reduces the function's outstanding issues

Source analysis is a scope-aware scan of the C++ token stream (namespaces, classes,
function definitions and declarations, call sites); no build or compile database is
needed. Callees are resolved by C++ name lookup order and argument count, not by full
overload resolution; every flagged issue is listed with its line so it can be checked.

Requires: python >= 3.9, git >= 2.38; the `gh` CLI for open-PR analysis.
"""

import argparse
import bisect
import collections
import datetime
import hashlib
import json
import pickle
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PARSER_VERSION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
DEFAULT_ROOTS = [
    "cpp/include/cudf",
    "cpp/src",
    "cpp/libcudf_streaming/include",
    "cpp/libcudf_streaming/src",
    "cpp/libcudf_kafka/include",
    "cpp/libcudf_kafka/src",
]
SOURCE_SUFFIXES = {".cu", ".cuh", ".cpp", ".hpp", ".h", ".cc", ".hh"}
STATES = ["unmigrated", "partial", "in_progress", "migrated", "not_applicable"]
RANK = {"unmigrated": 0, "partial": 1, "migrated": 2, "not_applicable": 3}
CURRENT_MR = "get_current_device_resource_ref"
LEGACY_MR_TYPES = {
    "device_async_resource_ref",
    "device_resource_ref",
    "host_resource_ref",
    "host_device_resource_ref",
}
MR_HINTS = {
    CURRENT_MR,
    "get_temporary_mr",
    "get_output_mr",
    "as_temporary",
    "memory_resources",
}
ALIAS_SYNTAX = {".", "->", "(", ")", "{", "}", ",", "::", "cudf", "std", "move"}
NOT_FUNCTIONS = {
    "if", "while", "for", "switch", "return", "decltype", "sizeof", "alignas",
    "alignof", "__attribute__", "noexcept", "requires", "static_assert", "throw",
    "catch", "__launch_bounds__", "typeof", "defined", "__declspec", "else", "do",
    "case", "new", "delete", "co_await", "co_return", "co_yield", "typeid",
    "static_cast", "dynamic_cast", "reinterpret_cast", "const_cast", "explicit",
    "template", "using", "typedef", "asm", "__asm__", "operator", "typename",
}  # fmt: skip
SKIP_DECL_HEADS = {"using", "typedef", "static_assert", "friend", "return", "extern"}
BUILTIN_TYPE_WORDS = {
    "int", "long", "short", "char", "unsigned", "signed", "double", "float", "bool",
    "void", "const", "auto",
}  # fmt: skip
TOKEN_RE = re.compile(
    r"""
    (?P<comment>//[^\n]*|/\*.*?\*/)
  | (?P<string>(?:u8|u|U|L)?R"(?P<d>[^()\s]{0,16})\(.*?\)(?P=d)"
               |(?:u8|u|U|L)?"(?:\\.|[^"\\\n])*")
  | (?P<num>0[xX][0-9a-fA-F']+[uUlLzZ]*|\d[\w.']*)
  | (?P<char>'(?:\\.|[^'\\\n])*')
  | (?P<ident>[A-Za-z_]\w*)
  | (?P<punct>::|->|\.\.\.|<<=|>>=|<<|==|!=|<=|>=|&&|\|\||\+\+|--|[-+*/%&|^]=|\S)
    """,
    re.X | re.S,
)
PP_RE = re.compile(r"^[ \t]*#(?:\\\n|[^\n])*", re.M)
LAUNCH_RE = re.compile(r"<<<.*?>>>", re.S)
CUDA_QUAL_RE = re.compile(
    r"\b(?:__global__|__device__|__host__|__forceinline__|__noinline__|__shared__"
    r"|__constant__|__managed__|__restrict__)\b"
)


def blank(match):
    return re.sub(r"[^\n]", " ", match.group(0))


def tokenize(source):
    source = PP_RE.sub(blank, source)
    source = LAUNCH_RE.sub(blank, source)
    source = CUDA_QUAL_RE.sub(blank, source)
    texts, lines = [], []
    line, last = 1, 0
    for m in TOKEN_RE.finditer(source):
        kind = m.lastgroup if m.lastgroup != "d" else "string"
        line += source.count("\n", last, m.start())
        last = m.start()
        if kind == "comment":
            continue
        texts.append('""' if kind == "string" else m.group(0))
        lines.append(line)
    return texts, lines


def is_ident(text):
    return text[:1].isalpha() or text[:1] == "_"


def is_macro_name(text):
    return len(text) >= 3 and text == text.upper() and any(c.isalpha() for c in text)


class Scanner:
    """Extracts function definitions/declarations from one translation unit."""

    def __init__(self, source):
        self.T, self.L = tokenize(source)
        self.records = []
        self.claimed = set()
        self.scopes = []

    def match(self, i, open_, close):
        depth = 0
        n = len(self.T)
        while i < n:
            t = self.T[i]
            if t == open_:
                depth += 1
            elif t == close:
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return n - 1

    def match_back_angle(self, k):
        depth = 0
        while k >= 0:
            t = self.T[k]
            if t == ">":
                depth += 1
            elif t == "<":
                depth -= 1
                if depth == 0:
                    return k
            k -= 1
        return -1

    def scope_names(self, kind):
        return tuple(n for k, names in self.scopes for n in names if k == kind)

    def run(self):
        T = self.T
        n = len(T)
        pending = []
        i = 0
        paren = 0
        while i < n:
            t = T[i]
            if t in "([":
                paren += 1
            elif t in ")]":
                paren = max(0, paren - 1)
            if paren:
                pending.append(i)
                i += 1
                continue
            if t == "{":
                i, pending = self.open_brace(i, pending)
                continue
            if t == "}":
                if self.scopes:
                    self.scopes.pop()
                pending = []
            elif t == ";":
                self.statement(pending)
                pending = []
            elif t == ":" and [T[j] for j in pending] in (
                ["public"],
                ["private"],
                ["protected"],
            ):
                pending = []
            else:
                pending.append(i)
            i += 1
        return self.records

    def strip_template(self, head):
        """Return (head without template header, is_template, is_instantiation)."""
        T = self.T
        is_template = instantiation = False
        while head and T[head[0]] == "template":
            if len(head) > 1 and T[head[1]] == "<":
                depth = 0
                for k, idx in enumerate(head[1:], 1):
                    depth += (T[idx] == "<") - (T[idx] == ">")
                    if depth == 0:
                        break
                head = head[k + 1 :]
                is_template = True
            else:
                head = head[1:]
                instantiation = True
        return head, is_template, instantiation

    def open_brace(self, i, pending):
        T = self.T
        head, _, _ = self.strip_template(pending)
        words = [T[j] for j in head]
        if words and words[0] == "inline":
            words = words[1:]
        if words[:1] == ["namespace"]:
            names = [w for w in words[1:] if w not in ("::", "inline")]
            names = [w for w in names if is_ident(w) and not is_macro_name(w)]
            names = names or ["(anon)"]
            self.scopes.append(("ns", names))
            return i + 1, []
        if words[:1] == ["extern"]:
            self.scopes.append(("extern", []))
            return i + 1, []
        func = self.parse_head(head)
        if func is None and self.class_name(head) is not None:
            name = self.class_name(head)
            if words[0] == "enum" or "enum" in words[:3]:
                return self.match(i, "{", "}") + 1, []
            self.scopes.append(("cls", [name]))
            return i + 1, []
        if func is not None:
            tail = [T[j] for j in func["tail"]]
            in_init_list = ":" in tail and (is_ident(T[head[-1]]) or T[head[-1]] == ">")
            if in_init_list and T[head[-1]] not in ("const", "noexcept", "override"):
                end = self.match(i, "{", "}")
                return end + 1, pending + list(range(i, end + 1))
            end = self.match(i, "{", "}")
            self.add_record(func, "definition", body=(i + 1, end), open_brace=i)
            return end + 1, []
        end = self.match(i, "{", "}")
        return end + 1, pending + [i]

    def class_name(self, head):
        T = self.T
        depth = 0
        for k, idx in enumerate(head):
            t = T[idx]
            depth += (t in "([") - (t in ")]")
            if depth == 0 and t in ("class", "struct", "union", "enum"):
                rest = [T[j] for j in head[k + 1 :]]
                if rest[:1] == ["class"] or rest[:1] == ["struct"]:
                    rest = rest[1:]
                names = []
                skip = 0
                for w in rest:
                    if skip:
                        skip += (w == "(") - (w == ")")
                        continue
                    if w in ("alignas", "__attribute__"):
                        skip = 1
                        continue
                    if w in (":", "final", "<", "{"):
                        break
                    if is_ident(w):
                        names.append(w)
                real = [w for w in names if not is_macro_name(w)] or names
                return real[0] if real else ""
        return None

    def statement(self, pending):
        T = self.T
        if (
            len(pending) > 1
            and T[pending[0]] == "extern"
            and T[pending[1]] == "template"
        ):
            pending = pending[1:]
        head, _, instantiation = self.strip_template(pending)
        if not head or T[head[0]] in SKIP_DECL_HEADS:
            return
        func = self.parse_head(head)
        if func is None:
            return
        tail = [T[j] for j in func["tail"]]
        if "delete" in tail or "default" in tail:
            return
        kind = "instantiation" if instantiation else "declaration"
        self.add_record(func, kind)

    def parse_head(self, head):
        """Locate `qualified::name(params) tail` in a declaration head."""
        T = self.T
        depth = ang = 0
        k = 0
        n = len(head)
        while k < n:
            t = T[head[k]]
            prev = T[head[k - 1]] if k else ""
            if t == "=" and depth == 0 and ang == 0 and prev != "operator":
                return None
            if t == "<" and prev != "operator" and (is_ident(prev) or prev == ">"):
                ang += 1
            elif t == ">" and ang and prev != "operator":
                ang -= 1
            elif t in "[{":
                depth += 1
            elif t in "]}":
                depth -= 1
            elif t == "(" and depth == 0 and ang == 0:
                found = None if prev == "operator" else self.name_before(head, k)
                end = self.match(head[k], "(", ")")
                if found is not None:
                    name, quals, start = found
                    close = bisect.bisect_left(head, end)
                    return {
                        "name": name,
                        "quals": quals,
                        "ret": head[:start],
                        "params": (head[k] + 1, end),
                        "tail": head[close + 1 :],
                        "line": self.L[head[k]],
                    }
                close = bisect.bisect_left(head, end)
                if close >= n or head[close] != end:
                    return None
                k = close
            k += 1
        return None

    def name_before(self, head, k):
        T = self.T
        pos = [T[j] for j in head]
        j = k - 1
        if j >= 0 and pos[j] == ">":
            j = bisect.bisect_left(head, self.match_back_angle(head[j])) - 1
        if (
            j >= 2
            and pos[j - 2] == "operator"
            and pos[j - 1 : j + 1] in (["(", ")"], ["[", "]"])
        ):
            name, s = "operator" + pos[j - 1] + pos[j], j - 2
        elif j >= 1 and pos[j - 1] == "operator":
            name, s = "operator" + pos[j], j - 1
        elif j >= 0 and is_ident(pos[j]) and pos[j] not in NOT_FUNCTIONS:
            name, s = pos[j], j
            if s > 0 and pos[s - 1] == "~":
                name, s = "~" + name, s - 1
        else:
            return None
        if is_macro_name(name.lstrip("~")):
            return None
        quals = []
        while s >= 2 and pos[s - 1] == "::":
            j = s - 2
            if pos[j] == ">":
                j = bisect.bisect_left(head, self.match_back_angle(head[j])) - 1
            if j < 0 or not is_ident(pos[j]):
                break
            quals.insert(0, pos[j])
            s = j
        return name, tuple(quals), s

    def split_args(self, a, b):
        """Split token range [a, b) at top-level commas into (start, end) ranges."""
        T = self.T
        if a >= b:
            return []
        out = []
        depth = ang = 0
        start = a
        for i in range(a, b):
            t = T[i]
            if t in "([{":
                depth += 1
            elif t in ")]}":
                depth -= 1
            elif t == "<" and depth == 0 and i > a and is_ident(T[i - 1]):
                ang += 1
            elif t == ">" and depth == 0 and ang:
                ang -= 1
            elif t == "," and depth == 0 and ang == 0:
                out.append((start, i))
                start = i + 1
        out.append((start, b))
        return out

    def parse_params(self, a, b):
        T = self.T
        params = []
        for s, e in self.split_args(a, b):
            toks = list(range(s, e))
            depth = ang = 0
            for k, idx in enumerate(toks):
                t = T[idx]
                depth += (t in "([{") - (t in ")]}")
                if t == "<" and depth == 0 and k and is_ident(T[toks[k - 1]]):
                    ang += 1
                elif t == ">" and depth == 0 and ang:
                    ang -= 1
                elif t == "=" and depth == 0 and ang == 0:
                    toks = toks[:k]
                    break
            has_default = len(toks) != e - s
            words = [T[j] for j in toks]
            if words in (["void"], []):
                continue
            name = ""
            if (
                len(words) >= 2
                and is_ident(words[-1])
                and words[-2] not in ("::", "<", ",")
                and words[-1] not in BUILTIN_TYPE_WORDS
            ):
                name = words[-1]
                words = words[:-1]
            idents = set(words)
            kind = (
                "old"
                if idents & LEGACY_MR_TYPES
                else "new" if "memory_resources" in idents else ""
            )
            params.append(
                {
                    "kind": kind,
                    "name": name,
                    "default": has_default,
                    "text": " ".join(words),
                    "norm": self.normalize_type(words, kind),
                }
            )
            self.claimed.update(s for s in toks)
        return params

    @staticmethod
    def normalize_type(words, kind):
        if kind:
            return "MR"
        out = []
        for k, w in enumerate(words):
            if w in ("const", "typename", "struct", "class", "::"):
                continue
            if k + 1 < len(words) and words[k + 1] == "::":
                continue
            out.append(w)
        return " ".join(out)

    def add_record(self, func, kind, body=None, open_brace=None):
        T = self.T
        a, b = func["params"]
        params = self.parse_params(a, b)
        ret = func["ret"]
        ret_words = [T[j] for j in ret]
        ns = self.scope_names("ns")
        cls = self.scope_names("cls")
        quals = func["quals"]
        full = quals if quals and quals[:1] == ns[:1] else ns + cls + quals
        local = "(anon)" in ns or ("static" in ret_words and not cls)
        rec = {
            "kind": kind,
            "name": func["name"],
            "qualified": full + (func["name"],),
            "scope": full,
            "local": local,
            "line": func["line"],
            "ret": " ".join(ret_words),
            "params": params,
            "buf_old": [],
            "buf_new": [],
            "current_mr": [],
            "calls": [],
        }
        if ret:
            self.scan_range(rec, ret[0], ret[-1] + 1, signature=True)
            rec["ret_old"] = bool(rec["buf_old"])
        else:
            rec["ret_old"] = False
        self.scan_range(rec, a, b, signature=True)
        if body:
            if func["tail"]:
                self.scan_range(rec, func["tail"][0], open_brace)
            self.scan_body(rec, body[0], body[1])
        self.records.append(rec)

    def scan_range(self, rec, a, b, signature=False):
        T, L = self.T, self.L
        for i in range(a, b):
            t = T[i]
            self.claimed.add(i)
            if t == "device_uvector" or t == "device_buffer":
                new = i >= 2 and T[i - 1] == "::" and T[i - 2] == "cuda"
                if t == "device_buffer" and new:
                    rec["buf_new"].append(L[i])
                elif i and T[i - 1] in (".", "->"):
                    continue
                else:
                    rec["buf_old"].append((L[i], t))
            elif t == CURRENT_MR and not signature:
                rec["current_mr"].append(L[i])

    def scan_body(self, rec, a, b):
        T, L = self.T, self.L
        self.scan_range(rec, a, b)
        mr_names = {p["name"] for p in rec["params"] if p["kind"] and p["name"]}
        for _ in range(2):
            i = a
            while i < b:
                if (
                    T[i] == "="
                    and i > a
                    and is_ident(T[i - 1])
                    and T[i - 1] not in mr_names
                ):
                    e = i
                    while e < b and T[e] != ";":
                        e += 1
                    rhs = T[i + 1 : e]
                    if any(w in mr_names or w in MR_HINTS for w in rhs) and all(
                        w in mr_names or w in MR_HINTS or w in ALIAS_SYNTAX for w in rhs
                    ):
                        mr_names.add(T[i - 1])
                i += 1
        for i in range(a, b):
            if T[i] != "(":
                continue
            j = i - 1
            if j > a and T[j] == ">":
                j = self.match_back_angle(j) - 1
            if j < a or not is_ident(T[j]) or T[j] in NOT_FUNCTIONS:
                continue
            name = T[j]
            quals = []
            s = j
            while s - 2 >= a and T[s - 1] == "::" and is_ident(T[s - 2]):
                quals.insert(0, T[s - 2])
                s -= 2
            member = s > a and T[s - 1] in (".", "->")
            end = self.match(i, "(", ")")
            args = self.split_args(i + 1, end)
            mr_args = tuple(
                k
                for k, (x, y) in enumerate(args)
                if any(T[z] in mr_names or T[z] in MR_HINTS for z in range(x, y))
            )
            rec["calls"].append((tuple(quals), name, len(args), mr_args, L[j], member))


def parse_file(path, source):
    scanner = Scanner(source)
    records = scanner.run()
    T, L = scanner.T, scanner.L
    legacy = LEGACY_MR_TYPES | {CURRENT_MR, "device_uvector", "device_buffer"}
    uncovered = [
        (L[i], t)
        for i, t in enumerate(T)
        if t in legacy
        and i not in scanner.claimed
        and not (t == "device_buffer" and T[i - 2 : i] == ["cuda", "::"])
    ]
    return {"records": records, "uncovered": uncovered}


def entity_key(path, rec):
    sig = tuple(p["norm"] for p in rec["params"])
    return (path if rec["local"] else "", rec["qualified"], sig)


class Entity:
    def __init__(self, key):
        self.key = key
        self.defs = []
        self.decls = []

    @property
    def qualified(self):
        return self.key[1]


def build_entities(parsed):
    entities = {}
    for path in sorted(parsed):
        for rec in parsed[path]["records"]:
            ent = entities.setdefault(
                entity_key(path, rec), Entity(entity_key(path, rec))
            )
            (ent.defs if rec["kind"] == "definition" else ent.decls).append((path, rec))
    by_name = collections.defaultdict(list)
    for ent in entities.values():
        by_name[ent.qualified[-1]].append(ent)
    return entities, by_name


def resolve_call(caller, call, by_name):
    quals, name, nargs, _, _, member = call
    if member:
        return []
    cands = by_name.get(name, [])
    if not cands:
        return []
    scope = caller["scope"]
    chosen = []
    for k in range(len(scope), -1, -1):
        want = tuple(scope[:k]) + quals + (name,)
        chosen = [c for c in cands if c.qualified == want]
        if chosen:
            break
    if not chosen:
        tail = quals + (name,)
        chosen = [c for c in cands if c.qualified[-len(tail) :] == tail]
        if len({c.qualified for c in chosen}) != 1:
            return []
    out = []
    for ent in chosen:
        rec = (ent.defs or ent.decls)[0][1]
        params = rec["params"]
        required = sum(1 for p in params if not p["default"])
        if required <= nargs <= len(params):
            out.append(rec)
    return out


def mr_slot(rec):
    for i, p in enumerate(rec["params"]):
        if p["kind"]:
            return i, p["kind"]
    return None


def classify_record(rec, by_name, has_body):
    kinds = [p["kind"] for p in rec["params"] if p["kind"]]
    mr_issues = []
    if kinds:
        mr_issues += [
            {"kind": "old_mr_param", "line": rec["line"], "detail": p["text"]}
            for p in rec["params"]
            if p["kind"] == "old"
        ]
        mr_issues += [
            {"kind": "current_resource", "line": ln, "detail": CURRENT_MR + "()"}
            for ln in rec["current_mr"]
        ]
    buf_issues = [
        {"kind": "rmm_buffer", "line": ln, "detail": t} for ln, t in rec["buf_old"]
    ]
    callee_buf = []
    if has_body:
        for call in rec["calls"]:
            _, _, nargs, mr_args, line, _ = call
            cands = resolve_call(rec, call, by_name)
            if not cands:
                continue
            label = "::".join(cands[0]["qualified"])
            slots = [mr_slot(c) for c in cands]
            if "new" in kinds and all(slots):
                if mr_args and all(
                    kind == "old" and idx in mr_args for idx, kind in slots
                ):
                    mr_issues.append(
                        {"kind": "callee_old_mr", "line": line, "detail": label}
                    )
                elif not mr_args and all(idx >= nargs for idx, _ in slots):
                    mr_issues.append(
                        {"kind": "callee_default_mr", "line": line, "detail": label}
                    )
            if all(c["ret_old"] for c in cands):
                callee_buf.append(
                    {"kind": "callee_returns_rmm", "line": line, "detail": label}
                )
    if not kinds:
        mr_status = "not_applicable"
    elif "new" not in kinds:
        mr_status = "unmigrated"
    else:
        mr_status = "partial" if mr_issues else "migrated"
    if buf_issues:
        buf_status = "partial" if rec["buf_new"] else "unmigrated"
    elif callee_buf:
        buf_status, buf_issues = "partial", callee_buf
    elif rec["buf_new"]:
        buf_status = "migrated"
    else:
        buf_status = "not_applicable"
    return (mr_status, mr_issues), (buf_status, buf_issues)


def needs_mr_param(rec, by_name):
    """Body-bearing functions without an MR param that allocate on the current resource."""
    if any(p["kind"] for p in rec["params"]):
        return []
    if rec["qualified"][-1] == CURRENT_MR or "memory_resources" in rec["qualified"]:
        return []
    found = [
        {"kind": "current_resource", "line": ln, "detail": CURRENT_MR + "()"}
        for ln in rec["current_mr"]
    ]
    for call in rec["calls"]:
        _, _, nargs, mr_args, line, _ = call
        if mr_args:
            continue
        cands = resolve_call(rec, call, by_name)
        slots = [mr_slot(c) for c in cands]
        if cands and all(slots) and all(idx >= nargs for idx, _ in slots):
            found.append(
                {
                    "kind": "callee_default_mr",
                    "line": line,
                    "detail": "::".join(cands[0]["qualified"]),
                }
            )
    return found


def mr_callers(entities, by_name):
    """Map entity key -> (resolved callers, callers that take an MR param)."""
    key_of = {id((e.defs or e.decls)[0][1]): k for k, e in entities.items()}
    out = collections.defaultdict(lambda: [0, 0])
    for ent in entities.values():
        for _, rec in ent.defs:
            has_mr = any(p["kind"] for p in rec["params"])
            for call in rec["calls"]:
                for callee in resolve_call(rec, call, by_name):
                    counts = out[key_of[id(callee)]]
                    counts[0] += 1
                    counts[1] += has_mr
    return out


def classify_entities(entities, by_name):
    out = {}
    for key, ent in entities.items():
        pairs = ent.defs or ent.decls
        has_body = bool(ent.defs)
        mr = ("not_applicable", [])
        buf = ("not_applicable", [])
        res = []
        for path, rec in pairs:
            (ms, mi), (bs, bi) = classify_record(rec, by_name, has_body)
            for issue in mi + bi:
                issue["file"] = path
            res.append(((ms, mi), (bs, bi)))
        for idx in (0, 1):
            worst = min((r[idx] for r in res), key=lambda s: RANK[s[0]])
            issues = [i for r in res for i in r[idx][1]]
            if idx == 0:
                mr = (worst[0], issues)
            else:
                buf = (worst[0], issues)
        wanted = []
        if has_body:
            for path, rec in pairs:
                for issue in needs_mr_param(rec, by_name):
                    issue["file"] = path
                    wanted.append(issue)
        out[key] = {"mr": mr, "buf": buf, "needs_mr_param": wanted}
    return out


def display_signature(ent):
    _, rec = (ent.defs or ent.decls)[0]
    params = ", ".join((p["text"] + " " + p["name"]).strip() for p in rec["params"])
    sig = f"{rec['ret']} {'::'.join(rec['qualified'])}({params})".strip()
    sig = re.sub(r"\s+(?=[,>)(<&*]|::)", "", sig)
    sig = re.sub(r"(<|\(|::)\s+", r"\1", sig)
    return sig[:400]


class Git:
    def __init__(self, git_dir=None, cwd=None):
        self.base = ["git"] + (["--git-dir", str(git_dir)] if git_dir else [])
        self.cwd = cwd

    def run(self, *args, input=None, check=True):
        proc = subprocess.run(
            self.base + list(args),
            cwd=self.cwd,
            input=input,
            capture_output=True,
            check=False,
        )
        if check and proc.returncode:
            raise RuntimeError(f"git {' '.join(args)}: {proc.stderr.decode()}")
        return proc

    def blobs(self, shas):
        shas = list(shas)
        if not shas:
            return {}
        proc = self.run("cat-file", "--batch", input="\n".join(shas).encode() + b"\n")
        data, pos, out = proc.stdout, 0, {}
        for sha in shas:
            eol = data.index(b"\n", pos)
            _, _, size = data[pos:eol].split()
            pos = eol + 1 + int(size)
            out[sha] = data[eol + 1 : pos]
            pos += 1
        return out

    def tree_files(self, treeish, roots):
        raw = self.run("ls-tree", "-r", "-z", "--full-tree", treeish, "--", *roots)
        files = {}
        for entry in raw.stdout.split(b"\0"):
            if not entry:
                continue
            meta, path = entry.decode().split("\t", 1)
            if is_source(path):
                files[path] = meta.split()[2]
        return files


def is_source(path):
    p = Path(path)
    return p.suffix in SOURCE_SUFFIXES and not p.name.endswith("_generated.h")


class ParseCache:
    def __init__(self, directory):
        self.file = Path(directory) / f"parse_v{PARSER_VERSION}.pkl"
        self.data = {}
        if self.file.exists():
            with open(self.file, "rb") as f:
                self.data = pickle.load(f)

    def parse(self, git, files):
        missing = {sha for sha in files.values() if sha not in self.data}
        if missing:
            blobs = git.blobs(sorted(missing))
            by_sha = {sha: path for path, sha in files.items() if sha in missing}
            for sha, content in blobs.items():
                self.data[sha] = parse_file(
                    by_sha[sha], content.decode("utf-8", "replace")
                )
            with open(self.file, "wb") as f:
                pickle.dump(self.data, f)
        return {path: self.data[sha] for path, sha in files.items()}


def analyze(parsed):
    entities, by_name = build_entities(parsed)
    return entities, classify_entities(entities, by_name)


def improves(which, before, after):
    """True if `after` moves an unmigrated/partial function toward done.

    For the memory_resources migration, losing the mr parameter altogether (for
    example, switching to a different resource-ref alias) is not progress.
    """
    if before[0] not in ("unmigrated", "partial"):
        return False
    if which == "mr" and after[0] == "not_applicable":
        return False
    if RANK[after[0]] != RANK[before[0]]:
        return RANK[after[0]] > RANK[before[0]]
    return len(after[1]) < len(before[1])


def addresses_needs_mr_param(key, before, after, after_by_name):
    """True if a PR gives a function that needs an MR param one, or drops its uses.

    Adding a parameter changes the entity key, so overloads are matched by name.
    """
    if not before["needs_mr_param"]:
        return False
    same = after.get(key)
    if same is not None:
        return len(same["needs_mr_param"]) < len(before["needs_mr_param"])
    return any(
        r["mr"][0] in ("migrated", "partial") for r in after_by_name.get(key[:2], [])
    )


def open_prs(args, repo_git):
    fields = "number,title,url,isDraft,baseRefName,author,files"
    try:
        proc = subprocess.run(
            ["gh", "pr", "list", "-R", args.pr_repo, "--state", "open", "--limit", "1000",
             "--json", fields],
            capture_output=True, text=True, check=True,
        )  # fmt: skip
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"Skipping open-PR analysis: `gh pr list` failed ({e})", file=sys.stderr)
        return []
    prs = []
    for pr in json.loads(proc.stdout):
        if pr["baseRefName"] != args.pr_base:
            continue
        paths = [f["path"] for f in pr["files"]]
        if len(paths) < 100 and not any(
            p.startswith(tuple(args.roots)) and is_source(p) for p in paths
        ):
            continue
        prs.append(pr)
    return prs


def pr_overlays(args, base_sha, base_files, cache, repo_git, prs):
    """Yield (pr, parsed files with the PR merged into base, conflicted paths).

    Files that conflict with the base ref are left at their base version; the overlay
    is None when nothing in the PR merges cleanly.
    """
    scratch = Path(args.cache_dir) / "pr.git"
    if not scratch.exists():
        subprocess.run(["git", "init", "--bare", "-q", str(scratch)], check=True)
        alt = Path(
            repo_git.run("rev-parse", "--git-common-dir").stdout.decode().strip()
        )
        alt = (Path(repo_git.cwd) / alt).resolve() / "objects"
        (scratch / "objects/info").mkdir(parents=True, exist_ok=True)
        (scratch / "objects/info/alternates").write_text(f"{alt}\n")
    sg = Git(scratch)
    sg.run("update-ref", "refs/base", base_sha)
    numbers = [pr["number"] for pr in prs]
    for i in range(0, len(numbers), 50):
        refspecs = [f"+refs/pull/{n}/head:refs/pr/{n}" for n in numbers[i : i + 50]]
        sg.run("fetch", "--no-tags", "-q", args.pr_remote, *refspecs, check=False)
    for pr in prs:
        ref = f"refs/pr/{pr['number']}"
        if sg.run("rev-parse", "-q", "--verify", ref, check=False).returncode:
            print(f"  PR #{pr['number']}: not fetched, skipped", file=sys.stderr)
            continue
        merge = sg.run("merge-tree", "--write-tree", "--name-only", "--no-messages",
                       base_sha, ref, check=False)  # fmt: skip
        lines = merge.stdout.decode().split("\n")
        tree = lines[0].strip()
        conflicted = {ln for ln in lines[1:] if ln}
        if not tree or len(tree) < 40:
            continue
        overlay = dict(base_files)
        diff = sg.run("diff-tree", "-r", "-z", "--raw", "--no-renames", base_sha, tree,
                      "--", *args.roots)  # fmt: skip
        parts = diff.stdout.split(b"\0")
        changed = 0
        skipped = []
        for meta, path in zip(parts[0::2], parts[1::2]):
            if not meta:
                continue
            path = path.decode()
            new_sha, status = meta.split()[3].decode(), meta.split()[4].decode()
            if not is_source(path):
                continue
            changed += 1
            if status == "D":
                overlay.pop(path, None)
            elif path in conflicted:
                skipped.append(path)
            else:
                overlay[path] = new_sha
        if not changed:
            continue
        if changed == len(skipped):
            yield pr, None, skipped
            continue
        missing = {s for s in overlay.values() if s not in cache.data}
        if missing:
            blobs = sg.blobs(sorted(missing))
            owner = {sha: path for path, sha in overlay.items() if sha in missing}
            for sha, content in blobs.items():
                cache.data[sha] = parse_file(
                    owner[sha], content.decode("utf-8", "replace")
                )
        yield pr, {path: cache.data[sha] for path, sha in overlay.items()}, skipped


def aggregate(functions):
    files = {}
    for fn in functions:
        entry = files.setdefault(
            fn["file"],
            {
                "path": fn["file"],
                "mr": collections.Counter(),
                "buf": collections.Counter(),
            },
        )
        for which in ("mr", "buf"):
            entry[which][fn[which]["status"]] += 1
    out = []
    for path in sorted(files):
        entry = files[path]
        out.append(
            {
                "path": path,
                "mr": {s: entry["mr"].get(s, 0) for s in STATES},
                "buf": {s: entry["buf"].get(s, 0) for s in STATES},
            }
        )
    return out


def totals(functions):
    return {
        which: {s: sum(1 for f in functions if f[which]["status"] == s) for s in STATES}
        for which in ("mr", "buf")
    }


def build_report(args, base_sha):
    repo = Path(args.repo).resolve()
    git = Git(cwd=repo)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = ParseCache(cache_dir)

    files = git.tree_files(base_sha, args.roots)
    print(f"Scanning {len(files)} files at {base_sha[:10]}", file=sys.stderr)
    parsed = cache.parse(git, files)
    entities, results = analyze(parsed)
    callers = mr_callers(entities, build_entities(parsed)[1])

    improvements = collections.defaultdict(
        lambda: {"mr": [], "buf": [], "needs_mr_param": []}
    )
    pr_info = {}
    if not args.no_prs:
        prs = open_prs(args, git)
        print(
            f"Analyzing {len(prs)} open PRs touching the scanned roots", file=sys.stderr
        )
        for pr, overlay, conflicted in pr_overlays(
            args, base_sha, files, cache, git, prs
        ):
            after = analyze(overlay)[1] if overlay else {}
            after_by_name = collections.defaultdict(list)
            for k, r in after.items():
                after_by_name[k[:2]].append(r)
            hit = 0
            for key, res in results.items():
                if after and addresses_needs_mr_param(key, res, after, after_by_name):
                    improvements[key]["needs_mr_param"].append(pr["number"])
                    hit += 1
                new = after.get(key)
                if new is None:
                    continue
                for which in ("mr", "buf"):
                    if improves(which, res[which], new[which]):
                        improvements[key][which].append(
                            {"number": pr["number"], "after": new[which][0]}
                        )
                        hit += 1
            pr_info[pr["number"]] = {
                "number": pr["number"],
                "title": pr["title"],
                "url": pr["url"],
                "draft": pr["isDraft"],
                "author": (pr.get("author") or {}).get("login", ""),
                "functions_improved": hit,
                "conflicted_files": conflicted,
            }
        with open(cache.file, "wb") as f:
            pickle.dump(cache.data, f)

    functions = []
    for key, ent in entities.items():
        path, rec = (ent.defs or ent.decls)[0]
        res = results[key]
        fn = {
            "name": "::".join(ent.qualified),
            "signature": display_signature(ent),
            "file": path,
            "line": rec["line"],
            "kind": "definition" if ent.defs else "declaration",
            "declared_in": sorted({f"{p}:{r['line']}" for p, r in ent.decls}),
            "defined_in": sorted({f"{p}:{r['line']}" for p, r in ent.defs}),
            "needs_mr_param": res["needs_mr_param"],
            "needs_mr_param_prs": improvements[key]["needs_mr_param"],
            "callers": callers[key][0],
            "callers_with_mr": callers[key][1],
        }
        for which in ("mr", "buf"):
            status, issues = res[which]
            prs_for = improvements[key][which]
            fn[which] = {
                "status": "in_progress" if prs_for else status,
                "base_status": status,
                "issues": issues,
                "prs": prs_for,
            }
        functions.append(fn)
    functions.sort(key=lambda f: (f["file"], f["line"], f["name"]))
    for i, fn in enumerate(functions):
        fn["id"] = i

    uncovered = {
        path: info["uncovered"] for path, info in parsed.items() if info["uncovered"]
    }
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base": {"ref": args.base_ref, "sha": base_sha},
        "repo_url": args.repo_url,
        "roots": args.roots,
        "states": STATES,
        "totals": totals(functions),
        "prs": sorted(pr_info.values(), key=lambda p: p["number"]),
        "files": aggregate(functions),
        "functions": functions,
        "unattributed_mentions": {
            "note": (
                "Mentions of tracked tokens outside any scanned function signature or "
                "body (class members, globals, macros); not counted in any status."
            ),
            "count": sum(len(v) for v in uncovered.values()),
            "by_file": uncovered,
        },
    }


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=str(here.parent / "cudf"))
    ap.add_argument("--base-ref", default="HEAD", help="git ref to analyze")
    ap.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    ap.add_argument("--output-dir", default=str(here))
    ap.add_argument(
        "--cache-dir",
        default=str(Path(tempfile.gettempdir()) / "cudf_migration_status"),
    )
    ap.add_argument("--no-prs", action="store_true", help="skip open-PR analysis")
    ap.add_argument("--pr-repo", default="NVIDIA/cudf")
    ap.add_argument("--pr-remote", default="https://github.com/NVIDIA/cudf.git")
    ap.add_argument("--pr-base", default="main", help="only PRs targeting this branch")
    ap.add_argument("--repo-url", default="https://github.com/NVIDIA/cudf")
    args = ap.parse_args()

    base_sha = (
        Git(cwd=args.repo)
        .run("rev-parse", f"{args.base_ref}^{{commit}}")
        .stdout.decode()
        .strip()
    )
    report = build_report(args, base_sha)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_path = out / "migration_status.json"
    data_path.write_text(json.dumps(report, indent=1))
    template = (here / "migration_status_template.html").read_text()
    payload = json.dumps(report, separators=(",", ":")).replace("</", "<\\/")
    (out / "migration_status.html").write_text(
        template.replace("/*__DATA__*/null", payload)
    )
    for which, label in (("mr", "memory_resources"), ("buf", "cuda::device_buffer")):
        print(f"{label}: {report['totals'][which]}", file=sys.stderr)
    print(f"Wrote {data_path} and {out / 'migration_status.html'}", file=sys.stderr)


if __name__ == "__main__":
    main()
