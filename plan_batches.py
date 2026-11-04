#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Partition the remaining cudf::memory_resources migration into a few large PRs.

Reads migration_status.json (from migration_status.py) and writes
migration_batches.json and migration_batches.md next to it.

Stages, in landing order:
  1. add-mr-param: functions with no MR parameter that allocate on the current
     resource. All review-required (adds API surface, decides output vs temporary).
  2. signature: unmigrated functions that only need old_mr_param -> memory_resources.
     Mechanical, except where the function also uses the current resource (those
     sites are deferred to stage 3 and listed there).
  3. temporary: get_current_device_resource_ref() sites and callees that are not
     migrated. Each site needs an output-vs-temporary decision.

Stage 2 and 3 are split by the area of the file that defines the function and
packed into balanced batches so each PR is a coherent slice of the tree.
"""

import argparse
import collections
import json
import pathlib

OPEN_PR_URL = "https://github.com/NVIDIA/cudf/pull/{}"


def area(fn):
    path = (fn["defined_in"] or [fn["file"]])[0].rsplit(":", 1)[0]
    parts = path.split("/")
    if parts[1] == "include":
        return "include/detail" if parts[3] == "detail" else "include/public"
    if parts[1] == "src":
        return "src/" + parts[2]
    return parts[1]


def pack(weights, bins):
    """Greedy longest-first packing of {area: weight} into `bins` groups."""
    groups = [[0, []] for _ in range(bins)]
    for name, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        g = min(groups, key=lambda g: g[0])
        g[0] += w
        g[1].append(name)
    return [sorted(g[1]) for g in groups if g[1]]


def site(fn, issue):
    return {
        "function": fn["signature"],
        "file": issue.get("file", fn["file"]),
        "line": issue["line"],
        "kind": issue["kind"],
        "detail": issue["detail"],
    }


def build(report, signature_bins, temporary_bins):
    fns = report["functions"]
    needs_all = [f for f in fns if f["needs_mr_param"]]
    in_flight = [f for f in needs_all if f["needs_mr_param_prs"]]
    no_source = [
        f for f in needs_all if not f["needs_mr_param_prs"] and not f["callers_with_mr"]
    ]
    needs = [
        f for f in needs_all if not f["needs_mr_param_prs"] and f["callers_with_mr"]
    ]
    unmigrated = [f for f in fns if f["mr"]["status"] == "unmigrated"]
    stage3 = [
        f
        for f in fns
        if f["mr"]["status"] in ("unmigrated", "partial") and f["mr"]["issues"]
    ]

    batches = []
    batches.append(
        {
            "id": "1-add-mr-param",
            "stage": "add-mr-param",
            "title": "Add MR parameters to APIs that allocate on the current resource",
            "areas": sorted({area(f) for f in needs}),
            "mechanical": [],
            "review_required": [site(f, i) for f in needs for i in f["needs_mr_param"]],
            "functions": len(needs),
        }
    )

    weights = collections.Counter(area(f) for f in unmigrated)
    for n, areas in enumerate(pack(weights, signature_bins), 1):
        members = [f for f in unmigrated if area(f) in areas]
        batches.append(
            {
                "id": f"2.{n}-signature",
                "stage": "signature",
                "title": "Migrate signatures to cudf::memory_resources: "
                + ", ".join(areas),
                "areas": areas,
                "mechanical": [
                    site(f, i)
                    for f in members
                    for i in f["mr"]["issues"]
                    if i["kind"] == "old_mr_param"
                ],
                "review_required": [],
                "functions": len(members),
            }
        )

    weights = collections.Counter()
    for f in stage3:
        weights[area(f)] += sum(i["kind"] != "old_mr_param" for i in f["mr"]["issues"])
    weights = {a: w for a, w in weights.items() if w}
    for n, areas in enumerate(pack(weights, temporary_bins), 1):
        members = [f for f in stage3 if area(f) in areas]
        sites = [
            site(f, i)
            for f in members
            for i in f["mr"]["issues"]
            if i["kind"] != "old_mr_param"
        ]
        batches.append(
            {
                "id": f"3.{n}-temporary",
                "stage": "temporary",
                "title": "Propagate temporary MRs: " + ", ".join(areas),
                "areas": areas,
                "mechanical": [s for s in sites if s["kind"] == "callee_old_mr"],
                "review_required": [s for s in sites if s["kind"] != "callee_old_mr"],
                "functions": len({f["id"] for f in members}),
            }
        )

    open_prs = [
        p
        for p in report["prs"]
        if p["functions_improved"] and not p["draft"] and not p["conflicted_files"]
    ]
    return {
        "base": report["base"],
        "batches": batches,
        "merge_first": [
            {
                k: p[k]
                for k in ("number", "title", "url", "author", "functions_improved")
            }
            for p in sorted(open_prs, key=lambda p: -p["functions_improved"])
        ],
        "needs_mr_param_in_flight": [
            {
                "function": f["signature"],
                "file": f["file"],
                "prs": f["needs_mr_param_prs"],
            }
            for f in in_flight
        ],
        "needs_mr_param_no_mr_caller": [
            {"function": f["signature"], "file": f["file"], "callers": f["callers"]}
            for f in no_source
        ],
        "blocked_in_flight": [
            {"number": p["number"], "title": p["title"], "url": p["url"]}
            for p in report["prs"]
            if p["functions_improved"] and (p["draft"] or p["conflicted_files"])
        ],
    }


def markdown(plan):
    lines = [
        "# cudf::memory_resources migration batches",
        "",
        f"Base: `{plan['base']['ref']}` ({plan['base']['sha'][:10]}). "
        f"{len(plan['batches'])} PRs.",
        "",
        "| PR | Stage | Functions | Mechanical sites | Review-required sites |",
        "|---|---|---:|---:|---:|",
    ]
    for b in plan["batches"]:
        lines.append(
            f"| {b['id']} | {b['title']} | {b['functions']} "
            f"| {len(b['mechanical'])} | {len(b['review_required'])} |"
        )
    lines += ["", "## Merge before opening new PRs", ""]
    lines += [
        f"- [#{p['number']}]({p['url']}) {p['title']} ({p['functions_improved']} functions, @{p['author']})"
        for p in plan["merge_first"]
    ]
    lines += ["", "## In flight but blocked (draft or conflicting)", ""]
    lines += [
        f"- [#{p['number']}]({p['url']}) {p['title']}"
        for p in plan["blocked_in_flight"]
    ]
    lines += [
        "",
        "## Functions needing an MR param, excluded from batch 1",
        "",
        f"- {len(plan['needs_mr_param_in_flight'])} are addressed by open PRs.",
        f"- {len(plan['needs_mr_param_no_mr_caller'])} have no caller holding an MR "
        "(for example, I/O writer internals); adding one needs an API decision upstream.",
    ]
    lines += ["", "## Review-required sites by batch", ""]
    for b in plan["batches"]:
        kinds = collections.Counter(s["kind"] for s in b["review_required"])
        lines.append(f"- **{b['id']}**: {dict(kinds) or 'none'}")
    return "\n".join(lines) + "\n"


def main():
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--report", type=pathlib.Path, default=here / "migration_status.json"
    )
    ap.add_argument("--output-dir", type=pathlib.Path, default=None)
    ap.add_argument("--signature-batches", type=int, default=5)
    ap.add_argument("--temporary-batches", type=int, default=3)
    args = ap.parse_args()
    out = args.output_dir or args.report.parent
    plan = build(
        json.loads(args.report.read_text()),
        args.signature_batches,
        args.temporary_batches,
    )
    (out / "migration_batches.json").write_text(json.dumps(plan, indent=1))
    (out / "migration_batches.md").write_text(markdown(plan))
    print(markdown(plan).split("## Merge")[0])


if __name__ == "__main__":
    main()
