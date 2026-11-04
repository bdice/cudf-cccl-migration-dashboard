#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import migration_status as m

OLD = "rmm::device_async_resource_ref mr"
NEW = "cudf::memory_resources mr"


def run(files):
    parsed = {path: m.parse_file(path, src) for path, src in files.items()}
    entities, results = m.analyze(parsed)
    by_name = {"::".join(k[1]): results[k] for k in entities}
    return by_name, parsed


def status(src, name, which="mr", extra=None):
    files = {"a.cu": src, **(extra or {})}
    by_name, _ = run(files)
    return by_name[name][which]


def test_unmigrated_and_migrated_and_not_applicable():
    src = f"""
namespace cudf {{
void old_fn(int x, {OLD}) {{}}
void new_fn(int x, {NEW}) {{}}
void plain(int x) {{}}
}}
"""
    assert status(src, "cudf::old_fn")[0] == "unmigrated"
    assert status(src, "cudf::new_fn")[0] == "migrated"
    assert status(src, "cudf::plain")[0] == "not_applicable"


def test_default_argument_does_not_count_as_current_resource_use():
    src = f"""
namespace cudf {{
void f(int x, cudf::memory_resources mr = cudf::get_current_device_resource_ref());
}}
"""
    assert status(src, "cudf::f")[0] == "migrated"


def test_callee_that_still_takes_old_ref_makes_caller_partial():
    src = f"""
namespace cudf {{
namespace detail {{
void callee(int x, {OLD}) {{}}
void caller(int x, {NEW}) {{ callee(x, mr.get_output_mr()); }}
}}
}}
"""
    state, issues = status(src, "cudf::detail::caller")
    assert state == "partial"
    assert [i["kind"] for i in issues] == ["callee_old_mr"]


def test_callee_migrated_keeps_caller_migrated():
    src = f"""
namespace cudf {{
namespace detail {{
void callee(int x, {NEW}) {{}}
void caller(int x, {NEW}) {{ callee(x, mr); }}
}}
}}
"""
    assert status(src, "cudf::detail::caller")[0] == "migrated"


def test_omitted_callee_mr_is_flagged():
    src = f"""
namespace cudf {{
void callee(int x, {NEW} = cudf::get_current_device_resource_ref());
void caller(int x, {NEW}) {{ callee(x); }}
}}
"""
    state, issues = status(src, "cudf::caller")
    assert state == "partial" and issues[0]["kind"] == "callee_default_mr"


def test_current_resource_in_body_of_migrated_function_is_partial():
    src = f"""
namespace cudf {{
void f({NEW}) {{ auto t = cudf::get_current_device_resource_ref(); }}
}}
"""
    assert status(src, "cudf::f")[0] == "partial"


def test_constructor_init_list_is_scanned():
    src = f"""
namespace cudf {{
struct s {{
  s({NEW}) : buf{{stream, cudf::get_current_device_resource_ref()}}, n(0) {{}}
}};
}}
"""
    assert status(src, "cudf::s::s")[0] == "partial"


def test_operator_and_macro_namespaces():
    src = f"""
namespace CUDF_EXPORT cudf {{
bool operator==(a const& l, a const& r) {{ return true; }}
struct f {{
  template <typename T>
  int operator()(int x, {OLD}) const {{ return 0; }}
}};
}}
"""
    assert status(src, "cudf::operator==")[0] == "not_applicable"
    assert status(src, "cudf::f::operator()")[0] == "unmigrated"


def test_declaration_and_definition_merge_across_files():
    header = f"namespace cudf {{ void f(int x, {NEW} = {{}}); }}"
    source = f"namespace cudf {{ void f(int x, {NEW}) {{}} }}"
    by_name, parsed = run({"a.hpp": header, "a.cu": source})
    assert list(by_name) == ["cudf::f"]


def test_buffer_states():
    src = f"""
namespace cudf {{
rmm::device_uvector<int> old_buf(int n) {{ return rmm::device_uvector<int>(n, s, mr); }}
void both(int n) {{ rmm::device_buffer a; cuda::device_buffer<int> b; }}
void done(int n) {{ cuda::device_buffer<int> b; }}
void none(int n) {{}}
void blocked(int n) {{ auto v = old_buf(n); }}
}}
"""
    got = {
        n: status(src, f"cudf::{n}", "buf")[0]
        for n in ("old_buf", "both", "done", "none", "blocked")
    }
    assert got == {
        "old_buf": "unmigrated",
        "both": "partial",
        "done": "migrated",
        "none": "not_applicable",
        "blocked": "partial",
    }


def test_improves_requires_real_progress():
    unm = ("unmigrated", [1])
    assert m.improves("mr", unm, ("migrated", []))
    assert m.improves("mr", unm, ("partial", [1]))
    assert not m.improves("mr", unm, ("not_applicable", []))
    assert not m.improves("mr", ("migrated", []), ("migrated", []))
    assert m.improves("buf", unm, ("not_applicable", []))
    assert m.improves("mr", ("partial", [1, 2]), ("partial", [1]))
    assert not m.improves("mr", ("partial", [1]), ("partial", [1]))


def test_needs_mr_param_flags_mrless_functions_using_current_resource():
    src = f"""
namespace cudf {{
void callee(int x, {NEW} = cudf::get_current_device_resource_ref());
void direct(int x) {{ auto r = cudf::get_current_device_resource_ref(); }}
void via_default(int x) {{ callee(x); }}
void passes_mr(int x, {NEW}) {{ callee(x, mr); }}
void clean(int x) {{ x++; }}
}}
"""
    files = {"a.cu": src}
    parsed = {p: m.parse_file(p, s) for p, s in files.items()}
    entities, results = m.analyze(parsed)
    wanted = {"::".join(k[1]): results[k]["needs_mr_param"] for k in entities}
    assert [i["kind"] for i in wanted["cudf::direct"]] == ["current_resource"]
    assert [i["kind"] for i in wanted["cudf::via_default"]] == ["callee_default_mr"]
    assert wanted["cudf::passes_mr"] == []
    assert wanted["cudf::clean"] == []


def test_plan_batches_orders_stages_and_splits_sites():
    import plan_batches

    def fn(i, path, status, issues=(), needs=()):
        return {
            "id": i,
            "signature": f"f{i}",
            "file": path,
            "defined_in": [path + ":1"],
            "mr": {"status": status, "issues": list(issues)},
            "needs_mr_param": list(needs),
            "needs_mr_param_prs": [],
            "callers": 1,
            "callers_with_mr": 1,
        }

    issue = lambda kind: {"kind": kind, "line": 1, "detail": "x"}
    report = {
        "base": {"ref": "r", "sha": "0" * 40},
        "prs": [],
        "functions": [
            fn(
                0, "cpp/src/a/x.cu", "not_applicable", needs=[issue("current_resource")]
            ),
            fn(1, "cpp/src/a/x.cu", "unmigrated", [issue("old_mr_param")]),
            fn(
                2,
                "cpp/src/b/y.cu",
                "unmigrated",
                [issue("old_mr_param"), issue("current_resource")],
            ),
            fn(3, "cpp/src/b/y.cu", "partial", [issue("callee_old_mr")]),
        ],
    }
    plan = plan_batches.build(report, 1, 1)
    stages = [b["stage"] for b in plan["batches"]]
    assert stages == ["add-mr-param", "signature", "temporary"]
    by_stage = {b["stage"]: b for b in plan["batches"]}
    assert len(by_stage["signature"]["mechanical"]) == 2
    assert [s["kind"] for s in by_stage["temporary"]["review_required"]] == [
        "current_resource"
    ]
    assert [s["kind"] for s in by_stage["temporary"]["mechanical"]] == ["callee_old_mr"]


def test_pr_adding_mr_param_addresses_needs_mr_param():
    callee = "void callee(int x, {NEW} = cudf::get_current_device_resource_ref());"
    before_src = f"""
namespace cudf {{
{callee.format(NEW=NEW)}
void helper(int x) {{ callee(x); }}
}}
"""
    after_src = f"""
namespace cudf {{
{callee.format(NEW=NEW)}
void helper(int x, {NEW}) {{ callee(x, mr); }}
}}
"""
    before_ents, before = m.analyze({"a.cu": m.parse_file("a.cu", before_src)})
    _, after = m.analyze({"a.cu": m.parse_file("a.cu", after_src)})
    by_name = {}
    for k, r in after.items():
        by_name.setdefault(k[:2], []).append(r)
    key = next(k for k in before_ents if k[1] == ("cudf", "helper"))
    assert key not in after
    assert m.addresses_needs_mr_param(key, before[key], after, by_name)


def test_mr_callers_counts_callers_holding_an_mr():
    src = f"""
namespace cudf {{
void helper(int x) {{ auto r = cudf::get_current_device_resource_ref(); }}
void a(int x, {OLD}) {{ helper(x); }}
void b(int x) {{ helper(x); }}
}}
"""
    parsed = {"a.cu": m.parse_file("a.cu", src)}
    entities, by_name = m.build_entities(parsed)
    counts = m.mr_callers(entities, by_name)
    key = next(k for k in entities if k[1] == ("cudf", "helper"))
    assert list(counts[key]) == [2, 1]


def test_resource_accessors_do_not_need_mr_param():
    src = """
namespace cudf {
inline rmm::device_async_resource_ref get_current_device_resource_ref()
{ return rmm::mr::get_current_device_resource_ref(); }
}
"""
    parsed = {"a.hpp": m.parse_file("a.hpp", src)}
    _, results = m.analyze(parsed)
    assert all(not r["needs_mr_param"] for r in results.values())


def test_default_repo_is_sibling_cudf(tmp_path, monkeypatch):
    import shutil
    import subprocess
    from pathlib import Path

    dashboard = tmp_path / "dashboard"
    dashboard.mkdir()
    shutil.copy(m.__file__, dashboard / "migration_status.py")
    shutil.copy(
        Path(m.__file__).with_name("migration_status_template.html"), dashboard
    )
    cudf = tmp_path / "cudf"
    subprocess.run(["git", "init", "-q", str(cudf)], check=True)
    subprocess.run(
        [
            "git", "-C", str(cudf), "-c", "user.name=Test",
            "-c", "user.email=test@example.com", "commit", "-qm", "Initial",
            "--allow-empty",
        ],
        check=True,
    )
    monkeypatch.setattr(m, "__file__", str(dashboard / "migration_status.py"))
    monkeypatch.setattr(
        "sys.argv", ["migration_status.py", "--no-prs", "--cache-dir", str(tmp_path / "cache")]
    )
    m.main()
    assert (dashboard / "migration_status.html").is_file()
