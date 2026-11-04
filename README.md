# cuDF CCCL migration dashboard

Track libcudf's migrations from RMM buffers to `cuda::device_buffer` and from
`rmm::device_async_resource_ref` to `cudf::memory_resources`, down to individual
functions and files. Open pull requests are overlaid onto the base revision to
show work in progress. A batch planner groups the remaining memory-resource
migration into reviewable PRs.

**[Open the dashboard](https://bdice.github.io/cudf-cccl-migration-dashboard/)**

## Run locally

Requires Python 3.9+, Git 2.38+, and an authenticated [GitHub CLI](https://cli.github.com/)
for open-PR analysis. No CUDA toolkit or cuDF build is needed.

```bash
git clone https://github.com/bdice/cudf-cccl-migration-dashboard.git
cd cudf-cccl-migration-dashboard
git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/cudf.git cudf
gh auth login
python3 migration_status.py --repo cudf --base-ref origin/main --output-dir site
python3 plan_batches.py --report site/migration_status.json
python3 -m http.server 8000 --directory site
```

Open <http://localhost:8000/migration_status.html>. The output directory also
contains the JSON report and migration batches in JSON and Markdown.

To analyze an existing checkout instead, pass `--repo /path/to/cudf --base-ref upstream/main`.
The default repository is a sibling `../cudf` checkout; the default revision is
`HEAD`. Use `--no-prs` for an offline report. Run `--help` for all options.

## GitHub Actions and Pages

The workflow tests the scanner, analyzes current `NVIDIA/cudf` main and open PRs,
and deploys the generated HTML and data directly to GitHub Pages. It runs daily
at 12:23 UTC, on pushes to `main`, and through **Actions → Dashboard → Run workflow**.
Pull requests run tests only. Generated reports are not committed to Git.

For another deployment, enable **Settings → Pages → Source → GitHub Actions**.
The workflow uses the built-in `GITHUB_TOKEN`; no personal token is required.

## How it works

The scanner reads committed C++ files from Git and performs scope-aware token
analysis, without a compilation database. PRs are merged into the base in a
separate temporary Git repository; conflicting files retain their base version.
It never checks out PR code or runs it.

Call resolution uses C++ name lookup and argument count, not full overload
resolution. Treat flagged sites as review aids rather than compiler-level proof.
The dashboard includes source links and issue details for inspection.

## Tests

```bash
python3 -m pip install pytest
python3 -m pytest -q
```

## License

[Apache License 2.0](LICENSE). Existing NVIDIA copyright notices are retained.
