# Homework-3: Crawling, Cleaning, and EDA

In this assignment you will crawl Google Jobs for data science job postings, clean the results twice (once manually and once with an LLM), and perform a visual exploratory data analysis (EDA) of the cleaned data.

Specific instructions are in each of the provided Jupyter notebooks. Work through them in order:

| Notebook | Task |
|---|---|
| `HW-3.1-crawl.ipynb` | Crawl Google Jobs with the SerpApi API and save the raw results to `data/` |
| `HW-3.2-cleaning-1.ipynb` | Clean and parse the raw results manually (no LLMs), save `data/processed-jobs-1.csv` |
| `HW-3.3-cleaning-2.ipynb` | Clean the raw results with an LLM, using either a cloud API or a local model, save `data/processed-jobs-2.csv` |
| `HW-3.4-visual-EDA.ipynb` | Visual EDA of both cleaned datasets |
| `HW-3.5-optional-bonus-resume-creator.ipynb` | Optional bonus: LLM + Quarto resume and cover letter pipeline |

## Using GitHub

You are expected to use Git and GitHub for this assignment.

1. Create your own **private** GitHub repository for this homework, and copy these starter files into it.
2. Make regular progress and **commit regularly** with meaningful commit messages (e.g. "Add search utility function", "Finish production crawl", "Parse salary ranges"). A healthy history has many small commits spread across the time you worked on the assignment, not one large commit at the end. Your commit history is part of your submission.
3. Push to GitHub regularly so that your work, and your crawled data, are backed up.
4. **Never commit API keys.** Keep them in a file outside the repository (e.g. `~/.api-keys.json`), and check that no notebook output cell prints a key before you commit.

When you are finished, export your commit history from the root of your repository:

```bash
git log --pretty=format:"%h | %ad | %an | %s" --date=iso > commit-history.txt
```

Open `commit-history.txt` to confirm it lists your commits.

## Submission

Upload the following to Canvas:

* `commit-history.txt`, your exported commit history
* Rendered HTML files for each of the Jupyter notebooks
  * Each notebook already starts with a front-matter cell that embeds resources (`format: html: embed-resources: true`), so each HTML file is self-contained.
  * Render with, e.g., `quarto render HW-3.1-crawl.ipynb`. By default Quarto does not re-execute notebooks, so rendering will not use up API searches or credits; just make sure the notebook was run and saved first.
  * Double check that the images are embedded correctly and render in Canvas.

## Local workflow

Part 1 defaults to an **offline preview**. Its live collection is not complete until all 30 titles have a completed or exhausted pagination chain in `reports/crawl-summary.csv`.

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_part1.py
```

For a new environment, first create it with `python3 -m venv .venv`.
Store the SerpApi key in `~/.api-keys.json` under the `serpapi` key, outside this repository. Once the offline tests pass, collect in stages and inspect each trial's saved JSON before the next stage:

```bash
python run_part1.py --live --stage trial
python run_part1.py --live --stage second
python run_part1.py --live --stage all
quarto render HW-3.1-crawl.ipynb --no-execute
```

Use `--keys-file /absolute/path/to/keys.json` if the key is stored elsewhere outside the repository. Run only one crawler at a time. Successful pages are cached in `data/`, request attempts are tracked in `reports/search-attempts.json`, and request pacing uses the live account hourly limit with a safety margin (75 seconds when the limit is unavailable). Request timestamps persist across restarts. The full crawl takes approximately 25 minutes with a 250-search/hour limit, or two hours with a 50-search/hour limit. Inspect the final audit for failures or incomplete titles before committing.

The student reviews, commits, and pushes each completed part before work starts on the next part. The runner performs no Git operations.

Part 2 runs entirely from the saved JSON, using deterministic Python rules and no network/model calls:

```bash
python run_part2.py
quarto render HW-3.2-cleaning-1.ipynb --no-execute
```

The cleaning implementation is in `manual_cleaning.py` and is also displayed in the notebook's implementation appendix, so the standalone HTML includes the extraction logic. Multivalued CSV columns are JSON arrays; missing scalars are blank. Salary assumptions, uncertain duplicate matches, multiple pay bands, and quality flags are documented in the notebook. The `reports/manual-cleaning-*` files contain coverage, provenance for every raw occurrence, duplicate links, review cases, regression examples, and the final audit.

Part 3 uses **Gemma 3 4B (`gemma3:4b`, Q4_K_M)** locally on an Apple M3 Pro with 36 GB memory. **The final scope is a frozen 122-posting sample out of 819 (14.9%), not full-population LLM cleaning.** The sample preserves 133 original occurrences and covers 29/30 search queries; Robotics Engineer is absent. `reports/llm-sample-selection.json` records the exact keys and cache identities. It is an availability sample of completed responses plus prototypes, not a designed probability sample. Small, uneven categories limit subgroup analysis.

Reproduce the submitted sample **without making any model requests**:

```bash
python run_part3.py --stage report
quarto render HW-3.3-cleaning-2.ipynb --no-execute
```

The notebook uses only the same 122 keys from the unchanged 819-row Part 2 CSV for comparisons. Reporting refuses missing/invalid declared caches and validates list/numeric/date CSV round trips. `reports/llm-sample-membership.csv` maps the selected rows to raw occurrences; `reports/llm-sample-query-coverage.csv` includes all 30 query categories, including the zero-count category. The audit explicitly records `COMPLETE_FOR_DECLARED_SAMPLE` and `population_complete=false`.

`llm_cleaning.py` implements prompts, schemas, local requests, cache/resume behavior, sample validation, and export; it is embedded in the HTML appendix. `reports/llm-*` also contains the selected five-job prototype, actual token counts, request timing, production wall time, evidence flags, matched comparisons, and reflection. Multivalued CSV fields are JSON arrays, missing scalars are blank, and salary assumptions are flagged. Coverage and extractor agreement are not accuracy measures.

All 122 final rows use the same v4 evidence-first prompt and Gemma 3 4B model. The selected model has no separate reasoning mode. Production used four independent requests through a temporary local server on port 11435. Local cloud API spend was $0; hardware/electricity and interrupted calls without returned counters are excluded from token estimates.

The wrapper supports explicit inference with `--stage pilot` or `--stage extract` and `--workers`. Reporting uses the frozen sample and requires no running model service. The runner drains active requests on Ctrl+C and resumes valid caches. The measured service had four parallel slots, context 16384, and flash attention enabled.

Part 4 runs offline EDA on both cleaned CSVs through the reusable `JobEDA` class in `job_eda.py`:

```bash
python run_part4.py
quarto render HW-3.4-visual-EDA.ipynb --no-execute
```

The notebook contains eight embedded figures, covering each dataset's profile, salary sensitivity and relationships, state-level geography, shared skill/method mentions, query-group skill profiles, pairwise Spearman correlations, and matched extractor comparison. The full manual dataset has 819 rows; direct comparisons use only the same 122 source identities. The paired-pay panel narrows further to 67 jointly eligible salary observations. Unknown categories stay in denominators. Salary exclusions, assumption sensitivity, shared-vocabulary counts, pair counts, display normalization and the unchanged input hashes are exported under `reports/eda/`. Geospatial analysis uses state labels and excludes ambiguous/multiple locations; no coordinates, maps, or external services are invented or required. Matplotlib 3.10+ is required for the plotting API.

See `SUBMISSION.md` for the four required HTML files and the command to export `commit-history.txt` **after the student commits Part 4**. Optional Part 5 has not been performed.

Final submission checklist:

- [x] Part 1: 30 titles, 90 real response files, 847 raw job records, completed collection audit, executed notebook, self-contained HTML.
- [x] Part 2: deterministic cleaning notebook, `data/processed-jobs-1.csv` (819 postings, 84 columns), provenance and quality reports, self-contained HTML.
- [x] Part 3 (declared sample scope): executed LLM notebook/reflection, `data/processed-jobs-2.csv` (122 postings, 92 columns), frozen selection/provenance/quality reports, self-contained HTML. The remaining 697 postings are not LLM-cleaned.
- [x] Part 4: reusable EDA class, both CSV profiles, eight figures, matched 122-job comparison, executed notebook, self-contained HTML.
- [ ] Optional Part 5, if completed: fictitious Jane Doe resume/cover-letter pipeline and rendered HTML.
- [ ] `commit-history.txt` exported **after** the final homework commit; open it and confirm every part appears.
- [ ] Open each final HTML and verify figures/resources are embedded; check for errors, placeholders, and leaked credentials before uploading to Canvas.
