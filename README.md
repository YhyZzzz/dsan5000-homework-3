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

Final submission checklist:

- [x] Part 1: 30 titles, 90 real response files, 847 raw job records, completed collection audit, executed notebook, self-contained HTML.
- [x] Part 2: deterministic cleaning notebook, `data/processed-jobs-1.csv` (819 postings, 84 columns), provenance and quality reports, self-contained HTML.
- [ ] Part 3: LLM cleaning notebook and reflection, `data/processed-jobs-2.csv`, self-contained HTML.
- [ ] Part 4: visual EDA of both datasets and comparison, executed notebook, self-contained HTML.
- [ ] Optional Part 5, if completed: fictitious Jane Doe resume/cover-letter pipeline and rendered HTML.
- [ ] `commit-history.txt` exported **after** the final homework commit; open it and confirm every part appears.
- [ ] Open each final HTML and verify figures/resources are embedded; check for errors, placeholders, and leaked credentials before uploading to Canvas.
