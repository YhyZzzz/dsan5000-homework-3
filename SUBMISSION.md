# Homework 3 submission

The four required parts and optional Part 5 bonus are implemented. Part 3 intentionally uses a declared 122-of-819 availability sample; its HTML and the Part 4 comparison state this scope and its limitations. Part 5 uses only the fictional Jane Doe profile and embeds all ten application documents in its standalone HTML report.

Upload these self-contained HTML files to Canvas:

- `HW-3.1-crawl.html`
- `HW-3.2-cleaning-1.html`
- `HW-3.3-cleaning-2.html`
- `HW-3.4-visual-EDA.html`
- `HW-3.5-optional-bonus-resume-creator.html` (bonus; contains five resume/cover-letter pairs)

Also upload `commit-history.txt`. **Create or refresh it after committing Part 5**, from the repository root:

```bash
git log --pretty=format:"%h | %ad | %an | %s" --date=iso > commit-history.txt
```

Open the result and confirm that it contains the Part 5 commit as well as the earlier parts. The Canvas history export need not create another Git commit. If you make any later commits, refresh the file again before upload.

The private GitHub repository contains the notebooks, code, raw JSON, both processed CSVs, selected model caches, tests and reports. Review and push Part 5 personally. The helper scripts perform no commits or pushes. Do not upload API keys.

Part 4 can be reproduced offline with:

```bash
python run_part4.py
quarto render HW-3.4-visual-EDA.ipynb --no-execute
```

Final checks: open each HTML, confirm its figures are visible without network access, confirm there are no error outputs, and ensure the exported history includes the final homework commit. The automated HTML/input checks are recorded in `reports/eda/submission-verification.json`.

Reproduce the completed bonus from caches without model calls:

```bash
python run_part5.py --stage render
python run_part5.py --stage report
quarto render HW-3.5-optional-bonus-resume-creator.ipynb --no-execute
```

The ten separate HTML documents and QMD sources are indexed in `applications/README.md`. They are also embedded in the bonus report, so the standalone bonus HTML is sufficient to view every pair. No actual employer is contacted. Bonus checks are recorded in `reports/bonus/submission-verification.json`.
