# Homework 3 submission

The four required parts are implemented. Part 3 intentionally uses a declared 122-of-819 availability sample; its HTML and the Part 4 comparison state this scope and its limitations. Optional Part 5 was not undertaken.

Upload these self-contained HTML files to Canvas:

- `HW-3.1-crawl.html`
- `HW-3.2-cleaning-1.html`
- `HW-3.3-cleaning-2.html`
- `HW-3.4-visual-EDA.html`

Also upload `commit-history.txt`. **Create or refresh it after committing Part 4**, from the repository root:

```bash
git log --pretty=format:"%h | %ad | %an | %s" --date=iso > commit-history.txt
```

Open the result and confirm that it contains the Part 4 commit as well as the earlier parts. The Canvas history export need not create another Git commit. If you make any later commits, refresh the file again before upload.

The private GitHub repository contains the notebooks, code, raw JSON, both processed CSVs, selected model caches, tests and reports. Review and push Part 4 personally. The helper scripts perform no commits or pushes. Do not upload API keys.

Part 4 can be reproduced offline with:

```bash
python run_part4.py
quarto render HW-3.4-visual-EDA.ipynb --no-execute
```

Final checks: open each HTML, confirm its figures are visible without network access, confirm there are no error outputs, and ensure the exported history includes the final homework commit. The automated HTML/input checks are recorded in `reports/eda/submission-verification.json`.
