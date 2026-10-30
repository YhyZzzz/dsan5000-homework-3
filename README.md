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
