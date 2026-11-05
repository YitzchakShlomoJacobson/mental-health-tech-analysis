# Workplace support profiles in the OSMI 2016 survey

Analysis code for an exploratory case study on workplace mental-health support and disclosure in technology-related employment.

## Set up the analysis

Use Python 3.12. In Windows PowerShell, open the extracted project folder and run:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS or Linux, use `python3 -m venv .venv` and `.venv/bin/python -m pip install -r requirements.txt`.

## Obtain the source data

Download the 2016 CSV from the [OSMI Mental Health in Tech Survey dataset](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016). Extract the CSV named `mental-heath-in-tech-2016_20161114.csv`, and save it as `data/survey_2016.csv` inside the project folder. Preserve the file's contents.

Source: Open Sourcing Mental Health, [Mental Health in Tech Survey 2016](https://www.osmihelp.org/research.html). OSMI distributes the survey data under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Run

In Windows PowerShell:

```powershell
.venv\Scripts\python.exe analyse.py
```

On macOS or Linux, run `.venv/bin/python analyse.py`. The script creates tables and summary metrics in `results/` and plots in `figures/`. It records package versions and a SHA-256 digest of the input file.

The workflow audits missingness, selects current workplace questions, encodes categorical responses, and compares k-means with average-linkage hierarchical clustering. It checks stability under subsampling and uses PCA and t-SNE to visualise the selected partition. The groupings describe exploratory patterns in workplace responses; they are not clinical diagnoses.

This repository includes the analysis script and dependencies. Obtain the survey CSV from the original publisher as described above, then generate the outputs locally.
