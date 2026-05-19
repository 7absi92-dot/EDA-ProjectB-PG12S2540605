# EDA Mini Project B — Time-Series Forecasting Starter

Student: Ahmed Al Habsi  
Student ID: PG12S2540605

This repository contains a starter Streamlit app for Mini Project B using a sliced local dataset at `data/dataset_sample.csv`.

## Files

- `app.py` — one-file Streamlit app
- `requirements.txt` — Python package requirements
- `README.md` — setup and submission instructions
- `data/dataset_sample.csv` — cleaned/sliced dataset sample

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud deployment

1. Create a public GitHub repository.
2. Upload these files exactly:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `data/dataset_sample.csv`
3. Go to Streamlit Community Cloud.
4. Select **New app**.
5. Connect your GitHub repository.
6. Set branch to `main`.
7. Set main file path to `app.py`.
8. Deploy.

## OpenRouter API key

The app checks for the OpenRouter API key in this order:

1. Streamlit Secrets: `OPENROUTER_API_KEY`
2. Environment variable: `OPENROUTER_API_KEY`
3. Password input field inside the app

Do not hardcode API keys in the repository.

## What to submit

Submit these items to your instructor:

- Streamlit deployed app URL
- GitHub repository URL
- Exported `submission.json`
- Exported `project_card.md`
- Required screenshots:
  - first 10 rows preview
  - metrics table after you add your model section
  - at least one dashboard plot after you add dashboard additions
