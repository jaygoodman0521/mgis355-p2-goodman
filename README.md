# Extended-Stay Risk at Admission

Ranks new inpatient admissions by their chance of running longer than usual for their diagnosis group, so case managers can start discharge planning on day one for the patients most likely to need it.

**MGIS 355 Project 2 · Jay Goodman · Rochester Institute of Technology**

| | |
|---|---|
| **App** | https://mgis355-p2-goodman-jnhd4affm2jxafjhgujbxc.streamlit.app/ |
| **Video** | *link added after recording* |
| **Data** | NYS SPARCS De-Identified Hospital Inpatient Discharges, 2022 (train) and 2023 (test), Finger Lakes region — see `data.md` |

## Result in one paragraph

A random forest using only what is known at admission (hospital, admission type, ED or not, age group, gender, payer, 3-digit ZIP, diagnosis category) was trained on 2022 and tested once on 2023. With a review capacity of 30% of admissions, it catches **10,843 of 24,331 extended stays (44.6%)**. Flagging the same number of admissions, the simple diagnosis-rate rule catches 8,981, so the model catches **1,862 more**, for an estimated **$3.1M** more in net savings under the stated cost assumptions. It is weakest for patients 70 and older (AUC 0.600 vs 0.639 overall). Details in `model_card.md`.

## What is in this repository

| File | What it is |
|---|---|
| `app.py` | Streamlit app: assess an admission, see how good the model is, see its limits |
| `model.ipynb` | Load, check, split, fit, evaluate, one step per cell; runs top to bottom |
| `model.joblib` | The trained random forest the app loads |
| `app_config.json` | Threshold, cost inputs and dropdown options saved by the notebook |
| `data/` | The two SPARCS extracts and the scored 2023 test rows |
| `model_card.md` | One-page model card |
| `data.md` | Source, license, filtering, and the problems the file hid |
| `verification.md` | Traced numbers, leakage check, duplicate check |
| `disclosure.md` | What the AI agent did and what I did |
| `requirements.txt` | Python packages |

## Run it locally

```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

To reproduce every number, open `model.ipynb` from this folder and choose **Restart → Run All**.

## Note

This is a student project built on public data. It is not an official tool of, or endorsed by, Rochester Regional Health or any hospital. Source: New York State Department of Health, SPARCS De-Identified Hospital Inpatient Discharges, 2022 and 2023, accessed November 7, 2026. The New York State Department of Health makes no representation, warranty or guarantee relating to the data or analyses derived from these data.
