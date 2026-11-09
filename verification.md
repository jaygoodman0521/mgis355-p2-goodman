# verification.md — Extended-Stay Risk at Admission

Every check below is a cell in `model.ipynb`; the step number says where.

## 1 · Two traced numbers (Step 16)

Both numbers are displayed in the app. Each was reproduced from the raw CSVs **in SQL with DuckDB**, without reusing the pandas code in Steps 4–14 or the app's code.

| Number in the app | Where the app shows it | Independent route | Result | Matched? |
|---|---|---|---|---|
| **$2,272** cost per inpatient day | "How good is this?" → cost assumptions | DuckDB reads the raw 2022 CSV as text, applies the exclusions and the 30-case rule in SQL, and takes `MEDIAN(Total Costs / Length of Stay)` | **$2,272.06** | Yes |
| **10,843 of 24,331** extended stays caught at the chosen threshold | "How good is this?" → Extended stays caught | DuckDB rebuilds the 2022 cutoffs (`QUANTILE_CONT(los, 0.75)` per DRG × severity, 30+ cases), the 2023 test rows and the target in SQL; the saved `model.joblib` scores those rows; count stays at or above the threshold stored in `app_config.json` | **10,843 of 24,331**, from 118,866 test rows | Yes |

One difference had to be handled for the match: DuckDB reads a blank ZIP as `NULL`, while pandas (with `keep_default_na=False`) reads it as an empty string. The SQL route recodes it to `"Unknown"` explicitly, as Step 4 does. Without that, those rows would have reached the model with a missing value.

## 2 · Leakage check (Step 15, supported by Step 11)

**Most suspicious feature: CCSR Diagnosis Code.** SPARCS records the *final* principal diagnosis, coded at discharge, and the model uses it as a stand-in for the admitting diagnosis. If the final diagnosis carries information learned during the stay, the model would look better than it could be in real use. The body system (APR MDC Code) is derived from the diagnosis, so it was removed too.

| Same forest settings, 2023 test year | Test AUC | Extended stays caught (flagging the same 36,723 admissions) |
|---|---|---|
| Deployed model (9 features) | 0.639 | 10,843 |
| Without diagnosis and body system (7 features) | 0.618 | 10,395 |
| Diagnosis rule (baseline) | 0.571 | 8,981 |

**What this shows.** The drop is modest: 0.021 AUC and 448 fewer stays caught. That is **not proof of leakage**, because the reason for admission legitimately predicts length of stay. It is **not proof of innocence** either, because the final diagnosis can differ from the admitting one (for example, chest pain on admission, heart attack on discharge). What settles the recommendation is the worst case: **with no diagnosis at all, the model still catches 1,414 more extended stays than the diagnosis rule at the same volume.** The model card marks the diagnosis features "approximately" known at prediction time for this reason.

**Supporting check (Step 11): the columns deliberately left out.** APR-DRG, Severity of Illness and Risk of Mortality are assigned by coders after discharge, so the deployed model never uses them. Adding them lifts test AUC only from **0.639 to 0.657**. The gain is small because the target is already defined *within* each DRG × severity group, so knowing the group says little about whether a stay runs long for that group. The target definition itself defuses the strongest leakage risk in the file.

## 3 · Duplicate rows (Steps 3, 4 and 11)

Exact duplicates cannot be confirmed or ruled out, because the public file has no patient ID. 128 of 136 duplicated 2022 rows are normal newborns, excluded by design; **4 duplicate rows remain in training and 7 in test.** Identical charges are common among distinct patients: 4,488 non-duplicate 2022 rows share hospital, DRG, length of stay and exact charges with another row. So the duplicates were kept. The time split means no stay can appear in both training and test.

**Sensitivity check (Step 11):** refitting without the 4 training duplicates moves test AUC from **0.6387 to 0.6391**, and recall at the 0.244 threshold from **0.4456 to 0.4503**. That is about what refitting a forest on slightly different rows does by chance, so keeping them does not change the result.

## 4 · Other checks that caught something

| Check | What it found | What changed |
|---|---|---|
| App vs notebook, same threshold | The first version of the app showed **10,782** caught instead of the notebook's **10,843**. Scores saved to `test_scores.csv` had been rounded to 4 decimals, pushing about 60 stays across the threshold. The slider also rounded the threshold to 0.244 from 0.2439…. | Saved unrounded scores (Step 14, revised); the app uses the exact stored threshold at the slider's default. The app now shows 10,843 and $17,149,787, matching Step 10. |
| Raw vs cleaned counts | An early agent message said 225 stays read `"120 +"` in 2022; that was the count after exclusions. The raw file has **286** (265 in 2023). | `data.md` reports the raw counts from Step 3. |
| Unseen test categories | Step 3 found 20 diagnosis codes only in 2023 (raw files); after exclusions the model faces **28 codes in 56 rows**. | Both counts explained in `data.md`; the encoder routes unseen codes to an "infrequent" column instead of failing. |
| Saved model reloads correctly | `model.joblib` reloaded gives identical 2023 scores (Step 14). | — |
| Notebook repeatability | Restart Kernel → Run All on a clean session reproduces every number above (one seed, `random_state=42`, used throughout). | — |
| Year-to-year stability | Base rates are stable across years (20.6% / 20.5%), and train, CV and test AUC agree for logistic regression (0.637 / 0.628 / 0.628), so 2023 behaves like 2022. | — |
