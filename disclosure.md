# disclosure.md — Extended-Stay Risk at Admission

**Agent used:** Claude (Anthropic), in one continuous working session from idea to deployment.

## What the agent did

- Proposed candidate project ideas, found and checked the public datasets (NYS SPARCS, CMS), and drafted the one-page checkpoint proposal.
- Wrote the code for every notebook step (comment lines and code), ran it on the same data first, and gave me the expected output for each step.
- Wrote `app.py` and `requirements.txt`, and tested the app before I ran it.
- Laid out the threshold options (pure cost, capacity-limited, larger review cost) with their numbers, and found the cost sources (Cochrane/AAFP, BLS).
- Drafted `data.md`, `model_card.md` and `verification.md` from the notebook outputs.

## What I did

- Chose to build the project around Rochester-area inpatient data, so the result could be useful to Rochester Regional Health.
- Downloaded the 2022 and 2023 SPARCS files, applied the Finger Lakes filter on export, and set up the repository.
- Ran every notebook step myself and compared each output with the expected output before moving on; the agent log in `model.ipynb` records each step.
- Made the threshold decision: review capacity of 30% of admissions with the random forest, after the agent laid out the options and recommended that approach. I tested it by asking whether a 2× lift over the base rate was realistic (it is only at the top 1% of admissions), and dropped the 2× bar as a pass/fail test.
- Deployed the app on Streamlit Community Cloud and checked it against the notebook numbers.

## Where I overrode or pushed back on the agent

- **Duplicate rows.** The agent's first answer was to keep the duplicates on the reasoning that, without a patient ID, they are most likely different patients. I asked whether they were verified and whether they could be. That led to the evidence in `verification.md` (most duplicates are excluded newborns, 4 remain in training) and the refit showing they do not move the result.
- **The 2× lift bar.** I questioned whether a 2× base-rate lift was achievable. The answer (only at very small volumes) changed how the model card reports performance: a full lift curve instead of a single pass/fail bar.

## One place the agent was wrong

The first version of the app showed **10,782** extended stays caught at the chosen threshold, while the notebook showed **10,843**. The agent had rounded the saved test scores to 4 decimals and let the slider round the threshold, which pushed about 60 stays across the line. Comparing the app with the notebook caught it, and Step 14 was revised to save unrounded scores. Separately, an early agent message gave 225 rows of `"120 +"` for 2022; the raw file has 286 (225 was the count after exclusions).

---

**Computed:** All figures in the notebook, app and documents come from `model.ipynb` run top to bottom on the two SPARCS files (`random_state=42`).
**Checked:** Every notebook step's output against the expected output; two app numbers ($2,272 per day; 10,843 of 24,331 caught) re-derived in DuckDB/SQL; the leakage refit without the diagnosis; the duplicate refit; the deployed app against the notebook.
**Unchecked:** The cost assumptions (0.73 days saved applied to all adults; 0.5 hours per review); whether the final principal diagnosis matches the admitting diagnosis in practice; why hospital rates differ; performance on any year after 2023 or any hospital outside the Finger Lakes region.
**Assisted:** Claude (Anthropic) wrote the notebook and app code and drafted the documents; I directed the project, ran and checked every step, made the threshold decision and deployed the app.
