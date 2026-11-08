"""Extended-Stay Risk at Admission — MGIS 355 Project 2 (Jay Goodman).

Loads the random forest trained in model.ipynb; never retrains.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent

st.set_page_config(page_title="Extended-Stay Risk at Admission", layout="wide")


# ---------- load once, never retrain ----------
@st.cache_resource
def load_model():
    return joblib.load(HERE / "model.joblib")


@st.cache_data
def load_config():
    with open(HERE / "app_config.json") as f:
        return json.load(f)


@st.cache_data
def load_test_scores():
    return pd.read_csv(HERE / "data" / "test_scores.csv")


model = load_model()
cfg = load_config()
scores = load_test_scores()

THRESHOLD = cfg["threshold"]
TARGET = "extended_stay"
N_TEST = len(scores)
N_EXTENDED = int(scores[TARGET].sum())
BASE_RATE = cfg["base_rate_test"]


def money(x):
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def evaluate(flags, y):
    tp = int((flags & (y == 1)).sum())
    fp = int((flags & (y == 0)).sum())
    fn = int((~flags & (y == 1)).sum())
    tn = int((~flags & (y == 0)).sum())
    flagged = tp + fp
    return {"flagged": flagged, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "precision": tp / flagged if flagged else 0.0}


def top_k_flags(score, k):
    """Flag the k highest scores (ties broken by row order) so two methods flag the same volume."""
    flags = np.zeros(len(score), dtype=bool)
    if k > 0:
        flags[np.argsort(-score.to_numpy(), kind="stable")[:k]] = True
    return flags


# ---------- first screen ----------
st.title("Extended-Stay Risk at Admission")
st.markdown(
    "**For inpatient case managers at Rochester-area hospitals: decide, on the day a patient is admitted, "
    "whether to start discharge planning now or wait for the routine review.**"
)
st.caption(
    f"Random forest trained on {cfg['train_year']} and tested on {cfg['test_year']} NYS SPARCS de-identified "
    "inpatient discharges, Finger Lakes region (17 hospitals). Student project on public data; "
    "not an official Rochester Regional Health tool."
)

tab_predict, tab_good, tab_limits = st.tabs(["Assess an admission", "How good is this?", "Limits"])

# ---------- prediction view ----------
with tab_predict:
    opts = cfg["options"]
    ccsr_desc = cfg["ccsr_description"]
    ccsr_codes = sorted(opts["CCSR Diagnosis Code"], key=lambda c: ccsr_desc.get(c, c))

    st.subheader("Enter what is known at admission")
    c1, c2, c3 = st.columns(3)
    with c1:
        hospital = st.selectbox("Hospital", opts["Facility Name"],
                                index=opts["Facility Name"].index("Rochester General Hospital"))
        age = st.selectbox("Age group", opts["Age Group"], index=opts["Age Group"].index("70 or Older"))
        gender = st.selectbox("Gender", opts["Gender"])
    with c2:
        admit_types = [t for t in opts["Type of Admission"] if t not in ("Newborn", "Not Available")]
        admit = st.selectbox("Type of admission", admit_types, index=admit_types.index("Emergency"))
        ed = st.radio("Came through the emergency department?", ["Y", "N"], horizontal=True,
                      format_func=lambda v: "Yes" if v == "Y" else "No")
        payer = st.selectbox("Primary payer", opts["Payment Typology 1"],
                             index=opts["Payment Typology 1"].index("Medicare"))
    with c3:
        dx = st.selectbox("Admitting diagnosis (CCSR category)", ccsr_codes,
                          index=ccsr_codes.index("CIR019") if "CIR019" in ccsr_codes else 0,
                          format_func=lambda c: f"{ccsr_desc.get(c, c)} ({c})")
        zip_labels = {"OOS": "Out of state", "Unknown": "Unknown / not recorded"}
        zips = opts["zip3"]
        zip3 = st.selectbox("Patient 3-digit ZIP", zips, index=zips.index("146") if "146" in zips else 0,
                            format_func=lambda z: zip_labels.get(z, z))

    mdc = cfg["ccsr_to_mdc"][dx]
    case = pd.DataFrame([{
        "Facility Name": hospital, "Type of Admission": admit, "Emergency Department Indicator": ed,
        "Age Group": age, "Gender": gender, "Payment Typology 1": payer, "zip3": zip3,
        "CCSR Diagnosis Code": dx, "APR MDC Code": mdc,
    }])[cfg["features"]]
    prob = float(model.predict_proba(case)[0, 1])
    flagged = prob >= THRESHOLD
    pct_rank = float((scores["score"] < prob).mean() * 100)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Chance this stay runs long", f"{prob:.0%}")
    m2.metric("Compared with the average admission", f"{prob / BASE_RATE:.1f}×",
              help=f"Average extended-stay rate in {cfg['test_year']}: {BASE_RATE:.1%}")
    m3.metric("Riskier than", f"{pct_rank:.0f}% of admissions",
              help=f"Share of {N_TEST:,} {cfg['test_year']} admissions with a lower predicted risk")

    if flagged:
        st.success(f"**Action: start discharge planning on day one.** At the chosen threshold ({THRESHOLD:.3f}), "
                   "this admission is in the group the review team can cover (about the top 30% of admissions).")
    else:
        st.info(f"**Action: routine review.** This admission is below the threshold ({THRESHOLD:.3f}); "
                "early-review slots go to higher-risk admissions first.")

    similar = scores[(scores["Age Group"] == age) & (scores["Type of Admission"] == admit)]
    if len(similar) >= 30:
        st.markdown(
            f"In {cfg['test_year']}, **{similar[TARGET].mean():.0%}** of {len(similar):,} admissions that were "
            f"{age.lower()} and {admit.lower()} ran longer than usual for their diagnosis group; "
            f"this patient's predicted chance is **{prob:.0%}**."
        )
    st.caption(f"Body system used by the model (from the diagnosis): {cfg['mdc_description'].get(mdc, mdc).title()}. "
               "An extended stay means longer than 75% of 2022 Finger Lakes stays with the same APR-DRG and severity.")

# ---------- how good is this ----------
with tab_good:
    st.subheader(f"Tested once on {N_TEST:,} admissions from {cfg['test_year']}, a year the model never saw")

    with st.expander("Cost assumptions (change them to see how the dollars move)"):
        e1, e2, e3 = st.columns(3)
        days_saved = e1.number_input("Days saved per extended stay caught", 0.0, 5.0, float(cfg["days_saved"]), 0.05,
                                     help="0.73 days: Cochrane review of discharge planning (AAFP summary, 2022)")
        cost_per_day = e2.number_input("Cost per inpatient day ($)", 500.0, 10000.0, float(round(cfg["cost_per_day"])), 50.0,
                                       help="Median of Total Costs / Length of Stay, 2022 SPARCS training rows")
        review_cost = e3.number_input("Cost of one early review ($)", 0.0, 1000.0,
                                      float(round(cfg["rn_hourly_wage"] * cfg["review_hours"], 2)), 1.0,
                                      help="0.5 hours (assumption) x $45.42/h, BLS mean RN wage, May 2023")
    saving = days_saved * cost_per_day

    thr = st.slider("Threshold: flag an admission when its predicted chance is at least…",
                    0.10, 0.45, float(round(THRESHOLD, 3)), 0.005, format="%.3f")
    if abs(thr - round(THRESHOLD, 3)) < 1e-9:
        thr = THRESHOLD  # the slider shows 3 decimals; use the exact locked threshold at its default position
    y = scores[TARGET].to_numpy()
    model_flags = scores["score"].to_numpy() >= thr
    m = evaluate(model_flags, y)
    r = evaluate(top_k_flags(scores["rule_score"], m["flagged"]), y)

    def net(res):
        return res["tp"] * saving - res["flagged"] * review_cost

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Admissions flagged", f"{m['flagged'] / N_TEST:.1%}", f"{m['flagged']:,} stays", delta_color="off")
    k2.metric("Extended stays caught", f"{m['recall']:.1%}", f"{m['tp']:,} of {N_EXTENDED:,}", delta_color="off")
    k3.metric("Flags that were right", f"{m['precision']:.1%}",
              f"{m['precision'] / BASE_RATE:.2f}× the {BASE_RATE:.1%} base rate", delta_color="off")
    k4.metric("Estimated net savings", money(net(m)), f"{money(net(m) - net(r))} vs diagnosis rule", delta_color="off")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Confusion matrix at this threshold**")
        cm = pd.DataFrame(
            {"Flagged for early planning": [f"{m['tp']:,} caught", f"{m['fp']:,} false alarms"],
             "Routine review": [f"{m['fn']:,} missed", f"{m['tn']:,} correctly left"]},
            index=["Stay ran long", "Stay did not run long"])
        st.table(cm)
        st.markdown(
            f"**What the mistakes cost:** {m['fn']:,} missed long stays leave about "
            f"{money(m['fn'] * saving)} of avoidable bed-days on the table; {m['fp']:,} false alarms use "
            f"{money(m['fp'] * review_cost)} of case-manager time."
        )
    with right:
        st.markdown("**Against the baselines, flagging the same number of admissions**")
        comp = pd.DataFrame([
            {"Method": "Flag nobody (always 'not extended')", "Caught": "0", "Recall": "0.0%",
             "Precision": "—", "Net savings": money(0)},
            {"Method": "Diagnosis rule (2022 rate of the diagnosis)", "Caught": f"{r['tp']:,}",
             "Recall": f"{r['recall']:.1%}", "Precision": f"{r['precision']:.1%}", "Net savings": money(net(r))},
            {"Method": "Random forest (this model)", "Caught": f"{m['tp']:,}",
             "Recall": f"{m['recall']:.1%}", "Precision": f"{m['precision']:.1%}", "Net savings": money(net(m))},
        ]).set_index("Method")
        st.table(comp)

    # trade-off curve
    grid = np.round(np.arange(0.10, 0.451, 0.005), 3)
    sc = scores["score"].to_numpy()
    curve = pd.DataFrame([{"thr": t, "flagged": (sc >= t).mean() * 100,
                           "recall": ((sc >= t) & (y == 1)).sum() / N_EXTENDED * 100} for t in grid])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["flagged"], y=curve["recall"], mode="lines", name="Random forest",
                             hovertemplate="%{x:.1f}% flagged → %{y:.1f}% caught<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random flagging",
                             line=dict(dash="dot", color="gray")))
    fig.add_trace(go.Scatter(x=[m["flagged"] / N_TEST * 100], y=[m["recall"] * 100], mode="markers",
                             name="Current threshold", marker=dict(size=12)))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="% of admissions flagged", yaxis_title="% of extended stays caught",
                      title="The trade as the threshold moves", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")

    gap = m["tp"] - r["tp"]
    st.markdown(
        f"> **At a threshold of {thr:.3f}, the model flags {m['flagged'] / N_TEST:.1%} of admissions and catches "
        f"{m['tp']:,} of {N_EXTENDED:,} extended stays ({m['recall']:.1%}) — {abs(gap):,} "
        f"{'more' if gap >= 0 else 'fewer'} than the diagnosis rule at the same volume — for an estimated "
        f"{money(net(m))} in net savings under the assumptions above.**"
    )
    if thr != THRESHOLD:
        st.caption(f"The chosen threshold is {THRESHOLD:.3f}: it fills a review capacity of 30% of admissions, set on 2022 data.")

# ---------- limits ----------
with tab_limits:
    st.subheader("When not to trust it")

    @st.cache_data
    def segment_table(df):
        rows = []
        for col in ["Age Group", "Type of Admission", "Payment Typology 1", "APR MDC Description"]:
            for value, g in df.groupby(col):
                if len(g) >= 1000 and g[TARGET].nunique() == 2:
                    caught = g.loc[g[TARGET] == 1, "score"] >= THRESHOLD
                    rows.append({"Segment": f"{col}: {str(value).title() if col == 'APR MDC Description' else value}",
                                 "Stays": len(g), "AUC": roc_auc_score(g[TARGET], g["score"]),
                                 "Extended stays caught": caught.mean()})
        return pd.DataFrame(rows).sort_values("AUC")

    seg = segment_table(scores)
    overall_auc = roc_auc_score(scores[TARGET], scores["score"])
    worst = seg.iloc[0]
    older = seg[seg["Segment"] == "Age Group: 70 or Older"]
    older_line = (f" Patients 70 and older score {older['AUC'].iloc[0]:.3f} on {int(older['Stays'].iloc[0]):,} stays: "
                  "the model mostly flags them for being older and cannot tell which of them will run long."
                  if len(older) else "")
    st.warning(
        f"**Weakest segment: {worst['Segment']}** — AUC {worst['AUC']:.3f} against {overall_auc:.3f} overall, "
        f"on {int(worst['Stays']):,} test stays.{older_line}"
    )
    show = seg.head(5).copy()
    show["Stays"] = show["Stays"].map("{:,}".format)
    show["AUC"] = show["AUC"].map("{:.3f}".format)
    show["Extended stays caught"] = show["Extended stays caught"].map("{:.1%}".format)
    st.markdown("**Five weakest segments (1,000+ test stays each)**")
    st.table(show.set_index("Segment"))

    st.markdown("**What it was trained on**")
    st.markdown(
        f"- {cfg['train_year']} inpatient discharges from the 17 hospitals in New York's Finger Lakes region "
        "(includes Rochester General, Unity, Newark-Wayne and Clifton Springs; not United Memorial in Batavia).\n"
        "- Adults and children; **newborn birth stays are excluded**, as are ungroupable stays and rare DRG × "
        "severity groups (under 30 cases in 2022).\n"
        "- Only what is known at admission: hospital, admission type, ED or not, age group, gender, payer, 3-digit ZIP "
        "and diagnosis category. The final DRG and severity are assigned after discharge and are not used.\n"
        "- Diagnoses with fewer than 30 training stays are grouped together, so the model knows little about them."
    )
    st.markdown("**Do not use it for**")
    st.markdown(
        "- Comparing hospital performance: hospital rates differ for reasons this data cannot separate "
        "(case mix within a DRG, post-acute bed availability).\n"
        "- Decisions about an individual patient's care: it ranks admissions for a review queue; "
        "it does not replace a case manager's assessment.\n"
        "- Hospitals outside the Finger Lakes region, or years far from 2022–2023, without re-testing."
    )
    st.caption("Dollar figures depend on the cost assumptions on the 'How good is this?' tab and are estimates, not results.")
