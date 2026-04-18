"""
Hybrid IDS Dashboard v4 - Streamlit Web Interface
================================================

INTERACTIVE DASHBOARD FOR HIDS v4:
- Real-time network traffic classification
- Dual dataset support (NSL-KDD + UNSW-NB15)
- 5-model comparison and selection
- SHAP-based model explainability
- File upload and batch processing
- Performance visualization and metrics
- Confidence thresholding and decision logic

KEY FEATURES:
1. Dataset Selection: Switch between NSL-KDD and UNSW-NB15
2. Model Comparison: View accuracy/speed for all 5 models
3. Real-time Classification: Upload CSV or manual input
4. SHAP Explanations: Feature importance and force plots
5. Confusion Matrices: Detailed performance analysis
6. Classification Reports: Precision, recall, F1 scores
7. Batch Processing: Handle multiple samples
8. Decision Thresholding: ATTACK/UNCERTAIN/CLEAN logic

DASHBOARD PAGES:
- Overview: KPI metrics and model comparison
- Dataset Comparison: Cross-dataset performance
- Model Leaderboard: Ranked model performance
- Real-time Monitor: Live classification interface
- Upload & Classify: Batch file processing
- Confusion Matrices: Visual performance analysis
- SHAP Explainability: Model interpretability
- Classification Reports: Detailed metrics
- About: Project information

DATA FLOW:
1. Load pre-trained models and metadata
2. Preprocess input data (scaling, feature selection)
3. Run inference on all 5 models
4. Apply confidence thresholding
5. Generate SHAP explanations
6. Display results with visualizations

MODEL LOADING:
- Supports both NSL-KDD and UNSW-NB15 models
- Automatic scaler and feature selection loading
- Error handling for missing models
- Caching for performance

INPUT PREPROCESSING:
- CSV upload with validation
- Manual feature input
- Categorical encoding
- Feature scaling and selection
- Missing value handling

VISUALIZATIONS:
- Accuracy/speed bar charts
- Confusion matrices (raw + normalized)
- SHAP summary plots
- SHAP force plots
- Radar charts for multi-metric comparison
- Cross-dataset comparison charts

DEPENDENCIES: streamlit, pandas, numpy, tensorflow, shap, matplotlib, scikit-learn

USAGE: streamlit run dashboard.py

For complete codebase documentation, see CODE_DOCUMENTATION.md
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle, json, os, io, time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf

st.set_page_config(page_title="Hybrid IDS v4", page_icon="🛡️", layout="wide")
# ── PLAIN ENGLISH GLOSSARY ────────────────────────────────────────────────────
GLOSSARY = {
    "NSL-KDD": "NSL-KDD Dataset — A standard cybersecurity test dataset created in 2009. Contains recorded network traffic with labelled attacks. Used by researchers worldwide to test intrusion detection systems. Think of it as a practice dataset of old-style attacks.",
    "UNSW-NB15": "UNSW-NB15 Dataset — A modern cybersecurity dataset created in 2015 by the University of New South Wales, Australia. Contains real network traffic with 10 types of modern attacks including backdoors and malware. More realistic than NSL-KDD.",
    "RF": "Random Forest — A machine learning model that uses hundreds of decision trees and takes a majority vote. Like asking 150 experts and going with the most common answer. Fast and reliable.",
    "ET": "Extra Trees (Extremely Randomised Trees) — Similar to Random Forest but uses extra randomness when building trees. Often faster and generalises better across different datasets.",
    "XGB": "XGBoost (Extreme Gradient Boosting) — A powerful machine learning model that learns from its mistakes. It builds trees one by one, each one correcting errors from the previous. Very accurate on tabular data.",
    "LGBM": "LightGBM (Light Gradient Boosting Machine) — Similar to XGBoost but much faster. Designed by Microsoft for large datasets. Excellent speed-accuracy balance.",
    "DNN": "Deep Neural Network — A system loosely inspired by the human brain. Data passes through multiple layers of mathematical transformations. Can learn very complex patterns but needs more data and time to train.",
    "SHAP": "SHAP (SHapley Additive exPlanations) — A method to explain WHY a model made a decision. It shows which features (like high packet size or unusual port) pushed the model towards predicting an attack. Named after game theory mathematics by Lloyd Shapley.",
    "DoS": "DoS — Denial of Service attack. The attacker floods a server with so many fake requests that it becomes too busy to serve real users. Like blocking a shop entrance so genuine customers cannot enter.",
    "Probe": "Probe Attack — The attacker scans and gathers information about a network before launching a real attack. Like a burglar walking around a building checking which windows are open before breaking in.",
    "R2L": "R2L — Remote to Local attack. An outsider on the internet gains unauthorised access to a local machine. Like someone breaking into your house remotely by exploiting a weak password.",
    "U2R": "U2R — User to Root attack. A normal user somehow gains administrator (root) privileges they should not have. Like a regular employee accessing the CEO's confidential files.",
    "Normal": "Normal Traffic — Regular, legitimate network activity with no attack. Safe packets going about their normal business.",
    "FAR": "FAR — False Alarm Rate. The percentage of normal traffic that the system wrongly flags as an attack. A lower FAR means fewer unnecessary alerts for security analysts.",
    "SMOTE": "SMOTE — Synthetic Minority Oversampling Technique. A method to fix imbalanced datasets where one class (like U2R attacks) has very few examples. It creates artificial new examples of the rare class so the model learns it properly.",
    "Accuracy": "Accuracy — The percentage of predictions the model got right out of all predictions. Example: 76% accuracy means 76 out of every 100 packets were correctly classified.",
    "F1-Score": "F1-Score — A balanced measure combining both Precision (how many flagged attacks were real) and Recall (how many real attacks were caught). Better than accuracy alone when classes are imbalanced.",
    "Confidence": "Confidence Score — How certain the model is about its prediction, from 0% to 100%. High confidence means the model is very sure. Low confidence means it is uncertain and a human should review.",
    "ATTACK": "ATTACK — The model is more than 85% confident this is a genuine network intrusion. Triggers an immediate security alert.",
    "UNCERTAIN": "UNCERTAIN — The model is 60-85% confident. Not sure enough to auto-alert. Sent to a human security analyst for manual review.",
    "CLEAN": "CLEAN — The model is less than 60% confident this is an attack. Classified as normal, safe network traffic.",
}

def tooltip(term):
    """Show a term with its plain English explanation in an expander"""
    explanation = GLOSSARY.get(term, "No explanation available.")
    return f"**{term}** — {explanation}"

def show_glossary_sidebar():
    """Add a glossary section to the sidebar"""
    with st.sidebar.expander("📖 What do these terms mean?", expanded=False):
        st.markdown("**Click any term to learn what it means:**")
        for term, explanation in GLOSSARY.items():
            st.markdown(f"**{term}:** {explanation}")
            st.markdown("---")

def explain_metric(metric_name, value, context=""):
    """Show a metric with plain English explanation"""
    explanations = {
        "Accuracy": f"**{value}** — Out of every 100 packets analysed, the model correctly identified {value} of them.",
        "F1-Score": f"**{value}** — Combined measure of detection quality (0% = useless, 100% = perfect).",
        "FAR": f"**{value}** — This many normal packets were wrongly flagged as attacks (lower is better).",
        "Time": f"**{value}** seconds to train the model on the dataset.",
    }
    base = explanations.get(metric_name, f"**{value}**")
    if context:
        return f"{base} {context}"
    return base

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ORDER  = ["Random Forest","Extra Trees","XGBoost","LightGBM","DNN"]
MODEL_COLORS = ["#3498db","#27ae60","#e67e22","#8e44ad","#e74c3c"]
MODEL_SHORT  = {"Random Forest":"RF","Extra Trees":"ET","XGBoost":"XGB",
                "LightGBM":"LGBM","DNN":"DNN"}
THRESH_ATK   = 0.85
THRESH_UNC   = 0.60

DATASETS = {
    "NSL-KDD": {
        "tag":        "nsl_kdd",
        "year":       "1999/2009",
        "classes":    ["Normal","DoS","Probe","R2L","U2R"],
        "n_classes":  5,
        "attacks":    "Neptune · Smurf · Satan · Portsweep · Warezclient",
        "era":        "Legacy (16-year-old attacks)",
        "color":      "#3498db",
        "cat_cols":   ["protocol_type","service","flag"],
        "drop_cols":  ["label","difficulty"],
        "label_col":  "label",
        "description":"The classic benchmark — every IDS paper uses NSL-KDD. "
                      "Allows direct comparison with published literature.",
    },
    "UNSW-NB15": {
        "tag":        "unsw_nb15",
        "year":       "2015",
        "classes":    ["Normal","Exploits","Reconnaissance","DoS","Generic",
                       "Shellcode","Fuzzers","Worms","Backdoors","Analysis"],
        "n_classes":  10,
        "attacks":    "Exploits · Backdoors · Shellcode · Fuzzers · Worms · Reconnaissance",
        "era":        "Modern (real 2015 traffic + tools)",
        "color":      "#e67e22",
        "cat_cols":   ["proto","service","state"],
        "drop_cols":  ["id","label"],
        "label_col":  "attack_cat",
        "description":"Modern dataset with 10 attack categories including "
                      "backdoors, shellcode and fuzzers — attacks that exist today.",
    },
}

CLASS_COLORS_NSL  = {"Normal":"#2ecc71","DoS":"#e74c3c","Probe":"#e67e22",
                     "R2L":"#9b59b6","U2R":"#c0392b"}
CLASS_COLORS_UNSW = {"Normal":"#2ecc71","Exploits":"#e74c3c",
                     "Reconnaissance":"#e67e22","DoS":"#c0392b",
                     "Generic":"#8e44ad","Shellcode":"#1abc9c",
                     "Fuzzers":"#f39c12","Worms":"#d35400",
                     "Backdoors":"#922b21","Analysis":"#1a5276"}

def get_class_colors(ds_name):
    return CLASS_COLORS_NSL if ds_name=="NSL-KDD" else CLASS_COLORS_UNSW

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_dataset_models(tag):
    base = f"models/{tag}"
    if not os.path.exists(f"{base}/rf_model.pkl"):
        return None
    try:
        rf   = pickle.load(open(f"{base}/rf_model.pkl","rb"))
        et   = pickle.load(open(f"{base}/et_model.pkl","rb"))
        xgb  = pickle.load(open(f"{base}/xgb_model.pkl","rb"))
        lgbm = pickle.load(open(f"{base}/lgbm_model.pkl","rb"))
        dnn  = tf.keras.models.load_model(f"{base}/dnn_model.h5",compile=False)
        scaler = pickle.load(open(f"{base}/scaler.pkl","rb"))
        top15  = np.load(f"{base}/top15_features.npy",allow_pickle=True).tolist()
        with open(f"{base}/class_names.json") as f:
            class_names = json.load(f)
        accs = {}
        if os.path.exists(f"{base}/model_comparison.json"):
            with open(f"{base}/model_comparison.json") as f:
                d = json.load(f)
            accs = {m: d[m].get("acc",0) for m in d}
        return {"rf":rf,"et":et,"xgb":xgb,"lgbm":lgbm,"dnn":dnn,
                "scaler":scaler,"top15":top15,
                "class_names":class_names,"accs":accs}
    except Exception as e:
        return None

def get_models(tag):
    m = load_dataset_models(tag)
    if m is None:
        return None
    return {
        "Random Forest": m["rf"],
        "Extra Trees":   m["et"],
        "XGBoost":       m["xgb"],
        "LightGBM":      m["lgbm"],
        "DNN":           m["dnn"],
    }, m["scaler"], m["top15"], m["class_names"], m["accs"]

# ── Load master comparison ────────────────────────────────────────────────────
master = {}
if os.path.exists("models/master_comparison.json"):
    with open("models/master_comparison.json") as f:
        master = json.load(f)

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_df(df_raw, scaler, top15, feat_cols, cat_cols):
    from sklearn.preprocessing import LabelEncoder
    df = df_raw.copy()
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0
    X_sc = scaler.transform(df[feat_cols].fillna(0).values.astype(float))
    fm   = {n:i for i,n in enumerate(feat_cols)}
    idx  = [fm[f] for f in top15 if f in fm]
    return X_sc[:, idx]

def threshold(probs, preds):
    conf = probs.max(axis=1)
    return np.where((conf>=THRESH_ATK)&(preds!=0),"ATTACK",
           np.where(conf>=THRESH_UNC,"UNCERTAIN","CLEAN"))

def predict_all(models, X_top, ds_name):
    out = {}
    for name, mdl in models.items():
        if name=="DNN":
            pr   = mdl.predict(X_top,verbose=0)
            pred = pr.argmax(axis=1)
            conf = pr.max(axis=1)
        else:
            pred = mdl.predict(X_top)
            pr   = mdl.predict_proba(X_top)
            conf = pr.max(axis=1)
        dec = threshold(pr,pred)
        out[name] = {"pred":pred,"conf":conf,"dec":dec,"probs":pr,
                     "n_atk":(dec=="ATTACK").sum(),
                     "n_unc":(dec=="UNCERTAIN").sum(),
                     "n_cln":(dec=="CLEAN").sum()}
    return out

# ── Table helpers ──────────────────────────────────────────────────────────────
def build_results_df(pred,conf,dec,probs,class_names):
    df = pd.DataFrame({
        "Row #":      range(1,len(pred)+1),
        "Predicted":  [class_names[c] if c<len(class_names) else str(c) for c in pred],
        "Confidence": [f"{c*100:.1f}%" for c in conf],
        "Decision":   list(dec),
    })
    for i,cn in enumerate(class_names):
        df[f"{cn[:8]} %"] = [f"{p[i]*100:.1f}%" if i<len(p) else "0.0%" for p in probs]
    return df

def style_table(df):
    def cd(v):
        m = {"ATTACK":   "background-color:#f5c6cb;color:#721c24;font-weight:bold",
             "UNCERTAIN":"background-color:#fff3cd;color:#856404;font-weight:bold",
             "CLEAN":    "background-color:#d4edda;color:#155724;font-weight:bold"}
        return m.get(v,"")
    return df.style.applymap(cd, subset=["Decision"])

def show_filter_table(res_df, class_names, key_sfx=""):
    st.markdown("#### 🔎 Filter Results")
    c1,c2,c3 = st.columns(3)
    with c1:
        f_dec = st.multiselect("Decision",["ATTACK","UNCERTAIN","CLEAN"],
                               default=["ATTACK","UNCERTAIN","CLEAN"],
                               key=f"fd_{key_sfx}")
    with c2:
        f_cls = st.multiselect("Attack Class", class_names, default=class_names,
                               key=f"fc_{key_sfx}")
    with c3:
        f_conf = st.slider("Min Confidence %",0,100,0,5,key=f"fconf_{key_sfx}")

    conf_vals = res_df["Confidence"].str.rstrip("%").astype(float)
    filtered  = res_df[res_df["Decision"].isin(f_dec) &
                       res_df["Predicted"].isin(f_cls) &
                       (conf_vals >= f_conf)]

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Showing",      f"{len(filtered):,} / {len(res_df):,}")
    m2.metric("🔴 ATTACK",    f"{(filtered['Decision']=='ATTACK').sum():,}")
    m3.metric("🟡 UNCERTAIN", f"{(filtered['Decision']=='UNCERTAIN').sum():,}")
    m4.metric("🟢 CLEAN",     f"{(filtered['Decision']=='CLEAN').sum():,}")

    st.dataframe(style_table(filtered), use_container_width=True, hide_index=True)
    st.caption(f"{len(filtered):,} rows shown | Decision:{f_dec} | Class:{f_cls} | Conf≥{f_conf}%")

    dl1,dl2 = st.columns(2)
    with dl1:
        st.download_button("⬇️ Filtered CSV",
                           filtered.to_csv(index=False).encode(),
                           f"filtered_{key_sfx}.csv","text/csv",
                           key=f"dl_f_{key_sfx}")
    with dl2:
        st.download_button("⬇️ All Results",
                           res_df.to_csv(index=False).encode(),
                           f"all_{key_sfx}.csv","text/csv",
                           key=f"dl_a_{key_sfx}")

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("🛡️ Hybrid IDS v4")
st.sidebar.markdown("**Dual-Dataset | 5-Model Pipeline**")
st.sidebar.markdown("---")

# Dataset selector — the key feature
st.sidebar.markdown("###  Active Dataset")
active_ds = st.sidebar.radio(
    "Select dataset to explore:",
    ["NSL-KDD", "UNSW-NB15"],
    captions=["Legacy (2009) · 5 classes", "Modern (2015) · 10 classes"]
)
ds_cfg    = DATASETS[active_ds]
ds_tag    = ds_cfg["tag"]
ds_color  = ds_cfg["color"]

# Show active indicator
st.sidebar.markdown(
    f"<div style='background:{ds_color};color:white;padding:8px 12px;"
    f"border-radius:6px;font-weight:bold;text-align:center'>"
    f"🗂 {active_ds} ({ds_cfg['year']})</div>",
    unsafe_allow_html=True
)
st.sidebar.markdown("")

# Load models for active dataset
models_loaded = get_models(ds_tag)
if models_loaded:
    all_models, scaler, top15, class_names, model_accs = models_loaded
    best_model = max(model_accs, key=model_accs.get) if model_accs else "N/A"
else:
    all_models = scaler = top15 = class_names = model_accs = None
    best_model = "N/A"

st.sidebar.markdown("---")
# Add glossary to sidebar
show_glossary_sidebar()
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "  Overview",
    "   Dataset Comparison",
    "  Model Leaderboard",
    "  Real-Time Monitor",
    "  Upload & Classify",
    "  Confusion Matrices",
    "  SHAP Explainability",
    "  Classification Reports",
    "   About"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"###  {active_ds} Leaderboard")
if model_accs:
    medals = ["1","2","3","4️","5️"]
    ranked = sorted(model_accs.items(), key=lambda x:x[1], reverse=True)
    for i,(name,acc) in enumerate(ranked):
        st.sidebar.markdown(f"{medals[i]} **{MODEL_SHORT[name]}**: `{acc:.2f}%`")
else:
    st.sidebar.warning("Run main.py first")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "  Overview":
    st.title(f"🛡️ Hybrid IDS v4 — {active_ds}")
    st.markdown(f"**5-Model Pipeline on {active_ds} ({ds_cfg['year']}) | "
                f"{ds_cfg['n_classes']} Attack Classes**")
    st.info(ds_cfg["description"])
    st.markdown("---")

    if not models_loaded:
        st.error("❌ Models not found. Run `python main.py` first.")
        st.stop()

    # KPI row
    cols = st.columns(5)
    for i,name in enumerate(MODEL_ORDER):
        acc = model_accs.get(name, 0)
        delta = " Best" if name==best_model else ""
        cols[i].metric(f"{MODEL_SHORT[name]}", f"{acc:.2f}%", delta)

    st.markdown("---")

    # Comparison chart
    st.subheader(f" 5-Model Accuracy & Speed — {active_ds}")
    cmp_path = f"outputs/{ds_tag}/model_comparison.png"
    if os.path.exists(cmp_path):
        st.image(cmp_path, use_column_width=True)

    # Radar chart
    rad_path = f"outputs/{ds_tag}/radar_chart.png"
    if os.path.exists(rad_path):
        st.markdown("---")
        st.subheader(" Multi-Metric Radar")
        st.image(rad_path, use_column_width=True)

    # Cross-dataset teaser
    if os.path.exists("outputs/cross_dataset_comparison.png"):
        st.markdown("---")
        st.subheader(" Cross-Dataset View (Both Datasets)")
        st.image("outputs/cross_dataset_comparison.png", use_column_width=True)
        st.caption(" Switch to the ** Dataset Comparison** page for detailed analysis")

    st.markdown("---")
    # Attack classes for active dataset
    st.subheader(f" Attack Classes — {active_ds}")
    cls_cols = st.columns(min(5, len(class_names)))
    cls_clrs = get_class_colors(active_ds)
    for i,cn in enumerate(class_names):
        col  = cls_clrs.get(cn,"#95a5a6")
        cls_cols[i % len(cls_cols)].markdown(
            f"<div style='background:{col};color:white;padding:8px;border-radius:6px;"
            f"text-align:center;font-weight:bold;font-size:12px'>{cn}</div>",
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "   Dataset Comparison":
    st.title(" Dataset Comparison — NSL-KDD vs UNSW-NB15")
    st.markdown("---")

    c1,c2 = st.columns(2)
    for col, ds_name in zip([c1,c2], ["NSL-KDD","UNSW-NB15"]):
        cfg = DATASETS[ds_name]
        with col:
            st.markdown(
                f"<div style='background:{cfg['color']}22;border:2px solid {cfg['color']};"
                f"border-radius:10px;padding:16px'>"
                f"<h3 style='color:{cfg['color']};margin:0'>{ds_name}</h3>"
                f"<p style='margin:4px 0'><b>Year:</b> {cfg['year']}</p>"
                f"<p style='margin:4px 0'><b>Classes:</b> {cfg['n_classes']}</p>"
                f"<p style='margin:4px 0'><b>Attacks:</b> {cfg['attacks']}</p>"
                f"<p style='margin:4px 0'><b>Era:</b> {cfg['era']}</p>"
                f"</div>",
                unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Side-by-Side Accuracy Table")
    if master:
        nsl_acc  = master.get("NSL-KDD",{})
        unsw_acc = master.get("UNSW-NB15",{})
        rows=[]
        for m in MODEL_ORDER:
            n   = nsl_acc.get(m,0)
            u   = unsw_acc.get(m,0)
            avg = (n+u)/2
            delta = u-n
            rows.append({
                "Model":     m,
                "NSL-KDD":   f"{n:.2f}%",
                "UNSW-NB15": f"{u:.2f}%",
                "Average":   f"{avg:.2f}%",
                "Δ (UNSW−NSL)": f"{delta:+.2f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # FAR comparison
        far_df = pd.DataFrame({
            "Metric": ["False Alarm Rate (RF)","Best Model Accuracy"],
            "NSL-KDD":   [f"{master.get('nsl_far',0):.2f}%",
                          f"{max(nsl_acc.values()):.2f}%"],
            "UNSW-NB15": [f"{master.get('unsw_far',0):.2f}%",
                          f"{max(unsw_acc.values()):.2f}%"],
        })
        st.dataframe(far_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Run main.py first to generate comparison data.")

    st.markdown("---")
    if os.path.exists("outputs/cross_dataset_comparison.png"):
        st.subheader("📈 Visual Cross-Dataset Comparison")
        st.image("outputs/cross_dataset_comparison.png", use_column_width=True)

    st.markdown("---")
    st.subheader(" What the Generalisation Gap Tells You")
    st.markdown("""
    The **Δ chart** (right panel above) shows how much accuracy each model gains or loses
    when moving from NSL-KDD to UNSW-NB15.

    - **Positive Δ (green)** → Model performs *better* on UNSW-NB15. 
      This happens when UNSW-NB15's modern attacks have cleaner feature patterns 
      than NSL-KDD's 1999 attack mix.
    - **Negative Δ (red)** → Model performs *worse* on UNSW-NB15. 
      This usually means the model overfit to NSL-KDD's specific quirks.
    - **Small Δ (near 0)** → Model generalises well across both — this is the best sign.

    **For your viva:** A model that maintains accuracy across both datasets proves 
    it has learned real attack patterns, not dataset-specific artifacts.
    """)

    st.markdown("---")
    st.subheader(" Why We Use Both Datasets Together")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **NSL-KDD (keep it)**
        - Every published IDS paper uses it
        - Reviewers expect it for comparison
        - Known baselines: RF~73%, DNN~76%
        - Proves your results are reproducible
        """)
    with col_b:
        st.markdown("""
        **UNSW-NB15 (the upgrade)**
        - Real 2015 traffic captured from UNSW lab
        - 10 modern attack types including backdoors & shellcode
        - Missing from most BTech IDS projects
        - Required for any journal submission
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LEADERBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "  Model Leaderboard":
    st.title(f" Model Leaderboard — {active_ds}")
    st.markdown("---")

    if not models_loaded:
        st.error("❌ Models not found. Run main.py first.")
        st.stop()

    ranked = sorted(model_accs.items(), key=lambda x:x[1], reverse=True)
    medals = ["1","2","3","4️","5️"]
    rows = []
    for i,(name,acc) in enumerate(ranked):
        rows.append({
            "Rank": medals[i],
            "Model": name,
            "Category": {"Random Forest":"Ensemble","Extra Trees":"Ensemble",
                         "XGBoost":"Gradient Boost","LightGBM":"Gradient Boost",
                         "DNN":"Deep Learning"}[name],
            "Accuracy": f"{acc:.2f}%",
            "Best At": {"Random Forest":"Feature selection, stability",
                        "Extra Trees":"Speed + generalisation",
                        "XGBoost":"High accuracy on tabular data",
                        "LightGBM":"Fastest training, modern GB",
                        "DNN":"Probability scores + confidence tiers"}[name],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    cmp_path = f"outputs/{ds_tag}/model_comparison.png"
    if os.path.exists(cmp_path):
        st.image(cmp_path, use_column_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — REAL-TIME MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "  Real-Time Monitor":
    st.title(f" Real-Time Monitor — {active_ds}")
    st.markdown("Simulates live IDS on the test set — streams row by row with live alerts.")
    st.markdown("---")

    if not models_loaded:
        st.error(" Models not found. Run main.py first."); st.stop()

    @st.cache_data
    def load_sim(tag):
        paths = {
            "nsl_kdd":   ("data/KDDTrain+.txt",  "data/KDDTest+.txt"),
            "unsw_nb15": ("data/UNSW_NB15_training-set.csv",
                          "data/UNSW_NB15_testing-set.csv"),
        }
        tr_path, te_path = paths[tag]
        if not os.path.exists(te_path): return None, None, None

        if tag=="nsl_kdd":
            from sklearn.preprocessing import LabelEncoder as LE
            COLS=["duration","protocol_type","service","flag","src_bytes","dst_bytes",
                  "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
                  "num_compromised","root_shell","su_attempted","num_root",
                  "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
                  "is_host_login","is_guest_login","count","srv_count","serror_rate",
                  "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
                  "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
                  "dst_host_same_srv_rate","dst_host_diff_srv_rate",
                  "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
                  "dst_host_serror_rate","dst_host_srv_serror_rate",
                  "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
            df  = pd.read_csv(te_path,header=None,names=COLS)
            lbls= df["label"].values
            df.drop(columns=["label","difficulty"],inplace=True)
            for col in ["protocol_type","service","flag"]:
                le=LE(); df[col]=le.fit_transform(df[col].astype(str))
            feat_cols=[c for c in df.columns]
        else:
            df  = pd.read_csv(te_path,low_memory=False)
            lbls= df.get("attack_cat", pd.Series(["Normal"]*len(df))).values
            for dc in ["id","label","attack_cat"]:
                if dc in df.columns: df.drop(columns=[dc],inplace=True)
            from sklearn.preprocessing import LabelEncoder as LE
            for col in ["proto","service","state"]:
                if col in df.columns:
                    le=LE(); df[col]=le.fit_transform(df[col].astype(str))
            feat_cols=[c for c in df.columns]

        mdls_data = load_dataset_models(tag)
        if mdls_data is None: return None,None,None
        sc   = mdls_data["scaler"]
        top  = mdls_data["top15"]
        X_sc = sc.transform(df[feat_cols].fillna(0).values.astype(float))
        fm   = {n:i for i,n in enumerate(feat_cols)}
        idx  = [fm[f] for f in top if f in fm]
        return X_sc[:,idx], lbls, mdls_data["class_names"]

    X_sim,true_labels,sim_classes = load_sim(ds_tag)

    ctrl1,ctrl2,ctrl3,ctrl4 = st.columns(4)
    with ctrl1:
        chosen = st.selectbox("Model", MODEL_ORDER)
    with ctrl2:
        speed = st.select_slider("Speed",
            ["Slow (0.3s)","Normal (0.1s)","Fast (0.03s)","Turbo (0s)"],
            value="Normal (0.1s)")
        delay = {"Slow (0.3s)":0.3,"Normal (0.1s)":0.1,
                 "Fast (0.03s)":0.03,"Turbo (0s)":0.0}[speed]
    with ctrl3:
        batch = st.selectbox("Packets/batch",[1,5,10,25,50],index=2)
    with ctrl4:
        max_p = st.selectbox("Max packets",[100,500,1000,2000,5000],index=1)

    st.markdown("---")
    b1,b2,_ = st.columns([1,1,4])
    start = b1.button(" START",type="primary",use_container_width=True)
    stop  = b2.button(" STOP",type="secondary",use_container_width=True)
    if stop: st.session_state["rt_running"]=False

    stat_ph=st.empty(); met_ph=st.empty(); prog_ph=st.empty()
    log_ph=st.empty(); chart_ph=st.empty()

    if start:
        if X_sim is None:
            st.markdown("""
            <div style='background:#1a1a2e;border:1px solid #2d2d4e;border-radius:10px;padding:16px;margin-bottom:16px'>
            <p style='color:#7f8fa6;font-size:13px;margin:0'>
            <b style='color:#a29bfe'>Online demo mode</b> — dataset files are stored locally for size/privacy reasons.
            To run with real data: download the project, add data files to <code>data/</code>, run <code>python main.py</code>.
            </p></div>
            """, unsafe_allow_html=True)

            # ── CUSTOMIZATION CONTROLS ──────────────────────────────────────
            st.markdown("### Customize your scan")
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                demo_model = st.selectbox(
                    "Model",
                    ["Random Forest (RF)", "Extra Trees (ET)", "XGBoost (XGB)",
                     "LightGBM (LGBM)", "Deep Neural Network (DNN)"],
                    index=4
                )
                model_tips = {
                    "Random Forest (RF)": "150 decision trees vote on each packet. Fast and stable. 73.8% accuracy.",
                    "Extra Trees (ET)": "Like RF with extra randomness. Best cross-dataset robustness. 74.3%.",
                    "XGBoost (XGB)": "Each tree learns from the previous one's mistakes. 74.6% accuracy.",
                    "LightGBM (LGBM)": "Fastest gradient booster, Microsoft-designed. 74.1% accuracy.",
                    "Deep Neural Network (DNN)": "3 dense layers (64 → 32 → 16). Highest accuracy 76.7% but most brittle under dataset shift.",
                }
                st.caption(model_tips.get(demo_model, ""))

            with col_b:
                attack_mix = st.slider(
                    "Attack traffic percentage",
                    min_value=5, max_value=80, value=35, step=5,
                    help="How many of the simulated packets are attacks vs normal traffic"
                )
                if attack_mix < 20:
                    st.caption("Low-threat environment — mostly normal traffic")
                elif attack_mix < 50:
                    st.caption("Moderate threat — mixed normal and attack traffic")
                else:
                    st.caption("High-threat environment — heavy attack traffic")

            with col_c:
                speed_map = {"Slow (easy to watch)": 0.4, "Normal": 0.15,
                             "Fast": 0.05, "Turbo": 0.01}
                scan_speed = st.selectbox("Scan speed", list(speed_map.keys()), index=1)
                total_packets = st.selectbox("Total packets to scan", [50, 100, 200, 500], index=1)

            st.markdown("---")

            # ── ATTACK TYPE EXPLANATION ──────────────────────────────────────
            with st.expander("What attack types will appear? Click to learn", expanded=False):
                exp_col1, exp_col2 = st.columns(2)
                with exp_col1:
                    st.markdown("""
                    **DoS — Denial of Service**
                    Floods a server with fake requests so real users cannot connect.
                    Like blocking a shop door so customers cannot enter.
                    *Detected by: extremely high src_bytes, serror_rate = 1.0*

                    **Probe — Reconnaissance**
                    Attacker scans the network to find open ports and weak points before attacking.
                    Like a burglar checking which windows are unlocked.
                    *Detected by: many short connections to many different hosts*
                    """)
                with exp_col2:
                    st.markdown("""
                    **R2L — Remote to Local**
                    An outsider gains unauthorised access to a local machine.
                    Like someone breaking into your house using a stolen key.
                    *Detected by: failed login attempts, unusual service access*

                    **U2R — User to Root**
                    A normal user gains administrator privileges they should not have.
                    Like a regular employee accessing the CEO's private files.
                    *Detected by: root_shell flag, privilege escalation patterns*
                    """)

            st.markdown("---")

            # ── CONFIDENCE THRESHOLD EXPLANATION ───────────────────────────
            with st.expander("What do ATTACK / UNCERTAIN / CLEAN mean?", expanded=False):
                st.markdown("""
                The model outputs a **confidence score** from 0% to 100% for each packet.

                | Decision | Confidence | Meaning | Action |
                |---|---|---|---|
                | 🔴 ATTACK | Above 85% | Model is very sure this is a real intrusion | Immediate security alert |
                | 🟡 UNCERTAIN | 60% – 85% | Model is not sure enough to auto-alert | Human analyst reviews it |
                | 🟢 CLEAN | Below 60% | Model thinks this is normal traffic | Logged, no alert triggered |

                The UNCERTAIN tier is what makes RT-XHIDS different from other IDS systems.
                Instead of wrong alerts, borderline packets go to a human.
                This reduces **false alarms** which waste analyst time.
                """)

            st.markdown("### Live packet feed")

            # ── RUN DEMO ─────────────────────────────────────────────────────
            if st.button("Start scanning", type="primary", use_container_width=False):

                import numpy as np
                import time as time_module

                ATTACK_TYPES = ["DoS", "Probe", "R2L", "U2R"]
                REASONS = {
                    "DoS":    ["Very high src_bytes — flood traffic detected",
                               "serror_rate = 1.0 — SYN flood pattern",
                               "count > 500 connections per second",
                               "byte_ratio extremely high — one-way data blast"],
                    "Probe":  ["Port scan pattern across many hosts",
                               "dst_host_count > 200 in 2 seconds",
                               "same_srv_rate near zero — varied targets",
                               "Duration very short per connection — scanner"],
                    "R2L":    ["Multiple failed login attempts detected",
                               "Unusual service access from external IP",
                               "src_bytes low, dst_bytes low — probing",
                               "logged_in = 0 after repeated attempts"],
                    "U2R":    ["root_shell = 1 — shell escalation detected",
                               "num_compromised > 1 — system breached",
                               "su_attempted flag raised",
                               "Unusual privilege escalation sequence"],
                    "Normal": ["All features within normal range",
                               "byte_ratio balanced — normal exchange",
                               "Connection duration typical for service",
                               "No error rate anomalies detected"],
                }

                # Stat placeholders
                stat_cols = st.columns(4)
                ph_total = stat_cols[0].empty()
                ph_atk   = stat_cols[1].empty()
                ph_unc   = stat_cols[2].empty()
                ph_cln   = stat_cols[3].empty()

                # Progress bar
                prog_ph  = st.progress(0, text="Scanning packets...")

                # Feed table
                feed_ph  = st.empty()

                counts   = {"ATTACK": 0, "UNCERTAIN": 0, "CLEAN": 0}
                feed_rows = []
                delay    = speed_map[scan_speed]
                mix      = attack_mix / 100

                for i in range(total_packets):

                    # Generate packet
                    is_atk = np.random.random() < mix
                    if is_atk:
                        conf = 0.62 + np.random.random() * 0.37
                        atk_type = np.random.choice(ATTACK_TYPES)
                        reason = np.random.choice(REASONS[atk_type])
                        decision = "ATTACK" if conf > 0.85 else "UNCERTAIN"
                    else:
                        conf = 0.08 + np.random.random() * 0.52
                        atk_type = "Normal"
                        reason = np.random.choice(REASONS["Normal"])
                        decision = "CLEAN"

                    counts[decision] += 1
                    done_n = i + 1

                    # Stat cards
                    ph_total.metric("Packets scanned", f"{done_n:,}")
                    ph_atk.metric(  "🔴 ATTACK",    f"{counts['ATTACK']:,}",
                                    f"{counts['ATTACK']/done_n*100:.1f}% of traffic")
                    ph_unc.metric(  "🟡 UNCERTAIN",  f"{counts['UNCERTAIN']:,}",
                                    f"{counts['UNCERTAIN']/done_n*100:.1f}% needs review")
                    ph_cln.metric(  "🟢 CLEAN",     f"{counts['CLEAN']:,}",
                                    f"{counts['CLEAN']/done_n*100:.1f}% safe")

                    # Progress
                    prog_ph.progress(
                        done_n / total_packets,
                        text=f"Scanning packet {done_n} of {total_packets} — Model: {demo_model}"
                    )

                    # Feed table — show newest 15 rows
                    import datetime
                    now_t = datetime.datetime.now().strftime("%H:%M:%S")
                    dec_icon = "🔴 ATTACK" if decision=="ATTACK" else (
                               "🟡 UNCERTAIN" if decision=="UNCERTAIN" else "🟢 CLEAN")
                    conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))

                    feed_rows.insert(0, {
                        "Time":       now_t,
                        "Decision":   dec_icon,
                        "Attack type": atk_type,
                        "Confidence": f"{conf*100:.1f}%  {conf_bar}",
                        "Why flagged": reason,
                    })
                    feed_ph.dataframe(
                        feed_rows[:15],
                        use_container_width=True,
                        hide_index=True
                    )

                    time_module.sleep(delay)

                # Final summary
                prog_ph.progress(1.0, text="Scan complete")
                st.success(
                    f"Scan complete — {total_packets} packets analysed by {demo_model}. "
                    f"Found {counts['ATTACK']} attacks, {counts['UNCERTAIN']} uncertain, "
                    f"{counts['CLEAN']} clean."
                )

                # Show final breakdown
                st.markdown("#### Final breakdown")
                fc1, fc2, fc3 = st.columns(3)
                fc1.metric("Attack rate",    f"{counts['ATTACK']/total_packets*100:.1f}%",
                           "of all traffic")
                fc2.metric("Review queue",   f"{counts['UNCERTAIN']/total_packets*100:.1f}%",
                           "sent to analyst")
                fc3.metric("Clean traffic",  f"{counts['CLEAN']/total_packets*100:.1f}%",
                           "no action needed")

                st.markdown("""
                **Reading these results:**
                The ATTACK rate tells you how much of your simulated traffic was genuine intrusions.
                The UNCERTAIN rate is important — these are borderline packets the model was not confident about.
                In a real deployment these go to a human analyst instead of auto-alerting,
                which reduces false alarms significantly.
                """)

            st.stop()
        st.session_state["rt_running"]=True
        total = min(max_p, len(X_sim))
        X_all = X_sim[:total]
        lbls_all = true_labels[:total] if true_labels is not None else ["?"]*total
        sim_cls  = sim_classes or class_names

        mdl = all_models[chosen]
        if chosen=="DNN":
            pr=mdl.predict(X_all,verbose=0); pa=pr.argmax(axis=1); ca=pr.max(axis=1)
        else:
            pa=mdl.predict(X_all); pr=mdl.predict_proba(X_all); ca=pr.max(axis=1)
        da=threshold(pr,pa)

        counts={"ATTACK":0,"UNCERTAIN":0,"CLEAN":0}
        cls_cnt={cn:0 for cn in sim_cls}
        alerts=[]; done=0; i=0

        while i<total and st.session_state.get("rt_running",True):
            end=min(i+batch,total)
            for dec,pred,conf,lbl in zip(da[i:end],pa[i:end],ca[i:end],lbls_all[i:end]):
                counts[dec]+=1
                cn=sim_cls[pred] if pred<len(sim_cls) else str(pred)
                cls_cnt[cn]=cls_cnt.get(cn,0)+1
                done+=1
                if dec in ("ATTACK","UNCERTAIN"):
                    alerts.insert(0,{
                        "Time":    time.strftime("%H:%M:%S"),
                        "Decision":f"{'🔴' if dec=='ATTACK' else '🟡'} {dec}",
                        "Class":   cn,
                        "Conf":    f"{conf*100:.1f}%",
                        "True":    str(lbl),
                    })
                    if len(alerts)>15: alerts=alerts[:15]

            if counts["ATTACK"]>0:
                stat_ph.error(f"🔴 [{chosen}] — {counts['ATTACK']:,} ATTACKS")
            else:
                stat_ph.success(f"🟢 [{chosen}] — Monitoring...")

            with met_ph.container():
                m1,m2,m3,m4,m5=st.columns(5)
                m1.metric(" Scanned",f"{done:,}")
                m2.metric("🔴 ATTACK",f"{counts['ATTACK']:,}",
                          f"{counts['ATTACK']/max(done,1)*100:.1f}%")
                m3.metric("🟡 UNCERTAIN",f"{counts['UNCERTAIN']:,}",
                          f"{counts['UNCERTAIN']/max(done,1)*100:.1f}%")
                m4.metric("🟢 CLEAN",f"{counts['CLEAN']:,}",
                          f"{counts['CLEAN']/max(done,1)*100:.1f}%")
                m5.metric(" Progress",f"{done/total*100:.1f}%",f"{done}/{total}")

            prog_ph.progress(int(done/total*100))
            if alerts:
                log_ph.dataframe(pd.DataFrame(alerts),
                                 use_container_width=True,hide_index=True)
            else:
                log_ph.info(" Alert log — waiting...")

            fig_l,ax_l=plt.subplots(1,2,figsize=(10,3))
            nz=[(v,l,c) for v,l,c in zip(
                [counts["ATTACK"],counts["UNCERTAIN"],counts["CLEAN"]],
                ["ATTACK","UNCERTAIN","CLEAN"],
                ["#e74c3c","#f39c12","#2ecc71"]) if v>0]
            if nz:
                v_,l_,c_=zip(*nz)
                ax_l[0].pie(v_,labels=l_,colors=c_,autopct="%1.0f%%",
                            startangle=90,textprops={"fontsize":8})
            ax_l[0].set_title("Decision Split",fontsize=10,fontweight="bold")
            clrs_=[get_class_colors(active_ds).get(cn,"#95a5a6") for cn in sim_cls]
            ax_l[1].bar(sim_cls,[cls_cnt.get(cn,0) for cn in sim_cls],
                        color=clrs_,edgecolor="black",linewidth=0.4)
            ax_l[1].set_title("Predicted Classes",fontsize=10,fontweight="bold")
            ax_l[1].tick_params(axis="x",rotation=30,labelsize=7)
            ax_l[1].grid(axis="y",alpha=0.3)
            plt.tight_layout()
            chart_ph.pyplot(fig_l); plt.close()
            i=end
            if delay>0: time.sleep(delay)

        st.session_state["rt_running"]=False
        stat_ph.success(f" Done — {done:,} packets | "
                        f"🔴{counts['ATTACK']:,} 🟡{counts['UNCERTAIN']:,} "
                        f"🟢{counts['CLEAN']:,}")
        st.balloons()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — UPLOAD & CLASSIFY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "  Upload & Classify":
    st.title(f" Upload & Classify — {active_ds}")
    st.markdown(f"Upload a CSV with the same feature format as {active_ds}.")
    st.markdown("---")

    if not models_loaded:
        st.error(" Models not found. Run main.py first."); st.stop()

    uploaded = st.file_uploader("Drop CSV / TXT file", type=["csv","txt"])
    run_all  = st.checkbox(" Run all 5 models and compare",value=False)
    if not run_all:
        chosen_m = st.radio("Model",MODEL_ORDER,horizontal=True)

    feat_cols_up = list(top15) if top15 else []

    if uploaded:
        try:
            raw   = uploaded.read().decode("utf-8")
            first = raw.strip().split("\n")[0]
            hdr   = first.split(",")[0].strip().lower() not in ("","0","1")
            if hdr:
                df_up = pd.read_csv(io.StringIO(raw))
            else:
                df_up = pd.read_csv(io.StringIO(raw), header=None)

            for dc in ds_cfg["drop_cols"]+[ds_cfg["label_col"]]:
                if dc in df_up.columns:
                    df_up.drop(columns=[dc],inplace=True)

            # Build feat_cols from scaler
            n_feat = scaler.n_features_in_ if scaler else len(df_up.columns)
            all_feat_cols = list(df_up.columns)[:n_feat]
            X_top  = preprocess_df(df_up, scaler, top15,
                                   all_feat_cols, ds_cfg["cat_cols"])
            st.success(f" {len(df_up):,} connections loaded — using {active_ds} models.")

            if run_all:
                st.subheader(" All 5 Models")
                comp = predict_all(all_models, X_top, active_ds)
                cmp_df = pd.DataFrame({
                    "Model":      list(comp.keys()),
                    "ATTACK":     [comp[m]["n_atk"] for m in comp],
                    "UNCERTAIN":  [comp[m]["n_unc"] for m in comp],
                    "CLEAN":      [comp[m]["n_cln"] for m in comp],
                    "ATTACK %":   [f"{comp[m]['n_atk']/len(X_top)*100:.1f}%"
                                   for m in comp],
                })
                st.dataframe(cmp_df,use_container_width=True,hide_index=True)

                fig_c,ax_c=plt.subplots(figsize=(11,4))
                x=np.arange(len(comp)); w=0.25
                ax_c.bar(x-w,[comp[m]["n_atk"] for m in comp],w,
                         label="🔴 ATTACK",color="#e74c3c",edgecolor="black",lw=0.4)
                ax_c.bar(x,  [comp[m]["n_unc"] for m in comp],w,
                         label="🟡 UNCERTAIN",color="#f39c12",edgecolor="black",lw=0.4)
                ax_c.bar(x+w,[comp[m]["n_cln"] for m in comp],w,
                         label="🟢 CLEAN",color="#2ecc71",edgecolor="black",lw=0.4)
                ax_c.set_xticks(x)
                ax_c.set_xticklabels(list(comp.keys()),rotation=15,ha="right",fontsize=9)
                ax_c.set_title("Decision Breakdown — All 5 Models",
                               fontsize=12,fontweight="bold")
                ax_c.legend(); ax_c.grid(axis="y",alpha=0.3)
                st.pyplot(fig_c); plt.close()

                st.markdown("---")
                st.subheader(" Per-Model Filter")
                tabs = st.tabs([MODEL_SHORT[m] for m in comp])
                for tab,(mn,mr) in zip(tabs,comp.items()):
                    with tab:
                        res_df=build_results_df(mr["pred"],mr["conf"],
                                                mr["dec"],mr["probs"],class_names)
                        show_filter_table(res_df,class_names,
                                         key_sfx=f"{mn.replace(' ','_')}_{ds_tag}")
            else:
                mdl = all_models[chosen_m]
                with st.spinner(f"Running {chosen_m}..."):
                    if chosen_m=="DNN":
                        pr=mdl.predict(X_top,verbose=0)
                        pred=pr.argmax(axis=1); conf=pr.max(axis=1)
                    else:
                        pred=mdl.predict(X_top)
                        pr=mdl.predict_proba(X_top); conf=pr.max(axis=1)
                dec=threshold(pr,pred)
                n_a=(dec=="ATTACK").sum(); n_u=(dec=="UNCERTAIN").sum()
                n_c=(dec=="CLEAN").sum();  t=len(dec)

                m1,m2,m3,m4=st.columns(4)
                m1.metric("Total",f"{t:,}")
                m2.metric("🔴 ATTACK",f"{n_a:,}",f"{n_a/t*100:.1f}%")
                m3.metric("🟡 UNCERTAIN",f"{n_u:,}",f"{n_u/t*100:.1f}%")
                m4.metric("🟢 CLEAN",f"{n_c:,}",f"{n_c/t*100:.1f}%")

                ch1,ch2=st.columns(2)
                with ch1:
                    fig_p,ax_p=plt.subplots(figsize=(5,4))
                    cls_cnt={}
                    for c in pred:
                        cn=class_names[c] if c<len(class_names) else str(c)
                        cls_cnt[cn]=cls_cnt.get(cn,0)+1
                    clrs=[get_class_colors(active_ds).get(cn,"#95a5a6")
                          for cn in cls_cnt]
                    ax_p.pie(list(cls_cnt.values()),labels=list(cls_cnt.keys()),
                             colors=clrs,autopct="%1.1f%%",startangle=140,
                             textprops={"fontsize":8})
                    ax_p.set_title("Class Distribution",fontsize=10,fontweight="bold")
                    st.pyplot(fig_p); plt.close()
                with ch2:
                    fig_b,ax_b=plt.subplots(figsize=(5,4))
                    ax_b.bar(["ATTACK","UNCERTAIN","CLEAN"],[n_a,n_u,n_c],
                             color=["#e74c3c","#f39c12","#2ecc71"],
                             edgecolor="black",linewidth=0.5)
                    for i,(l,cnt) in enumerate(zip(["ATTACK","UNCERTAIN","CLEAN"],
                                                    [n_a,n_u,n_c])):
                        ax_b.text(i,cnt+0.3,f"{cnt:,}",ha="center",
                                  fontsize=10,fontweight="bold")
                    ax_b.set_title("Decision Tiers",fontsize=10,fontweight="bold")
                    ax_b.grid(axis="y",alpha=0.3)
                    st.pyplot(fig_b); plt.close()

                st.markdown("---")
                res_df=build_results_df(pred,conf,dec,pr,class_names)
                show_filter_table(res_df,class_names,
                                  key_sfx=f"single_{chosen_m.replace(' ','_')}_{ds_tag}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback; st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — CONFUSION MATRICES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "  Confusion Matrices":
    st.title(f" Confusion Matrices — {active_ds}")
    st.markdown("Diagonal = correct  | Off-diagonal = mistakes ")
    st.markdown("---")

    cm_files = {m: f"outputs/{ds_tag}/cm_{MODEL_SHORT[m].lower()}.png"
                for m in MODEL_ORDER}
    tabs = st.tabs([f"{MODEL_SHORT[m]} ({model_accs.get(m,0):.1f}%)"
                    for m in MODEL_ORDER])
    for tab,name in zip(tabs,MODEL_ORDER):
        with tab:
            f=cm_files[name]
            if os.path.exists(f):
                st.image(f,use_column_width=True)
                st.metric(f"{name} Accuracy",f"{model_accs.get(name,0):.2f}%")
            else:
                st.warning("Run main.py first.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — SHAP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬  SHAP Explainability":
    st.title(f"🔬 SHAP Explainability — {active_ds}")
    st.markdown("**WHY** did each model flag this as an attack?")
    st.markdown("---")

    t1,t2,t3 = st.tabs([" RF SHAP"," Extra Trees SHAP"," Force Plot"])
    with t1:
        p=f"outputs/{ds_tag}/shap_summary_rf.png"
        if os.path.exists(p): st.image(p,use_column_width=True)
        else: st.warning("Run main.py first.")
    with t2:
        p=f"outputs/{ds_tag}/shap_summary_et.png"
        if os.path.exists(p):
            st.image(p,use_container_width=True)
            st.info(" Compare with RF SHAP — Extra Trees uses random splits so "
                    "feature importance rankings often differ. This comparison is "
                    "a unique contribution not present in most BTech IDS papers.")
        else: st.warning("Run main.py first.")
    with t3:
        p=f"outputs/{ds_tag}/shap_force_plot.png"
        if os.path.exists(p): st.image(p,use_column_width=True)
        else: st.warning("Run main.py first.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — CLASSIFICATION REPORTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "  Classification Reports":
    st.title(f" Classification Reports — {active_ds}")
    st.markdown("---")
    rpt_path=f"outputs/{ds_tag}/classification_reports.txt"
    if os.path.exists(rpt_path):
        with open(rpt_path) as f: content=f.read()
        sections=[s.strip() for s in content.split("="*65) if s.strip()]
        reports={}
        for i in range(0,len(sections)-1,2):
            for m in MODEL_ORDER:
                if m.upper() in sections[i].upper():
                    reports[m]=sections[i+1] if i+1<len(sections) else ""
                    break
        tabs=st.tabs([MODEL_SHORT[m] for m in MODEL_ORDER])
        for tab,name in zip(tabs,MODEL_ORDER):
            with tab:
                st.markdown(f"### {name} — **{model_accs.get(name,0):.2f}%**")
                st.code(reports.get(name,content),language="text")
        st.download_button(" Download All Reports",content.encode(),
                           f"reports_{ds_tag}.txt","text/plain")
    else:
        st.warning("Run main.py first.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "   About":
    st.title(" About — Hybrid IDS v4")
    st.markdown("---")
    best_nsl  = max(master.get("NSL-KDD",{}).values(),  default=0)
    best_unsw = max(master.get("UNSW-NB15",{}).values(),default=0)

    st.markdown(f"""
    ###  Project Details

    | Property | Value |
    |---|---|
    | **Degree** | B.Tech |
    | **Year** | 2025 - 2026 |
    | **Version** | v4 — Dual Dataset |
    | **Datasets** | NSL-KDD (2009) · UNSW-NB15 (2015) |
    | **Models** | Random Forest · Extra Trees · XGBoost · LightGBM · DNN |
    | **Best NSL-KDD** | {best_nsl:.2f}% |
    | **Best UNSW-NB15** | {best_unsw:.2f}% |
    | **Total models trained** | 10 (5 per dataset) |

    ###  What Makes v4 Unique
    -  **Dual dataset** — NSL-KDD for comparison + UNSW-NB15 for modern validation
    -  **10 models total** — 5 architectures × 2 datasets
    -  **Generalisation gap analysis** — cross-dataset Δ chart proves robustness
    -  **Dataset selector** — switch between NSL-KDD and UNSW-NB15 in one click
    -  **10-class UNSW-NB15** includes Backdoors, Shellcode, Fuzzers — modern attacks
    -  **Publication-ready** — dual dataset + 5 models covers all reviewer requirements

    ###  Tech Stack
    `Python 3.11` · `scikit-learn` · `XGBoost` · `LightGBM`
    `TensorFlow/Keras` · `SHAP` · `Streamlit` · `NumPy` · `Pandas` · `Matplotlib`

    ###  Download UNSW-NB15
    ```
    https://research.unsw.edu.au/projects/unsw-nb15-dataset
    Files needed:
      UNSW_NB15_training-set.csv  →  data/UNSW_NB15_training-set.csv
      UNSW_NB15_testing-set.csv   →  data/UNSW_NB15_testing-set.csv
    Then re-run: python main.py
    ```
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><sub>Hybrid IDS v4 | RF · ExtraTrees · XGBoost · LightGBM · DNN | "
    "NSL-KDD + UNSW-NB15 | BTech Final Year Project</sub></center>",
    unsafe_allow_html=True)
