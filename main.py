"""
Hybrid Intrusion Detection System (IDS) — v4
Dual-Dataset: NSL-KDD (2009) + UNSW-NB15 (2015)
5-Model Pipeline: Random Forest | Extra Trees | XGBoost | LightGBM | DNN
Author: BTech Final Year Project
"""

# ─── Suppress warnings ────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
import os, json, time, pickle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy  as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing  import LabelEncoder, MinMaxScaler
from sklearn.ensemble       import RandomForestClassifier, ExtraTreesClassifier
from sklearn.utils          import class_weight as cw_util
from sklearn.metrics        import (accuracy_score, confusion_matrix,
                                    classification_report,
                                    precision_score, recall_score, f1_score)
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.models      import Sequential
from tensorflow.keras.layers      import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks   import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers   import Adam

MODEL_ORDER  = ["Random Forest","Extra Trees","XGBoost","LightGBM","DNN"]
MODEL_COLORS = ["#3498db","#27ae60","#e67e22","#8e44ad","#e74c3c"]

# ══════════════════════════════════════════════════════════════════════════════
#  DATASET DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── NSL-KDD (1999 / 2009) ────────────────────────────────────────────────────
NSL_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]
NSL_CAT  = ["protocol_type","service","flag"]
NSL_CLS  = ["Normal","DoS","Probe","R2L","U2R"]
_DOS     = {"back","land","neptune","pod","smurf","teardrop","apache2",
            "udpstorm","processtable","mailbomb"}
_PROBE   = {"ipsweep","nmap","portsweep","satan","mscan","saint"}
_R2L     = {"ftp_write","guess_passwd","imap","multihop","phf","spy",
            "warezclient","warezmaster","sendmail","named","snmpgetattack",
            "snmpguess","xlock","xsnoop","httptunnel"}
_U2R     = {"buffer_overflow","loadmodule","perl","rootkit",
            "sqlattack","xterm","ps"}

def nsl_map(l):
    l = str(l).strip().lower()
    if l=="normal": return 0
    if l in _DOS:   return 1
    if l in _PROBE: return 2
    if l in _R2L:   return 3
    if l in _U2R:   return 4
    return 1

# ── UNSW-NB15 (2015) ──────────────────────────────────────────────────────────
# Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset
# Files: UNSW_NB15_training-set.csv  +  UNSW_NB15_testing-set.csv
UNSW_CLS  = ["Normal","Exploits","Reconnaissance","DoS","Generic",
             "Shellcode","Fuzzers","Worms","Backdoors","Analysis"]
UNSW_CAT  = ["proto","service","state"]
_UMAP     = {c.lower():i for i,c in enumerate(UNSW_CLS)}

def unsw_map(l):
    return _UMAP.get(str(l).strip().lower(), 0)

# ── Synthetic fallbacks ────────────────────────────────────────────────────────
def _nsl_synth(n):
    np.random.seed(42)
    data = {c: np.random.rand(n)
            for c in NSL_COLUMNS
            if c not in ("protocol_type","service","flag","label","difficulty")}
    data["protocol_type"] = np.random.choice(["tcp","udp","icmp"],n)
    data["service"]       = np.random.choice(
        ["http","ftp","smtp","ssh","dns","pop_3","telnet","eco_i","other"],n)
    data["flag"]          = np.random.choice(
        ["SF","S0","REJ","RSTO","SH","RSTR","S1","S2","OTH"],n)
    data["label"]         = np.random.choice(
        ["normal","neptune","satan","portsweep","smurf",
         "warezclient","guess_passwd","imap"],
        n, p=[0.5,0.15,0.1,0.1,0.05,0.04,0.03,0.03])
    data["difficulty"]    = np.random.randint(1,21,n)
    return pd.DataFrame(data, columns=NSL_COLUMNS)

def _unsw_synth(n):
    np.random.seed(99)
    num = ["dur","spkts","dpkts","sbytes","dbytes","rate","sttl","dttl",
           "sload","dload","sloss","dloss","sinpkt","dinpkt","sjit","djit",
           "swin","stcpb","dtcpb","dwin","tcprtt","synack","ackdat",
           "smean","dmean","trans_depth","response_body_len","ct_srv_src",
           "ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm","ct_dst_sport_ltm",
           "ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd","ct_flw_http_mthd",
           "ct_src_ltm","ct_srv_dst","is_sm_ips_ports"]
    data = {c: np.random.rand(n) for c in num}
    data["proto"]      = np.random.choice(["tcp","udp","icmp","arp","ospf"],n)
    data["service"]    = np.random.choice(["http","ftp","dns","smtp","ssh","-"],n)
    data["state"]      = np.random.choice(["FIN","INT","CON","REQ","RST"],n)
    data["attack_cat"] = np.random.choice(
        UNSW_CLS, n,
        p=[0.50,0.10,0.08,0.08,0.07,0.05,0.05,0.02,0.03,0.02])
    data["label"] = [0 if v=="Normal" else 1 for v in data["attack_cat"]]
    return pd.DataFrame(data)

# ══════════════════════════════════════════════════════════════════════════════
#  CORE TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(ds_name, train_df, test_df,
                 feat_cols, cat_cols, class_names):
    tag      = ds_name.lower().replace("-","_")
    n_cls    = len(class_names)
    os.makedirs(f"models/{tag}",  exist_ok=True)
    os.makedirs(f"outputs/{tag}", exist_ok=True)

    print(f"\n{'═'*65}")
    print(f"  PIPELINE: {ds_name}  ({n_cls} classes)")
    print(f"{'═'*65}")

    # Encode
    print("  [1] Encoding...")
    for col in cat_cols:
        if col not in train_df.columns: continue
        le = LabelEncoder()
        le.fit(pd.concat([train_df[col],test_df[col]]).astype(str))
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col]  = le.transform(test_df[col].astype(str))
# Scale
    print("  [2] Scaling...")
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(train_df[feat_cols].fillna(0).values)
    X_test  = scaler.transform(test_df[feat_cols].fillna(0).values)
    y_train = train_df["multi_label"].values
    y_test  = test_df["multi_label"].values
    print(f"    X_train:{X_train.shape}  X_test:{X_test.shape}")

    # RF feature selection — top 15
    print("  [3] Feature selection (top-15)...")
    rf_fs  = RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
    rf_fs.fit(X_train, y_train)
    top_idx  = np.argsort(rf_fs.feature_importances_)[::-1][:15]
    top_feat = [feat_cols[i] for i in top_idx] 
    Xtr15    = X_train[:, top_idx]
    Xte15    = X_test[:,  top_idx]
    print(f"    Top-5: {', '.join(top_feat[:5])} ...")

    # ── SMOTE CLASS BALANCING ─────────────────────────────
    from imblearn.over_sampling import SMOTE
    from collections import Counter

    print("  [4] Balancing classes with SMOTE...")
    print(f"      Before: {Counter(y_train)}")

    smote = SMOTE(random_state=42, k_neighbors=3)
    Xtr15, y_train = smote.fit_resample(Xtr15, y_train)

    print(f"      After:  {Counter(y_train)}")
    print(f"      New training size: {len(Xtr15):,}")
    # Class weights for DNN
    cls_u   = np.unique(y_train)
    cw_vals = cw_util.compute_class_weight("balanced",classes=cls_u,y=y_train)
    cw_dict = dict(zip(cls_u, cw_vals))

    # Train 5 models
    print("  [4] Training 5 models...")
    print(f"  {'─'*55}")
    res   = {}
    preds = {}

    def fit_eval(name, clf):
        t0   = time.time()
        clf.fit(Xtr15, y_train)
        pred = clf.predict(Xte15)
        acc  = accuracy_score(y_test, pred)
        el   = time.time()-t0
        print(f"    {name:<20} Acc:{acc*100:.2f}%  Time:{el:.1f}s")
        res[name]   = {"acc":round(acc*100,2),"time":round(el,1),"model":clf}
        preds[name] = pred

    fit_eval("Random Forest",
        RandomForestClassifier(n_estimators=150,random_state=42,n_jobs=-1))
    fit_eval("Extra Trees",
        ExtraTreesClassifier(n_estimators=150,random_state=42,n_jobs=-1))
    fit_eval("XGBoost",
        XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,
                      subsample=0.8,colsample_bytree=0.8,
                      use_label_encoder=False,eval_metric="mlogloss",
                      random_state=42,n_jobs=-1,verbosity=0))
    fit_eval("LightGBM",
        LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,
                       subsample=0.8,colsample_bytree=0.8,
                       random_state=42,n_jobs=-1,verbose=-1))

    # DNN
    print("    Training DNN (L2 + class weights + LR scheduler)...")
    reg = l2(1e-4)
    dnn = Sequential([
        Dense(512,activation="relu",kernel_regularizer=reg,input_shape=(15,)),
        BatchNormalization(),Dropout(0.3),
        Dense(256,activation="relu",kernel_regularizer=reg),
        BatchNormalization(),Dropout(0.25),
        Dense(128,activation="relu",kernel_regularizer=reg),
        BatchNormalization(),Dropout(0.2),
        Dense(64, activation="relu",kernel_regularizer=reg),
        BatchNormalization(),Dropout(0.15),
        Dense(32, activation="relu"),
        Dense(n_cls, activation="softmax"),
    ])
    dnn.compile(optimizer=Adam(1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    t0 = time.time()
    dnn.fit(Xtr15, y_train, validation_split=0.1,
            epochs=80, batch_size=128, class_weight=cw_dict,
            callbacks=[
                EarlyStopping(monitor="val_loss",patience=8,
                              restore_best_weights=True,verbose=0),
                ReduceLROnPlateau(monitor="val_loss",factor=0.5,
                                  patience=3,min_lr=1e-6,verbose=0)
            ], verbose=1)
    _,dnn_acc  = dnn.evaluate(Xte15,y_test,verbose=0)
    dnn_probs  = dnn.predict(Xte15,verbose=0)
    dnn_pred   = dnn_probs.argmax(axis=1)
    dnn_time   = time.time()-t0
    print(f"    {'DNN':<20} Acc:{dnn_acc*100:.2f}%  Time:{dnn_time:.1f}s")
    res["DNN"]   = {"acc":round(dnn_acc*100,2),"time":round(dnn_time,1),"model":dnn}
    preds["DNN"] = dnn_pred

    # Confidence thresholding
    mc   = dnn_probs.max(axis=1)
    decs = np.where((mc>=0.85)&(dnn_pred!=0),"ATTACK",
           np.where(mc>=0.60,"UNCERTAIN","CLEAN"))
    n_a  = (decs=="ATTACK").sum()
    n_u  = (decs=="UNCERTAIN").sum()
    n_c  = (decs=="CLEAN").sum()
    print(f"\n    DNN → ATTACK:{n_a:,}  UNCERTAIN:{n_u:,}  CLEAN:{n_c:,}")

    # SHAP
    print("  [5] SHAP explanations...")
    N_SH = min(500, Xte15.shape[0])
    for mname,outf in [("Random Forest",f"outputs/{tag}/shap_summary_rf.png"),
                        ("Extra Trees",  f"outputs/{tag}/shap_summary_et.png")]:
        exp  = shap.TreeExplainer(res[mname]["model"])
        svs  = exp.shap_values(Xte15[:N_SH])
        sv_p = svs[1] if isinstance(svs,list) else svs
        if isinstance(sv_p,np.ndarray) and sv_p.ndim==3: sv_p=sv_p[:,:,1]
        plt.figure()
        shap.summary_plot(sv_p,Xte15[:N_SH],
                          feature_names=top_feat,show=False,plot_size=(12,7))
        plt.tight_layout()
        plt.savefig(outf,dpi=150,bbox_inches="tight"); plt.close()
        print(f"    {mname} SHAP → {outf}")

    # Force plot
    exp_rf = shap.TreeExplainer(res["Random Forest"]["model"])
    svs_rf = exp_rf.shap_values(Xte15[:N_SH])
    ai     = np.where(y_test[:N_SH]!=0)[0]
    si     = ai[0] if len(ai)>0 else 0
    ev     = exp_rf.expected_value
    if isinstance(svs_rf,list):
        sv2=svs_rf[1][si]; ev_val=ev[1] if isinstance(ev,(list,np.ndarray)) else ev
    elif isinstance(svs_rf,np.ndarray) and svs_rf.ndim==3:
        sv2=svs_rf[si,:,1]; ev_val=ev[1] if isinstance(ev,(list,np.ndarray)) else ev
    else:
        sv2=svs_rf[si]; ev_val=ev if np.isscalar(ev) else ev[0]
    shap.force_plot(ev_val,sv2,Xte15[si],feature_names=top_feat,
                    matplotlib=True,show=False)
    fp=f"outputs/{tag}/shap_force_plot.png"
    plt.savefig(fp,dpi=150,bbox_inches="tight"); plt.close()
    print(f"    Force plot → {fp}")

    # Save models
    print(f"  [6] Saving to models/{tag}/...")
    for mname,fname in [("Random Forest","rf_model.pkl"),
                         ("Extra Trees",  "et_model.pkl"),
                         ("XGBoost",      "xgb_model.pkl"),
                         ("LightGBM",     "lgbm_model.pkl")]:
        with open(f"models/{tag}/{fname}","wb") as f:
            pickle.dump(res[mname]["model"],f)
    dnn.save(f"models/{tag}/dnn_model.h5")
    with open(f"models/{tag}/scaler.pkl","wb") as f: pickle.dump(scaler,f)
    np.save(f"models/{tag}/top15_features.npy",np.array(top_feat))
    with open(f"models/{tag}/class_names.json","w") as f: json.dump(class_names,f)
    cmp = {n:{"acc":r["acc"],"time":r["time"]} for n,r in res.items()}
    with open(f"models/{tag}/model_comparison.json","w") as f: json.dump(cmp,f,indent=2)
    print(f"    Saved.")

    # Classification reports
    print("  [7] Reports & charts...")
    with open(f"outputs/{tag}/classification_reports.txt","w") as f:
        for mname in MODEL_ORDER:
            rpt=classification_report(y_test,preds[mname],
                target_names=class_names,digits=4,zero_division=0)
            f.write(f"{'='*65}\n{mname}\n{'='*65}\n{rpt}\n")

    # Confusion matrices
    def save_cm(y_tr,y_pr,title,path):
        nc   = len(class_names)
        cm_r = confusion_matrix(y_tr,y_pr,labels=list(range(nc)))
        cm_n = cm_r.astype(float)
        rs   = cm_n.sum(axis=1,keepdims=True); rs[rs==0]=1
        cm_n /= rs
        fig,axes = plt.subplots(1,2,figsize=(6+nc*1.3,5))
        fig.suptitle(title,fontsize=12,fontweight="bold",y=1.02)
        for ax,data,cmap,fmt in [
            (axes[0],cm_r,"Blues",   lambda v:f"{int(v):,}"),
            (axes[1],cm_n,"RdYlGn", lambda v:f"{v:.2f}")]:
            im=ax.imshow(data,interpolation="nearest",cmap=cmap,
                         **({"vmin":0,"vmax":1} if cmap=="RdYlGn" else {}))
            fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
            sh=[c[:6] for c in class_names]
            ax.set_xticks(range(nc)); ax.set_xticklabels(sh,rotation=40,ha="right",fontsize=7)
            ax.set_yticks(range(nc)); ax.set_yticklabels(sh,fontsize=7)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            thr=data.max()/2 if cmap=="Blues" else 0.5
            for i in range(nc):
                for j in range(nc):
                    v=data[i,j]
                    col="white" if ((cmap=="Blues" and v>thr) or
                                    (cmap=="RdYlGn" and (v<0.3 or v>0.75))) else "black"
                    ax.text(j,i,fmt(v),ha="center",va="center",fontsize=6,color=col)
        axes[0].set_title("Raw Counts",fontsize=10,fontweight="bold")
        axes[1].set_title("Normalised %",fontsize=10,fontweight="bold")
        plt.tight_layout()
        plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()

    for mname,p in [("Random Forest",f"outputs/{tag}/cm_rf.png"),
                     ("Extra Trees",  f"outputs/{tag}/cm_et.png"),
                     ("XGBoost",      f"outputs/{tag}/cm_xgb.png"),
                     ("LightGBM",     f"outputs/{tag}/cm_lgbm.png"),
                     ("DNN",          f"outputs/{tag}/cm_dnn.png")]:
        save_cm(y_test,preds[mname],f"CM — {mname} [{ds_name}]",p)
    print(f"    Confusion matrices → outputs/{tag}/")

    # Accuracy + speed bar chart
    accs  = [res[m]["acc"]  for m in MODEL_ORDER]
    times = [res[m]["time"] for m in MODEL_ORDER]
    bi    = int(np.argmax(accs))
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    fig.suptitle(f"5-Model Comparison — {ds_name} (Top-15 Features)",
                 fontsize=13,fontweight="bold")
    bars_a=axes[0].bar(MODEL_ORDER,accs,color=MODEL_COLORS,edgecolor="black",width=0.6)
    for i,(bar,acc) in enumerate(zip(bars_a,accs)):
        axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,
                     f"{acc:.2f}%",ha="center",fontsize=10,fontweight="bold",
                     color="darkred" if i==bi else "black")
    axes[0].set_ylim(0,110); axes[0].set_ylabel("Accuracy (%)",fontsize=11)
    axes[0].set_title("Accuracy",fontsize=12,fontweight="bold")
    axes[0].grid(axis="y",alpha=0.3)
    axes[0].axhline(y=max(accs),color="gold",linestyle="--",linewidth=1.5,
                    label=f"Best:{max(accs):.2f}%")
    axes[0].legend(fontsize=9)
    axes[0].set_xticklabels(MODEL_ORDER,rotation=15,ha="right",fontsize=9)
    axes[0].text(bi,accs[bi]+2.5,"🏆",ha="center",fontsize=14)
    bars_t=axes[1].bar(MODEL_ORDER,times,color=MODEL_COLORS,edgecolor="black",width=0.6,alpha=0.85)
    for bar,t in zip(bars_t,times):
        axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
                     f"{t:.0f}s",ha="center",fontsize=10,fontweight="bold")
    axes[1].set_ylabel("Training Time (s)",fontsize=11)
    axes[1].set_title("Training Speed",fontsize=12,fontweight="bold")
    axes[1].grid(axis="y",alpha=0.3)
    axes[1].set_xticklabels(MODEL_ORDER,rotation=15,ha="right",fontsize=9)
    plt.tight_layout()
    plt.savefig(f"outputs/{tag}/model_comparison.png",dpi=150,bbox_inches="tight"); plt.close()

    # Radar chart
    def mvec(y_tr,y_pr):
        nc_=len(class_names); lbl=list(range(nc_))
        acc  = accuracy_score(y_tr,y_pr)*100
        prec = precision_score(y_tr,y_pr,average="weighted",zero_division=0,labels=lbl)*100
        rec  = recall_score(y_tr,y_pr,average="weighted",zero_division=0,labels=lbl)*100
        f1   = f1_score(y_tr,y_pr,average="weighted",zero_division=0,labels=lbl)*100
        cmr  = confusion_matrix(y_tr,y_pr,labels=lbl)
        tn   = cmr[0,0]; fp=cmr[0,1:].sum()
        far  = (1-fp/max(tn+fp,1))*100
        return [acc,prec,rec,f1,far]
    RL  = ["Accuracy","Precision","Recall","F1-Score","Low FAR"]
    ang = np.linspace(0,2*np.pi,5,endpoint=False).tolist(); ang+=ang[:1]
    fig_r,ax_r=plt.subplots(figsize=(7,7),subplot_kw={"projection":"polar"})
    for i,mname in enumerate(MODEL_ORDER):
        v=[x/100 for x in mvec(y_test,preds[mname])]; v.append(v[0])
        ax_r.plot(ang,v,"o-",linewidth=2,color=MODEL_COLORS[i],label=mname)
        ax_r.fill(ang,v,alpha=0.07,color=MODEL_COLORS[i])
    ax_r.set_xticks(ang[:-1]); ax_r.set_xticklabels(RL,fontsize=10)
    ax_r.set_ylim(0,1); ax_r.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax_r.set_yticklabels(["20%","40%","60%","80%","100%"],fontsize=7)
    ax_r.set_title(f"Radar — {ds_name}",fontsize=12,fontweight="bold",pad=20)
    ax_r.legend(loc="upper right",bbox_to_anchor=(1.35,1.15),fontsize=9)
    plt.tight_layout()
    plt.savefig(f"outputs/{tag}/radar_chart.png",dpi=150,bbox_inches="tight"); plt.close()
    print(f"    Charts saved to outputs/{tag}/")

    # FAR
    cmrf = confusion_matrix(y_test,preds["Random Forest"],labels=list(range(n_cls)))
    tn0  = cmrf[0,0]; fp0=cmrf[0,1:].sum()
    far  = fp0/max(tn0+fp0,1)*100

    # Leaderboard
    print(f"\n  {'─'*55}")
    print(f"  RESULTS — {ds_name}")
    print(f"  {'Model':<20} {'Accuracy':>10}  {'Time':>8}")
    medals={1:"🥇",2:"🥈",3:"🥉",4:"4️⃣",5:"5️⃣"}
    for rank,(mn,r) in enumerate(
            sorted(res.items(),key=lambda x:x[1]["acc"],reverse=True),1):
        print(f"  {medals[rank]}  {mn:<18} {r['acc']:>9.2f}%  {r['time']:>6.1f}s")
    print(f"  FAR (RF): {far:.2f}%  |  ATTACK:{n_a:,}  UNCERTAIN:{n_u:,}  CLEAN:{n_c:,}")

    return dict(results=res, preds=preds, top_feat=top_feat,
                class_names=class_names, far=round(far,2),
                n_atk=n_a, n_unc=n_u, n_cln=n_c,
                dnn_probs=dnn_probs, y_test=y_test)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("  Hybrid IDS v4  —  NSL-KDD + UNSW-NB15  |  5 Models Each")
print("="*65)

for d in ["data","models","outputs",
          "models/nsl_kdd","models/unsw_nb15",
          "outputs/nsl_kdd","outputs/unsw_nb15"]:
    os.makedirs(d,exist_ok=True)
print("[DIR] All directories ready.\n")

# ── Dataset 1: NSL-KDD ───────────────────────────────────────────────────────
print("▶  NSL-KDD")
NTR="data/KDDTrain+.txt"; NTE="data/KDDTest+.txt"
if not os.path.exists(NTR) or not os.path.exists(NTE):
    print("  [WARN] Not found — generating synthetic NSL-KDD...")
    _nsl_synth(5000).to_csv(NTR,header=False,index=False)
    _nsl_synth(1000).to_csv(NTE,header=False,index=False)
ntr=pd.read_csv(NTR,header=None,names=NSL_COLUMNS)
nte=pd.read_csv(NTE,header=None,names=NSL_COLUMNS)
for df in [ntr,nte]: df.drop(columns=["difficulty"],inplace=True,errors="ignore")
ntr["multi_label"]=ntr["label"].apply(nsl_map)
nte["multi_label"]=nte["label"].apply(nsl_map)
NSL_FEAT=[c for c in ntr.columns if c not in ("label","multi_label")]
print(f"  Train:{ntr.shape}  Test:{nte.shape}")
for k,v in ntr["multi_label"].value_counts().sort_index().items():
    print(f"    {NSL_CLS[k]:12s}: {v:,}")

nsl_out = run_pipeline("NSL-KDD",ntr.copy(),nte.copy(),NSL_FEAT,NSL_CAT,NSL_CLS)

# ── Dataset 2: UNSW-NB15 ─────────────────────────────────────────────────────
print("\n\n▶  UNSW-NB15")
UTR="data/UNSW_NB15_training-set.csv"; UTE="data/UNSW_NB15_testing-set.csv"
if not os.path.exists(UTR) or not os.path.exists(UTE):
    print("  [INFO] UNSW-NB15 files not found.")
    print("  [INFO] Download from: research.unsw.edu.au/projects/unsw-nb15-dataset")
    print("  [INFO] Place in data/ folder and re-run main.py")
    print("  [INFO] Using synthetic demo data for now...")
    ude_tr=_unsw_synth(5000); ude_te=_unsw_synth(1000)
else:
    ude_tr=pd.read_csv(UTR,low_memory=False)
    ude_te=pd.read_csv(UTE,low_memory=False)
    print(f"  Real UNSW-NB15 → Train:{ude_tr.shape}  Test:{ude_te.shape}")

for df in [ude_tr,ude_te]:
    for dc in ["id","label"]:
        if dc in df.columns: df.drop(columns=[dc],inplace=True)
    if "attack_cat" in df.columns:
        df["attack_cat"]=df["attack_cat"].fillna("Normal")
    else:
        df["attack_cat"]="Normal"
    df["multi_label"]=df["attack_cat"].apply(unsw_map)

UNSW_FEAT=[c for c in ude_tr.columns if c not in ("attack_cat","multi_label","label","id")]
UNSW_CAT_IN=[c for c in UNSW_CAT if c in ude_tr.columns]
print(f"  Features: {len(UNSW_FEAT)}")
for k,v in ude_tr["multi_label"].value_counts().sort_index().items():
    if k<len(UNSW_CLS): print(f"    {UNSW_CLS[k]:16s}: {v:,}")

unsw_out = run_pipeline("UNSW-NB15",ude_tr.copy(),ude_te.copy(),
                         UNSW_FEAT,UNSW_CAT_IN,UNSW_CLS)

# ── Cross-dataset comparison chart ───────────────────────────────────────────
print("\n[FINAL] Cross-dataset comparison chart...")
nsl_accs  = [nsl_out["results"][m]["acc"]  for m in MODEL_ORDER]
unsw_accs = [unsw_out["results"][m]["acc"] for m in MODEL_ORDER]
x=np.arange(len(MODEL_ORDER)); w=0.35

fig,axes=plt.subplots(1,2,figsize=(16,6))
fig.suptitle("NSL-KDD vs UNSW-NB15 — 5-Model Cross-Dataset Comparison",
             fontsize=14,fontweight="bold")

bn=axes[0].bar(x-w/2,nsl_accs, w,label="NSL-KDD (2009)",
               color="#3498db",edgecolor="black",alpha=0.9)
bu=axes[0].bar(x+w/2,unsw_accs,w,label="UNSW-NB15 (2015)",
               color="#e67e22",edgecolor="black",alpha=0.9)
for bar,a in zip(bn,nsl_accs):
    axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.4,
                 f"{a:.1f}%",ha="center",fontsize=8,fontweight="bold",color="#1a5276")
for bar,a in zip(bu,unsw_accs):
    axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.4,
                 f"{a:.1f}%",ha="center",fontsize=8,fontweight="bold",color="#7d4600")
axes[0].set_xticks(x)
axes[0].set_xticklabels(MODEL_ORDER,rotation=15,ha="right",fontsize=9)
axes[0].set_ylim(0,115); axes[0].set_ylabel("Accuracy (%)",fontsize=11)
axes[0].set_title("Accuracy — Both Datasets",fontsize=12,fontweight="bold")
axes[0].legend(fontsize=10); axes[0].grid(axis="y",alpha=0.3)

deltas=[u-n for u,n in zip(unsw_accs,nsl_accs)]
dc=[("#2ecc71" if d>=0 else "#e74c3c") for d in deltas]
bd=axes[1].bar(MODEL_ORDER,deltas,color=dc,edgecolor="black",width=0.5)
for bar,d in zip(bd,deltas):
    axes[1].text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+(0.3 if d>=0 else -1.2),
                 f"{d:+.1f}%",ha="center",fontsize=9,fontweight="bold")
axes[1].axhline(y=0,color="black",linewidth=1.5)
axes[1].set_ylabel("Accuracy Δ (UNSW − NSL)",fontsize=11)
axes[1].set_title("Generalisation Gap\n(green=UNSW higher, red=NSL higher)",
                  fontsize=12,fontweight="bold")
axes[1].set_xticklabels(MODEL_ORDER,rotation=15,ha="right",fontsize=9)
axes[1].grid(axis="y",alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/cross_dataset_comparison.png",dpi=150,bbox_inches="tight")
plt.close()
print("  outputs/cross_dataset_comparison.png ✓")

# Master JSON
master={
    "NSL-KDD":   {m:nsl_out["results"][m]["acc"]  for m in MODEL_ORDER},
    "UNSW-NB15": {m:unsw_out["results"][m]["acc"] for m in MODEL_ORDER},
    "nsl_far":   nsl_out["far"],
    "unsw_far":  unsw_out["far"],
}
with open("models/master_comparison.json","w") as f: json.dump(master,f,indent=2)
print("  models/master_comparison.json ✓")

# Final leaderboard
print("\n"+"="*65)
print("  FINAL CROSS-DATASET LEADERBOARD")
print("="*65)
print(f"\n  {'Model':<20} {'NSL-KDD':>10}  {'UNSW-NB15':>12}  {'Average':>9}")
print(f"  {'─'*57}")
avgs=[]
for m in MODEL_ORDER:
    n=nsl_out["results"][m]["acc"]; u=unsw_out["results"][m]["acc"]
    avg=(n+u)/2; avgs.append((m,n,u,avg))
    print(f"  {m:<20} {n:>9.2f}%  {u:>11.2f}%  {avg:>8.2f}%")
best=max(avgs,key=lambda x:x[3])
print(f"\n  🏆 Best overall: {best[0]}  ({best[3]:.2f}% avg)")
print(f"  NSL-KDD FAR: {nsl_out['far']:.2f}%   UNSW-NB15 FAR: {unsw_out['far']:.2f}%")
print("\n"+"="*65)
print("  Hybrid IDS v4 complete!  [2 Datasets × 5 Models = 10 trained]")
print("="*65)
print("\n  Next step:  streamlit run dashboard.py")
print("\n  To use real UNSW-NB15 data:")
print("    https://research.unsw.edu.au/projects/unsw-nb15-dataset")
print("    → Download UNSW_NB15_training-set.csv + UNSW_NB15_testing-set.csv")
print("    → Place in data/  →  re-run python main.py")