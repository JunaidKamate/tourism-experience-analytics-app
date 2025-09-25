
import streamlit as st
from pathlib import Path
import joblib, json
import pandas as pd, numpy as np

# MUST be first Streamlit call
st.set_page_config(layout='wide', page_title='Tourism Experience Analytics', page_icon=None)

# ==== Absolute paths (Windows) ====
BASE = Path(r"C:\Users\junai\tourism_raw_data")
PROC = BASE / "processed"
SPLITS = PROC / "splits"
MODELS = Path(r"C:\Users\junai\models")
RESULTS = Path(r"C:\Users\junai\results")

# Safe loader
def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"safe_load error for {path}: {e}")
        return None

# Load models/artifacts
rating_model = safe_load(MODELS / 'rating_model.pkl')
visitmode_model = safe_load(MODELS / 'visitmode_model.pkl')
recommender_svd = safe_load(MODELS / 'recommender_svd.pkl')
content_art = safe_load(MODELS / 'content_tfidf_artifacts.pkl')

visitmode_classes = None
if (MODELS / 'visitmode_classes.json').exists():
    try:
        visitmode_classes = json.load(open(MODELS / 'visitmode_classes.json','r'))['classes']
    except Exception as e:
        print("Could not load visitmode_classes.json:", e)

# feature lists / metadata fallback
model_meta = None
if (MODELS / 'model_metadata.json').exists():
    try:
        model_meta = json.load(open(MODELS / 'model_metadata.json','r'))
    except Exception:
        model_meta = None

feat_visitmode = None
if (MODELS / 'feature_cols_visitmode.json').exists():
    try:
        feat_visitmode = json.load(open(MODELS / 'feature_cols_visitmode.json'))['feature_cols']
    except Exception:
        feat_visitmode = None

feat_rating = model_meta.get('feature_cols') if model_meta and 'feature_cols' in model_meta else None

# utility: model expected features if available
def model_expected_features(model):
    if model is None:
        return None
    # sklearn models have feature_names_in_ (newer) or we can use feature metadata saved
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    # else fallback to feat_rating/feat_visitmode
    return None

rating_expected_feats = model_expected_features(rating_model)
visitmode_expected_feats = model_expected_features(visitmode_model)

# Load merged and train for lookups
@st.cache_data
def load_merged():
    p = PROC / 'merged_master.csv'
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    return None

@st.cache_data
def load_train():
    p = SPLITS / 'train.csv'
    if p.exists():
        return pd.read_csv(p)
    return None

merged = load_merged()
train_df = load_train()

# App title (change here if you want a different app title)
APP_TITLE = "Tourism Experience Analytics"
st.title(APP_TITLE)
st.caption("Classification · Prediction · Recommendation System")

# Sidebar inputs
st.sidebar.header("Inputs")
uid = int(st.sidebar.number_input("UserId", min_value=1, step=1, value=70456))
# attraction selector from merged/items if available
if merged is not None and 'AttractionId' in merged.columns:
    unique_atts = sorted(merged['AttractionId'].unique().tolist())
else:
    unique_atts = [640,841,748]
att = int(st.sidebar.selectbox("AttractionId", options=unique_atts, index=0))

st.sidebar.markdown("---")
st.sidebar.write("Models loaded:")
st.sidebar.write(f"Rating model: {'Yes' if rating_model is not None else 'No'}")
st.sidebar.write(f"VisitMode model: {'Yes' if visitmode_model is not None else 'No'}")
st.sidebar.write(f"Content TF-IDF: {'Yes' if content_art is not None else 'No'}")
st.sidebar.write(f"SVD artifact: {'Yes' if recommender_svd is not None else 'No'}")

# -------------------------
# Feature builder + alignment (guarantee columns and order)
# -------------------------
def prepare_features_for_user(user_id, attraction_id, requested_features=None, align_to_model=None):
    """
    Build a single-row DataFrame with columns requested_features (or fallback).
    If align_to_model is provided (list of feature names), ensure the returned DataFrame
    exactly contains those columns in same order (missing filled with 0).
    """
    # fallback feature list selection
    if requested_features is None:
        requested_features = feat_rating or feat_visitmode or []
    # build base row using merged representative rows
    if merged is None or merged.shape[0] == 0:
        base = pd.DataFrame([ {c: 0 for c in requested_features} ])
    else:
        # pick best candidate rows
        mask_both = (merged['UserId'] == user_id) & (merged['AttractionId'] == attraction_id)
        mask_user = (merged['UserId'] == user_id)
        mask_att  = (merged['AttractionId'] == attraction_id)
        row_both = merged[mask_both].iloc[0] if mask_both.any() else None
        row_user = merged[mask_user].iloc[0] if mask_user.any() else None
        row_att  = merged[mask_att].iloc[0]  if mask_att.any()  else None
        row_any  = merged.iloc[0]
        out = {}
        for c in requested_features:
            val = None
            # priority: row_both -> row_user -> row_att -> row_any
            for r in (row_both, row_user, row_att, row_any):
                if r is None: continue
                if c in r.index:
                    val = r[c]
                    break
            if pd.isna(val) or val is None:
                val = 0
            # coerce numpy scalars to native
            try:
                if hasattr(val, 'item'):
                    val = val.item()
            except Exception:
                pass
            out[c] = val
        base = pd.DataFrame([out], columns=requested_features)
    # Align to model expected features if provided
    if align_to_model:
        # Ensure all model columns exist; add missing with zeros; order accordingly
        for m in align_to_model:
            if m not in base.columns:
                base[m] = 0
        base = base.reindex(columns=align_to_model).fillna(0)
    return base

# -------------------------
# Recommender helpers (popularity/content/svd) - unchanged
# -------------------------
def recommend_popularity(k=5):
    if train_df is None:
        return []
    return train_df['AttractionId'].value_counts().index[:k].tolist()

def recommend_content(user_id, k=5):
    if content_art is None or train_df is None:
        return recommend_popularity(k)
    vec = content_art.get('vectorizer')
    items_meta = content_art.get('items_meta')
    item_idx = content_art.get('item_idx')
    user_hist = train_df[train_df['UserId']==user_id][['AttractionId','Rating']]
    if user_hist.empty:
        return recommend_popularity(k)
    mats=[]; weights=[]
    for aid, r in zip(user_hist['AttractionId'], user_hist['Rating']):
        if aid in item_idx:
            mats.append(vec.transform([items_meta.loc[aid,'meta_text']]).toarray().ravel())
            weights.append(r)
    if not mats:
        return recommend_popularity(k)
    profile = np.average(np.vstack(mats), axis=0, weights=np.array(weights))
    all_vecs = vec.transform(items_meta['meta_text'].values)
    sims = np.array(all_vecs.dot(profile))
    ranked = np.argsort(-sims)
    recs=[]; seen=set(user_hist['AttractionId'])
    for idx in ranked:
        aid = item_idx[idx]
        if aid not in seen:
            recs.append(aid)
        if len(recs)>=k:
            break
    return recs

def recommend_svd(user_id, k=5):
    try:
        ui = train_df.groupby(['UserId','AttractionId'])['Rating'].mean().unstack(fill_value=np.nan)
        if ui.shape[0]==0 or ui.shape[1]==0:
            return recommend_popularity(k)
        user_means = ui.mean(axis=1)
        ui_centered = ui.sub(user_means, axis=0).fillna(0)
        from sklearn.decomposition import TruncatedSVD
        n_comp = min(10, max(1, ui_centered.shape[1]-1))
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        user_factors = svd.fit_transform(ui_centered)
        item_factors = svd.components_.T
        recon = np.dot(user_factors, item_factors.T) + user_means.values.reshape(-1,1)
        recon_df = pd.DataFrame(recon, index=ui_centered.index, columns=ui_centered.columns)
        if user_id not in recon_df.index:
            return recommend_popularity(k)
        scores = recon_df.loc[user_id].sort_values(ascending=False)
        seen = set(train_df[train_df['UserId']==user_id]['AttractionId'].unique())
        recs = [int(i) for i in scores.index if int(i) not in seen][:k]
        return recs
    except Exception:
        return recommend_popularity(k)

# -------------------------
# UI actions
# -------------------------
tab = st.sidebar.selectbox("App page", ["Predict Rating","Predict Visit Mode","Recommend Top-5","EDA & Artifacts"])

if tab == "Predict Rating":
    st.header("Predict Rating for a User + Attraction")
    st.write("UserId:", uid, "AttractionId:", att)
    if rating_model is None:
        st.warning("Rating model not found. Train and save models to models/rating_model.pkl")
    else:
        # Determine target feature set for rating model
        align_feats = rating_expected_feats if rating_expected_feats is not None else (feat_rating or feat_visitmode)
        X = prepare_features_for_user(uid, att, requested_features=(feat_rating or feat_visitmode or []), align_to_model=align_feats)
        try:
            pred = rating_model.predict(X.fillna(0))[0]
            st.metric("Predicted Rating", round(float(pred), 3))
        except Exception as e:
            st.error("Prediction error: " + str(e))
            st.write("Model expected features (sample):", (align_feats[:20] if align_feats else "n/a"))
            st.write("Provided X columns (sample):", list(X.columns)[:40])

elif tab == "Predict Visit Mode":
    st.header("Predict Visit Mode for a User")
    st.write("UserId:", uid)
    if visitmode_model is None or visitmode_classes is None:
        st.warning("VisitMode classifier or classes mapping not found.")
    else:
        align_feats = visitmode_expected_feats if visitmode_expected_feats is not None else (feat_visitmode or [])
        X = prepare_features_for_user(uid, att, requested_features=(feat_visitmode or []), align_to_model=align_feats)
        try:
            pred_enc = visitmode_model.predict(X.fillna(0))[0]
            try:
                pred_label = visitmode_classes[int(pred_enc)]
            except Exception:
                pred_label = pred_enc
            st.write("Predicted Visit Mode:", pred_label)
        except Exception as e:
            st.error("VisitMode prediction error: " + str(e))
            st.write("Model expected features (sample):", (align_feats[:20] if align_feats else "n/a"))
            st.write("Provided X columns (sample):", list(X.columns)[:40])

elif tab == "Recommend Top-5":
    st.header("Recommendations (choose method)")
    method = st.selectbox("Method", ["popularity","content","svd"])
    if st.button("Get recommendations"):
        if method == "popularity":
            recs = recommend_popularity(k=5)
        elif method == "content":
            recs = recommend_content(uid, k=5)
        else:
            recs = recommend_svd(uid, k=5)
        st.write("Top-5 recommendations (AttractionIds):", recs)
        if merged is not None and 'AttractionId' in merged.columns:
            names = []
            for a in recs:
                r = merged[merged['AttractionId']==int(a)]
                names.append(r.iloc[0].get('att_Attraction', str(a)) if len(r) else str(a))
            st.write("Names:", names)

elif tab == "EDA & Artifacts":
    st.header("Quick EDA")
    if train_df is not None:
        st.subheader("Top attractions (train counts)")
        topk = train_df['AttractionId'].value_counts().head(10).reset_index().rename(columns={'index':'AttractionId','AttractionId':'visits'})
        st.table(topk)
    st.subheader("Models & artifacts present")
    st.write("rating_model:", (MODELS / 'rating_model.pkl').exists())
    st.write("visitmode_model:", (MODELS / 'visitmode_model.pkl').exists())
    st.write("content_tfidf:", (MODELS / 'content_tfidf_artifacts.pkl').exists())
    st.write("recommender_svd:", (MODELS / 'recommender_svd.pkl').exists())

st.markdown("---")
st.caption("Scaffold app — next: wire hybrid recommender, SHAP explainability and UI polish.")
