
"""
feature_utils.py

Helper functions for the Streamlit app to build model input features
consistent with the preprocessing used in modeling notebooks.

Functions:
 - load_merged() -> DataFrame (cached)
 - get_feature_lists() -> dict with keys 'rating' and 'visitmode' (lists of columns)
 - build_feature_vector(user_id, attraction_id, feature_list) -> DataFrame single row
"""
from functools import lru_cache
from pathlib import Path
import pandas as pd
import json
import numpy as np

PROC = Path('tourism_raw_data') / 'processed'
MERGED_PATH = PROC / 'merged_master.csv'
MODELS_DIR = Path('models')
@lru_cache(maxsize=1)
def load_merged():
    if MERGED_PATH.exists():
        df = pd.read_csv(MERGED_PATH, low_memory=False)
        return df
    raise FileNotFoundError(f"{MERGED_PATH} not found. Run preprocessing first.")

def get_feature_lists():
    """
    Try to read model metadata and saved feature lists.
    Returns dict: {'rating': [...], 'visitmode': [...]}
    """
    meta = {}
    # model_metadata.json may contain feature_cols for rating
    mm = MODELS_DIR / 'model_metadata.json'
    fc = MODELS_DIR / 'feature_cols_visitmode.json'
    rating_cols = None
    visit_cols = None
    if mm.exists():
        try:
            m = json.load(open(mm,'r',encoding='utf-8'))
            rating_cols = m.get('feature_cols')
        except Exception:
            rating_cols = None
    if fc.exists():
        try:
            fv = json.load(open(fc,'r',encoding='utf-8'))
            visit_cols = fv.get('feature_cols')
        except Exception:
            visit_cols = None
    return {'rating': rating_cols, 'visitmode': visit_cols}

def _pick_value_from_merged(row_candidate, col):
    """
    Helper to pick a value from a row candidate (Series) for a column name.
    Handles NaN.
    """
    if col in row_candidate.index:
        val = row_candidate[col]
        if pd.isna(val):
            return None
        return val
    return None

def build_feature_vector(user_id, attraction_id, feature_list):
    """
    Build a single-row DataFrame containing exactly the columns in feature_list.
    Strategy:
    - Load merged_master
    - Try to find:
       - a row for (UserId==user_id AND AttractionId==attraction_id)
       - else a row for (UserId==user_id)
       - else a representative row for the attraction
       - else fallback to first row of merged (global defaults)
    - For each column in feature_list, try to extract from user_row first, then att_row, else set 0/-1 depending on dtype
    - Returns pandas.DataFrame with single row and columns in same order as feature_list
    """
    if feature_list is None or len(feature_list)==0:
        return pd.DataFrame([{}])
    merged = load_merged()

    # find candidate rows
    mask_both = (merged['UserId'] == user_id) & (merged['AttractionId'] == attraction_id)
    att_mask = (merged['AttractionId'] == attraction_id)
    user_mask = (merged['UserId'] == user_id)

    if mask_both.any():
        cand_both = merged[mask_both].iloc[0]
    else:
        cand_both = None

    cand_user = merged[user_mask].iloc[0] if user_mask.any() else None
    cand_att  = merged[att_mask].iloc[0]  if att_mask.any()  else None
    cand_any  = merged.iloc[0]

    out = {}
    for c in feature_list:
        val = None
        # priority: explicit match row (user+att) -> user row -> att row -> any
        if cand_both is not None:
            val = _pick_value_from_merged(cand_both, c)
        if val is None and cand_user is not None:
            val = _pick_value_from_merged(cand_user, c)
        if val is None and cand_att is not None:
            val = _pick_value_from_merged(cand_att, c)
        if val is None:
            val = _pick_value_from_merged(cand_any, c)

        # If still None, fallbacks:
        if val is None:
            # common patterns: *_enc -> 0, numeric -> 0, categorical(enc or id) -> 0
            if c.endswith('_enc') or c.endswith('_Id') or c.endswith('_id') or c.endswith('Id'):
                val = 0
            else:
                val = 0

        # Coerce numpy/pandas scalars to Python native types for Streamlit safety
        if isinstance(val, (pd.Timestamp, )):
            # convert timestamp to ordinal or year if needed; default to 0
            try:
                val = int(getattr(val, 'year', 0))
            except Exception:
                val = 0
        if isinstance(val, (np.generic, )):
            val = val.item()
        out[c] = val
    # return DataFrame with single row (feature_list order)
    row = pd.DataFrame([out], columns=feature_list)
    return row
