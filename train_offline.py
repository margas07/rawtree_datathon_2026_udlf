import os
from typing import Any
from collections import Counter
import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
from dotenv import load_dotenv

load_dotenv()

API: str = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
HEADERS: dict[str, str] = {"Authorization": f"Bearer {os.environ.get('RAWTREE_API_KEY')}"}

def query_to_df(sql: str) -> pd.DataFrame:
    r: requests.Response = requests.post(f"{API}/v1/query", json={"sql": sql}, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return pd.DataFrame(r.json()["data"])

def get_catalog_and_stats() -> dict[str, dict[str, Any]]:
    """Fetches product metadata, historical organic CVR, and velocity."""
    catalog: pd.DataFrame = query_to_df("SELECT product_id::String as product_id, price_eur::Float64 as price, subcategory::String as subcategory FROM shop.catalog")
    
    stats_sql: str = """
        SELECT product_id::String as product_id, 
               countIf(event_type='purchase') / greatest(countIf(event_type='view'), 1) as org_cvr,
               count() / (60.0 * 24.0) as velocity
        FROM shop.history_events 
        GROUP BY product_id
    """
    stats_df: pd.DataFrame = query_to_df(stats_sql)
    
    merged: pd.DataFrame = catalog.merge(stats_df, on='product_id', how='left').fillna({'org_cvr': 0.0, 'velocity': 0.0})
    return merged.set_index('product_id').to_dict('index')

def get_covisit_matrix(limit_per_product: int = 20) -> dict[str, list[str]]:
    """Fetches top co-viewed products from history_events before test cut."""
    print("Fetching historical co-visit matrix...")
    sql: str = f"""
        SELECT a.product_id::String AS p, b.product_id::String AS q, count() AS n
        FROM (SELECT session_id::String AS session_id, product_id::String AS product_id FROM shop.history_events WHERE event_type = 'view' AND ts < '2026-11-07') AS a
        INNER JOIN (SELECT session_id::String AS session_id, product_id::String AS product_id FROM shop.history_events WHERE event_type = 'view' AND ts < '2026-11-07') AS b
          ON a.session_id = b.session_id
        WHERE a.product_id != b.product_id
        GROUP BY p, q ORDER BY n DESC LIMIT {limit_per_product} BY p
    """
    try:
        coviews: pd.DataFrame = query_to_df(sql)
        return coviews.groupby("p")["q"].apply(list).to_dict()
    except Exception as e:
        print(f"Warning: could not fetch covisit_matrix: {e}")
        return {}

def build_training_dataframe() -> pd.DataFrame:
    """Explodes history_sessions and computes the 6 features."""
    sessions: pd.DataFrame = query_to_df("""
        SELECT request_id, recent_views, shown, propensity, purchased 
        FROM shop.history_sessions 
        WHERE ts < '2026-11-07' AND purchased IS NOT NULL
        ORDER BY rand() LIMIT 50000
    """)
    
    catalog_dict: dict[str, dict[str, Any]] = get_catalog_and_stats()
    covisit_matrix: dict[str, list[str]] = get_covisit_matrix()
    rows: list[dict[str, Any]] = []
    
    for _, session in sessions.iterrows():
        recent_views: list[dict[str, Any]] = session['recent_views']
        shown_items: list[str] = session['shown']
        propensities: list[float] = session['propensity']
        purchased: str = session['purchased']
        
        recent_prices: list[float] = [catalog_dict.get(v['product_id'], {}).get('price', 0.0) for v in recent_views]
        avg_session_price: float = float(np.mean(recent_prices)) if recent_prices else 0.0
        recent_subcats: list[str] = [catalog_dict.get(v['product_id'], {}).get('subcategory', '') for v in recent_views]
        
        coview_scores: Counter[str] = Counter()
        for k, view in enumerate(reversed(recent_views)):
            for rank, candidate_id in enumerate(covisit_matrix.get(view["product_id"], [])):
                coview_scores[candidate_id] += (0.7 ** k) / (rank + 1)
        
        for i, candidate in enumerate(shown_items):
            cand_meta: dict[str, Any] = catalog_dict.get(candidate, {})
            cand_price: float = cand_meta.get('price', 0.0)
            cand_subcat: str = cand_meta.get('subcategory', '')
            
            price_ratio: float = (cand_price / avg_session_price) if avg_session_price > 0.0 else 1.0
            
            subcat_matches: int = sum(1 for sub in recent_subcats if sub == cand_subcat)
            subcat_affinity: float = (subcat_matches / len(recent_subcats)) if recent_subcats else 0.0
            
            rows.append({
                'request_id': session['request_id'],
                'product_id': candidate,
                'is_purchased': 1 if candidate == purchased else 0,
                'propensity': propensities[i],
                'coview_score': coview_scores.get(candidate, 0.0),
                'price_ratio': price_ratio,
                'subcat_affinity': subcat_affinity,
                'price_drop_flag': 0.0, 
                'organic_cvr': cand_meta.get('org_cvr', 0.0),
                'velocity': cand_meta.get('velocity', 0.0) 
            })
            
    return pd.DataFrame(rows)

def train_model() -> None:
    print("Building training dataset...")
    df: pd.DataFrame = build_training_dataframe()
    
    features: list[str] = ['coview_score', 'price_ratio', 'subcat_affinity', 'price_drop_flag', 'organic_cvr', 'velocity']
    X: pd.DataFrame = df[features]
    y: pd.Series = df['is_purchased']
    
    weights = np.clip(1.0 / np.maximum(df['propensity'].fillna(0.1).to_numpy(), 0.01), 0.5, 20.0)
    
    print("Training LightGBM...")
    model: lgb.LGBMClassifier = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y, sample_weight=weights)
    
    model.booster_.save_model('ranker_v2_model.txt')
    print("Model saved to ranker_v2_model.txt")

if __name__ == "__main__":
    train_model()
