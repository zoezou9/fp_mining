#!/usr/bin/env python3
import time, json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    results = {}
    # --- 1. 載入資料---
    t0 = time.time()
    # transfer t0 to utc -4 time
    print("Loading dataset...", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t0 - 14400)))

    df = pd.read_csv('./data/dataset_10000.csv', sep='delimiter', header=None, engine='python')
    txns = df[0].str.split(',').tolist()
    # txns = [[i for i in t if i != '-1' and i != ''] for t in txns]
    txns = [
        [item.strip() for item in txn
        if item.strip() and item.strip() != '-1']
        for txn in txns
    ]
    freq_counter = Counter(i for t in txns for i in t)
    unique_items = sorted(freq_counter)
    print(">> Unique items (before encoding):", unique_items, "count =", len(unique_items), flush=True)

    enc = LabelEncoder(); enc.fit(unique_items)
    N = len(txns)
    results['num_txns'] = N
    results['num_items'] = len(unique_items)
    results['preproc_time_s'] = time.time() - t0

    # --- 2. 特徵抽取 ---
    t1 = time.time()
    print("Extracting features...", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t1 - 14400)))
    real_root = type('FPNode', (), {'__init__': lambda s,code=None: setattr(s, 'children', {}) or setattr(s, 'count',0) or setattr(s,'code',code)})()
    real_root.count = 0
    features, labels = [], []
    for txn in txns:
        seq = sorted(txn, key=lambda x:-freq_counter[x])
        codes = enc.transform(seq)
        node = real_root; node.count += 1; prefix = []
        for c in codes:
            # build feature vector
            pv = np.zeros(len(unique_items), int)
            for p in prefix: pv[p] = 1
            
            cc = np.zeros(len(unique_items), int)
            for ch, nd in node.children.items(): cc[ch] = nd.count
            
            pc, pl = node.count, len(prefix)
            
            item = enc.inverse_transform([c])[0]
            
            p_c = freq_counter[item]/N


            # p_p = freq_counter[prefix[-1]]/N if prefix else 1.0
            if prefix:
                last_code = prefix[-1]
                last_item = enc.inverse_transform([last_code])[0]
                p_p = freq_counter[last_item] / N
            else:
                p_p = 1.0
            
            edge_ct = node.children.get(c, type(node)()).count
            p_edge = edge_ct / N
            p_bound = p_c * p_p * p_edge

            feat = np.concatenate([pv, cc, [pc], [pl], [p_bound]])
            features.append(feat); labels.append(c)

            # 插入節點
            if c not in node.children:
                nd = type(node)(code=c)
                nd.count = 0
                node.children[c] = nd
            node = node.children[c]; node.count += 1
            prefix.append(c)
            
            # if len(features) < 5:
            #     print("DBG:", item, p_c, p_p, p_edge, p_bound, flush=True)

    real_root.count = N
    X = np.array(features); y = np.array(labels)
    results['feature_count'] = X.shape[0]
    results['feature_time_s'] = time.time() - t1

    

    # --- 3. XGBoost 依商品拆分模型 ---
    t2 = time.time()
    print("Training models...", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t2 - 14400)))
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]

    item_models = {}
    for code in range(len(unique_items)):
        yb = (y_tr == code).astype(int)
        if yb.sum() < 30: continue
        mdl = XGBClassifier(
           objective='binary:logistic',
           use_label_encoder=False, eval_metric='logloss',
           n_estimators=100, max_depth=6, learning_rate=0.1
        )
        mdl.fit(X_tr, yb)
        item_models[code] = mdl
    results['num_trained_models'] = len(item_models)
    results['train_time_s'] = time.time() - t2

    # --- 4. 預測 & 評估 ---
    t3 = time.time()
    print("Evaluating models...", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t3 - 14400)))
    proba = np.zeros((len(te), len(unique_items)))
    for i, idx in enumerate(te):
        xi = X[idx].reshape(1, -1)
        for c, m in item_models.items():
            proba[i, c] = m.predict_proba(xi)[0,1]
    acc1 = accuracy_score(y_te, proba.argmax(axis=1))
    # acc3 = top_k_accuracy_score(y_te, proba, k=3)
    results.update({
        'top1_acc': acc1,
        'top2_acc': top_k_accuracy_score(y_te, proba, k=2),
        'top3_acc': top_k_accuracy_score(y_te, proba, k=3),
        'top4_acc': top_k_accuracy_score(y_te, proba, k=4),
        'top5_acc': top_k_accuracy_score(y_te, proba, k=5),
    })
    results['eval_time_s'] = time.time() - t3

    # --- 5. 輸出結果 ---
    # 存 JSON
    with open('results_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    # Bash 輸出
    print("--- Results Summary ---")
    for k,v in results.items():
        print(f"{k:20s}: {v}")

if __name__ == '__main__':
    main()
