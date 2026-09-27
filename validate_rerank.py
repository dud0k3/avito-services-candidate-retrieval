"""Validate the CatBoost reranker against the lexical baseline.

The validation search contexts are excluded from the reranker's training set.
The candidate corpus contains all unique train items, matching retrieval.py's
local validation setup. No benchmark labels are used.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from retrieval import ITEM_COLS, QUERY_COLS, CandidateRetriever, clean_text, make_validation, top_indices


TOKENS = re.compile(r"[a-zа-я0-9]+")
BASE_WEIGHTS = np.array([0.285, 0.240, 0.225, 0.250], dtype=np.float32)
LOG = logging.getLogger(__name__)


def rank_fraction(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = np.arange(len(scores), dtype=np.float32)
    return ranks / max(len(scores) - 1, 1)


def recall(pools, predictions, item_ids, positives):
    values = []
    for (signature, ids, local, _), scores in zip(pools, predictions):
        local_ids = top_indices(ids[local], scores[local], 50)
        if len(local_ids) < 50:
            remaining = top_indices(ids[~local], scores[~local], 50 - len(local_ids))
            local_ids = np.concatenate([local_ids, remaining])
        relevant = positives[signature]
        values.append(len(set(item_ids[local_ids]) & relevant) / len(relevant))
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-queries", type=int, default=1200)
    parser.add_argument("--validation-queries", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cols = list(dict.fromkeys(QUERY_COLS + ITEM_COLS + ["item_rating_reviews_count"]))
    train = pd.read_parquet(args.data_dir / "train.parquet", columns=cols)
    valid_queries, valid_positives = make_validation(train, args.validation_queries, args.seed)
    train["signature"] = pd.util.hash_pandas_object(
        train[QUERY_COLS].fillna(""), index=False
    ).to_numpy()
    other_queries = train.drop_duplicates("signature")
    other_queries = other_queries[~other_queries.signature.isin(valid_queries.signature)]
    fit_queries = other_queries.sample(args.train_queries, random_state=2026).reset_index(drop=True)
    fit_sigs = set(fit_queries.signature)
    fit_positives = defaultdict(set)
    for sig, item_id in zip(train.signature, train.item_id):
        if sig in fit_sigs:
            fit_positives[sig].add(str(item_id))
    items = train.drop_duplicates("item_id")
    retriever = CandidateRetriever(items)
    combined_queries = pd.concat([fit_queries, valid_queries], ignore_index=True)
    _, candidate_pools = retriever.retrieve(combined_queries, return_pools=True)

    titles = [clean_text(x, 100) for x in items.item_title_raw]
    title_tokens = [set(TOKENS.findall(x)) for x in titles]
    title_len = np.log1p(np.array([len(x) for x in titles], dtype=np.float32))
    ratings = pd.to_numeric(items.item_rating, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    reviews = pd.to_numeric(items.item_rating_reviews_count, errors="coerce").fillna(0)
    log_reviews = np.log1p(reviews.to_numpy(dtype=np.float32))

    feature_pools = []
    for row, pool in zip(combined_queries.itertuples(index=False), candidate_pools):
        ids, components, local = pool
        query = clean_text(row.search_query)
        q_tokens = set(TOKENS.findall(query))
        coverage = np.array([len(q_tokens & title_tokens[i]) / max(len(q_tokens), 1)
                             for i in ids], dtype=np.float32)
        exact = np.array([query in titles[i] for i in ids], dtype=np.float32)
        x = np.column_stack([
            components,
            local.astype(np.float32), coverage, exact, ratings[ids],
            log_reviews[ids], title_len[ids],
            np.full(len(ids), len(q_tokens), dtype=np.float32),
            np.full(len(ids), bool(clean_text(row.search_infm_params_text)), dtype=np.float32),
        ]).astype(np.float32)
        feature_pools.append((row.signature, ids, local, x))
    LOG.info("Constructed %s feature pools", len(feature_pools))

    rng = np.random.default_rng(42)
    xfit, yfit = [], []
    for signature, ids, _, x in feature_pools[:len(fit_queries)]:
        positive = np.isin(retriever.item_ids[ids], list(fit_positives[signature]))
        baseline = x[:, :4] @ BASE_WEIGHTS
        hard = np.argsort(baseline)[-200:]
        remainder = np.setdiff1d(np.arange(len(ids)), hard, assume_unique=True)
        extra = rng.choice(remainder, size=min(50, len(remainder)), replace=False)
        chosen = np.unique(np.concatenate([hard, extra, np.flatnonzero(positive)]))
        xfit.append(x[chosen])
        yfit.append(positive[chosen])
    xfit = np.concatenate(xfit)
    yfit = np.concatenate(yfit)
    LOG.info("Training rows=%s positives=%s", len(yfit), int(yfit.sum()))

    model = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=5,
        loss_function="Logloss", random_seed=42, thread_count=4,
        verbose=100,
    )
    model.fit(xfit, yfit)
    valid_pools = feature_pools[len(fit_queries):]
    baseline_scores = [x[:, :4] @ BASE_WEIGHTS for _, _, _, x in valid_pools]
    model_scores = [model.predict_proba(x)[:, 1] for _, _, _, x in valid_pools]
    print("baseline", recall(valid_pools, baseline_scores, retriever.item_ids, valid_positives), flush=True)
    print("catboost", recall(valid_pools, model_scores, retriever.item_ids, valid_positives), flush=True)
    for alpha in [0.2, 0.4, 0.6, 0.8]:
        blended = [(1 - alpha) * rank_fraction(base) + alpha * rank_fraction(model_score)
                   for base, model_score in zip(baseline_scores, model_scores)]
        print("blend_rank_alpha", alpha,
              recall(valid_pools, blended, retriever.item_ids, valid_positives), flush=True)
    print("feature_importance", model.get_feature_importance().round(1).tolist(), flush=True)


if __name__ == "__main__":
    main()
