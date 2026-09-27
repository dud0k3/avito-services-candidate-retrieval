"""Candidate retrieval followed by a locally trained CatBoost reranker.

Run with the supplied Parquet files and no network access. The model file is
saved separately, so a second run can use exactly the same trained model.
"""

from __future__ import annotations

import argparse
import gc
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from retrieval import (
    ITEM_COLS, QUERY_COLS, CandidateRetriever, clean_text, top_indices,
    validate_output,
)


LOG = logging.getLogger(__name__)
TOKENS = re.compile(r"[a-zа-я0-9]+")
BASE_WEIGHTS = np.array([0.285, 0.240, 0.225, 0.250], dtype=np.float32)
EXTRA_ITEM_COLS = ["item_rating_reviews_count"]


class FeatureBuilder:
    """Calculate the same query-item features for train and benchmark ads."""

    def __init__(self, items: pd.DataFrame):
        titles = [clean_text(x, 100) for x in items.item_title_raw]
        self.titles = titles
        self.title_tokens = [set(TOKENS.findall(x)) for x in titles]
        self.title_length = np.log1p(np.array([len(x) for x in titles], dtype=np.float32))
        self.ratings = pd.to_numeric(items.item_rating, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        reviews = pd.to_numeric(items.item_rating_reviews_count, errors="coerce").fillna(0)
        self.log_reviews = np.log1p(reviews.to_numpy(dtype=np.float32))

    def transform(self, query, pool) -> np.ndarray:
        ids, similarities, same_location = pool
        query_text = clean_text(query.search_query)
        query_tokens = set(TOKENS.findall(query_text))
        coverage = np.array([
            len(query_tokens & self.title_tokens[i]) / max(len(query_tokens), 1)
            for i in ids
        ], dtype=np.float32)
        exact = np.array([query_text in self.titles[i] for i in ids], dtype=np.float32)
        return np.column_stack([
            similarities,
            same_location.astype(np.float32), coverage, exact,
            self.ratings[ids], self.log_reviews[ids], self.title_length[ids],
            np.full(len(ids), len(query_tokens), dtype=np.float32),
            np.full(len(ids), bool(clean_text(query.search_infm_params_text)), dtype=np.float32),
        ]).astype(np.float32)


def fit_model(data_dir: Path, model_path: Path, n_contexts: int = 1200) -> CatBoostClassifier:
    """Learn from clicked query-item pairs, with hard negatives from retrieval."""
    columns = list(dict.fromkeys(QUERY_COLS + ITEM_COLS + EXTRA_ITEM_COLS))
    train = pd.read_parquet(data_dir / "train.parquet", columns=columns)
    train["signature"] = pd.util.hash_pandas_object(
        train[QUERY_COLS].fillna(""), index=False
    ).to_numpy()
    unique_queries = train.drop_duplicates("signature")
    fit_queries = unique_queries.sample(min(n_contexts, len(unique_queries)),
                                        random_state=2026).reset_index(drop=True)
    fit_signatures = set(fit_queries.signature)
    positives = defaultdict(set)
    for signature, item_id in zip(train.signature, train.item_id):
        if signature in fit_signatures:
            positives[signature].add(str(item_id))

    items = train[ITEM_COLS + EXTRA_ITEM_COLS].drop_duplicates("item_id").reset_index(drop=True)
    retriever = CandidateRetriever(items)
    _, pools = retriever.retrieve(fit_queries, return_pools=True)
    features = FeatureBuilder(items)

    random = np.random.default_rng(42)
    x_train, y_train = [], []
    for query, pool in zip(fit_queries.itertuples(index=False), pools):
        if pool is None:
            continue
        ids = pool[0]
        x = features.transform(query, pool)
        positive = np.isin(retriever.item_ids[ids], list(positives[query.signature]))
        baseline = x[:, :4] @ BASE_WEIGHTS
        hard_negatives = np.argsort(baseline)[-200:]
        remaining = np.setdiff1d(np.arange(len(ids)), hard_negatives, assume_unique=True)
        random_negatives = random.choice(remaining, size=min(50, len(remaining)), replace=False)
        chosen = np.unique(np.concatenate([hard_negatives, random_negatives,
                                           np.flatnonzero(positive)]))
        x_train.append(x[chosen])
        y_train.append(positive[chosen])
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    LOG.info("Reranker training rows=%s, positives=%s", len(y_train), int(y_train.sum()))

    model = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=5,
        loss_function="Logloss", random_seed=42, thread_count=4, verbose=False,
    )
    model.fit(x_train, y_train)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    LOG.info("Saved reranker to %s", model_path)
    return model


def predict(data_dir: Path, model: CatBoostClassifier, output: Path) -> None:
    """Retrieve ads for each benchmark query and rerank up to 50 of them."""
    queries = pd.read_parquet(data_dir / "benchmark_queries.parquet")
    items = pd.read_parquet(data_dir / "benchmark_items.parquet")
    retriever = CandidateRetriever(items)
    _, pools = retriever.retrieve(queries, return_pools=True)
    features = FeatureBuilder(items)
    predictions = []

    for row_number, (query, pool) in enumerate(zip(queries.itertuples(index=False), pools)):
        if pool is None:
            predictions.append(retriever.item_ids[:50].tolist())
            continue
        ids, _, same_location = pool
        x = features.transform(query, pool)
        model_score = model.predict_proba(x)[:, 1]
        chosen = top_indices(ids[same_location], model_score[same_location], 50)
        if len(chosen) < 50:
            remaining = top_indices(ids[~same_location], model_score[~same_location], 50 - len(chosen))
            chosen = np.concatenate([chosen, remaining])
        predictions.append(retriever.item_ids[chosen].tolist())
        if (row_number + 1) % 200 == 0:
            LOG.info("Reranked %s/%s queries", row_number + 1, len(queries))

    answer = pd.DataFrame({
        "query_id": queries.query_id.astype(str),
        "answer": [" ".join(ids) for ids in predictions],
    })
    validate_output(queries, items, answer)
    output.parent.mkdir(parents=True, exist_ok=True)
    answer.to_csv(output, index=False)
    LOG.info("Wrote %s", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("reranker.cbm"))
    parser.add_argument("--output", type=Path, default=Path("answer.csv"))
    parser.add_argument("--train-queries", type=int, default=1200)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.retrain or not args.model.exists():
        model = fit_model(args.data_dir, args.model, args.train_queries)
        gc.collect()
    else:
        model = CatBoostClassifier()
        model.load_model(str(args.model))
        LOG.info("Loaded reranker from %s", args.model)
    predict(args.data_dir, model, args.output)


if __name__ == "__main__":
    main()
