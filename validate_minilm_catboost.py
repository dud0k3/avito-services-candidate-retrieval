from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer

from finetune_minilm import build_query
from retrieval import (
    ITEM_COLS, QUERY_COLS, CandidateRetriever, clean_text, make_validation,
    recall_at_50, top_indices,
)
from rerank import BASE_WEIGHTS, EXTRA_ITEM_COLS, FeatureBuilder


LOG = logging.getLogger(__name__)
FEATURE_NAMES = [
    "title_word_tfidf", "title_char_tfidf", "full_text_tfidf", "params_tfidf",
    "same_location", "query_title_token_coverage", "exact_query_in_title",
    "rating", "log_reviews", "log_title_length", "query_token_count",
    "has_search_filter", "minilm_cosine", "minilm_rank_percentile", "minilm_source_hit",
]


def select_local(ids, scores, local, limit=50):
    selected = top_indices(ids[local], scores[local], limit)
    if len(selected) < limit:
        rest = top_indices(ids[~local], scores[~local], limit - len(selected))
        selected = np.concatenate([selected, rest])
    return selected


def recall_pools(pools, predictions, item_ids, positives):
    values = []
    for (signature, ids, local, _), scores in zip(pools, predictions):
        chosen = select_local(ids, scores, local)
        relevant = positives[signature]
        values.append(len(set(item_ids[chosen]) & relevant) / len(relevant))
    return float(np.mean(values))


def oracle_recall(pools, item_ids, positives):
    values = []
    for signature, ids, _, _ in pools:
        relevant = positives[signature]
        values.append(len(set(item_ids[ids]) & relevant) / len(relevant))
    return float(np.mean(values))


def rank_fraction(scores):
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = np.arange(len(scores), dtype=np.float32)
    return ranks / max(len(scores) - 1, 1)


def fit_reranker(pools, positives, item_ids, seed=42):
    rng = np.random.default_rng(seed)
    x_parts, y_parts = [], []
    for signature, ids, _, x in pools:
        positive = np.isin(item_ids[ids], list(positives[signature]))
        # Hard negatives are selected with the best retrieval blend measured
        # before this reranker experiment (0.4 lexical rank + 0.6 dense rank).
        lexical = x[:, :4] @ BASE_WEIGHTS
        lex_rank = rank_fraction(lexical)
        if x.shape[1] > 12:
            dense_rank = rank_fraction(x[:, 12])
            hard_score = 0.4 * lex_rank + 0.6 * dense_rank
        else:
            hard_score = lex_rank
        hard = np.argsort(hard_score, kind="stable")[-200:]
        remainder = np.setdiff1d(np.arange(len(ids)), hard, assume_unique=True)
        random_neg = rng.choice(remainder, size=min(50, len(remainder)), replace=False)
        chosen = np.unique(np.concatenate([hard, random_neg, np.flatnonzero(positive)]))
        x_parts.append(x[chosen])
        y_parts.append(positive[chosen])
    x_train = np.concatenate(x_parts).astype(np.float32, copy=False)
    y_train = np.concatenate(y_parts)
    LOG.info("Reranker rows=%s, positives=%s", len(y_train), int(y_train.sum()))
    model = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=5,
        loss_function="Logloss", random_seed=seed, thread_count=4,
        allow_writing_files=False, verbose=False,
    )
    model.fit(x_train, y_train)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--embedding-file", type=Path, required=True)
    parser.add_argument("--train-queries", type=int, default=1200)
    parser.add_argument("--validation-queries", type=int, default=600)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(8)

    columns = list(dict.fromkeys(QUERY_COLS + ITEM_COLS + EXTRA_ITEM_COLS))
    train = pd.read_parquet(args.data_dir / "train.parquet", columns=columns)
    valid_queries, valid_positives = make_validation(
        train, args.validation_queries, args.seed
    )
    train["signature"] = pd.util.hash_pandas_object(
        train[QUERY_COLS].fillna(""), index=False
    ).to_numpy()
    fit_queries = train.drop_duplicates("signature")
    fit_queries = fit_queries[~fit_queries.signature.isin(valid_queries.signature)]
    fit_queries = fit_queries.sample(args.train_queries, random_state=2026).reset_index(drop=True)
    fit_signatures = set(fit_queries.signature)
    fit_positives = defaultdict(set)
    for signature, item_id in zip(train.signature, train.item_id):
        if signature in fit_signatures:
            fit_positives[signature].add(str(item_id))

    items = train.drop_duplicates("item_id").reset_index(drop=True)
    retriever = CandidateRetriever(items)
    embeddings = np.load(args.embedding_file, mmap_mode="r")
    if len(embeddings) != len(items):
        raise ValueError("Embedding rows do not match deduplicated train item count")
    if embeddings.shape[1] != 384:
        raise ValueError(f"Expected MiniLM dimension 384, got {embeddings.shape[1]}")

    model = SentenceTransformer(str(args.model_dir), device=args.device)
    model.max_seq_length = 64
    all_queries = pd.concat([fit_queries, valid_queries], ignore_index=True)
    query_texts = [build_query(row) for row in all_queries.itertuples(index=False)]
    query_vectors = model.encode(
        query_texts, batch_size=128, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype(np.float32)
    del model
    item_tensor = torch.tensor(np.asarray(embeddings), device=args.device, dtype=torch.float16)
    LOG.info("Retrieving lexical candidates for %s contexts", len(all_queries))
    _, lexical_pools = retriever.retrieve(all_queries, return_pools=True)
    feature_builder = FeatureBuilder(items)
    query_matrices = {
        name: vectorizer.transform(
            [clean_text(value, 250) for value in all_queries.search_infm_params_text]
            if name == "params_word" else
            [clean_text(value) for value in all_queries.search_query]
        ).tocsr()
        for name, vectorizer in retriever.vectorizers.items()
    }

    lexical_feature_pools, hybrid_feature_pools = [], []
    lexical_only_preds, mini_only_preds, hybrid_blend_preds = [], [], []
    for q_index, (query, lexical_pool) in enumerate(
        zip(all_queries.itertuples(index=False), lexical_pools)
    ):
        qvec = torch.tensor(query_vectors[q_index], device=args.device, dtype=torch.float16)
        all_scores = item_tensor @ qvec
        global_ids = torch.topk(all_scores, min(150, len(items))).indices.cpu().numpy()
        local_np = np.flatnonzero(retriever.locations == int(query.search_location_id))
        if len(local_np):
            local_ids_t = torch.tensor(local_np, device=args.device)
            local_scores = all_scores[local_ids_t]
            local_ids = local_np[torch.topk(local_scores, min(150, len(local_np))).indices.cpu().numpy()]
        else:
            local_ids = np.array([], dtype=np.int32)
        dense_source_ids = np.unique(np.concatenate([global_ids, local_ids])).astype(np.int32)

        base_ids = lexical_pool[0] if lexical_pool is not None else np.array([], dtype=np.int32)
        union_ids = np.union1d(base_ids, dense_source_ids).astype(np.int32)
        local_mask = retriever.locations[union_ids] == int(query.search_location_id)
        components = np.column_stack([
            (query_matrices[name].getrow(q_index) @ retriever.matrices[name][union_ids].T)
            .toarray().ravel()
            for name in retriever.vectorizers
        ]).astype(np.float32)
        base_features = feature_builder.transform(query, (union_ids, components, local_mask))
        dense_scores = np.asarray(embeddings[union_ids], dtype=np.float32) @ query_vectors[q_index]
        dense_ranks = np.argsort(np.argsort(dense_scores, kind="stable"), kind="stable")
        dense_ranks = dense_ranks.astype(np.float32) / max(len(union_ids) - 1, 1)
        dense_hit = np.isin(union_ids, dense_source_ids, assume_unique=True).astype(np.float32)
        hybrid_x = np.column_stack([base_features, dense_scores, dense_ranks, dense_hit]).astype(np.float32)

        # The lexical model is trained and measured on the exact lexical pool.
        if len(base_ids):
            positions = np.searchsorted(union_ids, base_ids)
            lexical_x = hybrid_x[positions, :12]
            base_local = retriever.locations[base_ids] == int(query.search_location_id)
            lexical_feature_pools.append((query.signature, base_ids, base_local, lexical_x))
        else:
            lexical_feature_pools.append((query.signature, base_ids, np.array([], dtype=bool), np.empty((0, 12), dtype=np.float32)))
        hybrid_feature_pools.append((query.signature, union_ids, local_mask, hybrid_x))

        if q_index >= len(fit_queries):
            lex_scores = components @ BASE_WEIGHTS
            mini_only_preds.append(retriever.item_ids[select_local(union_ids, dense_scores, local_mask)].tolist())
            blend_score = 0.4 * rank_fraction(lex_scores) + 0.6 * rank_fraction(dense_scores)
            hybrid_blend_preds.append(retriever.item_ids[select_local(union_ids, blend_score, local_mask)].tolist())
            base_ids = lexical_feature_pools[-1][1]
            base_x = lexical_feature_pools[-1][3]
            base_local = lexical_feature_pools[-1][2]
            lexical_only_preds.append(retriever.item_ids[select_local(base_ids, base_x[:, :4] @ BASE_WEIGHTS, base_local)].tolist())
        if (q_index + 1) % 200 == 0:
            LOG.info("Built hybrid features for %s/%s contexts", q_index + 1, len(all_queries))

    fit_lex = lexical_feature_pools[:len(fit_queries)]
    fit_hybrid = hybrid_feature_pools[:len(fit_queries)]
    valid_lex = lexical_feature_pools[len(fit_queries):]
    valid_hybrid = hybrid_feature_pools[len(fit_queries):]
    base_model = fit_reranker(fit_lex, fit_positives, retriever.item_ids)
    hybrid_model = fit_reranker(fit_hybrid, fit_positives, retriever.item_ids)

    base_scores = [base_model.predict_proba(x)[:, 1] for _, _, _, x in valid_lex]
    hybrid_scores = [hybrid_model.predict_proba(x)[:, 1] for _, _, _, x in valid_hybrid]
    base_recall = recall_pools(valid_lex, base_scores, retriever.item_ids, valid_positives)
    hybrid_recall = recall_pools(valid_hybrid, hybrid_scores, retriever.item_ids, valid_positives)

    print("validation_queries", len(valid_queries), flush=True)
    print("train_queries", len(fit_queries), flush=True)
    print("tfidf_retrieval_recall", recall_at_50(valid_queries, lexical_only_preds, valid_positives), flush=True)
    print("tfidf_candidate_oracle_recall", oracle_recall(valid_lex, retriever.item_ids, valid_positives), flush=True)
    print("tfidf_plus_catboost_recall", base_recall, flush=True)
    print("tfidf_minilm_union_candidate_oracle_recall", oracle_recall(valid_hybrid, retriever.item_ids, valid_positives), flush=True)
    print("tfidf_minilm_union_retrieval_blend_recall", recall_at_50(valid_queries, hybrid_blend_preds, valid_positives), flush=True)
    print("minilm_only_recall", recall_at_50(valid_queries, mini_only_preds, valid_positives), flush=True)
    print("tfidf_minilm_union_plus_catboost_recall", hybrid_recall, flush=True)
    print("catboost_delta_pp", (hybrid_recall - base_recall) * 100, flush=True)
    print("hybrid_feature_importance", dict(zip(FEATURE_NAMES, hybrid_model.get_feature_importance().round(2).tolist())), flush=True)


if __name__ == "__main__":
    main()
