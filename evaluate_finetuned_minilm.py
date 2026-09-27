"""Evaluate a fine-tuned MiniLM dense retriever on held-out Avito queries."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from finetune_minilm import build_document, build_query
from retrieval import (
    ITEM_COLS, QUERY_COLS, CandidateRetriever, clean_text, make_validation,
    recall_at_50, top_indices,
)


LOG = logging.getLogger(__name__)
BASE_WEIGHTS = np.array([0.285, 0.240, 0.225, 0.250], dtype=np.float32)


def rank_fraction(scores):
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = np.arange(len(scores), dtype=np.float32)
    return ranks / max(len(scores) - 1, 1)


def recall_candidates(pools, item_ids, positives):
    vals = []
    for query, ids, _, _ in pools:
        relevant = positives[query]
        vals.append(len(set(item_ids[ids]) & relevant) / len(relevant))
    return float(np.mean(vals))


def select_local(ids, scores, local, limit=50):
    selected = top_indices(ids[local], scores[local], limit)
    if len(selected) < limit:
        rest = top_indices(ids[~local], scores[~local], limit - len(selected))
        selected = np.concatenate([selected, rest])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--embedding-file", type=Path, required=True)
    parser.add_argument("--n-queries", type=int, default=600)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    torch.set_num_threads(8)
    columns = list(dict.fromkeys(QUERY_COLS + ITEM_COLS))
    train = pd.read_parquet(args.data_dir / "train.parquet", columns=columns)
    queries, positives = make_validation(train, args.n_queries, args.seed)
    items = train.drop_duplicates("item_id").reset_index(drop=True)
    retriever = CandidateRetriever(items)
    item_ids = retriever.item_ids
    n_items = len(items)
    model = SentenceTransformer(str(args.model_dir), device=args.device)
    model.max_seq_length = 64

    args.embedding_file.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.lib.format.open_memmap(
        args.embedding_file, mode="w+", dtype=np.float16,
        shape=(n_items, model.get_embedding_dimension()),
    )
    documents = [build_document(row) for row in items.itertuples(index=False)]
    for start in range(0, n_items, 4096):
        end = min(start + 4096, n_items)
        vectors = model.encode(
            documents[start:end], batch_size=128, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        embeddings[start:end] = vectors.astype(np.float16)
        if (end % 20_000 < 4096) or end == n_items:
            embeddings.flush()
            LOG.info("Encoded %s/%s corpus items", end, n_items)
    del documents
    embeddings.flush()

    query_texts = [build_query(row) for row in queries.itertuples(index=False)]
    query_vectors = model.encode(
        query_texts, batch_size=128, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype(np.float32)
    item_tensor = torch.tensor(np.asarray(embeddings), device=args.device, dtype=torch.float16)
    _, lexical_pools = retriever.retrieve(queries, return_pools=True)
    q_matrices = {
        name: vectorizer.transform(
            [clean_text(x, 250) for x in queries.search_infm_params_text]
            if name == "params_word" else
            [clean_text(x) for x in queries.search_query]
        ).tocsr()
        for name, vectorizer in retriever.vectorizers.items()
    }

    pools, cosine_scores, lexical_scores = [], [], []
    dense_only_predictions, lexical_predictions = [], []
    for query_index, query in enumerate(queries.itertuples(index=False)):
        q = torch.tensor(query_vectors[query_index], device=args.device, dtype=torch.float16)
        all_scores = item_tensor @ q
        global_ids = torch.topk(all_scores, min(150, n_items)).indices.cpu().numpy()
        local_np = np.flatnonzero(retriever.locations == int(query.search_location_id))
        if len(local_np):
            local_tensor_ids = torch.tensor(local_np, device=args.device)
            local_scores = all_scores[local_tensor_ids]
            k = min(150, len(local_np))
            local_ids = local_np[torch.topk(local_scores, k).indices.cpu().numpy()]
        else:
            local_ids = np.array([], dtype=np.int32)

        base = lexical_pools[query_index]
        base_ids = base[0] if base is not None else np.array([], dtype=np.int32)
        ids = np.unique(np.concatenate([base_ids, global_ids, local_ids])).astype(np.int32)
        local = retriever.locations[ids] == int(query.search_location_id)
        components = np.column_stack([
            (q_matrices[name].getrow(query_index) @ retriever.matrices[name][ids].T)
            .toarray().ravel()
            for name in retriever.vectorizers
        ])
        dense = np.asarray(embeddings[ids], dtype=np.float32) @ query_vectors[query_index]
        lexical = components @ BASE_WEIGHTS
        pools.append((query.signature, ids, local, components))
        cosine_scores.append(dense)
        lexical_scores.append(lexical)
        dense_only_predictions.append(
            item_ids[select_local(ids, dense, local)].tolist()
        )
        lexical_ids = (retriever.item_ids[retriever.select(base, 50, 0.0)].tolist()
                       if base is not None else retriever.item_ids[:50].tolist())
        lexical_predictions.append(lexical_ids)
        if (query_index + 1) % 100 == 0:
            LOG.info("Scored dense retrieval for %s/%s queries",
                     query_index + 1, len(queries))

    print("tfidf_recall", recall_at_50(queries, lexical_predictions, positives), flush=True)
    print("finetuned_dense_recall", recall_at_50(queries, dense_only_predictions, positives), flush=True)
    print("tfidf_dense_union_candidate_recall",
          recall_candidates(pools, item_ids, positives), flush=True)
    for dense_weight in [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]:
        predictions = []
        for (signature, ids, local, _), lex, dense in zip(pools, lexical_scores, cosine_scores):
            combined = (1 - dense_weight) * rank_fraction(lex) + dense_weight * rank_fraction(dense)
            chosen = select_local(ids, combined, local)
            predictions.append(item_ids[chosen].tolist())
        print("tfidf_finetuned_dense_rank_blend", dense_weight,
              recall_at_50(queries, predictions, positives), flush=True)


if __name__ == "__main__":
    main()
