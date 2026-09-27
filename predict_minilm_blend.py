"""создаем ответы, смешивая обычный поиск и модель."""

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
    ITEM_COLS, CandidateRetriever, clean_text, top_indices, validate_output,
)


LOG = logging.getLogger(__name__)
LEXICAL_WEIGHTS = np.array([0.285, 0.240, 0.225, 0.250], dtype=np.float32)
DENSE_WEIGHT = 0.60


def rank_fraction(scores: np.ndarray) -> np.ndarray:
    # ставим каждому объявлению место в списке: выше оценка — лучше место.
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = np.arange(len(scores), dtype=np.float32)
    return ranks / max(len(scores) - 1, 1)


def select_local(ids: np.ndarray, scores: np.ndarray, local: np.ndarray) -> np.ndarray:
    # сначала выбираем объявления из города запроса, потом добираем другие.
    selected = top_indices(ids[local], scores[local], 50)
    if len(selected) < 50:
        other = top_indices(ids[~local], scores[~local], 50 - len(selected))
        selected = np.concatenate([selected, other])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Fine-tuned MiniLM directory created by finetune_minilm.py")
    parser.add_argument("--output", type=Path, default=Path("answer.csv"))
    parser.add_argument("--device", default="mps", help="For example: mps, cuda, or cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(8)
    # читаем запросы и объявления, для которых нужно составить ответ.
    queries = pd.read_parquet(args.data_dir / "benchmark_queries.parquet")
    items = pd.read_parquet(args.data_dir / "benchmark_items.parquet", columns=ITEM_COLS)
    retriever = CandidateRetriever(items)

    # готовим модель и считаем векторы заголовков и параметров объявлений.
    model = SentenceTransformer(str(args.model_dir), device=args.device)
    model.max_seq_length = 64
    documents = [build_document(row) for row in items.itertuples(index=False)]
    item_vectors = model.encode(
        documents, batch_size=128, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    ).astype(np.float16)
    query_vectors = model.encode(
        [build_query(row) for row in queries.itertuples(index=False)],
        batch_size=128, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    ).astype(np.float32)
    del model, documents

    # добавляем кандидатов из поиска по словам.
    LOG.info("Finding lexical candidates for %s queries", len(queries))
    _, lexical_pools = retriever.retrieve(queries, return_pools=True)
    query_matrices = {
        name: vectorizer.transform(
            [clean_text(x, 250) for x in queries.search_infm_params_text]
            if name == "params_word" else
            [clean_text(x) for x in queries.search_query]
        ).tocsr()
        for name, vectorizer in retriever.vectorizers.items()
    }
    item_tensor = torch.tensor(item_vectors, device=args.device, dtype=torch.float16)
    # соединяем два списка кандидатов и готовим ответы для запросов.
    predictions: list[list[str]] = []

    for query_index, query in enumerate(queries.itertuples(index=False)):
        query_tensor = torch.tensor(query_vectors[query_index], device=args.device,
                                    dtype=torch.float16)
        dense_all = item_tensor @ query_tensor
        global_ids = torch.topk(dense_all, min(150, len(items))).indices.cpu().numpy()
        local_pool = np.flatnonzero(
            retriever.locations == int(query.search_location_id)
        )
        if len(local_pool):
            local_tensor = torch.tensor(local_pool, device=args.device)
            local_ids = local_pool[
                torch.topk(dense_all[local_tensor], min(150, len(local_pool)))
                .indices.cpu().numpy()
            ]
        else:
            local_ids = np.array([], dtype=np.int32)

        lexical_pool = lexical_pools[query_index]
        lexical_ids = (lexical_pool[0] if lexical_pool is not None
                       else np.array([], dtype=np.int32))
        ids = np.unique(np.concatenate([lexical_ids, global_ids, local_ids])).astype(np.int32)
        local = retriever.locations[ids] == int(query.search_location_id)
        components = np.column_stack([
            (query_matrices[name].getrow(query_index) @ retriever.matrices[name][ids].T)
            .toarray().ravel()
            for name in retriever.vectorizers
        ]).astype(np.float32)
        lexical_score = components @ LEXICAL_WEIGHTS
        dense_score = item_vectors[ids].astype(np.float32) @ query_vectors[query_index]
        blended_score = (
            (1.0 - DENSE_WEIGHT) * rank_fraction(lexical_score)
            + DENSE_WEIGHT * rank_fraction(dense_score)
        )
        chosen = select_local(ids, blended_score, local)
        predictions.append(retriever.item_ids[chosen].tolist())

        if (query_index + 1) % 100 == 0:
            LOG.info("Processed %s/%s queries", query_index + 1, len(queries))

    # собираем таблицу и проверяем ее перед сохранением.
    answer = pd.DataFrame({
        "query_id": queries.query_id.astype(str),
        "answer": [" ".join(ids) for ids in predictions],
    })
    validate_output(queries, items, answer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    answer.to_csv(args.output, index=False)
    LOG.info("Wrote validated answer to %s", args.output)


if __name__ == "__main__":
    main()
