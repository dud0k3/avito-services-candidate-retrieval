from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


LOG = logging.getLogger(__name__)
ITEM_COLS = [
    "item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
    "item_location_id", "item_microcat_id", "item_category_id", "item_rating",
]
QUERY_COLS = [
    "search_query", "search_location_id", "search_is_delivery_search",
    "search_infm_params_text", "search_category",
]


def clean_text(value: object, limit: int | None = None) -> str:
    """приводит текст к общему виду перед поиском."""
    if not isinstance(value, str):
        return ""
    value = value.casefold().replace("ё", "е").replace("\xa0", " ")
    return value[:limit] if limit is not None else value


def top_indices(indices: np.ndarray, scores: np.ndarray, limit: int) -> np.ndarray:
    # возвращаем номера объявлений с самыми высокими оценками.
    if limit <= 0:
        return indices[:0]
    if len(indices) <= limit:
        return indices[np.argsort(-scores, kind="stable")]
    keep = np.argpartition(scores, -limit)[-limit:]
    keep = keep[np.argsort(-scores[keep], kind="stable")]
    return indices[keep]


class CandidateRetriever:
    def __init__(self, items: pd.DataFrame):
        # сохраняем id и город каждого объявления.
        self.items = items.reset_index(drop=True).copy()
        self.item_ids = self.items.item_id.astype(str).to_numpy()
        self.locations = self.items.item_location_id.fillna(-1).to_numpy(dtype=np.int64)

        titles = [clean_text(x, 100) for x in self.items.item_title_raw]
        params = [clean_text(x, 400) for x in self.items.item_infm_params_text]
        descriptions = [clean_text(x, 650) for x in self.items.item_description_raw]
        full = [f"{t} {t} {p} {d}" for t, p, d in zip(titles, params, descriptions)]

        # заголовок часто сразу называет услугу, а полный текст помогает найти подробности.
        self.vectorizers = {
            "title_word": TfidfVectorizer(
                min_df=2, max_df=0.9, max_features=200_000,
                ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32,
            ),
            "title_char": TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                max_features=450_000, sublinear_tf=True, dtype=np.float32,
            ),
            "full_word": TfidfVectorizer(
                min_df=2, max_df=0.9, max_features=450_000,
                ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32,
            ),
            "params_word": TfidfVectorizer(
                min_df=2, max_df=0.9, max_features=250_000,
                ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32,
            ),
        }
        self.matrices = {}
        # строим отдельный поиск по каждому набору текстовых полей.
        for name, vectorizer in self.vectorizers.items():
            LOG.info("Vectorizing %s", name)
            texts = full if name == "full_word" else params if name == "params_word" else titles
            self.matrices[name] = vectorizer.fit_transform(texts).tocsr()
            LOG.info("%s: shape=%s, nnz=%s", name, self.matrices[name].shape, self.matrices[name].nnz)

    def retrieve(
        self,
        queries: pd.DataFrame,
        limit: int = 50,
        local_quota: int = 50,
        location_bonus: float = 0.0,
        source_limit: int = 150,
        return_pools: bool = False,
    ):
        # чистим текст запроса и готовим его для каждого текстового поиска.
        query_texts = [clean_text(x) for x in queries.search_query]
        filter_texts = [clean_text(x, 250) for x in queries.search_infm_params_text]
        q_matrices = {
            name: vectorizer.transform(filter_texts if name == "params_word" else query_texts).tocsr()
            for name, vectorizer in self.vectorizers.items()
        }
        weights = {"title_word": 0.285, "title_char": 0.240,
                   "full_word": 0.225, "params_word": 0.250}
        result: list[list[str]] = []
        pools = []

        for row_number, row in enumerate(queries.itertuples(index=False)):
            # собираем лучших кандидатов из каждого поиска.
            location = int(row.search_location_id)
            candidate_ids: set[int] = set()
            score_rows: dict[str, sparse.csr_matrix] = {}

            for name, item_matrix in self.matrices.items():
                similarities = (q_matrices[name].getrow(row_number) @ item_matrix.T).tocsr()
                score_rows[name] = similarities
                ids = similarities.indices
                scores = similarities.data
                if len(ids):
                    candidate_ids.update(top_indices(ids, scores, source_limit).tolist())
                    local = self.locations[ids] == location
                    if local.any():
                        candidate_ids.update(
                            top_indices(ids[local], scores[local], source_limit).tolist()
                        )

            if not candidate_ids:
                # если совпадений нет, возвращаем первые id из корпуса.
                result.append(self.item_ids[:limit].tolist())
                pools.append(None)
                continue

            ids = np.fromiter(sorted(candidate_ids), dtype=np.int32)
            components = np.column_stack([
                score_rows[name][:, ids].toarray().ravel()
                for name in weights
            ])
            is_local = self.locations[ids] == location
            pool = (ids, components, is_local)
            pools.append(pool)
            chosen = self.select(pool, local_quota, location_bonus, limit)
            result.append(self.item_ids[chosen].tolist())

            if (row_number + 1) % 200 == 0:
                LOG.info("Processed %s/%s queries", row_number + 1, len(queries))
        return (result, pools) if return_pools else result

    @staticmethod
    def select(pool, local_quota: int, location_bonus: float, limit: int = 50,
               weights=(0.285, 0.240, 0.225, 0.250)) -> np.ndarray:
        # складываем оценки и сначала выбираем объявления из нужного города.
        ids, components, is_local = pool
        combined = components @ np.asarray(weights, dtype=np.float32)
        combined += location_bonus * is_local
        # после местных объявлений добираем лучшие оставшиеся варианты.
        local_ids = top_indices(ids[is_local], combined[is_local], min(local_quota, limit))
        used = np.isin(ids, local_ids)
        remaining = top_indices(ids[~used], combined[~used], limit - len(local_ids))
        return np.concatenate([local_ids, remaining])[:limit]


def make_validation(train: pd.DataFrame, n_queries: int, seed: int):
    # делим одинаковые поисковые запросы на группы и выбираем часть для проверки.
    train = train.copy()
    # для одного запроса сохраняем все выбранные объявления без повторов.
    train["signature"] = pd.util.hash_pandas_object(
        train[QUERY_COLS].fillna(""), index=False
    ).to_numpy()
    unique_queries = train.drop_duplicates("signature")
    selected = unique_queries.sample(min(n_queries, len(unique_queries)), random_state=seed)
    selected_sigs = set(selected.signature)
    positives = defaultdict(set)
    for signature, item_id in zip(train.signature, train.item_id):
        if signature in selected_sigs:
            positives[signature].add(str(item_id))
    return selected.reset_index(drop=True), positives


def recall_at_50(queries: pd.DataFrame, predictions: list[list[str]], positives: dict) -> float:
    # считаем среднюю долю нужных объявлений среди первых пятидесяти.
    values = []
    for signature, guessed in zip(queries.signature, predictions):
        target = positives[signature]
        values.append(len(set(guessed) & target) / len(target))
    return float(np.mean(values))


def validate_output(queries: pd.DataFrame, items: pd.DataFrame, answer: pd.DataFrame):
    # проверяем колонки, id, повторы и число объявлений в ответе.
    if list(answer.columns) != ["query_id", "answer"]:
        raise ValueError("CSV must have exactly query_id and answer columns")
    expected = set(queries.query_id.astype(str))
    if len(answer) != len(queries) or set(answer.query_id) != expected:
        raise ValueError("Missing, extra or duplicate query_id")
    if not answer.query_id.map(lambda x: isinstance(x, str) and len(x) == 16).all():
        raise ValueError("query_id must be a 16-character string")
    valid_items = set(items.item_id.astype(str))
    for text in answer.answer:
        ids = text.split()
        if (len(ids) > 50 or len(ids) != len(set(ids)) or
                any(re.fullmatch(r"[0-9a-f]{16}", item_id) is None for item_id in ids) or
                not set(ids) <= valid_items):
            raise ValueError("Invalid item IDs in answer")


def main():
    # читаем запросы и объявления, затем ищем кандидатов.
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("answer.csv"))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validation-queries", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-quota", type=int, default=50)
    parser.add_argument("--location-bonus", type=float, default=0.0)
    parser.add_argument("--source-limit", type=int, default=150)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # при проверке берем метки из train, для ответа — запросы benchmark.
    if args.validate:
        columns = list(dict.fromkeys(QUERY_COLS + ITEM_COLS))
        train = pd.read_parquet(args.data_dir / "train.parquet", columns=columns)
        queries, positives = make_validation(train, args.validation_queries, args.seed)
        items = train[ITEM_COLS].drop_duplicates("item_id")
    else:
        queries = pd.read_parquet(args.data_dir / "benchmark_queries.parquet")
        items = pd.read_parquet(args.data_dir / "benchmark_items.parquet")

    retriever = CandidateRetriever(items)
    retrieved = retriever.retrieve(
        queries,
        local_quota=args.local_quota,
        location_bonus=args.location_bonus,
        source_limit=args.source_limit,
        return_pools=args.sweep or args.diagnose,
    )
    predictions, pools = retrieved if args.sweep or args.diagnose else (retrieved, None)
    if args.validate:
        # показываем качество поиска и при необходимости проверяем промахи.
        print(f"Recall@50: {recall_at_50(queries, predictions, positives):.6f}")
        if args.sweep:
            oracle = []
            for signature, pool in zip(queries.signature, pools):
                oracle.append(len(set(retriever.item_ids[pool[0]]) & positives[signature]) /
                              len(positives[signature]) if pool else 0.0)
            print(f"Recall in candidate union: {np.mean(oracle):.6f}")
            for quota in [35, 45, 50]:
                for filter_weight in [0.0, 0.05, 0.10, 0.15, 0.25, 0.40]:
                    text_weights = np.array([0.38, 0.32, 0.30]) * (1.0 - filter_weight)
                    weights = (*text_weights, filter_weight)
                    trial_predictions = [
                        retriever.item_ids[retriever.select(pool, quota, 0.0,
                                                            weights=weights)].tolist()
                        if pool else retriever.item_ids[:50].tolist()
                        for pool in pools
                    ]
                    print(f"quota={quota:2d} filter_weight={filter_weight:.2f} "
                          f"Recall@50={recall_at_50(queries, trial_predictions, positives):.6f}")
        if args.diagnose:
            item_locations = dict(zip(retriever.item_ids, retriever.locations))
            item_titles = dict(zip(retriever.item_ids, retriever.items.item_title_raw))
            for row, predicted, pool in zip(queries.itertuples(index=False), predictions, pools):
                target = positives[row.signature]
                missed = target - set(predicted)
                if not missed:
                    continue
                pool_ids = set(retriever.item_ids[pool[0]]) if pool else set()
                for item_id in missed:
                    reason = "ranked_out" if item_id in pool_ids else "not_retrieved"
                    same_city = item_locations[item_id] == row.search_location_id
                    print(f"MISS {reason} same_city={same_city} query={row.search_query!r} "
                          f"filters={row.search_infm_params_text!r} "
                          f"item={item_titles[item_id]!r}")
    else:
        # записываем готовые id в нужный csv формат.
        answer = pd.DataFrame({
            "query_id": queries.query_id.astype(str),
            "answer": [" ".join(p) for p in predictions],
        })
        validate_output(queries, items, answer)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        answer.to_csv(args.output, index=False)
        LOG.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
