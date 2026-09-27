"""дообучаем модель на запросах и выбранных объявлениях."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader

from retrieval import ITEM_COLS, QUERY_COLS, clean_text, make_validation


LOG = logging.getLogger(__name__)


def build_query(row) -> str:
    # добавляем фильтр к запросу, если пользователь его указал.
    query = clean_text(row.search_query)
    filters = clean_text(row.search_infm_params_text, 250)
    return f"{query} [filters] {filters}" if filters else query


def build_document(row) -> str:
    # для объявления берем заголовок и его параметры.
    title = clean_text(row.item_title_raw, 100)
    params = clean_text(row.item_infm_params_text, 300)
    return f"{title} [params] {params}" if params else title


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # читаем обучающие строки и оставляем отдельные запросы для проверки.
    cols = list(dict.fromkeys(QUERY_COLS + ITEM_COLS))
    train = pd.read_parquet(args.data_dir / "train.parquet", columns=cols)
    validation_queries, _ = make_validation(train, 600, args.seed)
    train["signature"] = pd.util.hash_pandas_object(
        train[QUERY_COLS].fillna(""), index=False
    ).to_numpy()
    validation_signatures = set(validation_queries.signature)

    fit_pairs = train[~train.signature.isin(validation_signatures)].copy()
    fit_pairs = fit_pairs.groupby("signature", sort=False, group_keys=False).sample(
        n=1, random_state=args.seed
    )
    # не повторяем одно объявление, чтобы пары в одной пачке не мешали обучению.
    fit_pairs = fit_pairs.drop_duplicates("item_id")
    fit_pairs = fit_pairs.sample(
        n=min(args.max_pairs, len(fit_pairs)), random_state=args.seed
    ).reset_index(drop=True)
    LOG.info("Fine-tuning on %s query/item pairs; validation contexts excluded",
             len(fit_pairs))

    # превращаем строки в пары текста для обучения.
    examples = [
        InputExample(texts=[build_query(row), build_document(row)])
        for row in fit_pairs.itertuples(index=False)
    ]
    # загружаем готовую открытую модель и настраиваем длину текста.
    model = SentenceTransformer(str(args.model_dir), device="cpu")
    model.max_seq_length = 64
    loss_fn = MultipleNegativesRankingLoss(model)
    loader = DataLoader(
        examples, batch_size=args.batch_size, shuffle=True,
        collate_fn=model.smart_batching_collate, num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup_steps,
                         max(0.0, (total_steps - step) / max(total_steps - warmup_steps, 1))),
    )

    # обучаем модель несколько раз по всем выбранным парам.
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(features, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += float(loss.detach())
            if global_step % 100 == 0:
                LOG.info("epoch=%s step=%s/%s loss=%.4f", epoch + 1,
                         global_step, total_steps, float(loss.detach()))
        LOG.info("Epoch %s complete; mean loss=%.4f", epoch + 1,
                 epoch_loss / max(len(loader), 1))

    # сохраняем модель, чтобы потом использовать ее для поиска.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output_dir))
    LOG.info("Saved fine-tuned MiniLM to %s", args.output_dir)


if __name__ == "__main__":
    main()
