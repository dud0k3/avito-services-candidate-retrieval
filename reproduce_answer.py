"""запускает поиск и проверяет совпадение с отправленным ответом."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "minilm_seed17"
EXPECTED = ROOT / "answer.csv"
DATA_HASHES = {
    "train.parquet": "e150ab7a5c98769f643b95ee3a02dc0f664b647facd69a7d280ec6d48ca554a7",
    "benchmark_items.parquet": "193b3a3961464620cbf8d8797152b7f90cae1826ca749fb7817a210984a09899",
    "benchmark_queries.parquet": "e49de4fb76f03979a4af96034817767d39ae5de9f94e254fed028243188dda06",
}
ANSWER_HASH = "7b73162eb481f3e7389ce737d4e77725e711b808f1fcdcd53dd083a32549021b"


def sha256(path: Path) -> str:
    # считаем хэш файла по частям, чтобы не загружать его целиком в память.
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    parser.add_argument("--output", type=Path, default=ROOT / "answer.reproduced.csv")
    args = parser.parse_args()

    # сначала проверяем, что на входе лежат исходные файлы задания.
    for name, expected_hash in DATA_HASHES.items():
        path = args.data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"не найден файл с данными: {path}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"не совпал хэш данных: {name}")

    # проверяем веса, токенизатор и настройки модели.
    manifest = MODEL_DIR / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(f"не найден список хэшей модели: {manifest}")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected_hash, name = line.split(maxsplit=1)
        path = MODEL_DIR / name.strip()
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"не совпал файл модели: {name.strip()}")
    if sha256(EXPECTED) != ANSWER_HASH:
        raise ValueError("сохраненный answer.csv отличается от проверенного файла")

    # собираем ответ тем же скриптом, который использовался для отправки.
    command = [
        sys.executable,
        str(ROOT / "predict_minilm_blend.py"),
        "--data-dir", str(args.data_dir),
        "--model-dir", str(MODEL_DIR),
        "--output", str(args.output),
        "--device", args.device,
    ]
    subprocess.run(command, check=True)

    # сверяем весь файл, включая порядок строк и объявлений.
    if sha256(args.output) != ANSWER_HASH:
        raise SystemExit(
            "ответ отличается от отправленного. проверь версии из requirements и устройство запуска."
        )
    print(f"готово: {args.output} полностью совпадает с answer.csv")


if __name__ == "__main__":
    main()
