# Сохраненная MiniLM

Это веса модели, которыми был собран приложенный `answer.csv`.

- исходная модель: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- исходная версия на Hugging Face: `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`
- обучение: 20 000 пар, 2 эпохи, batch size 32, seed 17, Multiple Negatives Ranking Loss
- размер вектора: 384
- SHA-256 файла `model.safetensors`: `1627486fae2a9877fa0da572b6bf9a594640757e5dc29f8519e96973053d3cee`
- лицензия исходной модели: Apache-2.0

Веса сохранены в Git LFS. При клонировании установите Git LFS, чтобы получить сам файл, а не короткий текстовый указатель.
