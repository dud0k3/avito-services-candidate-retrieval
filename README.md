# Кандидатогенерация объявлений услуг

В тестовом задании нужно по короткому запросу вернуть до 50 `item_id` из корпуса объявлений. Главная метрика — Recall@50: доля объявлений, которые пользователь выбрал по этому запросу и которые попали в список кандидатов. Порядок внутри списка не влияет на метрику.

Я сравнил лексический поиск, BM25, готовую и дообученную MiniLM, а также CatBoost-ранжирование. Ключевой результат на отложенных данных: добавление дообученной MiniLM в пул кандидатов и признаки CatBoost подняло Recall@50 с **0,7537 до 0,7631** на 600 запросах (seed 17). Подробные условия, метрики и ограничения собраны в [EXPERIMENTS.md](EXPERIMENTS.md).

## Текущий рабочий вариант

Поиск строит четыре TF-IDF-сигнала. Каждый предлагает до 150 объявлений по всему корпусу и до 150 из локации запроса. CatBoost выбирает финальные 50 из объединённого пула.

1. Слова и пары слов в заголовке помогают находить прямое совпадение с названием услуги.
2. Символьные фрагменты длиной 3–5 символов помогают с опечатками, окончаниями и разным написанием.
3. Слова в заголовке, параметрах и начале описания помогают, если нужная формулировка есть не только в заголовке.
4. Фильтры запроса сопоставляются с параметрами объявления.

Помимо четырёх текстовых оценок, CatBoost получает совпадение локации, покрытие слов запроса заголовком, точное вхождение запроса, рейтинг, число отзывов и несколько простых признаков длины/наличия фильтра. Подробности и результаты по каждому источнику — в отчёте экспериментов.

## Запуск решения

Положите `train.parquet`, `benchmark_queries.parquet` и `benchmark_items.parquet` в один каталог. Данные в репозитории не хранятся.

```bash
python -m pip install -r requirements.txt
python retrieval.py --data-dir /path/to/data --validate
python validate_rerank.py --data-dir /path/to/data --train-queries 1200 --validation-queries 600 --seed 42
python rerank.py --data-dir /path/to/data --model reranker.cbm --output answer.csv
```

Последняя команда сохраняет CSV и проверяет число запросов, формат ID, отсутствие дубликатов и лимит в 50 кандидатов. Добавьте `--retrain`, чтобы заново обучить CatBoost.

## Эксперименты с MiniLM

Экспериментальные скрипты доступны отдельно от production-пайплайна. Они не меняют `answer.csv` и текущую модель CatBoost. Нужны локальные данные и необязательные зависимости:

```bash
python -m pip install -r requirements-transformers.txt
python finetune_minilm.py --data-dir /path/to/data \
  --model-dir sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --output-dir /path/to/finetuned-minilm --max-pairs 20000 \
  --epochs 2 --batch-size 32 --seed 17

python evaluate_finetuned_minilm.py --data-dir /path/to/data \
  --model-dir /path/to/finetuned-minilm \
  --embedding-file /path/to/cache/train_items.npy --n-queries 600 --seed 17

python validate_minilm_catboost.py --data-dir /path/to/data \
  --model-dir /path/to/finetuned-minilm \
  --embedding-file /path/to/cache/train_items.npy \
  --train-queries 1200 --validation-queries 600 --seed 17
```

Первый скрипт дообучает SentenceTransformer на парах «запрос — выбранное объявление», второй считает dense retrieval, третий сравнивает TF-IDF + CatBoost с TF-IDF ∪ MiniLM + CatBoost на одном и том же holdout. Веса модели и рассчитанные эмбеддинги не включены в Git: они занимают сотни мегабайт и строятся локально. Скрипты используют открытый `paraphrase-multilingual-MiniLM-L12-v2`; после загрузки весов обращения к внешним API нет.

## Ограничения

- Все числа — локальные оценки по выбранным взаимодействиям из `train.parquet`, не скрытый результат Stepik.
- Seed 17 проверен на 600 контекстах. Для вывода о стабильном приросте нужны дополнительные разбиения.
- В тесте все подходящие объявления есть в `benchmark_items.parquet`; локальная проверка использует 344 825 объявлений train-корпуса. Эти корпуса отличаются.
- Клик/выбор объявления — доступная разметка релевантности, но она отражает поведение пользователей и не покрывает все подходящие объявления.
- Список ошибок и честное сравнение всех вариантов приведены в [EXPERIMENTS.md](EXPERIMENTS.md).
