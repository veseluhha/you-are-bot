# You are bot — проект

Задача из соревнования [you-are-bot](https://stackmorelayers.ai/competitions/you-are-bot):
по диалогу двух участников предсказать вероятность того, что участник — бот. Метрика — LogLoss.

## Запуск

Сырые файлы соревнования (`train.json`, `ytrain.csv`, `test.json`, `ytest.csv`,
`sample_submission.csv`) лежат в `data/raw/`. Весь эксперимент запускается одной командой:

```bash
./run.sh
```

Скрипт ставит окружение из `uv.lock` (`uv sync --frozen`) и выполняет `run_experiment.py`:

1. подготовка данных: ручные признаки + перплексия LLM для каждой реплики (кэш в
   `data/processed/llm_cache.parquet`);
2. fine-tune rubert-tiny2 по 5 фолдам -> OOF-предсказания для стекинга;
3. все конфиги из `configs/*.yaml` с кросс-валидацией;
4. онлайн-эксперимент: качество по первым k репликам диалога.

Всё пишется в MLflow. Лучший по `cv_logloss` сабмит копируется в `outputs/submission.csv`,
сводная таблица — в `outputs/leaderboard.csv`.

Этапы 1–2 используют torch и рассчитаны на GPU (CUDA или Apple MPS). На Linux torch ставится
из индекса PyTorch под CUDA 12.6, на macOS — с PyPI. Запуск на сервере с несколькими GPU:

```bash
CUDA_VISIBLE_DEVICES=3 ./run.sh
```

Без GPU-этапов (только CPU, около 5 минут):

```bash
./run.sh --no-llm --no-transformer
```

Посмотреть эксперименты:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Отдельные этапы:

```bash
uv run python prepare_data.py                          # data/raw -> data/processed
uv run python train_transformer.py                     # fine-tune rubert-tiny2
uv run python train.py configs/11_pair_lgbm_swap.yaml  # одна модель
uv run python prefix_experiment.py                     # онлайн-постановка
uv run python eda.py                                   # графики EDA -> reports/
```

Для отдельных скриптов трекинг-сервер задаётся через `MLFLOW_TRACKING_URI`
(в `run.sh` по умолчанию `sqlite:///mlflow.db`).

## Подходы

| Конфиг | Что проверяем |
|---|---|
| `01`–`03` | бейзлайны: константа, ручные признаки, TF-IDF |
| `04`–`05` | признаки обоих участников; пересчёт вероятностей «в диалоге не больше одного бота» |
| `06`–`09` | LightGBM, блендинг с логрегом |
| `10`–`11` | **совместная модель пары**: объект — диалог, 3 исхода (нет бота / бот 0 / бот 1), ограничение встроено в постановку; `11` — аугментация перестановкой участников |
| `12`–`13` | **перплексия LLM (Binoculars)**: NLL реплики под Qwen2.5-0.5B и кросс-перплексия с Qwen2.5-0.5B-Instruct |
| `14`–`15` | **стекинг с дообученным rubert-tiny2**: трансформер читает весь диалог с маркерами «Я:» / «Собеседник:», его OOF-вероятности — признак модели пары |
| `prefix_online` | **онлайн-постановка**: качество по первым k репликам; обучение на полных диалогах против обучения с префиксами |

## Структура

| Путь | Что там |
|---|---|
| `src/youarebot/data.py` | чтение json/csv, разворот диалогов в строки «участник диалога» |
| `src/youarebot/features.py` | ручные признаки: стиль текста, эхо собеседника, самоповторы |
| `src/youarebot/llm_features.py` | перплексия и Binoculars-скор реплик, кэш по md5 текста |
| `src/youarebot/prepare.py` | этап подготовки данных, md5 сырых данных |
| `src/youarebot/models.py` | сборка sklearn-пайплайна по конфигу, пост-обработка пары |
| `src/youarebot/pair.py` | совместная модель пары (sklearn-совместимый классификатор) |
| `src/youarebot/transformer.py` | fine-tune rubert-tiny2 по фолдам |
| `src/youarebot/prefix.py` | онлайн-эксперимент на префиксах диалога |
| `src/youarebot/train.py` | CV по диалогам, финальная модель, сабмит, логирование в MLflow |
| `src/youarebot/utils.py` | конфиги, сиды, git-метаданные, выбор устройства |
| `configs/*.yaml`, `configs/transformer/` | один файл — один эксперимент |
| `run_experiment.py`, `run.sh` | end-to-end запуск |
| `eda.py`, `reports/` | EDA и графики |

Этапы на torch (перплексия, fine-tune) запускаются отдельными процессами: libomp из torch
и из lightgbm/sklearn в одном процессе на macOS дают segfault.

## Что логируется в MLflow

Эксперимент `you-are-bot`. Родительский run `pipeline`, у него дочерние run'ы: по одному на
конфиг, `transformer_rubert_tiny2` и `prefix_online`.

- параметры: весь конфиг в плоском виде (`model.params.C`, `features.own_text_tfidf.ngram_range`, …),
  размеры выборок, число признаков;
- теги: `git_commit`, `git_dirty`, `data_md5` (хеш сырых файлов), `config_file`, устройство для torch;
- метрики: `cv_logloss`, `cv_roc_auc`, `cv_accuracy`, `cv_brier`, `fold_logloss` по фолдам,
  те же метрики на трейне; для онлайн-эксперимента `{full,prefix}_logloss_k{k}`;
- артефакты: конфиг, `meta.json` данных (с именами LLM), OOF-предсказания, сабмит, график
  калибровки, важность признаков, модель (`model`, sklearn-пайплайн с сигнатурой), кривая
  качества по числу реплик.

Воспроизводимость: зависимости зафиксированы в `pyproject.toml` (точные версии) и `uv.lock`,
сид задаётся в конфиге и передаётся в CV-сплит, модели и torch, разбиение на фолды
группируется по `dialog_id`, чтобы оба участника одного диалога всегда были в одном фолде.
