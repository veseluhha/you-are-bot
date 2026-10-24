# You are bot — проект

Задача из соревнования [you-are-bot](https://stackmorelayers.ai/competitions/you-are-bot):
по диалогу двух участников предсказать вероятность того, что участник — бот. Метрика — LogLoss.

## Запуск

Сырые файлы соревнования (`train.json`, `ytrain.csv`, `test.json`, `ytest.csv`,
`sample_submission.csv`) лежат в `data/raw/`. Весь эксперимент запускается одной командой:

```bash
./run.sh
```

Скрипт ставит окружение из `uv.lock` (`uv sync --frozen`), готовит признаки, обучает
все конфиги из `configs/` с кросс-валидацией и пишет всё в MLflow. Лучший по `cv_logloss`
сабмит копируется в `outputs/submission.csv`, сводная таблица — в `outputs/leaderboard.csv`.

Посмотреть эксперименты:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Отдельные этапы:

```bash
uv run python prepare_data.py                                # data/raw -> data/processed
uv run python train.py configs/05_tfidf_handcrafted_logreg_pair.yaml   # одна модель
uv run python run_experiment.py --configs configs/06_*.yaml  # выбранные конфиги
uv run python eda.py                                         # графики EDA -> reports/
```

Для `train.py` трекинг-сервер задаётся через `MLFLOW_TRACKING_URI`
(в `run.sh` по умолчанию `sqlite:///mlflow.db`).

## Структура

| Путь | Что там |
|---|---|
| `src/youarebot/data.py` | чтение json/csv, разворот диалогов в строки «участник диалога» |
| `src/youarebot/features.py` | ручные признаки: стиль текста, эхо собеседника, самоповторы |
| `src/youarebot/prepare.py` | этап подготовки данных, md5 сырых данных |
| `src/youarebot/models.py` | сборка sklearn-пайплайна по конфигу, пост-обработка пары |
| `src/youarebot/train.py` | CV по диалогам, финальная модель, сабмит, логирование в MLflow |
| `configs/*.yaml` | один файл — один эксперимент |
| `run_experiment.py`, `run.sh` | end-to-end запуск |
| `eda.py`, `reports/` | EDA и графики |

## Что логируется в MLflow

Эксперимент `you-are-bot`. Родительский run `pipeline`, у него дочерние run'ы по конфигам.

- параметры: весь конфиг в плоском виде (`model.params.C`, `features.own_text_tfidf.ngram_range`, …),
  размеры выборок, число признаков;
- теги: `git_commit`, `git_dirty`, `data_md5` (хеш сырых файлов), `config_file`;
- метрики: `cv_logloss`, `cv_roc_auc`, `cv_accuracy`, `cv_brier`, `fold_logloss` по фолдам,
  те же метрики на трейне;
- артефакты: конфиг, `meta.json` данных, OOF-предсказания, сабмит, график калибровки,
  важность признаков, модель (`model`, sklearn-пайплайн с сигнатурой).

Воспроизводимость: зависимости зафиксированы в `pyproject.toml` (точные версии) и `uv.lock`,
сид задаётся в конфиге и передаётся в CV-сплит и модели, разбиение на фолды группируется
по `dialog_id`, чтобы оба участника одного диалога всегда были в одном фолде.
