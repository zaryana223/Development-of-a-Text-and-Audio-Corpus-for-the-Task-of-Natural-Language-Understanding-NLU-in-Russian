# Development of a Text and Audio Corpus for Natural Language Understanding (NLU) in Russian

**Разработка текстового и аудио корпуса для задачи понимания естественного языка (NLU) на русском языке**

Выпускная квалификационная работа, НИУ ВШЭ, 2026 — **Дамашова Заряна Алексеевна**

Репозиторий: подготовка **обучающих данных** (перевод, адаптация), **код метрик NLU**, **таблицы ошибок** из экспериментов ВКР.

Связанные репозитории: [XSID-ru-NLP](https://github.com/zaryana223/XSID-ru-NLP) (бенчмарк), [NLU](https://github.com/zaryana223/NLU) (MaChAmp, обучение).

---

## Структура репозитория

### Предобработка и обучающие данные

| Файл | Назначение |
|------|------------|
| `clean_up.ipynb` | Очистка и нормализация английского XSID |
| `translation.py` | Перевод тренировочного корпуса на русский (LLM API) |
| `adaptation.ipynb` | Культурная адаптация сущностей |
| `en.train.reference.conll` | Очищенный английский train |
| `en.train.unique_ids.conll` | Версия для перевода (без слотов в токенах) |
| `ru.train.conll` | Переведённый русский train (~37k) |
| `ru.train_adapt.conll` | Адаптированная версия |
| `ru.train.unique_ids.conll` | Уникальные id для train |

### Метрики NLU (`nlu_metrics/`)

Код для оценки **intent accuracy/F1**, **Span F1**, **Joint**, для GigaAM — **Slot F1 (/N)**.

```bash
pip install -r requirements.txt
python run_metrics.py --gold gold.conll --pred pred.conll --model my_model --output-dir results
```

Подробнее: раздел **Метрики** ниже и [docs/DATA_AND_ERRORS.md](docs/DATA_AND_ERRORS.md).

### Ошибки моделей (`data/errors/`)

Excel и CSV с разбором ошибок (mDeBERTa, mmBERT, генеративные LLM, GigaAM).  
**Что в каждом файле и как читать** — [docs/DATA_AND_ERRORS.md](docs/DATA_AND_ERRORS.md).

---

## Метрики

| Метрика | Описание |
|---------|----------|
| Intent Accuracy | Доля верных интентов |
| Intent F1 (weighted) | F1 по классам интентов |
| Span F1 | Span-level F1 (seqeval, BIO) |
| Slot F1 (/N) | Для имён с `gigaam`: среднее F1 по всем 500 репликам |
| Joint | Среднее Intent F1 и Slot/Span F1 (см. README в коде) |

Выход: `comparison_<model>.csv`, `metrics_summary.csv`, опционально `per_slot_<model>.csv`.

---

## English

Russian NLU corpus resources (xSID-based): data prep notebooks, **metric evaluation code**, and **error analysis spreadsheets** from the HSE thesis (intent/slot filling, encoders, LLMs, GigaAM ASR→NLU). See [docs/DATA_AND_ERRORS.md](docs/DATA_AND_ERRORS.md).

---

## License

MIT — see [LICENSE](LICENSE).
