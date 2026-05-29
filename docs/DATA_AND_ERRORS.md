# Данные об ошибках: что лежит в `data/errors/` и как этим пользоваться

Папка **`data/errors/`** — готовые таблицы из ВКР для разбора ошибок моделей. Форматы: **Excel (.xlsx)** и **CSV**. Их можно открыть в Excel, Google Sheets или pandas; к метрикам из `nlu_metrics/` они не привязаны напрямую, но описывают те же прогоны.

---

## 1. Энкодеры (mDeBERTa, mmBERT)

### `ERRORS_mDeBERTa_best_test.xlsx`

**Зачем:** детальный разбор **лучших тестовых** прогонов mDeBERTa-v3-base на двух сценариях.

**Листы (два блока — `trans_trans` и `adapt_Russia`):**

| Лист | Содержание |
|------|------------|
| `*_сводка` | Краткие цифры: сколько реплик, ошибок по интентам и слотам |
| `*_метрики` | Intent Acc, Span F1, Joint и др. |
| `*_интенты` | Ошибки по каждому интенту (всего / ошибок / доля) |
| `*_путаницы` | Матрица: gold intent → pred intent |
| `*_ошибки_I` | Реплики с неверным интентом (id, gold, pred, текст при наличии) |
| `*_ошибки_S` | Реплики с ошибками слотов при верном интенте |
| `*_слоты` | Per-slot F1 (precision, recall, F1) |
| `*_худшие` | Реплики с наибольшим числом ошибок слотов |

**Как использовать:** начни с `trans_trans__сводка` и `trans_trans__интенты`; для главы про ошибки — листы `*_путаницы` и `*_худшие`.

---

### `ERRORS_mdeberta_trans_trans_RU_best_val.xlsx`

**Зачем:** ошибки mDeBERTa на **validation** (сценарий trans train → trans val, русский).

**Листы:** Сводка, F1 по интентам, Путаницы интентов, Ошибки интентов, Ошибки слотов, Худшие реплики.

**Как использовать:** сравни с тестовым файлом выше — видно, переносятся ли ошибки с val на test.

---

### `ERRORS_mmBERT_adapt_Russia_val_runner.xlsx`

**Зачем:** то же для **mmBERT** на adapt_Russia validation (сильный сценарий по Joint на adapt-тесте).

**Структура листов** — как у mdeberta val файла.

---

### `ANALYSIS_detailed.xlsx`

**Зачем:** сводка по **всем ~80 конфигурациям** энкодеров + сравнение с GigaAM.

**Важные листы:**

| Лист | Содержание |
|------|------------|
| `Все 80 конфигураций` | Полная таблица метрик |
| `Энкодеры vs GigaAM` | Текстовый NLU vs речь+NLU |
| `Joint по сценариям` / `Joint ключевые сценарии` | Лучшие сочетания train/test |
| `Путаницы интентов` | Агрегированные путаницы |
| `По числу спанов` | Связь числа спанов и ошибок |
| `Интенты mDeBERTa adapt` / `Слоты mDeBERTa adapt` | Разбор по классам |

**Как использовать:** для таблиц в дипломе и общих выводов; для одной модели удобнее узкие `ERRORS_*.xlsx`.

---

### `PER_intent_mDeBERTa_test.xlsx` / `PER_slots_mDeBERTa_test.xlsx`

**Зачем:** компактные таблицы **per-intent** и **per-slot** F1 на тесте mDeBERTa (trans-trans).

---

## 2. Генеративные LLM

### `generative_error_analysis_tables.xlsx`

**Зачем:** все прогоны **zero-shot / few-shot** (Qwen, Gemma, Phi и др.) + разбор ошибок.

**Блоки листов:**

| Префикс | Смысл |
|---------|--------|
| `all_runs`, `pivot_*`, `best_*` | Сводные метрики и лучшие конфигурации |
| `zs_*` | Zero-shot: путаницы интентов, галлюцинации слотов, уверенность, BIO vs span |
| `best_*` | То же для лучшего прогона (Gemma-2-9B-it) |

**Типичные листы:** `*_intent_errors`, `*_confusions`, `*_hallucinated`, `*_слоты_по_типу`, `*_худшие_слоты`.

**Как использовать:** для раздела 6.1 ВКР (косвенные запросы, лишние слоты, нормализация чисел/имён).

---

## 3. GigaAM (речь → NLU)

CSV из пайплайна ASR + MaChAmp (тест 500 реплик, trans-trans-Russian):

| Файл | Содержание |
|------|------------|
| `gigaam_trans_trans_intent_error_stats.csv` | По каждому интенту: всего, ошибок, доля |
| `gigaam_trans_trans_intent_confusions.csv` | Пары gold → pred intent |
| `gigaam_trans_trans_slot_errors_summary.csv` | По типу слота: пропущено / лишних / всего |
| `gigaam_trans_trans_slot_errors_detail.csv` | По репликам: missing/extra spans |
| `gigaam_trans_trans_slot_wrong_type_pairs.csv` | Пересечения с неверным типом слота |

**Как использовать:** для таблиц ошибок GigaAM в LaTeX; детальный CSV — для поиска примеров в `.conll` по `id`.

**Метрики GigaAM** считай через `run_metrics.py` с `gigaam` в имени модели — включится **Slot F1 (/500)**.

---

## 4. Связь с кодом метрик

```
gold.conll + pred.conll  →  run_metrics.py  →  comparison_*.csv, metrics_summary.csv
```

Excel в `data/errors/` — **готовый анализ** после сравнения gold/pred (MaChAmp, generative JSON, GigaAM). Чтобы **пересчитать** метрики на новых предсказаниях — только `nlu_metrics/` (см. README).

---

## 5. Быстрый выбор файла

| Вопрос | Файл |
|--------|------|
| Лучший энкодер на тесте, ошибки по интентам | `ERRORS_mDeBERTa_best_test.xlsx` |
| Все 80 конфигураций | `ANALYSIS_detailed.xlsx` |
| Ошибки LLM (Gemma zero-shot) | `generative_error_analysis_tables.xlsx` → листы `zs_*` / `best_*` |
| Ошибки после GigaAM | `gigaam_trans_trans_*.csv` |
| mmBERT на adapt validation | `ERRORS_mmBERT_adapt_Russia_val_runner.xlsx` |

---

## English (short)

The `data/errors/` folder stores thesis-ready error analysis tables for encoders, generative LLMs, and the GigaAM ASR→NLU pipeline. Use Excel for human-readable breakdowns; use `run_metrics.py` to recompute metrics from new CoNLL predictions.
