# REPORT (DSGD-Auto)

Этот файл — единый отчёт/документация для репродукции экспериментов и подготовки статьи.

Дата обновления: 2026-02-01

## Актуальные артефакты для статьи

- Основной LaTeX: `paper_with_complete_table_and_figure.tex`
- Глобальные графики метрик для статьи (полный test split, без outlier/inlier разбиения) — лежат в `Common_code/results/`:
  - `Common_code/results/ALL_DATASETS_Accuracy.png`
  - `Common_code/results/ALL_DATASETS_F1Score.png`
  - `Common_code/results/ALL_DATASETS_NLL.png`
  - `Common_code/results/ALL_DATASETS_ECE.png`
  - `Common_code/results/ALL_DATASETS_Precision.png`
  - `Common_code/results/ALL_DATASETS_Recall.png`
  - `Common_code/results/ALL_DATASETS_Uncertainty.png`
  - `Common_code/results/ALL_DATASETS_metrics.csv` (объединённая таблица метрик по всем датасетам)

- Отдельно (диагностика на outliers/inliers + статистики базы правил) — в `Common_code/results/outlier_plots/`:
  - `Common_code/results/outlier_plots/rule_counts_pastel.png`
  - `Common_code/results/outlier_plots/combined_rule_literals.png`
  - `Common_code/results/outlier_plots/ALL_DATASETS_Accuracy_OUTLIERS_INLIERS.png`
  - `Common_code/results/outlier_plots/ALL_DATASETS_F1Score_OUTLIERS_INLIERS.png`
  - `Common_code/results/outlier_plots/ALL_DATASETS_NLL_OUTLIERS_INLIERS.png`
  - `Common_code/results/outlier_plots/ALL_DATASETS_ECE_OUTLIERS_INLIERS.png`
  - `Common_code/results/outlier_plots/ALL_DATASETS_Precision_OUTLIERS_INLIERS.png`
  - `Common_code/results/outlier_plots/ALL_DATASETS_Recall_OUTLIERS_INLIERS.png`
  - `Common_code/results/outlier_plots/ALL_DATASETS_Uncertainty_OUTLIERS_INLIERS.png`
  - `Common_code/results/ALL_DATASETS_metrics_OUTLIERS_INLIERS.csv` (объединённая таблица для outliers/inliers)

## Примечание про удалённые экспериментальные файлы

Ряд промежуточных скриптов/тестов, которые использовались для генерации отчётов/таблиц, удалён из репозитория, чтобы оставить только материалы, которые реально входят в paper.
Исторические описания/команды сохранены в архивной части ниже (см. раздел ``Архив'').

## Примечание про «удалённые» отчёты

Ранее в репозитории были отдельные файлы отчётов (`RULE_INSPECTOR_REPORT*.md`, `DEEP_RULE_QUALITY_REPORT*.md`, `UNIFIED_RULES_REPORT.md`, и т.д.).
Чтобы не держать много производных артефактов в корне, отдельные файлы отчётов были удалены, но их содержимое сохранено ниже в архивной секции.

---

# Архив (исторический UNIFIED_RULES_REPORT.md)

# Единый отчёт по инспекции и качеству правил (DSGD-Auto)

Этот файл объединяет все перечисленные артефакты в один документ и добавляет пояснения по тому, **что именно измеряется**, **как читать метрики** и **какие файлы за что отвечают**.

## Как устроены артефакты (кратко)

- **Rule Inspector report** (`RULE_INSPECTOR_REPORT*.md`) — примеры по отдельным объектам: сколько правил сработало, какая итоговая неопределённость `Ω`, какие правила внесли вклад, и как выглядит “Combined Rule” (только как объяснение).
- **Deep Rule Quality report** (`DEEP_RULE_QUALITY_REPORT*.md`) — агрегированные метрики качества (Accuracy/F1/NLL/Brier/ECE), DST-поведение (`unc_rule` vs `unc_comb`), глубина фьюжна, диагностические показатели конфликтов, и интерпретируемость combined-rule как “фильтра”.
- **Deep analysis** (`DEEP_RULE_INSPECTOR_ANALYSIS.md`) — разбор причин, почему STATIC часто ведёт себя иначе, чем RIPPER/FOIL (в частности, из‑за плотной базы правил и глубокого DST‑слияния).
- **LaTeX-таблица** (`complete_rule_table.tex`) — снимок плотности баз правил (число правил и средняя длина) по датасетам/алгоритмам.
- **Визуализация** (`create_rule_visualization.py`) — скрипт, который рисует схему пайплайна (индукция правил → активации → DST‑фьюжн → pignistic → предсказание).
- **Генераторы отчётов** (`Common_code/generate_*_report.py`) — как получались Markdown отчёты (команды ниже).

## Ключевые понятия (как читать отчёты)

- **Rule base scale**: `R` и статистики длины правила (число литералов).
- **Fusion depth**: прокси‑метрика — сколько правил “сработало” на объекте/в среднем. Чем больше, тем глубже DST‑слияние источников.
- **Ω / uncertainty**:
  - `unc_rule` — средняя `Ω` по активным правилам *до* слияния (насколько “осторожны” сами правила).
  - `unc_comb` — `Ω` *после* DST‑комбинации активных правил (что стало с неопределённостью после фьюжна).
- **Combined Rule**: строится из положительных вкладов, агрегируется в конъюнкцию условий и **не используется для инференса** (explanation‑only).
- **Conflict diagnostics**: доля конфликтов и соотношение согласующихся/конфликтующих правил в explain‑подвыборке.

## Как перегенерировать отчёты (пример)

Команды запускаются из корня репозитория:

```bash
# Rule Inspector (примеры по объектам)
python3 Common_code/generate_rule_inspection_report.py --out RULE_INSPECTOR_REPORT.md

# Deep Rule Quality (агрегированная оценка)
python3 Common_code/generate_deep_rule_quality_report.py --out DEEP_RULE_QUALITY_REPORT.md
```

Примечание: `DEEP_RULE_QUALITY_REPORT_ALL.md` и `RULE_INSPECTOR_REPORT_SAHEART.md` — обычно получаются теми же генераторами, но с другим набором `--datasets`/`--out`.

---

## Вложенные исходные файлы

Ниже содержимое всех перечисленных файлов (Markdown — сдвиг заголовков на +1 уровень, чтобы не ломать структуру общего документа).

---

## Deep Analysis of Rule Inspector (DSGD-Auto): STATIC vs RIPPER vs FOIL
Источник: `DEEP_RULE_INSPECTOR_ANALYSIS.md`

Этот файл — «глубокий» разбор того, что именно показывает `Common_code/sample_rule_inspector.py`, как строится “Combined Rule” (скомбинированное правило), и почему поведение STATIC радикально отличается от RIPPER/FOIL на датасетах `SAheart.csv` и `df_wine.csv`.

Связанные артефакты с конкретными примерами (табличные отчёты):
- `RULE_INSPECTOR_REPORT.md:1` — Adult/Bank/BrainTumor/BreastCancer.
- `RULE_INSPECTOR_REPORT_SAHEART.md:1` — SAheart/df_wine.

---

### 1) Что именно делает Rule Inspector

`Common_code/sample_rule_inspector.py` — это утилита для “forensic” просмотра одной точки данных:

1) **Берёт сохранённую модель правил** (из `Common_code/pkl_rules/*_dst.pkl`) и выбранный датасет.
2) **Выбирает один sample** (по индексу в выбранном split).
3) **Считает, какие правила сработали** (rule firing / activation).
4) **Комбинирует активные правила через DST (Dempster/Yager/Vote)** и выводит:
   - предсказанный класс,
   - оценки (pignistic probabilities) и неопределённость,
   - список вкладов правил и “Combined Rule” в виде конъюнкции условий.

Ключевой момент: **Combined Rule — это объяснение (explanation-only)**. Оно *не участвует* в инференсе. Инференс идёт по реальным правилам и их DST-комбинации.

---

### 2) Откуда берутся “сработавшие правила” (активации)

Всё сводится к двум шагам в `Common_code/DSModelMultiQ.py`:

1) **Вычисление литералов** (одиночных условий вида `feature < t`, `feature > t`, `feature == v`):
   - `_eval_literals(...)` и кеши литералов: `Common_code/DSModelMultiQ.py:351`.
2) **Правило — это конъюнкция литералов** (AND). Правило считается активным, если активны *все* его литералы:
   - `_activation_matrix(...)`: `Common_code/DSModelMultiQ.py:415`.

То есть “глубина фьюжна” (fusion depth), которую мы используем в отчётах, — это просто:

> **fusion depth(x) ≈ число правил, которые сработали на sample x**  
> (потому что DST-комбинация применяется последовательно по активному набору).

---

### 3) Что такое DSGD/DSGD++ слой в контексте этих правил

В терминах текущего кода (см. `Common_code/DSModelMultiQ.py`):

- У нас есть набор правил `r_j(x) ∈ {0,1}`.
- Каждому правилу соответствует **обучаемый вектор масс** `m_j` над `K` классами + `Ω` (ignorance).
- При инференсе для sample `x` берётся подмножество активных правил и выполняется DST-комбинация:
  - `forward(...)`: `Common_code/DSModelMultiQ.py:562`.
  - неопределённость:
    - `unc_rule` = средняя `Ω` по активным правилам (до комбинации),
    - `unc_comb` = `Ω` после комбинации: `Common_code/DSModelMultiQ.py:792`.

Очень важная диагностическая разница:

- `unc_rule` показывает **“как осторожны сами правила”**.
- `unc_comb` показывает **“что стало с неопределённостью после слияния множества источников”**.

Если `unc_rule` высокий (например, ~0.8), но `unc_comb` ≈ 0, это означает **fusion-driven ignorance collapse**:
много осторожных правил при Dempster-слиянии дают почти нулевую итоговую `Ω` из‑за нормализации и накопления конфликтов.

---

### 4) Как строится “Combined Rule” (скомбинированное правило)

#### 4.1 Rule contributions (вклады правил)

Для одного sample `get_combined_rule(...)` строит список вкладов `rule_contributions`:
- `Common_code/DSModelMultiQ.py:1103`.

Для `dempster/yager` веса выводятся из:
- “certainty” правила (≈ `1-Ω`),
- массы на предсказанный класс,
- и знака:
  - **положительный вес** если правило “согласуется” с предсказанием,
  - **отрицательный** если правило поддерживает другой класс.

Практическое следствие: в одном sample могут быть одновременно
“agreeing” и “conflicting” правила — именно это и есть конфликт источников.

#### 4.2 Combined condition (агрегация литералов)

Дальше строится агрегированная конъюнкция литералов:
- `_combine_literals(...)`: `Common_code/DSModelMultiQ.py:896`.

Алгоритм упрощённо такой:
1) Берутся **только положительные вклады** (weight > 0).
2) Вес правила распределяется по его литералам (примерно равномерно по длине правила).
3) Литералы группируются по `(feature, op)`:
   - для `>` берётся “самый сильный” порог (максимальный),
   - для `<` берётся “самый сильный” порог (минимальный),
   - для `==` берётся значение с максимальной суммой весов.
4) Собирается строка `f1 op v1 AND f2 op v2 AND ...`.

**Почему иногда Combined Rule может быть `<empty>`**
— если в sample все веса ≤ 0 (или положительные веса не имеют валидных литералов), то объединять нечего. Это нормальное ограничение “позитив‑only” объяснения.

**Почему Combined Rule не равен “истинному” дереву рассуждений**
— потому что:
- отрицательные вклады *не включаются* в Combined condition,
- несовместимые условия могут “сплющиться” (например, разные пороги по одному признаку),
- Combined condition не проверяется как отдельное правило на данных (и не используется в инференсе).

---

### 5) Что реально означает “STATIC / RIPPER / FOIL” в этой работе

Ниже — строго “по коду”, чтобы можно было писать в статье без домыслов.

#### 5.1 STATIC (baseline, “старый” метод генерации правил)

Файл: `Common_code/rule_generator.py`.

STATIC строит большой пул простых литералов:
- **числовые признаки**: пороги по квантилям → условия вида `x_i < t` и `x_i > t`;
- **категориальные признаки**: частые значения → `x_i == v`.

Дальше STATIC расширяет пул в **низко‑порядковые конъюнкции**:
- пары и тройки (pair/triple) “топовых” признаков по эвристике разнообразия/вариативности.

Смысл: это действительно близко к “brute force” по пространству коротких правил.

Плюсы:
- высокая покрываемость: почти всегда что‑то срабатывает;
- простая логика правил.

Минусы (важно для твоего текста):
- правил много, и они **мелкие/поверхностные** (обычно 1–3 литерала);
- на одном sample срабатывает **очень много правил**, поэтому DST‑слияние становится очень глубоким;
- глубокое Dempster‑слияние часто приводит к **collapse** (`unc_comb → 0`) и делает “неопределённость” неинформативной, даже если каждое правило осторожно.

#### 5.2 RIPPER (rule list, separate-and-conquer + pruning)

Файл: `Common_code/rule_generator.py` (`algo="ripper"`).

По сути:
- правила растут “separate-and-conquer” по классам,
- есть разделение grow/prune и reduced‑error pruning (в отличие от FOIL),
- поиск идёт по “precision tiers” (сначала строгая точность, затем слабее).

Практический эффект:
- правил **меньше**,
- каждое правило **длиннее** (обычно больше литералов),
- на sample срабатывает **мало правил** → fusion depth маленький.

#### 5.3 FOIL (greedy growth by gain, без RIPPER‑pruning)

Файл: `Common_code/rule_generator.py` (`algo="foil"`).

FOIL‑ветка использует:
- жадный рост правила по gain‑критерию,
- без reduced‑error pruning.

Практический эффект:
- правил обычно больше, чем у RIPPER, но меньше “по плотности”, чем STATIC,
- длина правила часто средняя/высокая,
- depth умеренный.

---

### 6) SAheart.csv: глубокое сравнение поведения (STATIC vs RIPPER vs FOIL)

Ниже — метрики, измеренные на **случайных 200 sample** (seed=0) из `SAheart.csv` для каждого алгоритма.

Обозначения:
- `rules` — число правил в модели.
- `len_mean / len_p90` — средняя/90‑й процентиль длины правила (число литералов).
- `depth_*` — число сработавших правил на sample (fusion depth proxy).
- `holes` — доля sample, где не сработало ни одно правило.
- `unc_rule_mean` — средняя Ω по активным правилам (до фьюжна).
- `unc_comb_mean` — средняя Ω после DST‑слияния.
- `conf_rate` — доля sample, где есть хотя бы 1 конфликтующее правило (`n_conflicting > 0`).

| algo | rules | len_mean | len_p90 | depth_mean | depth_p90 | depth_max | holes | unc_rule_mean | unc_comb_mean | conf_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STATIC | 796 | 2.63 | 3 | 407.83 | 641 | 705 | 0.0% | 0.8003 | ~0.0000000040 | 72.0% |
| RIPPER | 25 | 3.12 | 4.6 | 3.56 | 5 | 7 | 0.0% | 0.6862 | 0.2918 | 94.5% |
| FOIL | 78 | 4.91 | 7 | 1.72 | 3 | 7 | 1.5% | 0.2825 | 0.1804 | 4.0% |

#### 6.1 Главный вывод по SAheart

**STATIC**:
- Очень высокая “глубина” (`depth_mean ~ 408`), то есть на один sample “стреляет” сотни правил.
- При этом `unc_rule_mean ~ 0.80` (каждое правило само по себе осторожно).
- Но `unc_comb_mean ≈ 0` — почти полная потеря `Ω` после Dempster‑слияния.

Интерпретация:
- STATIC создаёт много “слабых” источников, которые массово активируются;
- Dempster‑нормализация при многократном слиянии приводит к collapse, и итоговая неопределённость перестаёт быть полезной как “я не знаю”.

**RIPPER**:
- Depth маленький (в среднем ~3–4 активных правила).
- `unc_comb_mean` остаётся заметным (~0.29), то есть неопределённость после фьюжна сохраняется.
- `conf_rate` высокий (почти всегда есть конфликт), но из‑за малого числа правил конфликт не приводит к collapse так, как в STATIC.

**FOIL**:
- Depth ещё меньше.
- `unc_rule_mean` низкий (~0.28): правила в среднем более “уверенные” (низкая Ω у самих правил).
- `conf_rate` почти нулевой: FOIL на SAheart чаще строит правила, которые согласованы с итоговым предсказанием.

#### 6.2 Как выглядят Combined Rules на SAheart (пример)

Смотри конкретные примеры в `RULE_INSPECTOR_REPORT_SAHEART.md:7`.

Характерный паттерн:
- STATIC: Combined Rule включает много простых порогов (`age > ...`, `tobacco > ...`, `alcohol < ...`) и категориальный литерал `famhist == Present`.
- RIPPER/FOIL: Combined Rule обычно короче и отражает более “смысловые” конъюнкции.

---

### 7) df_wine.csv: глубокое сравнение поведения (STATIC vs RIPPER vs FOIL)

Метрики на **случайных 200 sample** (seed=0) из `df_wine.csv`.

| algo | rules | len_mean | len_p90 | depth_mean | depth_p90 | depth_max | holes | unc_rule_mean | unc_comb_mean | conf_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STATIC | 542 | 2.64 | 3 | 207.47 | 298 | 298 | 0.0% | 0.8010 | 0.0000020 | 94.5% |
| RIPPER | 104 | 4.29 | 5 | 4.65 | 6 | 7 | 0.0% | 0.6559 | 0.1973 | 99.0% |
| FOIL | 164 | 4.78 | 6 | 3.03 | 5 | 7 | 0.0% | 0.6223 | 0.3023 | 78.5% |

#### 7.1 Главный вывод по df_wine

**STATIC** опять даёт экстремальную глубину фьюжна (`depth_mean ~ 207`) и практически нулевую итоговую Ω (`unc_comb ~ 2e-6`), при этом `unc_rule ~ 0.80`.

Это классический сигнал:
- правило‑база слишком плотная,
- DS‑слой “пересобирает” сотни источников и теряет информативную неопределённость.

RIPPER/FOIL:
- depth на порядок меньше,
- `unc_comb` остаётся заметным,
- конфликтов много (особенно у RIPPER), но они не приводят к “полной уверенности” как в STATIC.

#### 7.2 Как выглядят Combined Rules на df_wine (пример)

Смотри `RULE_INSPECTOR_REPORT_SAHEART.md:89`.

Паттерн:
- STATIC: многие пороги/квантили по химическим признакам → Combined Rule получается как “сборная солянка” порогов.
- RIPPER/FOIL: правила длиннее, но их мало; Combined Rule обычно короткая (потому что складывается из нескольких сильных вкладов).

---

### 8) Почему STATIC “работает”, но даёт мало научно‑полезной интерпретации

Это ключевой текст для твоей статьи (как ты просил):

1) STATIC даёт неплохую точность, потому что покрывает пространство множеством простых разбиений.
2) Но интерпретация отдельного правила слабая:
   - правило короткое,
   - часто “общее” (`feature > threshold` без контекста),
   - таких правил сотни и они повторяют друг друга.
3) DST‑слой в STATIC‑режиме часто работает в режиме “очень глубокого фьюжна”, что:
   - увеличивает конфликтность,
   - приводит к collapse итоговой `Ω`,
   - делает uncertainty‑сигнал менее осмысленным.

В противоположность этому, RIPPER/FOIL создают:
- меньшее число более структурных правил,
- меньшую глубину фьюжна,
- более стабильное (и интерпретируемое) поведение неопределённости.

---

### 9) Что писать в статье про “глубину” и “комбинированные правила”

#### 9.1 Определения (коротко и строго)

- **Rule firing / activation**: правило активно, если все его литералы истинны на sample.
- **Fusion depth**: число активных правил на sample (сколько источников фьюзится).
- **Two Omega signals**:
  - `unc_rule`: средняя Ω по активным правилам до фьюжна,
  - `unc_comb`: Ω после DS‑фьюжна активных правил.
- **Combined Rule** (explanation-only): конъюнкция литералов, полученная из *положительных* вкладов активных правил; не используется в предсказании.

#### 9.2 Почему важно различать unc_rule и unc_comb

SAheart/df_wine показывают, что можно иметь:
- высокую осторожность правил (`unc_rule ~ 0.8`),
- но почти нулевую неопределённость после слияния (`unc_comb → 0`) из‑за глубины и конфликтов.

Это и есть научно важный эффект: **индуктор правил меняет динамику DST‑слияния**, а не только точность.

---

### 10) Как воспроизвести и расширить анализ

1) Табличные отчёты с примерами:
   - `python3 Common_code/generate_rule_inspection_report.py --out RULE_INSPECTOR_REPORT_SAHEART.md --datasets SAheart.csv df_wine.csv --algos STATIC RIPPER FOIL`
2) Точечный forensic‑просмотр конкретного sample:
   - `python3 Common_code/sample_rule_inspector.py --dataset SAheart.csv --algo STATIC --idx 0 --split test --show-combined --combine-rule dempster`

Если хочешь, я сделаю ещё один скрипт, который автоматически добавит в этот deep‑report:
- распределения depth (гистограммы),
- таблицы “unc_rule vs unc_comb” по всем датасетам,
- примеры “high conflict” / “high depth” / “holes” кейсов для каждого индуктора.

---

## Deep Rule Quality Report (DSGD-Auto)
Источник: `DEEP_RULE_QUALITY_REPORT.md`

This report evaluates rule-induced DSGD models across datasets, focusing on predictive quality, DST uncertainty behavior, fusion depth, and interpretability of the combined (explanation-only) rule.

Notes:
- Train/test split is stratified with `test_size` and `seed`.
- Probabilities are pignistic; for reproducibility the pignistic prior is set to inverse class frequency on the train split.
- “Combined rule” is built from positive contributions only and is not used for inference; we still evaluate it as a *filter* to measure how sharply it describes the predicted situation and how well it generalizes.

### SAheart.csv

#### STATIC

- Data: n_total=462 (train=388, test=74), d=9, label_col=`chd`
- Classes (K=2): 0, 1 | counts=[302, 160]
- Rule base: R=796; literals/rule mean=2.63, p90=3.0, max=3
- Fusion depth (rules fired/sample): mean=419.61, p90=652.9, max=705; holes=0.00%

Predictive quality (test):
- Accuracy=0.7297 | F1_macro=0.6980 | F1_weighted=0.7271
- NLL=0.5405 | Brier=0.3681 | ECE(15)=0.1237

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.8003
- Ω combined (unc_comb) mean=0.0000

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=0.0% | literals mean=11.58 (p90=16.7) | distinct features mean=7.50
- Combined rule as filter: test coverage mean=2.41% precision mean=74.74%
- Generalization (train→test): coverage 1.56%→2.41% | precision 76.23%→74.74%

Conflict diagnostics (explain subset):
- conflict rate=62.2% | mean agreeing rules=285.26 | mean conflicting rules=134.35

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 781 | 0 | 91.89% | 69.1% | 0.798 | age < 60 → class 0 |
| 712 | 0 | 91.89% | 63.2% | 0.797 | ldl > 2.66 → class 0 |
| 685 | 0 | 89.19% | 62.1% | 0.813 | sbp > 118 → class 0 |
| 780 | 0 | 89.19% | 60.6% | 0.792 | age > 21.75 → class 0 |
| 684 | 0 | 87.84% | 67.7% | 0.782 | sbp < 160.625 → class 0 |
| 711 | 0 | 87.84% | 67.7% | 0.781 | ldl < 7.0775 → class 0 |
| 740 | 0 | 87.84% | 64.6% | 0.791 | typea > 42 → class 0 |
| 726 | 0 | 86.49% | 67.2% | 0.820 | adiposity < 34.46 → class 0 |
| 725 | 0 | 86.49% | 59.4% | 0.806 | adiposity > 15.16 → class 0 |
| 698 | 0 | 86.49% | 68.8% | 0.766 | tobacco < 7.9625 → class 0 |
| 753 | 0 | 86.49% | 60.9% | 0.830 | obesity > 21.705 → class 0 |
| 739 | 0 | 85.14% | 61.9% | 0.817 | typea < 64 → class 0 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=60, row=317) true=`0` pred_idx=0 fired=422 Ω=0.0000 [OK]
  - combined: alcohol > 22.832499504089355 AND alcohol < 42.9150013923645 AND typea < 50.0 AND tobacco > 0.817500002682209 AND sbp < 118.0 AND age < 31.0 AND adiposity < 15.15999984741211 AND obesity > 22.947500705718994 AND ldl > 3.7812499701976776 AND tobacco < 1.85999995470047 AND ldl < 4.355000019073486 AND age > 21.75
  - combined-as-filter: train cov=0.00% prec=nan% | test cov=1.35% prec=100.00%
  - top contributions:
    - [agree] w=+0.0608 :: age < 39 → class 0
    - [agree] w=+0.0575 :: alcohol < 42.915 & tobacco < 7.9625 & typea < 60 → class 0
    - [agree] w=+0.0552 :: alcohol < 42.915 & sbp < 160.625 & typea < 64 → class 0
- sample(test_idx=25, row=430) true=`0` pred_idx=0 fired=108 Ω=0.0000 [OK]
  - combined: alcohol < 0.5775000154972076 AND age < 21.75 AND tobacco < 0.0774999987334013 AND sbp < 124.0 AND obesity < 21.705000162124634 AND ldl < 4.355000019073486 AND typea > 64.0 AND adiposity < 19.739999771118164 AND ldl > 3.7812499701976776 AND adiposity > 15.15999984741211 AND famhist == Absent
  - combined-as-filter: train cov=0.00% prec=nan% | test cov=1.35% prec=100.00%
  - top contributions:
    - [agree] w=+0.0648 :: age < 21.75 → class 0
    - [agree] w=+0.0608 :: age < 39 → class 0
    - [agree] w=+0.0521 :: alcohol < 42.915 & sbp < 160.625 & tobacco < 7.9625 → class 0
- sample(test_idx=2, row=157) true=`0` pred_idx=1 fired=351 Ω=0.0000 [WRONG]
  - combined: sbp > 160.625 AND tobacco > 7.962500035762787 AND adiposity > 34.459999084472656 AND age > 50.0
  - combined-as-filter: train cov=0.52% prec=100.00% | test cov=2.70% prec=50.00%
  - top contributions:
    - [agree] w=+0.0463 :: sbp > 160.625 → class 1
    - [agree] w=+0.0442 :: tobacco > 7.9625 → class 1
    - [agree] w=+0.0335 :: age > 50 → class 1

#### RIPPER

- Data: n_total=462 (train=388, test=74), d=9, label_col=`chd`
- Classes (K=2): 0, 1 | counts=[302, 160]
- Rule base: R=25; literals/rule mean=3.12, p90=4.6, max=5
- Fusion depth (rules fired/sample): mean=3.49, p90=5.0, max=6; holes=0.00%

Predictive quality (test):
- Accuracy=0.8514 | F1_macro=0.8322 | F1_weighted=0.8491
- NLL=0.3376 | Brier=0.1990 | ECE(15)=0.1137

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.6896
- Ω combined (unc_comb) mean=0.2964

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=0.0% | literals mean=4.89 (p90=7.0) | distinct features mean=4.66
- Combined rule as filter: test coverage mean=7.72% precision mean=88.76%
- Generalization (train→test): coverage 7.08%→7.72% | precision 80.82%→88.76%

Conflict diagnostics (explain subset):
- conflict rate=91.9% | mean agreeing rules=2.43 | mean conflicting rules=1.05

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 12 | 1 | 59.46% | 20.5% | 0.771 | tobacco < 1.86 |
| 23 | 0 | 47.30% | 62.9% | 0.890 | typea > 53 |
| 11 | 1 | 40.54% | 56.7% | 0.925 | tobacco > 1.86 |
| 19 | 0 | 29.73% | 90.9% | 0.609 | tobacco < 1.18 & typea < 56 |
| 22 | 0 | 25.68% | 63.2% | 0.791 | adiposity > 26.1 & ldl < 5.407 |
| 13 | 0 | 22.97% | 94.1% | 0.632 | age < 49 & ldl < 3.531 & obesity > 21.72 |
| 15 | 0 | 21.62% | 100.0% | 0.554 | age < 38 & sbp < 144 & tobacco < 1.86 & typea < 65 |
| 17 | 0 | 20.27% | 86.7% | 0.650 | obesity < 25.72 & tobacco < 0.4 |
| 16 | 0 | 12.16% | 88.9% | 0.384 | alcohol > 7.01 & famhist == Absent & obesity > 25.72 & tobacco < 7.398 |
| 1 | 1 | 8.11% | 83.3% | 0.298 | adiposity < 26.48 & age > 38.4 & famhist == Present & tobacco < 5.44 |
| 2 | 1 | 6.76% | 100.0% | 0.382 | adiposity < 29.86 & age > 51 & ldl > 3.102 & tobacco > 1.86 & typea < 53 |
| 20 | 0 | 6.76% | 80.0% | 0.410 | adiposity < 26.1 & age < 53 & tobacco > 4.03 & typea < 60.7 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=26, row=75) true=`0` pred_idx=0 fired=3 Ω=0.3051 [OK]
  - combined: tobacco < 0.4000000059604645 AND typea < 56.0 AND obesity < 25.72000026702881
  - combined-as-filter: train cov=11.08% prec=90.70% | test cov=12.16% prec=100.00%
  - top contributions:
    - [agree] w=+0.1326 :: tobacco < 1.18 & typea < 56
    - [agree] w=+0.1105 :: obesity < 25.72 & tobacco < 0.4
    - [conflict] w=-0.0000 :: tobacco < 1.86
- sample(test_idx=66, row=368) true=`0` pred_idx=1 fired=2 Ω=0.8231 [WRONG]
  - combined: tobacco > 1.85999995470047
  - combined-as-filter: train cov=52.58% prec=46.57% | test cov=40.54% prec=56.67%
  - top contributions:
    - [conflict] w=-0.0037 :: typea > 53
    - [agree] w=+0.0000 :: tobacco > 1.86
- sample(test_idx=21, row=128) true=`1` pred_idx=0 fired=4 Ω=0.1719 [WRONG]
  - combined: ldl > 7.096000003814697 AND typea < 49.2 AND famhist == Absent AND obesity > 25.72000026702881
  - combined-as-filter: train cov=2.32% prec=88.89% | test cov=2.70% prec=50.00%
  - top contributions:
    - [agree] w=+0.2971 :: famhist == Absent & ldl > 5.407 & typea < 49.2
    - [conflict] w=-0.1561 :: adiposity < 29.86 & age > 51 & ldl > 3.102 & tobacco > 1.86 & typea < 53
    - [agree] w=+0.1019 :: ldl > 7.096 & obesity > 25.72 & typea < 51.2

#### FOIL

- Data: n_total=462 (train=388, test=74), d=9, label_col=`chd`
- Classes (K=2): 0, 1 | counts=[302, 160]
- Rule base: R=78; literals/rule mean=4.91, p90=7.0, max=8
- Fusion depth (rules fired/sample): mean=1.66, p90=3.0, max=4; holes=2.70%

Predictive quality (test):
- Accuracy=0.9459 | F1_macro=0.9396 | F1_weighted=0.9454
- NLL=0.2534 | Brier=0.1002 | ECE(15)=0.0981

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.2773
- Ω combined (unc_comb) mean=0.1758

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=2.7% | literals mean=6.74 (p90=10.0) | distinct features mean=6.34
- Combined rule as filter: test coverage mean=3.21% precision mean=93.19%
- Generalization (train→test): coverage 1.87%→3.21% | precision 95.56%→93.19%

Conflict diagnostics (explain subset):
- conflict rate=5.4% | mean agreeing rules=1.59 | mean conflicting rules=0.07

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 40 | 0 | 16.22% | 100.0% | 0.234 | adiposity > 12.13 & age < 40 & obesity < 27.83 & tobacco < 0.4 & typea < 66 |
| 49 | 0 | 8.11% | 100.0% | 0.260 | adiposity < 17.16 & age < 41.1 & ldl < 4.395 & sbp > 122.1 & typea > 50 |
| 58 | 0 | 8.11% | 66.7% | 0.281 | adiposity > 29.86 & ldl < 5.407 & tobacco < 1.86 & typea < 63 |
| 43 | 0 | 6.76% | 80.0% | 0.226 | famhist == Absent & obesity > 24.11 & sbp < 134 & tobacco < 7.398 & typea > 57.4 |
| 48 | 0 | 6.76% | 80.0% | 0.216 | adiposity > 26.1 & age < 62.95 & sbp < 154 & tobacco < 0.751 & typea < 56 |
| 64 | 0 | 6.76% | 100.0% | 0.228 | age < 56.4 & alcohol > 0.544 & ldl < 4.355 & obesity < 23.63 & sbp < 154.8 & tobacco > 0.4 & typea < 62.8 |
| 67 | 0 | 6.76% | 100.0% | 0.242 | alcohol > 7.494 & ldl < 5.11 & obesity > 24.97 & tobacco < 0.846 & typea < 57.3 |
| 54 | 0 | 5.41% | 100.0% | 0.287 | adiposity < 16.3 & ldl < 5.163 & obesity > 22.01 & typea < 62 |
| 60 | 0 | 5.41% | 100.0% | 0.252 | adiposity > 21.1 & age < 53 & alcohol > 18.51 & ldl < 6.649 & obesity < 26.76 & sbp > 126.1 & tobacco < 7.96 & typea > 53 |
| 0 | 1 | 5.41% | 100.0% | 0.253 | age > 38.95 & alcohol > 0 & famhist == Present & ldl > 5.04 & obesity < 34.6 & sbp > 152 |
| 22 | 1 | 4.05% | 100.0% | 0.297 | alcohol > 17.17 & ldl > 3.186 & sbp < 147.3 & tobacco > 1.86 & typea < 44.15 |
| 63 | 0 | 4.05% | 66.7% | 0.261 | age < 54.4 & alcohol < 1.54 & obesity < 26.53 & sbp > 126.1 & tobacco < 1.67 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=0, row=262) true=`0` pred_idx=1 fired=2 Ω=0.1772 [WRONG]
  - combined: adiposity < 26.104999542236328 AND age > 34.0 AND famhist == Present AND obesity < 22.010000228881836
  - combined-as-filter: train cov=1.80% prec=85.71% | test cov=1.35% prec=0.00%
  - top contributions:
    - [agree] w=+0.4756 :: adiposity < 26.1 & age > 34 & famhist == Present & obesity < 22.01
    - [conflict] w=-0.2503 :: age > 59 & alcohol < 7.01 & obesity < 26.33 & sbp < 166 & tobacco < 6 & typea < 63
- sample(test_idx=73, row=118) true=`1` pred_idx=1 fired=0 Ω=1.0000 [OK]
  - combined: <empty>
  - combined-as-filter: train cov=0.00% prec=nan% | test cov=0.00% prec=nan%
- sample(test_idx=8, row=258) true=`1` pred_idx=0 fired=2 Ω=0.1453 [WRONG]
  - combined: adiposity < 31.399999618530273 AND age > 47.0 AND ldl > 4.869999885559082 AND obesity > 23.630999183654787 AND sbp < 144.0 AND tobacco > 1.7999999523162842 AND typea > 49.0
  - combined-as-filter: train cov=1.80% prec=100.00% | test cov=1.35% prec=0.00%
  - top contributions:
    - [agree] w=+0.5853 :: adiposity < 31.4 & age > 47 & ldl > 4.87 & obesity > 23.63 & sbp < 144 & tobacco > 1.8 & typea > 49
    - [conflict] w=-0.2623 :: adiposity > 23.59 & age > 28.25 & alcohol > 11.32 & ldl > 4.455 & sbp < 146.7 & tobacco > 7.398 & typea > 49

### df_wine.csv

#### STATIC

- Data: n_total=6497 (train=5458, test=1039), d=6, label_col=`labels`
- Classes (K=2): 0, 1 | counts=[2827, 3670]
- Rule base: R=542; literals/rule mean=2.64, p90=3.0, max=3
- Fusion depth (rules fired/sample): mean=198.15, p90=297.0, max=298; holes=0.00%

Predictive quality (test):
- Accuracy=0.6266 | F1_macro=0.6120 | F1_weighted=0.6218
- NLL=0.6523 | Brier=0.4602 | ECE(15)=0.0306

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.8009
- Ω combined (unc_comb) mean=0.0000

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=0.8% | literals mean=7.74 (p90=10.0) | distinct features mean=5.64
- Combined rule as filter: test coverage mean=1.51% precision mean=67.75%
- Generalization (train→test): coverage 1.35%→1.51% | precision 70.75%→67.75%

Conflict diagnostics (explain subset):
- conflict rate=92.5% | mean agreeing rules=130.07 | mean conflicting rules=68.25

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 477 | 1 | 87.78% | 56.5% | 0.801 | density < 0.99818 → class 1 |
| 506 | 1 | 86.81% | 55.4% | 0.807 | sulphates > 0.38 → class 1 |
| 505 | 1 | 85.95% | 61.3% | 0.782 | sulphates < 0.69 → class 1 |
| 491 | 1 | 85.95% | 58.6% | 0.815 | pH < 3.4 → class 1 |
| 519 | 1 | 85.85% | 58.9% | 0.794 | alcohol < 12.1 → class 1 |
| 478 | 1 | 85.76% | 58.8% | 0.812 | density > 0.991 → class 1 |
| 492 | 1 | 85.76% | 55.1% | 0.803 | pH > 3.04 → class 1 |
| 520 | 1 | 83.93% | 52.2% | 0.789 | alcohol > 9.2 → class 1 |
| 36 | 1 | 80.27% | 60.4% | 0.811 | alcohol < 12.1 & density > 0.991 → class 1 |
| 540 | 1 | 78.73% | 57.2% | 0.808 | good == 0 → class 1 |
| 34 | 1 | 77.48% | 53.8% | 0.793 | alcohol > 9.2 & density < 0.99818 → class 1 |
| 27 | 1 | 75.84% | 57.7% | 0.809 | alcohol < 12.1 & sulphates > 0.38 → class 1 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=176, row=2293) true=`1` pred_idx=1 fired=198 Ω=0.0000 [OK]
  - combined: alcohol < 9.800000190734863 AND pH > 3.109999895095825 AND density > 0.9981799721717834 AND sulphates > 0.4300000071525574 AND quality == 6.0 AND pH < 3.2100000381469727 AND good == 0.0 AND alcohol > 9.199999809265137 AND sulphates < 0.6899999976158142
  - combined-as-filter: train cov=0.20% prec=54.55% | test cov=0.10% prec=100.00%
  - top contributions:
    - [agree] w=+0.0526 :: pH < 3.32 → class 1
    - [agree] w=+0.0492 :: alcohol < 11.3 & quality == 6 & sulphates < 0.69 → class 1
    - [agree] w=+0.0486 :: density > 0.991 & good == 0 & pH < 3.4 → class 1
- sample(test_idx=328, row=5271) true=`0` pred_idx=1 fired=51 Ω=0.0011 [WRONG]
  - combined: pH < 3.0399999618530273 AND density < 0.9923999905586243 AND sulphates < 0.3799999952316284 AND alcohol > 9.5 AND good == 1.0 AND quality == 8.0
  - combined-as-filter: train cov=0.05% prec=100.00% | test cov=0.10% prec=0.00%
  - top contributions:
    - [agree] w=+0.0526 :: pH < 3.32 → class 1
    - [agree] w=+0.0388 :: sulphates < 0.51 → class 1
    - [agree] w=+0.0348 :: pH < 3.16 → class 1
- sample(test_idx=905, row=5139) true=`0` pred_idx=1 fired=69 Ω=0.0001 [WRONG]
  - combined: pH < 3.0399999618530273 AND density < 0.9923999905586243 AND sulphates < 0.3799999952316284 AND alcohol > 9.5 AND quality == 6.0 AND good == 0.0
  - combined-as-filter: train cov=0.44% prec=45.83% | test cov=0.48% prec=60.00%
  - top contributions:
    - [agree] w=+0.0526 :: pH < 3.32 → class 1
    - [agree] w=+0.0437 :: good == 0 & pH < 3.4 & sulphates < 0.69 → class 1
    - [agree] w=+0.0421 :: density < 0.99699 & pH < 3.4 & quality == 6 → class 1

#### RIPPER

- Data: n_total=6497 (train=5458, test=1039), d=6, label_col=`labels`
- Classes (K=2): 0, 1 | counts=[2827, 3670]
- Rule base: R=104; literals/rule mean=4.29, p90=5.0, max=6
- Fusion depth (rules fired/sample): mean=4.81, p90=6.0, max=9; holes=0.00%

Predictive quality (test):
- Accuracy=0.7151 | F1_macro=0.7068 | F1_weighted=0.7132
- NLL=0.5613 | Brier=0.3725 | ECE(15)=0.0768

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.6538
- Ω combined (unc_comb) mean=0.1879

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=0.0% | literals mean=5.59 (p90=7.0) | distinct features mean=4.87
- Combined rule as filter: test coverage mean=3.57% precision mean=75.57%
- Generalization (train→test): coverage 3.51%→3.57% | precision 74.21%→75.57%

Conflict diagnostics (explain subset):
- conflict rate=99.0% | mean agreeing rules=2.83 | mean conflicting rules=2.03

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 99 | 1 | 53.61% | 60.7% | 0.682 | alcohol < 11.87 & density < 0.9966 |
| 56 | 0 | 31.76% | 58.8% | 0.715 | alcohol > 11.05 |
| 98 | 1 | 30.22% | 56.4% | 0.739 | density > 0.9966 |
| 59 | 0 | 29.64% | 24.4% | 0.681 | alcohol < 9.6 |
| 48 | 0 | 25.41% | 24.6% | 0.622 | alcohol < 11.05 & density < 0.9975 & sulphates < 0.51 |
| 50 | 0 | 25.22% | 57.6% | 0.649 | density < 0.9966 & pH > 3.13 & sulphates > 0.5 |
| 103 | 1 | 16.07% | 41.3% | 0.819 | alcohol > 11.9 |
| 32 | 0 | 15.98% | 44.6% | 0.679 | alcohol > 9.7 & density < 0.9954 & pH < 3.35 & quality == 6 |
| 68 | 1 | 13.76% | 77.6% | 0.648 | alcohol < 11.05 & pH > 2.99 & quality == 5 & sulphates < 0.51 |
| 61 | 1 | 12.51% | 95.4% | 0.536 | alcohol < 9.5 & density > 0.9946 & pH < 3.25 & sulphates < 0.54 |
| 97 | 1 | 11.93% | 58.9% | 0.645 | alcohol > 9.6 & density > 0.9949 & sulphates < 0.609 |
| 16 | 0 | 11.26% | 75.2% | 0.455 | alcohol > 9.4 & density > 0.9961 & pH > 3.15 & sulphates > 0.51 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=573, row=4718) true=`0` pred_idx=1 fired=5 Ω=0.1868 [WRONG]
  - combined: alcohol < 11.053333473205564 AND density < 0.9912600010633469 AND sulphates < 0.44999998807907104 AND pH > 2.990000009536743 AND quality == 5.0
  - combined-as-filter: train cov=0.05% prec=33.33% | test cov=0.10% prec=0.00%
  - top contributions:
    - [agree] w=+0.0938 :: alcohol < 11.05 & pH > 2.99 & quality == 5 & sulphates < 0.51
    - [agree] w=+0.0607 :: alcohol < 11.87 & density < 0.9966
    - [agree] w=+0.0000 :: density < 0.9913 & sulphates < 0.45
- sample(test_idx=497, row=6268) true=`1` pred_idx=0 fired=2 Ω=0.5365 [WRONG]
  - combined: alcohol > 11.053333473205564
  - combined-as-filter: train cov=29.99% prec=55.96% | test cov=31.76% prec=58.79%
  - top contributions:
    - [agree] w=+0.0456 :: alcohol > 11.05
    - [conflict] w=-0.0303 :: alcohol < 11.87 & density < 0.9966
- sample(test_idx=655, row=5439) true=`0` pred_idx=1 fired=5 Ω=0.2498 [WRONG]
  - combined: density < 0.9912600010633469 AND alcohol < 11.870000123977661 AND sulphates < 0.38999998569488525 AND alcohol > 10.5 AND pH < 3.2100000381469727 AND quality == 6.0
  - combined-as-filter: train cov=0.29% prec=25.00% | test cov=0.19% prec=50.00%
  - top contributions:
    - [agree] w=+0.0637 :: alcohol > 10.5 & density < 0.9958 & pH < 3.21 & quality == 6 & sulphates < 0.39
    - [agree] w=+0.0607 :: alcohol < 11.87 & density < 0.9966
    - [conflict] w=-0.0228 :: alcohol > 11.05

#### FOIL

- Data: n_total=6497 (train=5458, test=1039), d=6, label_col=`labels`
- Classes (K=2): 0, 1 | counts=[2827, 3670]
- Rule base: R=164; literals/rule mean=4.78, p90=6.0, max=6
- Fusion depth (rules fired/sample): mean=2.98, p90=4.0, max=7; holes=0.29%

Predictive quality (test):
- Accuracy=0.7276 | F1_macro=0.7169 | F1_weighted=0.7241
- NLL=0.5530 | Brier=0.3688 | ECE(15)=0.0538

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.6194
- Ω combined (unc_comb) mean=0.3098

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=0.8% | literals mean=5.93 (p90=8.0) | distinct features mean=5.22
- Combined rule as filter: test coverage mean=3.82% precision mean=74.47%
- Generalization (train→test): coverage 3.84%→3.82% | precision 74.71%→74.47%

Conflict diagnostics (explain subset):
- conflict rate=77.8% | mean agreeing rules=1.92 | mean conflicting rules=1.06

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 88 | 1 | 26.56% | 68.8% | 0.598 | alcohol < 11.05 & density < 0.9963 & good == 0 & pH > 2.96 & sulphates < 0.6 |
| 50 | 0 | 16.46% | 49.7% | 0.712 | alcohol > 9.3 & density > 0.9949 & pH < 3.3 & sulphates > 0.43 |
| 42 | 0 | 11.65% | 42.1% | 0.664 | density < 0.9943 & good == 0 & pH < 3.3 & sulphates > 0.45 |
| 59 | 1 | 10.49% | 89.0% | 0.449 | alcohol < 11.05 & density > 0.9928 & pH < 3.28 & quality == 6 & sulphates < 0.56 |
| 41 | 0 | 8.85% | 46.7% | 0.809 | alcohol > 9.7 & density < 0.9939 & good == 1 & pH > 3.21 |
| 47 | 0 | 8.66% | 57.8% | 0.598 | alcohol > 9.1 & density > 0.9928 & pH > 3.29 & quality == 6 |
| 46 | 0 | 8.18% | 45.9% | 0.736 | alcohol < 9.6 & density > 0.994 & pH > 3.09 & sulphates > 0.51 |
| 51 | 0 | 7.12% | 50.0% | 0.798 | alcohol > 11.4 & density < 0.9937 & pH < 3.3 & sulphates < 0.46 |
| 56 | 1 | 6.93% | 94.4% | 0.457 | alcohol < 9.5 & density > 0.9947 & pH < 3.25 & quality == 5 & sulphates < 0.54 |
| 141 | 1 | 6.64% | 43.5% | 0.715 | alcohol > 11.05 & density < 0.9913 & good == 0 & sulphates < 0.58 |
| 31 | 0 | 6.54% | 60.3% | 0.597 | alcohol > 9.4 & density > 0.9924 & pH > 3.22 & quality == 5 & sulphates < 0.66 |
| 136 | 1 | 6.06% | 31.7% | 0.719 | alcohol > 11.05 & density < 0.994 & pH > 3.164 & sulphates > 0.51 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=551, row=4785) true=`0` pred_idx=1 fired=3 Ω=0.3017 [WRONG]
  - combined: alcohol < 11.053333473205564 AND density < 0.9962999820709229 AND good == 0.0 AND pH > 2.9600000381469727 AND sulphates < 0.6000000238418579
  - combined-as-filter: train cov=27.37% prec=68.01% | test cov=26.56% prec=68.84%
  - top contributions:
    - [agree] w=+0.1102 :: alcohol < 11.05 & density < 0.9963 & good == 0 & pH > 2.96 & sulphates < 0.6
    - [conflict] w=-0.0936 :: alcohol > 9.7 & density < 0.9966 & pH < 3.13 & quality == 4 & sulphates < 0.58
    - [conflict] w=-0.0000 :: density < 0.9943 & good == 0 & pH < 3.3 & sulphates > 0.45
- sample(test_idx=564, row=1279) true=`0` pred_idx=0 fired=0 Ω=1.0000 [OK]
  - combined: <empty>
  - combined-as-filter: train cov=0.00% prec=nan% | test cov=0.00% prec=nan%
- sample(test_idx=618, row=531) true=`0` pred_idx=1 fired=3 Ω=0.2693 [WRONG]
  - combined: alcohol > 9.600000381469727 AND density > 0.9948599934577942 AND pH < 3.2230000257492066 AND sulphates < 0.6224999815225601
  - combined-as-filter: train cov=6.67% prec=69.78% | test cov=5.58% prec=68.97%
  - top contributions:
    - [agree] w=+0.1286 :: alcohol > 9.6 & density > 0.9949 & pH < 3.223 & sulphates < 0.6225
    - [conflict] w=-0.1283 :: alcohol > 9.5 & density > 0.9949 & good == 0 & pH > 3.13 & quality == 5 & sulphates > 0.55
    - [conflict] w=-0.0000 :: alcohol > 9.3 & density > 0.9949 & pH < 3.3 & sulphates > 0.43

---

## Deep Rule Quality Report (DSGD-Auto)
Источник: `DEEP_RULE_QUALITY_REPORT_ALL.md`

This report evaluates rule-induced DSGD models across datasets, focusing on predictive quality, DST uncertainty behavior, fusion depth, and interpretability of the combined (explanation-only) rule.

Notes:
- Train/test split is stratified with `test_size` and `seed`.
- Probabilities are pignistic; for reproducibility the pignistic prior is set to inverse class frequency on the train split.
- “Combined rule” is built from positive contributions only and is not used for inference; we still evaluate it as a *filter* to measure how sharply it describes the predicted situation and how well it generalizes.

### adult.csv

#### STATIC

- Data: n_total=30162 (train=25336, test=4826), d=14, label_col=`labels`
- Classes (K=2): 0, 1 | counts=[22654, 7508]
- Rule base: R=392; literals/rule mean=2.27, p90=3.0, max=3
- Fusion depth (rules fired/sample): mean=64.26, p90=83.0, max=133; holes=0.00%

Predictive quality (test):
- Accuracy=0.7772 | F1_macro=0.7146 | F1_weighted=0.7818
- NLL=0.4609 | Brier=0.3038 | ECE(15)=0.0200

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.7882
- Ω combined (unc_comb) mean=0.0005

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=14.0% | literals mean=11.20 (p90=15.0) | distinct features mean=9.50
- Combined rule as filter (mean over explain subset): test coverage=0.326% precision=84.40%
- Generalization (mean train→test): coverage 0.337%→0.326% | precision 82.37%→84.40%
- Combined rule distribution (test): coverage p50=0.021% p90=0.186% | precision p50=100.00% p90=100.00%
- Literal decoding (combined rules): decoded categorical share=100.0% (lower means more code-like, less human-readable)

Conflict diagnostics (explain subset):
- conflict rate=34.0% | mean agreeing rules=45.68 | mean conflicting rules=18.93

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 380 | 0 | 90.72% (4378) | 74.7% | 0.800 | native.country == United-States → class 0 |
| 293 | 0 | 88.09% (4251) | 75.0% | 0.842 | fnlwgt > 79385.1 → class 0 |
| 294 | 0 | 87.48% (4222) | 74.6% | 0.702 | fnlwgt < 308091 → class 0 |
| 279 | 0 | 87.01% (4199) | 75.9% | 0.850 | age < 55 → class 0 |
| 280 | 0 | 86.18% (4159) | 71.2% | 0.817 | age > 23 → class 0 |
| 373 | 0 | 86.01% (4151) | 74.0% | 0.800 | race == White → class 0 |
| 309 | 0 | 85.16% (4110) | 72.0% | 0.779 | hours.per.week > 30 → class 0 |
| 310 | 0 | 80.36% (3878) | 79.6% | 0.840 | hours.per.week < 50 → class 0 |
| 27 | 0 | 79.69% (3846) | 74.3% | 0.800 | fnlwgt < 308091 & native.country == United-States → class 0 |
| 24 | 0 | 79.34% (3829) | 74.5% | 0.800 | fnlwgt > 79385.1 & native.country == United-States → class 0 |
| 6 | 0 | 76.83% (3708) | 75.7% | 0.801 | age < 55 & fnlwgt > 79385.1 → class 0 |
| 9 | 0 | 75.88% (3662) | 71.1% | 0.817 | age > 23 & fnlwgt > 79385.1 → class 0 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=272, row=15715) true=`1` pred_idx=1 fired=64 Ω=0.0003 [OK]
  - combined: <empty>
  - combined-as-filter: train matches=0/25336 cov=0.000% prec=nan% | test matches=0/4826 cov=0.000% prec=nan%
  - top contributions:
    - [conflict] w=-0.0432 :: fnlwgt < 308091 → class 0
    - [conflict] w=-0.0396 :: fnlwgt < 201442 → class 0
    - [conflict] w=-0.0307 :: age < 55 & fnlwgt > 79385.1 & hours.per.week < 45 → class 0
- sample(test_idx=4596, row=22595) true=`0` pred_idx=0 fired=37 Ω=0.0057 [OK]
  - combined: fnlwgt < 79385.125 AND hours.per.week > 50.0 AND age > 42.0 AND education.num == 10.0 AND native.country == United-States AND occupation == Sales AND age < 47.0 AND relationship == Not-in-family AND race == White AND marital.status == Never-married AND workclass == Self-emp-not-inc AND education == Some-college
  - combined-as-filter: train matches=0/25336 cov=0.000% prec=nan% | test matches=1/4826 cov=0.021% prec=100.00%
  - top contributions:
    - [agree] w=+0.0863 :: fnlwgt < 308091 → class 0
    - [agree] w=+0.0792 :: fnlwgt < 201442 → class 0
    - [agree] w=+0.0552 :: education.num == 10 & fnlwgt < 308091 → class 0
- sample(test_idx=402, row=3844) true=`0` pred_idx=1 fired=96 Ω=0.0000 [WRONG]
  - combined: capital.gain > 0.0 AND fnlwgt > 117682.5 AND fnlwgt < 308090.875 AND age > 28.0 AND hours.per.week < 45.0 AND race == White AND education == Some-college AND occupation == Prof-specialty AND workclass == Private AND education.num == 10.0
  - combined-as-filter: train matches=11/25336 cov=0.043% prec=72.73% | test matches=3/4826 cov=0.062% prec=66.67%
  - top contributions:
    - [conflict] w=-0.1222 :: hours.per.week < 40 → class 0
    - [agree] w=+0.0828 :: age > 23 & capital.gain > 0 & fnlwgt > 79385.1 → class 1
    - [agree] w=+0.0741 :: capital.gain > 0 & fnlwgt < 308091 & hours.per.week < 45 → class 1

#### RIPPER

- Data: n_total=30162 (train=25336, test=4826), d=14, label_col=`labels`
- Classes (K=2): 0, 1 | counts=[22654, 7508]
- Rule base: R=304; literals/rule mean=4.66, p90=7.0, max=8
- Fusion depth (rules fired/sample): mean=6.32, p90=9.0, max=15; holes=0.00%

Predictive quality (test):
- Accuracy=0.8521 | F1_macro=0.8058 | F1_weighted=0.8534
- NLL=0.3243 | Brier=0.2062 | ECE(15)=0.0115

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.7201
- Ω combined (unc_comb) mean=0.2026

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=0.3% | literals mean=7.58 (p90=10.0) | distinct features mean=7.32
- Combined rule as filter (mean over explain subset): test coverage=0.427% precision=88.92%
- Generalization (mean train→test): coverage 0.397%→0.427% | precision 88.65%→88.92%
- Combined rule distribution (test): coverage p50=0.062% p90=0.769% | precision p50=100.00% p90=100.00%
- Literal decoding (combined rules): decoded categorical share=100.0% (lower means more code-like, less human-readable)

Conflict diagnostics (explain subset):
- conflict rate=68.0% | mean agreeing rules=4.90 | mean conflicting rules=1.35

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 303 | 0 | 69.15% (3337) | 80.6% | 0.851 | age < 45 |
| 117 | 1 | 40.16% (1938) | 35.7% | 0.998 | age > 24 & fnlwgt > 1.308e+05 & sex == Male |
| 302 | 0 | 33.42% (1613) | 62.4% | 0.923 | age > 43 |
| 98 | 1 | 23.95% (1156) | 47.4% | 0.998 | age > 29 & hours.per.week > 40 |
| 195 | 0 | 16.56% (799) | 94.1% | 0.697 | age < 36 & capital.gain < 5054 & sex == Female |
| 155 | 0 | 15.13% (730) | 97.9% | 0.437 | age < 37 & capital.gain < 5054 & marital.status == Never-married & sex == Male |
| 259 | 0 | 14.71% (710) | 60.6% | 0.855 | capital.gain < 5054 & education == Bachelors |
| 299 | 0 | 14.55% (702) | 98.9% | 0.480 | relationship == Own-child |
| 274 | 0 | 11.79% (569) | 58.9% | 0.886 | capital.gain < 6849 & occupation == Prof-specialty |
| 125 | 0 | 10.30% (497) | 99.8% | 0.236 | age < 37 & capital.gain < 5054 & hours.per.week < 50 & marital.status == Never-married & sex == Female |
| 228 | 0 | 10.24% (494) | 94.7% | 0.711 | capital.gain < 5054 & relationship == Unmarried |
| 301 | 0 | 9.66% (466) | 84.5% | 0.903 | age < 45 & education == Some-college & fnlwgt < 1.993e+05 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=106, row=21331) true=`1` pred_idx=1 fired=6 Ω=0.2026 [OK]
  - combined: age > 32.0 AND hours.per.week > 40.0 AND education == Doctorate AND fnlwgt > 130794.0 AND marital.status == Married-civ-spouse AND occupation == Prof-specialty AND sex == Male
  - combined-as-filter: train matches=68/25336 cov=0.268% prec=89.71% | test matches=8/4826 cov=0.166% prec=75.00%
  - top contributions:
    - [agree] w=+0.4309 :: age > 27 & education == Doctorate & hours.per.week > 40
    - [agree] w=+0.0344 :: age > 32 & education == Doctorate & fnlwgt > 1.308e+05 & hours.per.week > 32 & marital.status == Married-civ-spouse & occupation == Prof-specialty
    - [conflict] w=-0.0039 :: capital.gain < 6849 & occupation == Prof-specialty
- sample(test_idx=2341, row=12101) true=`0` pred_idx=1 fired=4 Ω=0.9007 [WRONG]
  - combined: age > 36.0 AND hours.per.week > 40.0 AND education == HS-grad AND marital.status == Married-civ-spouse
  - combined-as-filter: train matches=826/25336 cov=3.260% prec=45.04% | test matches=155/4826 cov=3.212% prec=45.81%
  - top contributions:
    - [conflict] w=-0.0019 :: age > 43
    - [conflict] w=-0.0002 :: fnlwgt < 3.958e+04
    - [agree] w=+0.0000 :: age > 29 & hours.per.week > 40
- sample(test_idx=3621, row=18551) true=`0` pred_idx=1 fired=4 Ω=0.6659 [WRONG]
  - combined: education == Some-college AND marital.status == Married-civ-spouse AND relationship == Husband
  - combined-as-filter: train matches=1968/25336 cov=7.768% prec=44.00% | test matches=399/4826 cov=8.268% prec=43.86%
  - top contributions:
    - [conflict] w=-0.0221 :: fnlwgt < 1.784e+05 & race == Black
    - [conflict] w=-0.0019 :: age > 43
    - [conflict] w=-0.0017 :: age > 37 & capital.gain < 5054 & education == Some-college

#### FOIL

- Data: n_total=30162 (train=25336, test=4826), d=14, label_col=`labels`
- Classes (K=2): 0, 1 | counts=[22654, 7508]
- Rule base: R=614; literals/rule mean=5.45, p90=7.7, max=8
- Fusion depth (rules fired/sample): mean=4.28, p90=8.0, max=16; holes=0.48%

Predictive quality (test):
- Accuracy=0.8601 | F1_macro=0.8215 | F1_weighted=0.8632
- NLL=0.2935 | Brier=0.1912 | ECE(15)=0.0081

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.5767
- Ω combined (unc_comb) mean=0.2320

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=2.3% | literals mean=8.37 (p90=11.0) | distinct features mean=7.82
- Combined rule as filter (mean over explain subset): test coverage=0.311% precision=89.11%
- Generalization (mean train→test): coverage 0.291%→0.311% | precision 88.74%→89.11%
- Combined rule distribution (test): coverage p50=0.062% p90=0.622% | precision p50=100.00% p90=100.00%
- Literal decoding (combined rules): decoded categorical share=100.0% (lower means more code-like, less human-readable)

Conflict diagnostics (explain subset):
- conflict rate=36.7% | mean agreeing rules=3.98 | mean conflicting rules=0.43

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 546 | 0 | 18.98% (916) | 87.0% | 0.788 | capital.gain < 5054 & education == HS-grad & fnlwgt < 2.2e+05 & hours.per.week < 50 |
| 515 | 0 | 10.05% (485) | 79.0% | 0.730 | capital.gain < 5054 & education == Some-college & hours.per.week < 50 & sex == Male |
| 518 | 0 | 9.88% (477) | 88.1% | 0.744 | age > 32 & capital.gain < 5054 & fnlwgt > 1.308e+05 & relationship == Not-in-family |
| 199 | 1 | 9.28% (448) | 26.1% | 0.937 | age < 45 & education == HS-grad & hours.per.week > 32 & marital.status == Married-civ-spouse |
| 232 | 0 | 6.17% (298) | 100.0% | 0.172 | age < 33 & capital.gain < 5054 & education == HS-grad & fnlwgt > 3.836e+04 & hours.per.week < 43 & marital.status == Never-married |
| 315 | 0 | 5.93% (286) | 100.0% | 0.320 | age < 34 & capital.gain < 5054 & fnlwgt > 1.776e+05 & marital.status == Never-married & relationship == Own-child & workclass == Private |
| 252 | 0 | 5.78% (279) | 100.0% | 0.517 | age < 28 & capital.gain < 5054 & fnlwgt > 8.662e+04 & hours.per.week < 32 & marital.status == Never-married |
| 219 | 0 | 5.74% (277) | 100.0% | 0.274 | age < 30 & capital.gain < 5054 & education == HS-grad & fnlwgt > 3.862e+04 & hours.per.week < 50 & marital.status == Never-married |
| 314 | 0 | 5.10% (246) | 98.0% | 0.475 | capital.gain < 5054 & education == HS-grad & fnlwgt < 2.892e+05 & hours.per.week < 50 & relationship == Not-in-family & workclass == Private |
| 214 | 0 | 5.01% (242) | 100.0% | 0.732 | age < 30 & capital.gain < 5054 & education == HS-grad & hours.per.week < 42 & marital.status == Never-married & workclass == Private |
| 523 | 0 | 4.68% (226) | 84.5% | 0.761 | age < 37 & capital.gain < 5054 & education == Bachelors & hours.per.week < 42 |
| 476 | 0 | 4.62% (223) | 98.2% | 0.734 | capital.gain < 5054 & hours.per.week < 42 & occupation == Other-service & sex == Female & workclass == Private |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=2989, row=21067) true=`1` pred_idx=1 fired=4 Ω=0.2358 [OK]
  - combined: age > 36.0 AND marital.status == Married-civ-spouse AND occupation == Adm-clerical AND fnlwgt < 178419.0 AND relationship == Wife AND education == Bachelors AND hours.per.week < 50.0
  - combined-as-filter: train matches=9/25336 cov=0.036% prec=77.78% | test matches=3/4826 cov=0.062% prec=33.33%
  - top contributions:
    - [agree] w=+0.1434 :: age > 27 & fnlwgt < 1.784e+05 & marital.status == Married-civ-spouse & occupation == Adm-clerical & relationship == Wife
    - [agree] w=+0.0645 :: age > 36 & education == Bachelors & hours.per.week < 50 & marital.status == Married-civ-spouse & occupation == Adm-clerical
    - [conflict] w=-0.0525 :: age > 43 & capital.gain < 5054 & occupation == Adm-clerical & sex == Female
- sample(test_idx=3716, row=2079) true=`1` pred_idx=1 fired=0 Ω=1.0000 [OK]
  - combined: <empty>
  - combined-as-filter: train matches=0/25336 cov=0.000% prec=nan% | test matches=0/4826 cov=0.000% prec=nan%
- sample(test_idx=3621, row=18551) true=`0` pred_idx=1 fired=2 Ω=0.7144 [WRONG]
  - combined: education == Some-college AND marital.status == Married-civ-spouse AND occupation == Adm-clerical AND sex == Male
  - combined-as-filter: train matches=126/25336 cov=0.497% prec=40.48% | test matches=27/4826 cov=0.559% prec=62.96%
  - top contributions:
    - [conflict] w=-0.0289 :: capital.gain < 5054 & education == Some-college & hours.per.week < 50 & sex == Male
    - [agree] w=+0.0000 :: education == Some-college & marital.status == Married-civ-spouse & occupation == Adm-clerical & sex == Male

### bank-full.csv

#### STATIC

- Data: n_total=45211 (train=37977, test=7234), d=16, label_col=`labels`
- Classes (K=2): 0, 1 | counts=[39922, 5289]
- Rule base: R=800; literals/rule mean=2.63, p90=3.0, max=3
- Fusion depth (rules fired/sample): mean=220.02, p90=303.0, max=398; holes=0.00%

Predictive quality (test):
- Accuracy=0.8468 | F1_macro=0.6892 | F1_weighted=0.8588
- NLL=0.3439 | Brier=0.2185 | ECE(15)=0.0155

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.7966
- Ω combined (unc_comb) mean=0.0000

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=16.7% | literals mean=15.95 (p90=21.0) | distinct features mean=13.07
- Combined rule as filter (mean over explain subset): test coverage=0.056% precision=95.84%
- Generalization (mean train→test): coverage 0.047%→0.056% | precision 94.32%→95.84%
- Combined rule distribution (test): coverage p50=0.014% p90=0.014% | precision p50=100.00% p90=100.00%
- Literal decoding (combined rules): decoded categorical share=100.0% (lower means more code-like, less human-readable)

Conflict diagnostics (explain subset):
- conflict rate=21.0% | mean agreeing rules=184.77 | mean conflicting rules=31.75

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 775 | 0 | 98.02% (7091) | 88.2% | 0.799 | default == no → class 0 |
| 715 | 0 | 87.90% (6359) | 88.4% | 0.786 | day < 28 → class 0 |
| 730 | 0 | 87.54% (6333) | 86.7% | 0.806 | duration > 67 → class 0 |
| 701 | 0 | 87.12% (6302) | 88.8% | 0.788 | balance < 2983 → class 0 |
| 729 | 0 | 87.09% (6300) | 93.0% | 0.951 | duration < 485 → class 0 |
| 750 | 0 | 86.99% (6293) | 89.3% | 0.810 | pdays < 166 → class 0 |
| 743 | 0 | 86.91% (6287) | 87.5% | 0.828 | campaign < 5 → class 0 |
| 687 | 0 | 86.87% (6284) | 89.1% | 0.862 | age < 55 → class 0 |
| 716 | 0 | 86.52% (6259) | 88.8% | 0.839 | day > 5 → class 0 |
| 688 | 0 | 84.37% (6103) | 89.2% | 0.783 | age > 30 → class 0 |
| 702 | 0 | 84.31% (6099) | 87.4% | 0.803 | balance > 0 → class 0 |
| 779 | 0 | 83.78% (6061) | 87.3% | 0.800 | loan == no → class 0 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=3137, row=10766) true=`0` pred_idx=0 fired=220 Ω=0.0000 [OK]
  - combined: balance < 1430.0 AND balance > 789.0 AND duration < 485.0 AND duration > 318.0 AND day > 16.0 AND pdays < 166.0 AND age > 43.0 AND age < 48.0 AND campaign > 5.0 AND day < 19.0 AND previous < 1.0 AND job == blue-collar
  - combined-as-filter: train matches=0/37977 cov=0.000% prec=nan% | test matches=1/7234 cov=0.014% prec=100.00%
  - top contributions:
    - [agree] w=+0.1336 :: day > 13 → class 0
    - [agree] w=+0.1197 :: age < 55 & balance > 0 & duration < 485 → class 0
    - [agree] w=+0.1100 :: balance > 0 & campaign > 1 & duration < 485 → class 0
- sample(test_idx=2579, row=41919) true=`0` pred_idx=1 fired=76 Ω=0.0000 [WRONG]
  - combined: <empty>
  - combined-as-filter: train matches=0/37977 cov=0.000% prec=nan% | test matches=0/7234 cov=0.000% prec=nan%
  - top contributions:
    - [conflict] w=-0.0668 :: day > 13 → class 0
    - [conflict] w=-0.0278 :: balance > 1430 → class 0
    - [conflict] w=-0.0269 :: balance > 2983 → class 0
- sample(test_idx=3567, row=41053) true=`0` pred_idx=1 fired=97 Ω=0.0000 [WRONG]
  - combined: <empty>
  - combined-as-filter: train matches=0/37977 cov=0.000% prec=nan% | test matches=0/7234 cov=0.000% prec=nan%
  - top contributions:
    - [conflict] w=-0.0668 :: day > 13 → class 0
    - [conflict] w=-0.0507 :: balance > 0 & duration < 485 & previous < 1 → class 0
    - [conflict] w=-0.0278 :: balance > 1430 → class 0

#### RIPPER

- Data: n_total=45211 (train=37977, test=7234), d=16, label_col=`labels`
- Classes (K=2): 0, 1 | counts=[39922, 5289]
- Rule base: R=267; literals/rule mean=4.85, p90=7.0, max=8
- Fusion depth (rules fired/sample): mean=7.00, p90=10.0, max=18; holes=0.00%

Predictive quality (test):
- Accuracy=0.8937 | F1_macro=0.7838 | F1_weighted=0.9019
- NLL=0.2299 | Brier=0.1457 | ECE(15)=0.0122

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.6825
- Ω combined (unc_comb) mean=0.1556

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=0.3% | literals mean=10.17 (p90=14.0) | distinct features mean=9.36
- Combined rule as filter (mean over explain subset): test coverage=0.454% precision=89.82%
- Generalization (mean train→test): coverage 0.448%→0.454% | precision 91.07%→89.82%
- Combined rule distribution (test): coverage p50=0.069% p90=0.677% | precision p50=100.00% p90=100.00%
- Literal decoding (combined rules): decoded categorical share=100.0% (lower means more code-like, less human-readable)

Conflict diagnostics (explain subset):
- conflict rate=59.0% | mean agreeing rules=6.03 | mean conflicting rules=1.07

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 265 | 0 | 60.55% (4380) | 89.7% | 0.923 | campaign > 1 |
| 262 | 0 | 29.06% (2102) | 92.4% | 0.930 | balance < 131 |
| 256 | 0 | 22.95% (1660) | 84.6% | 0.941 | age < 35 & balance > 131 |
| 263 | 0 | 19.98% (1445) | 96.5% | 0.918 | age > 34 & campaign > 2 & duration < 437 |
| 258 | 0 | 19.45% (1407) | 90.7% | 0.885 | age > 36 & campaign < 2 & duration < 507.1 |
| 84 | 1 | 18.74% (1356) | 22.3% | 0.981 | pdays > -1 |
| 183 | 0 | 16.60% (1201) | 96.3% | 0.741 | day < 10 & duration < 279 & pdays < 80 |
| 176 | 0 | 16.51% (1194) | 99.2% | 0.431 | day > 15 & duration < 117 |
| 155 | 0 | 16.28% (1178) | 98.2% | 0.605 | contact == unknown & duration < 768.4 & month == may & poutcome == unknown |
| 73 | 1 | 15.88% (1149) | 22.9% | 0.996 | balance > 131 & campaign > 1 & contact == cellular & duration > 150.1 |
| 205 | 0 | 13.91% (1006) | 89.7% | 0.904 | month == aug |
| 111 | 0 | 13.67% (989) | 99.5% | 0.379 | contact == unknown & duration < 539 & housing == yes & month == may & poutcome == unknown |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=4698, row=39512) true=`1` pred_idx=0 fired=7 Ω=0.1639 [WRONG]
  - combined: duration < 180.0 AND poutcome == failure AND balance > 450.0 AND campaign < 3.0 AND month == may AND age < 37.0 AND balance < 3091.5 AND day > 10.0 AND education == tertiary AND job == technician AND campaign > 1.0 AND previous > 1.0
  - combined-as-filter: train matches=1/37977 cov=0.003% prec=100.00% | test matches=1/7234 cov=0.014% prec=0.00%
  - top contributions:
    - [agree] w=+0.3564 :: duration < 180 & poutcome == failure
    - [agree] w=+0.1284 :: balance > 450 & campaign < 3 & duration < 413 & month == may
    - [agree] w=+0.0484 :: age < 37 & balance < 3092 & day > 10 & duration < 611 & education == tertiary & job == technician
- sample(test_idx=3265, row=27122) true=`1` pred_idx=1 fired=4 Ω=0.9009 [OK]
  - combined: duration > 437.0 AND campaign < 2.0 AND contact == cellular AND previous < 1.0 AND balance > 131.0 AND housing == no
  - combined-as-filter: train matches=383/37977 cov=1.009% prec=49.61% | test matches=66/7234 cov=0.912% prec=50.00%
  - top contributions:
    - [conflict] w=-0.0016 :: age > 34 & balance > 176 & duration > 289.8 & education == secondary & previous < 2
    - [agree] w=+0.0000 :: campaign < 2 & contact == cellular & duration > 146.7 & previous < 1
    - [agree] w=+0.0000 :: balance > 131 & campaign < 2 & duration > 126 & housing == no
- sample(test_idx=3567, row=41053) true=`0` pred_idx=1 fired=7 Ω=0.5828 [WRONG]
  - combined: campaign < 2.0 AND duration > 211.0 AND housing == no AND age > 57.0 AND job == retired AND contact == cellular AND previous < 1.0 AND balance > 131.0
  - combined-as-filter: train matches=98/37977 cov=0.258% prec=48.98% | test matches=18/7234 cov=0.249% prec=44.44%
  - top contributions:
    - [agree] w=+0.0328 :: age > 57 & campaign < 2 & duration > 211 & housing == no & job == retired
    - [conflict] w=-0.0060 :: age > 36 & campaign < 2 & duration < 507.1
    - [conflict] w=-0.0041 :: month == aug

#### FOIL

- Data: n_total=45211 (train=37977, test=7234), d=16, label_col=`labels`
- Classes (K=2): 0, 1 | counts=[39922, 5289]
- Rule base: R=417; literals/rule mean=5.31, p90=7.0, max=8
- Fusion depth (rules fired/sample): mean=5.61, p90=9.0, max=21; holes=0.17%

Predictive quality (test):
- Accuracy=0.8983 | F1_macro=0.7994 | F1_weighted=0.9073
- NLL=0.2077 | Brier=0.1338 | ECE(15)=0.0182

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.5367
- Ω combined (unc_comb) mean=0.1650

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=3.0% | literals mean=11.09 (p90=15.0) | distinct features mean=10.01
- Combined rule as filter (mean over explain subset): test coverage=0.401% precision=92.17%
- Generalization (mean train→test): coverage 0.389%→0.401% | precision 92.36%→92.17%
- Combined rule distribution (test): coverage p50=0.055% p90=0.597% | precision p50=100.00% p90=100.00%
- Literal decoding (combined rules): decoded categorical share=100.0% (lower means more code-like, less human-readable)

Conflict diagnostics (explain subset):
- conflict rate=33.7% | mean agreeing rules=5.13 | mean conflicting rules=0.50

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 265 | 0 | 22.75% (1646) | 95.6% | 0.709 | balance < 1125 & day > 10 & duration < 873.7 & marital == married & poutcome == unknown |
| 377 | 0 | 16.19% (1171) | 95.5% | 0.869 | day < 13 & duration < 756.2 & housing == yes & previous < 2 |
| 183 | 0 | 11.97% (866) | 99.0% | 0.494 | age < 59 & day > 4 & duration < 520.1 & education == secondary & month == may & poutcome == unknown |
| 384 | 0 | 10.95% (792) | 88.6% | 0.955 | balance < 1125 & day < 16 & duration < 380.5 & housing == no |
| 47 | 1 | 10.02% (725) | 26.3% | 0.990 | balance > 334.5 & contact == cellular & duration > 115.3 & education == tertiary |
| 375 | 0 | 9.65% (698) | 91.4% | 0.936 | balance < 1125 & contact == cellular & day > 16 & housing == yes |
| 37 | 1 | 9.47% (685) | 32.6% | 0.998 | campaign < 2 & duration > 136 & housing == no & loan == no |
| 297 | 0 | 9.19% (665) | 95.5% | 0.864 | age < 37 & balance < 1841 & duration < 270.4 & marital == single & pdays < 82.6 |
| 97 | 0 | 7.87% (569) | 100.0% | 0.030 | age > 26 & campaign > 1 & contact == unknown & day > 6 & duration < 382 & housing == yes & previous < 1 |
| 68 | 0 | 7.76% (561) | 100.0% | 0.029 | age > 30 & contact == unknown & day < 26 & duration < 437 & housing == yes & month == may & previous < 1 |
| 65 | 0 | 7.67% (555) | 100.0% | 0.132 | age > 33 & contact == unknown & duration < 310 & housing == yes & month == may & previous < 1 |
| 56 | 0 | 7.67% (555) | 100.0% | 0.100 | contact == unknown & day < 27 & duration < 258 & housing == yes & month == may & previous < 1 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=2579, row=41919) true=`0` pred_idx=0 fired=6 Ω=0.1527 [OK]
  - combined: balance > 1125.0 AND previous < 1.0 AND contact == cellular AND duration > 756.1999999999971 AND month == oct AND age > 38.0 AND campaign < 2.0 AND day > 10.0
  - combined-as-filter: train matches=2/37977 cov=0.005% prec=100.00% | test matches=1/7234 cov=0.014% prec=100.00%
  - top contributions:
    - [agree] w=+0.6006 :: balance > 1125 & contact == cellular & duration > 756.2 & month == oct & previous < 1
    - [conflict] w=-0.0910 :: age > 59 & contact == cellular & duration > 365 & job == retired & poutcome == unknown
    - [conflict] w=-0.0080 :: age > 59 & balance > 318.5 & duration > 180 & housing == no
- sample(test_idx=5166, row=31085) true=`1` pred_idx=1 fired=0 Ω=1.0000 [OK]
  - combined: <empty>
  - combined-as-filter: train matches=0/37977 cov=0.000% prec=nan% | test matches=0/7234 cov=0.000% prec=nan%
- sample(test_idx=3567, row=41053) true=`0` pred_idx=1 fired=6 Ω=0.2834 [WRONG]
  - combined: duration > 365.0 AND age > 59.0 AND contact == cellular AND job == retired AND poutcome == unknown AND housing == no AND balance > 318.5 AND campaign < 2.0 AND loan == no
  - combined-as-filter: train matches=36/37977 cov=0.095% prec=63.89% | test matches=9/7234 cov=0.124% prec=55.56%
  - top contributions:
    - [agree] w=+0.1819 :: age > 59 & contact == cellular & duration > 365 & job == retired & poutcome == unknown
    - [conflict] w=-0.0731 :: age > 46 & day < 15 & duration < 756.2 & education == secondary & month == aug & poutcome == unknown
    - [agree] w=+0.0160 :: age > 59 & balance > 318.5 & duration > 180 & housing == no

### BrainTumor.csv

#### STATIC

- Data: n_total=3762 (train=3160, test=602), d=13, label_col=`labels`
- Classes (K=2): 0, 1 | counts=[2079, 1683]
- Rule base: R=853; literals/rule mean=2.52, p90=3.0, max=3
- Fusion depth (rules fired/sample): mean=498.42, p90=750.7, max=769; holes=0.00%

Predictive quality (test):
- Accuracy=0.9767 | F1_macro=0.9765 | F1_weighted=0.9768
- NLL=0.0584 | Brier=0.0327 | ECE(15)=0.0138

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.7864
- Ω combined (unc_comb) mean=0.0000

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=0.0% | literals mean=21.41 (p90=25.0) | distinct features mean=12.94
- Combined rule as filter (mean over explain subset): test coverage=1.450% precision=99.72%
- Generalization (mean train→test): coverage 1.471%→1.450% | precision 99.77%→99.72%
- Combined rule distribution (test): coverage p50=0.664% p90=3.156% | precision p50=100.00% p90=100.00%
- Literal decoding (combined rules): decoded categorical share=nan% (lower means more code-like, less human-readable)

Conflict diagnostics (explain subset):
- conflict rate=100.0% | mean agreeing rules=303.97 | mean conflicting rules=197.39

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 852 | 0 | 100.00% (602) | 55.3% | 0.797 | Coarseness == 0 → class 0 |
| 754 | 0 | 89.53% (539) | 51.4% | 0.764 | Kurtosis > 5.77742 → class 0 |
| 740 | 0 | 89.37% (538) | 51.1% | 0.819 | Skewness > 2.32286 → class 0 |
| 684 | 0 | 88.87% (535) | 57.9% | 0.749 | Mean > 3.37515 → class 0 |
| 741 | 0 | 88.87% (535) | 61.3% | 0.739 | Skewness < 5.95781 → class 0 |
| 755 | 0 | 88.70% (534) | 61.6% | 0.765 | Kurtosis < 36.8853 → class 0 |
| 838 | 0 | 88.04% (530) | 57.0% | 0.803 | Correlation > 0.935306 → class 0 |
| 825 | 0 | 88.04% (530) | 61.3% | 0.798 | Dissimilarity < 6.63431 → class 0 |
| 769 | 0 | 88.04% (530) | 58.1% | 0.816 | Contrast < 205.338 → class 0 |
| 698 | 0 | 87.38% (526) | 52.9% | 0.781 | Variance > 227.831 → class 0 |
| 712 | 0 | 87.38% (526) | 52.9% | 0.788 | Standard Deviation > 15.0941 → class 0 |
| 18 | 0 | 87.38% (526) | 52.9% | 0.786 | Standard Deviation > 15.0941 & Variance > 227.831 → class 0 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=447, row=2314) true=`0` pred_idx=0 fired=497 Ω=0.0000 [OK]
  - combined: Variance < 365.8471145629883 AND Variance > 227.83107566833496 AND ASM > 0.07025238871574402 AND Kurtosis < 16.088111400604248 AND Entropy > 0.09039528109133244 AND Contrast > 72.54677391052246 AND Homogeneity > 0.5123224258422852 AND Kurtosis > 7.251713275909424 AND Standard Deviation < 19.12712812423706 AND Energy > 0.2650516629219055 AND Contrast < 131.896240234375 AND Correlation < 0.9471380561590195
  - combined-as-filter: train matches=14/3160 cov=0.443% prec=100.00% | test matches=2/602 cov=0.332% prec=100.00%
  - top contributions:
    - [agree] w=+0.4228 :: ASM > 0.0510134 → class 0
    - [agree] w=+0.3521 :: Entropy > 0.0668942 → class 0
    - [agree] w=+0.1971 :: Homogeneity > 0.512322 → class 0
- sample(test_idx=125, row=421) true=`0` pred_idx=0 fired=151 Ω=0.0000 [OK]
  - combined: Homogeneity > 0.6213260963559151 AND ASM > 0.12290586438030005 AND Variance < 227.83107566833496 AND Entropy > 0.15228710882365704 AND Correlation < 0.9471380561590195 AND Standard Deviation < 15.094074010848999 AND Contrast < 52.42851448059082 AND Energy > 0.35057931393384933 AND Kurtosis > 7.251713275909424 AND Dissimilarity < 2.823696345090866 AND Mean < 10.56283950805664 AND Skewness > 2.618549644947052
  - combined-as-filter: train matches=5/3160 cov=0.158% prec=100.00% | test matches=1/602 cov=0.166% prec=100.00%
  - top contributions:
    - [agree] w=+0.4228 :: ASM > 0.0510134 → class 0
    - [agree] w=+0.3521 :: Entropy > 0.0668942 → class 0
    - [agree] w=+0.3263 :: Homogeneity > 0.575229 → class 0
- sample(test_idx=230, row=3219) true=`1` pred_idx=0 fired=646 Ω=0.0000 [WRONG]
  - combined: Variance > 365.8471145629883 AND Variance < 968.5724792480469 AND Kurtosis < 7.251713275909424 AND Contrast > 72.54677391052246 AND Contrast < 88.18684101104736 AND Mean > 13.241966247558594 AND Skewness < 2.618549644947052 AND Standard Deviation > 19.12712812423706 AND Homogeneity > 0.4377995394170284 AND Energy > 0.131877813488245 AND Entropy > 0.023764305049553514 AND ASM > 0.017391758039593697
  - combined-as-filter: train matches=105/3160 cov=3.323% prec=98.10% | test matches=11/602 cov=1.827% prec=90.91%
  - top contributions:
    - [agree] w=+0.2414 :: Kurtosis < 9.58856 → class 0
    - [conflict] w=-0.1972 :: Entropy < 0.0668942 → class 1
    - [conflict] w=-0.1869 :: Energy < 0.225861 → class 1

#### RIPPER

- Data: n_total=3762 (train=3160, test=602), d=13, label_col=`labels`
- Classes (K=2): 0, 1 | counts=[2079, 1683]
- Rule base: R=23; literals/rule mean=2.83, p90=4.0, max=4
- Fusion depth (rules fired/sample): mean=4.83, p90=7.0, max=9; holes=0.00%

Predictive quality (test):
- Accuracy=0.9884 | F1_macro=0.9882 | F1_weighted=0.9884
- NLL=0.0366 | Brier=0.0168 | ECE(15)=0.0149

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.4344
- Ω combined (unc_comb) mean=0.0260

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=0.0% | literals mean=6.35 (p90=8.0) | distinct features mean=6.15
- Combined rule as filter (mean over explain subset): test coverage=11.453% precision=99.94%
- Generalization (mean train→test): coverage 10.779%→11.453% | precision 99.79%→99.94%
- Combined rule distribution (test): coverage p50=7.807% p90=28.904% | precision p50=100.00% p90=100.00%
- Literal decoding (combined rules): decoded categorical share=nan% (lower means more code-like, less human-readable)

Conflict diagnostics (explain subset):
- conflict rate=21.7% | mean agreeing rules=4.48 | mean conflicting rules=0.27

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 22 | 0 | 51.83% (312) | 98.1% | 0.426 | Entropy > 0.065 |
| 7 | 1 | 49.17% (296) | 88.9% | 0.869 | Entropy < 0.06689 |
| 15 | 0 | 40.86% (246) | 99.2% | 0.227 | Contrast < 158.8 & Energy > 0.2111 & Entropy > 0.04169 & Mean < 16.16 |
| 0 | 1 | 40.86% (246) | 100.0% | 0.118 | Entropy < 0.04646 & Kurtosis > 5.992 |
| 8 | 1 | 35.88% (216) | 86.6% | 0.648 | Homogeneity < 0.5629 & Kurtosis > 12.38 |
| 2 | 1 | 30.40% (183) | 99.5% | 0.117 | Correlation > 0.9487 & Energy < 0.2148 & Kurtosis > 5.992 & Skewness > 2.291 |
| 12 | 0 | 29.73% (179) | 99.4% | 0.517 | Contrast < 161.7 & Entropy > 0.05613 & Homogeneity > 0.5629 |
| 9 | 1 | 29.07% (175) | 70.3% | 0.962 | Variance > 945.4 |
| 10 | 1 | 28.41% (171) | 80.1% | 0.601 | Kurtosis > 19.49 |
| 11 | 0 | 27.57% (166) | 100.0% | 0.642 | Dissimilarity < 4.482 & Energy > 0.2259 & Entropy > 0.05919 & Kurtosis < 12.4 |
| 1 | 1 | 21.59% (130) | 99.2% | 0.190 | Entropy < 0.05307 & Homogeneity < 0.5021 & Variance > 869.1 |
| 16 | 0 | 15.28% (92) | 98.9% | 0.189 | Entropy > 0.05136 & Mean > 12.2 & Variance < 1625 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=415, row=2089) true=`0` pred_idx=0 fired=5 Ω=0.0145 [OK]
  - combined: Entropy > 0.06500032842159272 AND Energy > 0.21110250800848007 AND Homogeneity > 0.5123224258422852 AND Variance > 655.5255126953125 AND Contrast < 158.78328552246091 AND Mean < 16.155856323242187
  - combined-as-filter: train matches=247/3160 cov=7.816% prec=100.00% | test matches=47/602 cov=7.807% prec=100.00%
  - top contributions:
    - [agree] w=+0.7183 :: Energy > 0.2033 & Homogeneity > 0.5123 & Variance > 655.5
    - [agree] w=+0.5929 :: Contrast < 158.8 & Energy > 0.2111 & Entropy > 0.04169 & Mean < 16.16
    - [agree] w=+0.3232 :: Entropy > 0.065
- sample(test_idx=340, row=229) true=`0` pred_idx=0 fired=3 Ω=0.5873 [OK]
  - combined: Contrast > 149.01501312255857 AND Entropy > 0.03786910437047482 AND Homogeneity > 0.5047804653644562 AND Mean > 5.759965515136718
  - combined-as-filter: train matches=198/3160 cov=6.266% prec=95.45% | test matches=28/602 cov=4.651% prec=100.00%
  - top contributions:
    - [agree] w=+0.1090 :: Contrast > 149 & Entropy > 0.03787 & Homogeneity > 0.5048 & Mean > 5.76
    - [conflict] w=-0.0075 :: Entropy < 0.06689
    - [conflict] w=-0.0005 :: Variance > 945.4
- sample(test_idx=311, row=213) true=`1` pred_idx=0 fired=4 Ω=0.1058 [WRONG]
  - combined: Contrast < 158.78328552246091 AND Entropy > 0.05612604152411223 AND Energy > 0.21110250800848007 AND Mean < 16.155856323242187 AND Homogeneity > 0.5629000842571258
  - combined-as-filter: train matches=781/3160 cov=24.715% prec=99.74% | test matches=175/602 cov=29.070% prec=99.43%
  - top contributions:
    - [agree] w=+0.5929 :: Contrast < 158.8 & Energy > 0.2111 & Entropy > 0.04169 & Mean < 16.16
    - [agree] w=+0.2329 :: Contrast < 161.7 & Entropy > 0.05613 & Homogeneity > 0.5629
    - [conflict] w=-0.0647 :: Kurtosis > 19.49

#### FOIL

- Data: n_total=3762 (train=3160, test=602), d=13, label_col=`labels`
- Classes (K=2): 0, 1 | counts=[2079, 1683]
- Rule base: R=60; literals/rule mean=4.32, p90=6.0, max=8
- Fusion depth (rules fired/sample): mean=4.05, p90=7.0, max=11; holes=0.00%

Predictive quality (test):
- Accuracy=0.9983 | F1_macro=0.9983 | F1_weighted=0.9983
- NLL=0.0163 | Brier=0.0047 | ECE(15)=0.0085

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.2246
- Ω combined (unc_comb) mean=0.0193

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=0.0% | literals mean=7.78 (p90=11.0) | distinct features mean=7.62
- Combined rule as filter (mean over explain subset): test coverage=5.827% precision=99.89%
- Generalization (mean train→test): coverage 5.731%→5.827% | precision 100.00%→99.89%
- Combined rule distribution (test): coverage p50=2.658% p90=17.143% | precision p50=100.00% p90=100.00%
- Literal decoding (combined rules): decoded categorical share=nan% (lower means more code-like, less human-readable)

Conflict diagnostics (explain subset):
- conflict rate=0.3% | mean agreeing rules=3.99 | mean conflicting rules=0.00

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 41.20% (248) | 100.0% | 0.106 | Entropy < 0.0487 & Kurtosis > 5.992 |
| 6 | 1 | 28.74% (173) | 100.0% | 0.367 | Correlation > 0.9507 & Energy < 0.2105 & Entropy < 0.06689 & Kurtosis > 5.614 & Skewness > 2.37 |
| 30 | 0 | 25.58% (154) | 100.0% | 0.132 | Contrast < 88.82 & Energy > 0.2259 & Entropy > 0.05906 & Homogeneity > 0.5185 & Kurtosis < 19.49 |
| 23 | 1 | 25.42% (153) | 100.0% | 0.455 | Correlation > 0.9442 & Entropy < 0.0619 & Homogeneity < 0.5347 & Kurtosis > 8.123 & Skewness > 3.049 |
| 2 | 1 | 22.09% (133) | 100.0% | 0.266 | Entropy < 0.06034 & Homogeneity < 0.5123 & Kurtosis > 19.49 & Skewness > 2.37 |
| 1 | 1 | 21.26% (128) | 100.0% | 0.171 | Entropy < 0.05215 & Homogeneity < 0.5007 & Kurtosis > 4.999 & Variance > 869.1 |
| 33 | 0 | 20.10% (121) | 100.0% | 0.317 | Energy > 0.2633 & Entropy > 0.05387 & Kurtosis < 12.38 & Skewness < 4.967 |
| 9 | 1 | 19.44% (117) | 99.1% | 0.343 | Correlation > 0.9507 & Entropy < 0.06689 & Kurtosis > 8.982 & Mean > 4.598 |
| 4 | 1 | 18.11% (109) | 100.0% | 0.638 | Correlation > 0.9645 & Energy < 0.2089 & Entropy < 0.06453 & Kurtosis > 5.648 |
| 39 | 0 | 17.11% (103) | 100.0% | 0.135 | Contrast < 115.3 & Correlation > 0.9599 & Energy > 0.2196 & Entropy > 0.05451 & Homogeneity > 0.5029 & Kurtosis < 32.63 & Variance < 628.6 |
| 31 | 0 | 13.29% (80) | 100.0% | 0.139 | Contrast < 161.3 & Dissimilarity > 3.616 & Entropy > 0.05641 & Homogeneity > 0.5123 & Kurtosis < 19.49 |
| 51 | 0 | 11.96% (72) | 100.0% | 0.283 | Contrast < 156.2 & Energy > 0.2875 & Entropy > 0.06027 & Mean < 8.462 & Skewness < 4.393 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=58, row=464) true=`1` pred_idx=1 fired=4 Ω=0.0153 [OK]
  - combined: Entropy < 0.060335636138916016 AND Kurtosis > 19.486724662780748 AND Skewness > 3.0491999864578245 AND Correlation > 0.9507338464260101 AND Homogeneity < 0.5123224258422852 AND Mean > 4.598040771484374 AND Energy < 0.2104817032814026
  - combined-as-filter: train matches=289/3160 cov=9.146% prec=100.00% | test matches=49/602 cov=8.140% prec=100.00%
  - top contributions:
    - [agree] w=+0.5384 :: Entropy < 0.06034 & Homogeneity < 0.5123 & Kurtosis > 19.49 & Skewness > 2.37
    - [agree] w=+0.4312 :: Correlation > 0.9507 & Entropy < 0.06689 & Kurtosis > 8.982 & Mean > 4.598
    - [agree] w=+0.4012 :: Correlation > 0.9507 & Energy < 0.2105 & Entropy < 0.06689 & Kurtosis > 5.614 & Skewness > 2.37
- sample(test_idx=311, row=213) true=`1` pred_idx=1 fired=1 Ω=0.3433 [OK]
  - combined: Correlation > 0.9507338464260101 AND Entropy < 0.06689422205090523 AND Kurtosis > 8.981733226776122 AND Mean > 4.598040771484374
  - combined-as-filter: train matches=627/3160 cov=19.842% prec=100.00% | test matches=117/602 cov=19.435% prec=99.15%
  - top contributions:
    - [agree] w=+0.4312 :: Correlation > 0.9507 & Entropy < 0.06689 & Kurtosis > 8.982 & Mean > 4.598
- sample(test_idx=561, row=330) true=`0` pred_idx=1 fired=1 Ω=0.0733 [WRONG]
  - combined: Correlation > 0.9644977152347565 AND Dissimilarity < 4.626947212219238 AND Entropy < 0.08327286094427108 AND Homogeneity < 0.5123224258422852 AND Mean > 15.79658203125 AND Skewness > 2.08024183511734
  - combined-as-filter: train matches=26/3160 cov=0.823% prec=100.00% | test matches=3/602 cov=0.498% prec=66.67%
  - top contributions:
    - [agree] w=+0.8588 :: Correlation > 0.9645 & Dissimilarity < 4.627 & Entropy < 0.08327 & Homogeneity < 0.5123 & Mean > 15.8 & Skewness > 2.08

### breast-cancer-wisconsin.csv

#### STATIC

- Data: n_total=683 (train=574, test=109), d=9, label_col=`labels`
- Classes (K=2): 0, 1 | counts=[444, 239]
- Rule base: R=559; literals/rule mean=2.56, p90=3.0, max=3
- Fusion depth (rules fired/sample): mean=25.22, p90=41.0, max=41; holes=0.00%

Predictive quality (test):
- Accuracy=0.9725 | F1_macro=0.9699 | F1_weighted=0.9726
- NLL=0.2407 | Brier=0.1177 | ECE(15)=0.1671

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.7779
- Ω combined (unc_comb) mean=0.0530

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=0.0% | literals mean=8.39 (p90=9.0) | distinct features mean=8.39
- Combined rule as filter (mean over explain subset): test coverage=1.355% precision=97.25%
- Generalization (mean train→test): coverage 0.671%→1.355% | precision 100.00%→97.25%
- Combined rule distribution (test): coverage p50=0.917% p90=2.752% | precision p50=100.00% p90=100.00%
- Literal decoding (combined rules): decoded categorical share=nan% (lower means more code-like, less human-readable)

Conflict diagnostics (explain subset):
- conflict rate=45.0% | mean agreeing rules=24.21 | mean conflicting rules=1.01

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 550 | 0 | 78.90% (86) | 79.1% | 0.846 | mitoses == 1 → class 0 |
| 540 | 0 | 63.30% (69) | 89.9% | 0.819 | normal_nucleoli == 1 → class 0 |
| 500 | 0 | 55.05% (60) | 93.3% | 0.736 | marginal_adhesion == 1 → class 0 |
| 520 | 0 | 52.29% (57) | 98.2% | 0.728 | bare_nucleoli == 1 → class 0 |
| 480 | 0 | 52.29% (57) | 100.0% | 0.732 | size_uniformity == 1 → class 0 |
| 510 | 0 | 52.29% (57) | 96.5% | 0.724 | epithelial_size == 2 → class 0 |
| 490 | 0 | 46.79% (51) | 100.0% | 0.728 | shape_uniformity == 1 → class 0 |
| 60 | 0 | 46.79% (51) | 100.0% | 0.832 | normal_nucleoli == 1 & size_uniformity == 1 → class 0 |
| 18 | 0 | 45.87% (50) | 98.0% | 0.810 | bare_nucleoli == 1 & normal_nucleoli == 1 → class 0 |
| 343 | 0 | 44.95% (49) | 98.0% | 0.733 | bare_nucleoli == 1 & mitoses == 1 & normal_nucleoli == 1 → class 0 |
| 54 | 0 | 44.04% (48) | 100.0% | 0.746 | shape_uniformity == 1 & size_uniformity == 1 → class 0 |
| 167 | 0 | 43.12% (47) | 100.0% | 0.822 | bare_nucleoli == 1 & mitoses == 1 & size_uniformity == 1 → class 0 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=14, row=213) true=`0` pred_idx=0 fired=26 Ω=0.0147 [OK]
  - combined: bare_nucleoli == 1.0 AND size_uniformity == 1.0 AND marginal_adhesion == 1.0 AND bland_chromatin == 3.0 AND epithelial_size == 2.0 AND normal_nucleoli == 1.0 AND mitoses == 1.0 AND shape_uniformity == 3.0
  - combined-as-filter: train matches=1/574 cov=0.174% prec=100.00% | test matches=1/109 cov=0.917% prec=100.00%
  - top contributions:
    - [agree] w=+0.0670 :: bare_nucleoli == 1 → class 0
    - [agree] w=+0.0636 :: bare_nucleoli == 1 & epithelial_size == 2 & size_uniformity == 1 → class 0
    - [agree] w=+0.0635 :: bare_nucleoli == 1 & bland_chromatin == 3 → class 0
- sample(test_idx=7, row=610) true=`1` pred_idx=1 fired=9 Ω=0.2521 [OK]
  - combined: normal_nucleoli == 7.0 AND shape_uniformity == 6.0 AND size_uniformity == 6.0 AND epithelial_size == 7.0 AND bare_nucleoli == 6.0 AND bland_chromatin == 7.0 AND mitoses == 3.0 AND marginal_adhesion == 5.0
  - combined-as-filter: train matches=0/574 cov=0.000% prec=nan% | test matches=1/109 cov=0.917% prec=100.00%
  - top contributions:
    - [agree] w=+0.0706 :: normal_nucleoli == 7 → class 1
    - [agree] w=+0.0242 :: shape_uniformity == 6 → class 1
    - [agree] w=+0.0159 :: size_uniformity == 6 → class 1
- sample(test_idx=38, row=226) true=`0` pred_idx=1 fired=16 Ω=0.0656 [WRONG]
  - combined: epithelial_size == 3.0 AND bland_chromatin == 4.0 AND size_uniformity == 4.0 AND shape_uniformity == 6.0 AND clump_thickness == 8.0 AND normal_nucleoli == 3.0
  - combined-as-filter: train matches=0/574 cov=0.000% prec=nan% | test matches=1/109 cov=0.917% prec=0.00%
  - top contributions:
    - [agree] w=+0.0716 :: epithelial_size == 3 → class 1
    - [agree] w=+0.0541 :: bland_chromatin == 4 → class 1
    - [conflict] w=-0.0335 :: bare_nucleoli == 1 → class 0

#### RIPPER

- Data: n_total=683 (train=574, test=109), d=9, label_col=`labels`
- Classes (K=2): 0, 1 | counts=[444, 239]
- Rule base: R=18; literals/rule mean=1.33, p90=2.0, max=3
- Fusion depth (rules fired/sample): mean=2.94, p90=5.0, max=5; holes=2.75%

Predictive quality (test):
- Accuracy=0.8899 | F1_macro=0.8857 | F1_weighted=0.8923
- NLL=0.5561 | Brier=0.3655 | ECE(15)=0.2880

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.7527
- Ω combined (unc_comb) mean=0.4879

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=11.9% | literals mean=3.22 (p90=6.0) | distinct features mean=3.22
- Combined rule as filter (mean over explain subset): test coverage=16.817% precision=99.83%
- Generalization (mean train→test): coverage 20.447%→16.817% | precision 99.48%→99.83%
- Combined rule distribution (test): coverage p50=22.936% p90=33.028% | precision p50=100.00% p90=100.00%
- Literal decoding (combined rules): decoded categorical share=nan% (lower means more code-like, less human-readable)

Conflict diagnostics (explain subset):
- conflict rate=15.6% | mean agreeing rules=2.72 | mean conflicting rules=0.23

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 15 | 0 | 55.05% (60) | 93.3% | 0.691 | marginal_adhesion == 1 |
| 13 | 0 | 46.79% (51) | 100.0% | 0.696 | shape_uniformity == 1 |
| 10 | 0 | 43.12% (47) | 100.0% | 0.692 | bare_nucleoli == 1 & size_uniformity == 1 |
| 12 | 0 | 41.28% (45) | 100.0% | 0.692 | bare_nucleoli == 1 & epithelial_size == 2 |
| 11 | 0 | 32.11% (35) | 100.0% | 0.692 | epithelial_size == 2 & normal_nucleoli == 1 & shape_uniformity == 1 |
| 0 | 1 | 19.27% (21) | 100.0% | 0.855 | bare_nucleoli == 10 |
| 3 | 1 | 13.76% (15) | 100.0% | 0.805 | size_uniformity == 10 |
| 14 | 0 | 11.01% (12) | 83.3% | 0.859 | shape_uniformity == 2 |
| 1 | 1 | 6.42% (7) | 100.0% | 0.767 | clump_thickness == 10 |
| 2 | 1 | 6.42% (7) | 100.0% | 0.836 | normal_nucleoli == 10 |
| 8 | 1 | 5.50% (6) | 83.3% | 0.679 | clump_thickness == 8 |
| 4 | 1 | 4.59% (5) | 100.0% | 0.866 | marginal_adhesion == 4 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=9, row=420) true=`1` pred_idx=1 fired=3 Ω=0.4877 [OK]
  - combined: clump_thickness == 10.0 AND bare_nucleoli == 10.0
  - combined-as-filter: train matches=36/574 cov=6.272% prec=100.00% | test matches=3/109 cov=2.752% prec=100.00%
  - top contributions:
    - [agree] w=+0.0375 :: clump_thickness == 10
    - [conflict] w=-0.0276 :: marginal_adhesion == 1
    - [agree] w=+0.0093 :: bare_nucleoli == 10
- sample(test_idx=37, row=285) true=`0` pred_idx=1 fired=0 Ω=1.0000 [WRONG]
  - combined: <empty>
  - combined-as-filter: train matches=0/574 cov=0.000% prec=nan% | test matches=0/109 cov=0.000% prec=nan%
- sample(test_idx=87, row=470) true=`0` pred_idx=1 fired=1 Ω=0.6956 [WRONG]
  - combined: <empty>
  - combined-as-filter: train matches=0/574 cov=0.000% prec=nan% | test matches=0/109 cov=0.000% prec=nan%
  - top contributions:
    - [conflict] w=-0.0318 :: shape_uniformity == 1

#### FOIL

- Data: n_total=683 (train=574, test=109), d=9, label_col=`labels`
- Classes (K=2): 0, 1 | counts=[444, 239]
- Rule base: R=42; literals/rule mean=2.17, p90=3.0, max=4
- Fusion depth (rules fired/sample): mean=2.75, p90=5.0, max=5; holes=0.00%

Predictive quality (test):
- Accuracy=0.4587 | F1_macro=0.4261 | F1_weighted=0.3846
- NLL=0.6745 | Brier=0.4830 | ECE(15)=0.2660

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.7952
- Ω combined (unc_comb) mean=0.5766

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=53.2% | literals mean=1.72 (p90=5.2) | distinct features mean=1.72
- Combined rule as filter (mean over explain subset): test coverage=1.498% precision=98.04%
- Generalization (mean train→test): coverage 1.280%→1.498% | precision 100.00%→98.04%
- Combined rule distribution (test): coverage p50=0.000% p90=5.505% | precision p50=100.00% p90=100.00%
- Literal decoding (combined rules): decoded categorical share=nan% (lower means more code-like, less human-readable)

Conflict diagnostics (explain subset):
- conflict rate=53.2% | mean agreeing rules=1.12 | mean conflicting rules=1.63

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 23 | 0 | 43.12% (47) | 100.0% | 0.880 | bare_nucleoli == 1 & size_uniformity == 1 |
| 25 | 0 | 41.28% (45) | 100.0% | 0.865 | bare_nucleoli == 1 & epithelial_size == 2 |
| 26 | 0 | 35.78% (39) | 100.0% | 0.780 | marginal_adhesion == 1 & normal_nucleoli == 1 & shape_uniformity == 1 |
| 24 | 0 | 32.11% (35) | 100.0% | 0.690 | epithelial_size == 2 & normal_nucleoli == 1 & shape_uniformity == 1 |
| 30 | 0 | 20.18% (22) | 100.0% | 0.693 | bland_chromatin == 1 & size_uniformity == 1 |
| 3 | 1 | 13.76% (15) | 100.0% | 0.835 | size_uniformity == 10 |
| 0 | 1 | 9.17% (10) | 100.0% | 0.823 | bare_nucleoli == 10 & marginal_adhesion == 10 |
| 1 | 1 | 8.26% (9) | 100.0% | 0.815 | bare_nucleoli == 10 & bland_chromatin == 7 |
| 5 | 1 | 6.42% (7) | 100.0% | 0.838 | clump_thickness == 10 |
| 8 | 1 | 6.42% (7) | 100.0% | 0.877 | normal_nucleoli == 10 |
| 28 | 0 | 5.50% (6) | 100.0% | 0.748 | bare_nucleoli == 2 & size_uniformity == 1 |
| 4 | 1 | 4.59% (5) | 100.0% | 0.664 | bare_nucleoli == 10 & clump_thickness == 5 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=40, row=408) true=`0` pred_idx=1 fired=3 Ω=0.5558 [WRONG]
  - combined: <empty>
  - combined-as-filter: train matches=0/574 cov=0.000% prec=nan% | test matches=0/109 cov=0.000% prec=nan%
  - top contributions:
    - [conflict] w=-0.0376 :: bland_chromatin == 2 & clump_thickness == 5 & normal_nucleoli == 1
    - [conflict] w=-0.0030 :: bare_nucleoli == 1 & epithelial_size == 2
    - [conflict] w=-0.0010 :: bare_nucleoli == 1 & size_uniformity == 1
- sample(test_idx=99, row=435) true=`0` pred_idx=1 fired=1 Ω=0.8799 [WRONG]
  - combined: <empty>
  - combined-as-filter: train matches=0/574 cov=0.000% prec=nan% | test matches=0/109 cov=0.000% prec=nan%
  - top contributions:
    - [conflict] w=-0.0010 :: bare_nucleoli == 1 & size_uniformity == 1
- sample(test_idx=12, row=458) true=`0` pred_idx=1 fired=5 Ω=0.3425 [WRONG]
  - combined: <empty>
  - combined-as-filter: train matches=0/574 cov=0.000% prec=nan% | test matches=0/109 cov=0.000% prec=nan%
  - top contributions:
    - [conflict] w=-0.0332 :: epithelial_size == 2 & normal_nucleoli == 1 & shape_uniformity == 1
    - [conflict] w=-0.0324 :: bland_chromatin == 1 & size_uniformity == 1
    - [conflict] w=-0.0124 :: marginal_adhesion == 1 & normal_nucleoli == 1 & shape_uniformity == 1

### SAheart.csv

#### STATIC

- Data: n_total=462 (train=388, test=74), d=9, label_col=`chd`
- Classes (K=2): 0, 1 | counts=[302, 160]
- Rule base: R=796; literals/rule mean=2.63, p90=3.0, max=3
- Fusion depth (rules fired/sample): mean=419.61, p90=652.9, max=705; holes=0.00%

Predictive quality (test):
- Accuracy=0.7297 | F1_macro=0.6980 | F1_weighted=0.7271
- NLL=0.5405 | Brier=0.3681 | ECE(15)=0.1237

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.8003
- Ω combined (unc_comb) mean=0.0000

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=0.0% | literals mean=11.58 (p90=16.7) | distinct features mean=7.50
- Combined rule as filter (mean over explain subset): test coverage=2.411% precision=74.74%
- Generalization (mean train→test): coverage 1.564%→2.411% | precision 76.23%→74.74%
- Combined rule distribution (test): coverage p50=1.351% p90=4.054% | precision p50=100.00% p90=100.00%
- Literal decoding (combined rules): decoded categorical share=100.0% (lower means more code-like, less human-readable)

Conflict diagnostics (explain subset):
- conflict rate=62.2% | mean agreeing rules=285.26 | mean conflicting rules=134.35

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 781 | 0 | 91.89% (68) | 69.1% | 0.798 | age < 60 → class 0 |
| 712 | 0 | 91.89% (68) | 63.2% | 0.797 | ldl > 2.66 → class 0 |
| 685 | 0 | 89.19% (66) | 62.1% | 0.813 | sbp > 118 → class 0 |
| 780 | 0 | 89.19% (66) | 60.6% | 0.792 | age > 21.75 → class 0 |
| 684 | 0 | 87.84% (65) | 67.7% | 0.782 | sbp < 160.625 → class 0 |
| 711 | 0 | 87.84% (65) | 67.7% | 0.781 | ldl < 7.0775 → class 0 |
| 740 | 0 | 87.84% (65) | 64.6% | 0.791 | typea > 42 → class 0 |
| 726 | 0 | 86.49% (64) | 67.2% | 0.820 | adiposity < 34.46 → class 0 |
| 725 | 0 | 86.49% (64) | 59.4% | 0.806 | adiposity > 15.16 → class 0 |
| 698 | 0 | 86.49% (64) | 68.8% | 0.766 | tobacco < 7.9625 → class 0 |
| 753 | 0 | 86.49% (64) | 60.9% | 0.830 | obesity > 21.705 → class 0 |
| 739 | 0 | 85.14% (63) | 61.9% | 0.817 | typea < 64 → class 0 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=60, row=317) true=`0` pred_idx=0 fired=422 Ω=0.0000 [OK]
  - combined: alcohol > 22.832499504089355 AND alcohol < 42.9150013923645 AND typea < 50.0 AND tobacco > 0.817500002682209 AND sbp < 118.0 AND age < 31.0 AND adiposity < 15.15999984741211 AND obesity > 22.947500705718994 AND ldl > 3.7812499701976776 AND tobacco < 1.85999995470047 AND ldl < 4.355000019073486 AND age > 21.75
  - combined-as-filter: train matches=0/388 cov=0.000% prec=nan% | test matches=1/74 cov=1.351% prec=100.00%
  - top contributions:
    - [agree] w=+0.0608 :: age < 39 → class 0
    - [agree] w=+0.0575 :: alcohol < 42.915 & tobacco < 7.9625 & typea < 60 → class 0
    - [agree] w=+0.0552 :: alcohol < 42.915 & sbp < 160.625 & typea < 64 → class 0
- sample(test_idx=25, row=430) true=`0` pred_idx=0 fired=108 Ω=0.0000 [OK]
  - combined: alcohol < 0.5775000154972076 AND age < 21.75 AND tobacco < 0.0774999987334013 AND sbp < 124.0 AND obesity < 21.705000162124634 AND ldl < 4.355000019073486 AND typea > 64.0 AND adiposity < 19.739999771118164 AND ldl > 3.7812499701976776 AND adiposity > 15.15999984741211 AND famhist == Absent
  - combined-as-filter: train matches=0/388 cov=0.000% prec=nan% | test matches=1/74 cov=1.351% prec=100.00%
  - top contributions:
    - [agree] w=+0.0648 :: age < 21.75 → class 0
    - [agree] w=+0.0608 :: age < 39 → class 0
    - [agree] w=+0.0521 :: alcohol < 42.915 & sbp < 160.625 & tobacco < 7.9625 → class 0
- sample(test_idx=2, row=157) true=`0` pred_idx=1 fired=351 Ω=0.0000 [WRONG]
  - combined: sbp > 160.625 AND tobacco > 7.962500035762787 AND adiposity > 34.459999084472656 AND age > 50.0
  - combined-as-filter: train matches=2/388 cov=0.515% prec=100.00% | test matches=2/74 cov=2.703% prec=50.00%
  - top contributions:
    - [agree] w=+0.0463 :: sbp > 160.625 → class 1
    - [agree] w=+0.0442 :: tobacco > 7.9625 → class 1
    - [agree] w=+0.0335 :: age > 50 → class 1

#### RIPPER

- Data: n_total=462 (train=388, test=74), d=9, label_col=`chd`
- Classes (K=2): 0, 1 | counts=[302, 160]
- Rule base: R=25; literals/rule mean=3.12, p90=4.6, max=5
- Fusion depth (rules fired/sample): mean=3.49, p90=5.0, max=6; holes=0.00%

Predictive quality (test):
- Accuracy=0.8514 | F1_macro=0.8322 | F1_weighted=0.8491
- NLL=0.3376 | Brier=0.1990 | ECE(15)=0.1137

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.6896
- Ω combined (unc_comb) mean=0.2964

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=0.0% | literals mean=4.89 (p90=7.0) | distinct features mean=4.66
- Combined rule as filter (mean over explain subset): test coverage=7.725% precision=88.76%
- Generalization (mean train→test): coverage 7.084%→7.725% | precision 80.82%→88.76%
- Combined rule distribution (test): coverage p50=5.405% p90=14.054% | precision p50=100.00% p90=100.00%
- Literal decoding (combined rules): decoded categorical share=100.0% (lower means more code-like, less human-readable)

Conflict diagnostics (explain subset):
- conflict rate=91.9% | mean agreeing rules=2.43 | mean conflicting rules=1.05

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 12 | 1 | 59.46% (44) | 20.5% | 0.771 | tobacco < 1.86 |
| 23 | 0 | 47.30% (35) | 62.9% | 0.890 | typea > 53 |
| 11 | 1 | 40.54% (30) | 56.7% | 0.925 | tobacco > 1.86 |
| 19 | 0 | 29.73% (22) | 90.9% | 0.609 | tobacco < 1.18 & typea < 56 |
| 22 | 0 | 25.68% (19) | 63.2% | 0.791 | adiposity > 26.1 & ldl < 5.407 |
| 13 | 0 | 22.97% (17) | 94.1% | 0.632 | age < 49 & ldl < 3.531 & obesity > 21.72 |
| 15 | 0 | 21.62% (16) | 100.0% | 0.554 | age < 38 & sbp < 144 & tobacco < 1.86 & typea < 65 |
| 17 | 0 | 20.27% (15) | 86.7% | 0.650 | obesity < 25.72 & tobacco < 0.4 |
| 16 | 0 | 12.16% (9) | 88.9% | 0.384 | alcohol > 7.01 & famhist == Absent & obesity > 25.72 & tobacco < 7.398 |
| 1 | 1 | 8.11% (6) | 83.3% | 0.298 | adiposity < 26.48 & age > 38.4 & famhist == Present & tobacco < 5.44 |
| 2 | 1 | 6.76% (5) | 100.0% | 0.382 | adiposity < 29.86 & age > 51 & ldl > 3.102 & tobacco > 1.86 & typea < 53 |
| 20 | 0 | 6.76% (5) | 80.0% | 0.410 | adiposity < 26.1 & age < 53 & tobacco > 4.03 & typea < 60.7 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=26, row=75) true=`0` pred_idx=0 fired=3 Ω=0.3051 [OK]
  - combined: tobacco < 0.4000000059604645 AND typea < 56.0 AND obesity < 25.72000026702881
  - combined-as-filter: train matches=43/388 cov=11.082% prec=90.70% | test matches=9/74 cov=12.162% prec=100.00%
  - top contributions:
    - [agree] w=+0.1326 :: tobacco < 1.18 & typea < 56
    - [agree] w=+0.1105 :: obesity < 25.72 & tobacco < 0.4
    - [conflict] w=-0.0000 :: tobacco < 1.86
- sample(test_idx=66, row=368) true=`0` pred_idx=1 fired=2 Ω=0.8231 [WRONG]
  - combined: tobacco > 1.85999995470047
  - combined-as-filter: train matches=204/388 cov=52.577% prec=46.57% | test matches=30/74 cov=40.541% prec=56.67%
  - top contributions:
    - [conflict] w=-0.0037 :: typea > 53
    - [agree] w=+0.0000 :: tobacco > 1.86
- sample(test_idx=21, row=128) true=`1` pred_idx=0 fired=4 Ω=0.1719 [WRONG]
  - combined: ldl > 7.096000003814697 AND typea < 49.2 AND famhist == Absent AND obesity > 25.72000026702881
  - combined-as-filter: train matches=9/388 cov=2.320% prec=88.89% | test matches=2/74 cov=2.703% prec=50.00%
  - top contributions:
    - [agree] w=+0.2971 :: famhist == Absent & ldl > 5.407 & typea < 49.2
    - [conflict] w=-0.1561 :: adiposity < 29.86 & age > 51 & ldl > 3.102 & tobacco > 1.86 & typea < 53
    - [agree] w=+0.1019 :: ldl > 7.096 & obesity > 25.72 & typea < 51.2

#### FOIL

- Data: n_total=462 (train=388, test=74), d=9, label_col=`chd`
- Classes (K=2): 0, 1 | counts=[302, 160]
- Rule base: R=78; literals/rule mean=4.91, p90=7.0, max=8
- Fusion depth (rules fired/sample): mean=1.66, p90=3.0, max=4; holes=2.70%

Predictive quality (test):
- Accuracy=0.9459 | F1_macro=0.9396 | F1_weighted=0.9454
- NLL=0.2534 | Brier=0.1002 | ECE(15)=0.0981

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.2773
- Ω combined (unc_comb) mean=0.1758

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=2.7% | literals mean=6.74 (p90=10.0) | distinct features mean=6.34
- Combined rule as filter (mean over explain subset): test coverage=3.214% precision=93.19%
- Generalization (mean train→test): coverage 1.867%→3.214% | precision 95.56%→93.19%
- Combined rule distribution (test): coverage p50=2.027% p90=6.351% | precision p50=100.00% p90=100.00%
- Literal decoding (combined rules): decoded categorical share=100.0% (lower means more code-like, less human-readable)

Conflict diagnostics (explain subset):
- conflict rate=5.4% | mean agreeing rules=1.59 | mean conflicting rules=0.07

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 40 | 0 | 16.22% (12) | 100.0% | 0.234 | adiposity > 12.13 & age < 40 & obesity < 27.83 & tobacco < 0.4 & typea < 66 |
| 49 | 0 | 8.11% (6) | 100.0% | 0.260 | adiposity < 17.16 & age < 41.1 & ldl < 4.395 & sbp > 122.1 & typea > 50 |
| 58 | 0 | 8.11% (6) | 66.7% | 0.281 | adiposity > 29.86 & ldl < 5.407 & tobacco < 1.86 & typea < 63 |
| 43 | 0 | 6.76% (5) | 80.0% | 0.226 | famhist == Absent & obesity > 24.11 & sbp < 134 & tobacco < 7.398 & typea > 57.4 |
| 48 | 0 | 6.76% (5) | 80.0% | 0.216 | adiposity > 26.1 & age < 62.95 & sbp < 154 & tobacco < 0.751 & typea < 56 |
| 64 | 0 | 6.76% (5) | 100.0% | 0.228 | age < 56.4 & alcohol > 0.544 & ldl < 4.355 & obesity < 23.63 & sbp < 154.8 & tobacco > 0.4 & typea < 62.8 |
| 67 | 0 | 6.76% (5) | 100.0% | 0.242 | alcohol > 7.494 & ldl < 5.11 & obesity > 24.97 & tobacco < 0.846 & typea < 57.3 |
| 54 | 0 | 5.41% (4) | 100.0% | 0.287 | adiposity < 16.3 & ldl < 5.163 & obesity > 22.01 & typea < 62 |
| 60 | 0 | 5.41% (4) | 100.0% | 0.252 | adiposity > 21.1 & age < 53 & alcohol > 18.51 & ldl < 6.649 & obesity < 26.76 & sbp > 126.1 & tobacco < 7.96 & typea > 53 |
| 0 | 1 | 5.41% (4) | 100.0% | 0.253 | age > 38.95 & alcohol > 0 & famhist == Present & ldl > 5.04 & obesity < 34.6 & sbp > 152 |
| 22 | 1 | 4.05% (3) | 100.0% | 0.297 | alcohol > 17.17 & ldl > 3.186 & sbp < 147.3 & tobacco > 1.86 & typea < 44.15 |
| 63 | 0 | 4.05% (3) | 66.7% | 0.261 | age < 54.4 & alcohol < 1.54 & obesity < 26.53 & sbp > 126.1 & tobacco < 1.67 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=0, row=262) true=`0` pred_idx=1 fired=2 Ω=0.1772 [WRONG]
  - combined: adiposity < 26.104999542236328 AND age > 34.0 AND famhist == Present AND obesity < 22.010000228881836
  - combined-as-filter: train matches=7/388 cov=1.804% prec=85.71% | test matches=1/74 cov=1.351% prec=0.00%
  - top contributions:
    - [agree] w=+0.4756 :: adiposity < 26.1 & age > 34 & famhist == Present & obesity < 22.01
    - [conflict] w=-0.2503 :: age > 59 & alcohol < 7.01 & obesity < 26.33 & sbp < 166 & tobacco < 6 & typea < 63
- sample(test_idx=73, row=118) true=`1` pred_idx=1 fired=0 Ω=1.0000 [OK]
  - combined: <empty>
  - combined-as-filter: train matches=0/388 cov=0.000% prec=nan% | test matches=0/74 cov=0.000% prec=nan%
- sample(test_idx=8, row=258) true=`1` pred_idx=0 fired=2 Ω=0.1453 [WRONG]
  - combined: adiposity < 31.399999618530273 AND age > 47.0 AND ldl > 4.869999885559082 AND obesity > 23.630999183654787 AND sbp < 144.0 AND tobacco > 1.7999999523162842 AND typea > 49.0
  - combined-as-filter: train matches=7/388 cov=1.804% prec=100.00% | test matches=1/74 cov=1.351% prec=0.00%
  - top contributions:
    - [agree] w=+0.5853 :: adiposity < 31.4 & age > 47 & ldl > 4.87 & obesity > 23.63 & sbp < 144 & tobacco > 1.8 & typea > 49
    - [conflict] w=-0.2623 :: adiposity > 23.59 & age > 28.25 & alcohol > 11.32 & ldl > 4.455 & sbp < 146.7 & tobacco > 7.398 & typea > 49

### df_wine.csv

#### STATIC

- Data: n_total=6497 (train=5458, test=1039), d=6, label_col=`labels`
- Classes (K=2): 0, 1 | counts=[2827, 3670]
- Rule base: R=542; literals/rule mean=2.64, p90=3.0, max=3
- Fusion depth (rules fired/sample): mean=198.15, p90=297.0, max=298; holes=0.00%

Predictive quality (test):
- Accuracy=0.6266 | F1_macro=0.6120 | F1_weighted=0.6218
- NLL=0.6523 | Brier=0.4602 | ECE(15)=0.0306

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.8009
- Ω combined (unc_comb) mean=0.0000

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=0.0% | literals mean=7.75 (p90=10.0) | distinct features mean=5.71
- Combined rule as filter (mean over explain subset): test coverage=1.431% precision=70.07%
- Generalization (mean train→test): coverage 1.267%→1.431% | precision 70.97%→70.07%
- Combined rule distribution (test): coverage p50=0.385% p90=2.993% | precision p50=75.00% p90=100.00%
- Literal decoding (combined rules): decoded categorical share=nan% (lower means more code-like, less human-readable)

Conflict diagnostics (explain subset):
- conflict rate=96.0% | mean agreeing rules=123.29 | mean conflicting rules=76.15

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 477 | 1 | 87.78% (912) | 56.5% | 0.801 | density < 0.99818 → class 1 |
| 506 | 1 | 86.81% (902) | 55.4% | 0.807 | sulphates > 0.38 → class 1 |
| 505 | 1 | 85.95% (893) | 61.3% | 0.782 | sulphates < 0.69 → class 1 |
| 491 | 1 | 85.95% (893) | 58.6% | 0.815 | pH < 3.4 → class 1 |
| 519 | 1 | 85.85% (892) | 58.9% | 0.794 | alcohol < 12.1 → class 1 |
| 478 | 1 | 85.76% (891) | 58.8% | 0.812 | density > 0.991 → class 1 |
| 492 | 1 | 85.76% (891) | 55.1% | 0.803 | pH > 3.04 → class 1 |
| 520 | 1 | 83.93% (872) | 52.2% | 0.789 | alcohol > 9.2 → class 1 |
| 36 | 1 | 80.27% (834) | 60.4% | 0.811 | alcohol < 12.1 & density > 0.991 → class 1 |
| 540 | 1 | 78.73% (818) | 57.2% | 0.808 | good == 0 → class 1 |
| 34 | 1 | 77.48% (805) | 53.8% | 0.793 | alcohol > 9.2 & density < 0.99818 → class 1 |
| 27 | 1 | 75.84% (788) | 57.7% | 0.809 | alcohol < 12.1 & sulphates > 0.38 → class 1 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=797, row=5978) true=`1` pred_idx=1 fired=198 Ω=0.0000 [OK]
  - combined: alcohol < 9.5 AND pH < 3.0399999618530273 AND density < 0.9948599934577942 AND sulphates > 0.4300000071525574 AND good == 0.0 AND quality == 5.0 AND sulphates < 0.550000011920929 AND density > 0.9936000108718872 AND alcohol > 9.199999809265137
  - combined-as-filter: train matches=1/5458 cov=0.018% prec=100.00% | test matches=1/1039 cov=0.096% prec=100.00%
  - top contributions:
    - [agree] w=+0.0526 :: pH < 3.32 → class 1
    - [agree] w=+0.0486 :: density > 0.991 & good == 0 & pH < 3.4 → class 1
    - [agree] w=+0.0480 :: alcohol < 10.8 → class 1
- sample(test_idx=110, row=6385) true=`1` pred_idx=0 fired=61 Ω=0.0003 [WRONG]
  - combined: alcohol > 12.100000381469727 AND pH > 3.4000000953674316 AND density < 0.9909999966621399
  - combined-as-filter: train matches=27/5458 cov=0.495% prec=51.85% | test matches=11/1039 cov=1.059% prec=36.36%
  - top contributions:
    - [agree] w=+0.0469 :: alcohol > 12.1 → class 0
    - [agree] w=+0.0362 :: alcohol > 10.8 → class 0
    - [agree] w=+0.0351 :: alcohol > 9.8 → class 0
- sample(test_idx=850, row=844) true=`0` pred_idx=1 fired=248 Ω=0.0000 [WRONG]
  - combined: alcohol < 10.800000190734863 AND sulphates > 0.4300000071525574 AND density < 0.9969900250434875 AND pH > 3.1600000858306885 AND quality == 6.0 AND good == 0.0 AND pH < 3.2100000381469727 AND density > 0.9958599805831909 AND alcohol > 9.5
  - combined-as-filter: train matches=8/5458 cov=0.147% prec=62.50% | test matches=3/1039 cov=0.289% prec=33.33%
  - top contributions:
    - [agree] w=+0.0526 :: pH < 3.32 → class 1
    - [agree] w=+0.0486 :: density > 0.991 & good == 0 & pH < 3.4 → class 1
    - [agree] w=+0.0480 :: alcohol < 10.8 → class 1

#### RIPPER

- Data: n_total=6497 (train=5458, test=1039), d=6, label_col=`labels`
- Classes (K=2): 0, 1 | counts=[2827, 3670]
- Rule base: R=104; literals/rule mean=4.29, p90=5.0, max=6
- Fusion depth (rules fired/sample): mean=4.81, p90=6.0, max=9; holes=0.00%

Predictive quality (test):
- Accuracy=0.7151 | F1_macro=0.7068 | F1_weighted=0.7132
- NLL=0.5613 | Brier=0.3725 | ECE(15)=0.0768

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.6538
- Ω combined (unc_comb) mean=0.1879

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=0.0% | literals mean=5.58 (p90=7.0) | distinct features mean=4.79
- Combined rule as filter (mean over explain subset): test coverage=3.328% precision=75.94%
- Generalization (mean train→test): coverage 3.305%→3.328% | precision 76.29%→75.94%
- Combined rule distribution (test): coverage p50=1.155% p90=8.951% | precision p50=75.00% p90=100.00%
- Literal decoding (combined rules): decoded categorical share=nan% (lower means more code-like, less human-readable)

Conflict diagnostics (explain subset):
- conflict rate=98.3% | mean agreeing rules=2.89 | mean conflicting rules=1.94

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 99 | 1 | 53.61% (557) | 60.7% | 0.682 | alcohol < 11.87 & density < 0.9966 |
| 56 | 0 | 31.76% (330) | 58.8% | 0.715 | alcohol > 11.05 |
| 98 | 1 | 30.22% (314) | 56.4% | 0.739 | density > 0.9966 |
| 59 | 0 | 29.64% (308) | 24.4% | 0.681 | alcohol < 9.6 |
| 48 | 0 | 25.41% (264) | 24.6% | 0.622 | alcohol < 11.05 & density < 0.9975 & sulphates < 0.51 |
| 50 | 0 | 25.22% (262) | 57.6% | 0.649 | density < 0.9966 & pH > 3.13 & sulphates > 0.5 |
| 103 | 1 | 16.07% (167) | 41.3% | 0.819 | alcohol > 11.9 |
| 32 | 0 | 15.98% (166) | 44.6% | 0.679 | alcohol > 9.7 & density < 0.9954 & pH < 3.35 & quality == 6 |
| 68 | 1 | 13.76% (143) | 77.6% | 0.648 | alcohol < 11.05 & pH > 2.99 & quality == 5 & sulphates < 0.51 |
| 61 | 1 | 12.51% (130) | 95.4% | 0.536 | alcohol < 9.5 & density > 0.9946 & pH < 3.25 & sulphates < 0.54 |
| 97 | 1 | 11.93% (124) | 58.9% | 0.645 | alcohol > 9.6 & density > 0.9949 & sulphates < 0.609 |
| 16 | 0 | 11.26% (117) | 75.2% | 0.455 | alcohol > 9.4 & density > 0.9961 & pH > 3.15 & sulphates > 0.51 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=573, row=4718) true=`0` pred_idx=1 fired=5 Ω=0.1868 [WRONG]
  - combined: alcohol < 11.053333473205564 AND density < 0.9912600010633469 AND sulphates < 0.44999998807907104 AND pH > 2.990000009536743 AND quality == 5.0
  - combined-as-filter: train matches=3/5458 cov=0.055% prec=33.33% | test matches=1/1039 cov=0.096% prec=0.00%
  - top contributions:
    - [agree] w=+0.0938 :: alcohol < 11.05 & pH > 2.99 & quality == 5 & sulphates < 0.51
    - [agree] w=+0.0607 :: alcohol < 11.87 & density < 0.9966
    - [agree] w=+0.0000 :: density < 0.9913 & sulphates < 0.45
- sample(test_idx=457, row=5089) true=`1` pred_idx=0 fired=2 Ω=0.5365 [WRONG]
  - combined: alcohol > 11.053333473205564
  - combined-as-filter: train matches=1637/5458 cov=29.993% prec=55.96% | test matches=330/1039 cov=31.761% prec=58.79%
  - top contributions:
    - [agree] w=+0.0456 :: alcohol > 11.05
    - [conflict] w=-0.0303 :: alcohol < 11.87 & density < 0.9966
- sample(test_idx=387, row=4569) true=`1` pred_idx=0 fired=5 Ω=0.2313 [WRONG]
  - combined: alcohol > 11.053333473205564 AND density < 0.9927999973297119 AND pH < 3.130000114440918 AND sulphates < 0.4399999976158142
  - combined-as-filter: train matches=162/5458 cov=2.968% prec=57.41% | test matches=35/1039 cov=3.369% prec=65.71%
  - top contributions:
    - [agree] w=+0.0624 :: alcohol > 11.05 & density < 0.9928 & pH < 3.13 & sulphates < 0.44
    - [agree] w=+0.0456 :: alcohol > 11.05
    - [conflict] w=-0.0303 :: alcohol < 11.87 & density < 0.9966

#### FOIL

- Data: n_total=6497 (train=5458, test=1039), d=6, label_col=`labels`
- Classes (K=2): 0, 1 | counts=[2827, 3670]
- Rule base: R=164; literals/rule mean=4.78, p90=6.0, max=6
- Fusion depth (rules fired/sample): mean=2.98, p90=4.0, max=7; holes=0.29%

Predictive quality (test):
- Accuracy=0.7276 | F1_macro=0.7169 | F1_weighted=0.7241
- NLL=0.5530 | Brier=0.3688 | ECE(15)=0.0538

Uncertainty / DS behavior:
- Ω rule-avg (unc_rule) mean=0.6194
- Ω combined (unc_comb) mean=0.3098

Combined-rule interpretability (explain subset; explanation-only):
- Combined empty rate=1.0% | literals mean=5.96 (p90=8.0) | distinct features mean=5.27
- Combined rule as filter (mean over explain subset): test coverage=3.385% precision=76.45%
- Generalization (mean train→test): coverage 3.405%→3.385% | precision 76.74%→76.45%
- Combined rule distribution (test): coverage p50=1.636% p90=6.930% | precision p50=81.82% p90=100.00%
- Literal decoding (combined rules): decoded categorical share=nan% (lower means more code-like, less human-readable)

Conflict diagnostics (explain subset):
- conflict rate=73.7% | mean agreeing rules=1.94 | mean conflicting rules=1.02

Most-used rules on test (top coverage):
| rule_id | label | coverage | precision | Ω | caption |
|---:|---:|---:|---:|---:|---|
| 88 | 1 | 26.56% (276) | 68.8% | 0.598 | alcohol < 11.05 & density < 0.9963 & good == 0 & pH > 2.96 & sulphates < 0.6 |
| 50 | 0 | 16.46% (171) | 49.7% | 0.712 | alcohol > 9.3 & density > 0.9949 & pH < 3.3 & sulphates > 0.43 |
| 42 | 0 | 11.65% (121) | 42.1% | 0.664 | density < 0.9943 & good == 0 & pH < 3.3 & sulphates > 0.45 |
| 59 | 1 | 10.49% (109) | 89.0% | 0.449 | alcohol < 11.05 & density > 0.9928 & pH < 3.28 & quality == 6 & sulphates < 0.56 |
| 41 | 0 | 8.85% (92) | 46.7% | 0.809 | alcohol > 9.7 & density < 0.9939 & good == 1 & pH > 3.21 |
| 47 | 0 | 8.66% (90) | 57.8% | 0.598 | alcohol > 9.1 & density > 0.9928 & pH > 3.29 & quality == 6 |
| 46 | 0 | 8.18% (85) | 45.9% | 0.736 | alcohol < 9.6 & density > 0.994 & pH > 3.09 & sulphates > 0.51 |
| 51 | 0 | 7.12% (74) | 50.0% | 0.798 | alcohol > 11.4 & density < 0.9937 & pH < 3.3 & sulphates < 0.46 |
| 56 | 1 | 6.93% (72) | 94.4% | 0.457 | alcohol < 9.5 & density > 0.9947 & pH < 3.25 & quality == 5 & sulphates < 0.54 |
| 141 | 1 | 6.64% (69) | 43.5% | 0.715 | alcohol > 11.05 & density < 0.9913 & good == 0 & sulphates < 0.58 |
| 31 | 0 | 6.54% (68) | 60.3% | 0.597 | alcohol > 9.4 & density > 0.9924 & pH > 3.22 & quality == 5 & sulphates < 0.66 |
| 136 | 1 | 6.06% (63) | 31.7% | 0.719 | alcohol > 11.05 & density < 0.994 & pH > 3.164 & sulphates > 0.51 |

Examples (decoded combined rule + generalization of that combined rule):
- sample(test_idx=877, row=1416) true=`0` pred_idx=0 fired=3 Ω=0.3151 [OK]
  - combined: alcohol > 9.5 AND density > 0.9948599934577942 AND sulphates > 0.550000011920929 AND good == 0.0 AND pH > 3.130000114440918 AND quality == 5.0 AND pH < 3.299999952316284
  - combined-as-filter: train matches=73/5458 cov=1.337% prec=80.82% | test matches=16/1039 cov=1.540% prec=93.75%
  - top contributions:
    - [agree] w=+0.2566 :: alcohol > 9.5 & density > 0.9949 & good == 0 & pH > 3.13 & quality == 5 & sulphates > 0.55
    - [conflict] w=-0.0205 :: alcohol < 10.3 & density > 0.9966 & pH < 3.36 & quality == 5 & sulphates > 0.51
    - [agree] w=+0.0000 :: alcohol > 9.3 & density > 0.9949 & pH < 3.3 & sulphates > 0.43
- sample(test_idx=269, row=2044) true=`0` pred_idx=0 fired=0 Ω=1.0000 [OK]
  - combined: <empty>
  - combined-as-filter: train matches=0/5458 cov=0.000% prec=nan% | test matches=0/1039 cov=0.000% prec=nan%
- sample(test_idx=387, row=4569) true=`1` pred_idx=0 fired=4 Ω=0.3403 [WRONG]
  - combined: alcohol > 11.399999618530273 AND density < 0.9927999973297119 AND pH < 3.2200000286102295 AND sulphates < 0.4399999976158142 AND good == 0.0
  - combined-as-filter: train matches=130/5458 cov=2.382% prec=53.85% | test matches=25/1039 cov=2.406% prec=48.00%
  - top contributions:
    - [agree] w=+0.0508 :: alcohol > 9.4 & density < 0.9928 & good == 0 & pH < 3.22 & sulphates < 0.44
    - [agree] w=+0.0000 :: alcohol > 11.4 & density < 0.9937 & pH < 3.3 & sulphates < 0.46
    - [conflict] w=-0.0000 :: alcohol > 11.05 & density < 0.9913 & good == 0 & sulphates < 0.58

---

## Rule Inspector Report (DSGD-Auto)
Источник: `RULE_INSPECTOR_REPORT.md`

This report shows, per dataset and rule-induction method, (1) rule-base scale, (2) fusion depth proxy (rules fired/sample), and (3) combined-rule examples produced by `DSModelMultiQ.get_combined_rule` (explanation-only).

Combined condition is built from positive rule contributions, aggregated per feature/op into a single conjunctive string; it is not used for inference.

### adult.csv

#### adult.csv — STATIC

- Rules: 392; mean literals/rule: 2.27; p90: 3
- Fusion depth (rules fired/sample): mean 63.95; p90 84; max 132; holes 0.0%

| sample | row | true | pred_idx | fired | Ω | combined (top literals) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 103 | 0.0000 | hours.per.week < 30.0 AND fnlwgt < 152157.875 AND fnlwgt > 117682.5 AND age > 55.0 AND education.num == 9.0 AND capital.loss > 0.0 AND education == HS-grad AND native.country == United-States AND occupation == Exec-managerial AND relationship == Not-in-family |

Top contributing rules:
- [agree] class=0 w=+0.2444 :: hours.per.week < 40 → class 0
- [agree] class=0 w=+0.0907 :: education.num == 9 → class 0
- [agree] class=0 w=+0.0863 :: fnlwgt < 308091 → class 0

| 123 | 123 | 0 | 0 | 47 | 0.0013 | fnlwgt < 79385.125 AND age > 55.0 AND education.num == 9.0 AND hours.per.week > 45.0 AND native.country == United-States AND education == HS-grad AND capital.loss > 0.0 AND relationship == Husband AND race == White AND occupation == Farming-fishing |

Top contributing rules:
- [agree] class=0 w=+0.0907 :: education.num == 9 → class 0
- [agree] class=0 w=+0.0863 :: fnlwgt < 308091 → class 0
- [agree] class=0 w=+0.0849 :: education.num == 9 & fnlwgt < 308091 → class 0

| 1024 | 1024 | 0 | 1 | 117 | 0.0000 | education.num == 15.0 AND capital.loss > 0.0 AND fnlwgt > 117682.5 AND age > 28.0 AND fnlwgt < 308090.875 AND occupation == Prof-specialty AND hours.per.week > 30.0 AND age < 55.0 AND native.country == United-States AND education == Prof-school |

Top contributing rules:
- [agree] class=1 w=+0.6426 :: education.num == 15 → class 1
- [agree] class=1 w=+0.1149 :: age > 23 & capital.loss > 0 & fnlwgt < 308091 → class 1
- [agree] class=1 w=+0.0688 :: capital.loss > 0 & fnlwgt > 79385.1 → class 1


#### adult.csv — RIPPER

- Rules: 304; mean literals/rule: 4.66; p90: 7
- Fusion depth (rules fired/sample): mean 6.31; p90 9; max 14; holes 0.0%

| sample | row | true | pred_idx | fired | Ω | combined (top literals) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 6 | 0.1806 | hours.per.week < 40.0 AND capital.gain < 5054.25 AND relationship == 1.0 AND education == 11.0 AND age > 58.0 AND occupation == 3.0 |

Top contributing rules:
- [agree] class=0 w=+0.2482 :: capital.gain < 5054 & hours.per.week < 40 & relationship == Not-in-family
- [agree] class=0 w=+0.2449 :: capital.gain < 5054 & education == HS-grad & hours.per.week < 40
- [agree] class=0 w=+0.0151 :: age > 58 & hours.per.week < 42

| 123 | 123 | 0 | 0 | 9 | 0.1521 | occupation == 4.0 AND age > 45.0 AND capital.gain < 5054.25 AND hours.per.week < 55.0 AND education == 11.0 AND hours.per.week > 40.0 AND fnlwgt < 39581.0 |

Top contributing rules:
- [agree] class=0 w=+0.4172 :: age > 45 & capital.gain < 5054 & hours.per.week < 55 & occupation == Farming-fishing
- [conflict] class=1 w=-0.0805 :: age > 36 & capital.loss > 0 & hours.per.week > 32 & marital.status == Married-civ-spouse & native.country > Mexico
- [conflict] class=1 w=-0.0470 :: age > 26 & capital.loss > 0 & education == HS-grad & hours.per.week > 42 & sex == Male

| 1024 | 1024 | 0 | 0 | 4 | 0.4511 | capital.gain < 5054.25 AND relationship == 4.0 AND fnlwgt < 178419.0 AND race == 2.0 AND occupation == 9.0 AND age > 43.0 |

Top contributing rules:
- [agree] class=0 w=+0.0793 :: capital.gain < 5054 & relationship == Unmarried
- [agree] class=0 w=+0.0442 :: fnlwgt < 1.784e+05 & race == Black
- [agree] class=0 w=+0.0079 :: capital.gain < 6849 & occupation == Prof-specialty


#### adult.csv — FOIL

- Rules: 614; mean literals/rule: 5.45; p90: 7
- Fusion depth (rules fired/sample): mean 4.23; p90 8; max 13; holes 0.8%

| sample | row | true | pred_idx | fired | Ω | combined (top literals) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 4 | 0.1255 | capital.gain < 5054.25 AND relationship == 1.0 AND fnlwgt < 178419.0 AND hours.per.week < 40.0 AND sex == 0.0 AND education == 11.0 AND workclass == 2.0 AND age > 32.0 AND fnlwgt > 130794.0 |

Top contributing rules:
- [agree] class=0 w=+0.2978 :: capital.gain < 5054 & fnlwgt < 1.784e+05 & hours.per.week < 40 & relationship == Not-in-family & sex == Female
- [agree] class=0 w=+0.2710 :: capital.gain < 5054 & education == HS-grad & fnlwgt < 2.892e+05 & hours.per.week < 50 & relationship == Not-in-family & workclass == Private
- [agree] class=0 w=+0.0582 :: age > 32 & capital.gain < 5054 & fnlwgt > 1.308e+05 & relationship == Not-in-family

| 123 | 123 | 0 | 0 | 3 | 0.5230 | capital.gain < 5054.25 AND education == 11.0 AND fnlwgt < 178419.0 AND age > 36.0 AND workclass == 4.0 AND hours.per.week > 45.0 |

Top contributing rules:
- [agree] class=0 w=+0.0791 :: age > 36 & capital.gain < 5054 & education == HS-grad & fnlwgt < 2.857e+05 & workclass == Self-emp-not-inc
- [agree] class=0 w=+0.0231 :: capital.gain < 5054 & education == HS-grad & fnlwgt < 1.784e+05 & hours.per.week > 45
- [conflict] class=1 w=-0.0000 :: fnlwgt < 1.784e+05 & hours.per.week > 48.7 & marital.status == Married-civ-spouse & workclass == Self-emp-not-inc

| 1024 | 1024 | 0 | 0 | 1 | 0.2641 | capital.gain < 6497.0 AND fnlwgt < 220010.0 AND relationship == 4.0 AND workclass == 1.0 |

Top contributing rules:
- [agree] class=0 w=+0.5218 :: capital.gain < 6497 & fnlwgt < 2.2e+05 & relationship == Unmarried & workclass == Local-gov


### bank-full.csv

#### bank-full.csv — STATIC

- Rules: 800; mean literals/rule: 2.63; p90: 3
- Fusion depth (rules fired/sample): mean 224.95; p90 303; max 397; holes 0.0%

| sample | row | true | pred_idx | fired | Ω | combined (top literals) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 175 | 0.0000 | balance > 1430.0 AND duration < 318.0 AND balance < 2983.0 AND age > 55.0 AND duration > 236.0 AND day < 8.0 AND pdays < 166.0 AND campaign < 2.0 AND previous < 1.0 AND month == may |

Top contributing rules:
- [agree] class=0 w=+0.1043 :: age > 30 & balance < 2983 & duration < 485 → class 0
- [agree] class=0 w=+0.1014 :: balance > 0 & duration < 485 & previous < 1 → class 0
- [agree] class=0 w=+0.0952 :: age > 30 & balance > 0 & duration < 318 → class 0

| 123 | 123 | 0 | 0 | 271 | 0.0000 | duration < 139.0 AND balance < 73.0 AND balance > 0.0 AND pdays < 166.0 AND duration > 103.0 AND day < 8.0 AND age > 39.0 AND age < 48.0 AND campaign < 3.0 AND previous < 1.0 |

Top contributing rules:
- [agree] class=0 w=+0.2399 :: duration < 139 → class 0
- [agree] class=0 w=+0.1591 :: duration < 236 → class 0
- [agree] class=0 w=+0.1197 :: age < 55 & balance > 0 & duration < 485 → class 0

| 1024 | 1024 | 0 | 0 | 287 | 0.0000 | duration < 103.0 AND balance < 789.0 AND balance > 450.0 AND pdays < 166.0 AND duration > 67.0 AND day < 8.0 AND age > 39.0 AND age < 48.0 AND campaign < 2.0 AND previous < 1.0 |

Top contributing rules:
- [agree] class=0 w=+0.3650 :: duration < 103 → class 0
- [agree] class=0 w=+0.2399 :: duration < 139 → class 0
- [agree] class=0 w=+0.1591 :: duration < 236 → class 0


#### bank-full.csv — RIPPER

- Rules: 267; mean literals/rule: 4.85; p90: 7
- Fusion depth (rules fired/sample): mean 7.25; p90 11; max 15; holes 0.0%

| sample | row | true | pred_idx | fired | Ω | combined (top literals) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 10 | 0.0000 | duration < 279.0 AND month == 8.0 AND housing == 1.0 AND contact == 2.0 AND day < 10.0 AND previous < 1.0 AND age > 38.0 AND poutcome == 3.0 AND campaign < 2.0 AND balance > 450.0 |

Top contributing rules:
- [agree] class=0 w=+0.8942 :: day < 16 & duration < 279 & housing == yes & month == may
- [agree] class=0 w=+0.8848 :: age > 30 & contact == unknown & duration < 426 & housing == yes & month == may & previous < 1
- [agree] class=0 w=+0.7008 :: contact == unknown & day < 10 & duration < 507 & previous < 1

| 123 | 123 | 0 | 0 | 15 | 0.0000 | duration < 160.0 AND month == 8.0 AND day < 10.0 AND housing == 1.0 AND contact == 2.0 AND previous < 1.0 AND poutcome == 3.0 AND age > 38.0 AND balance < 131.0 AND pdays < 8.0 |

Top contributing rules:
- [agree] class=0 w=+0.8942 :: day < 16 & duration < 279 & housing == yes & month == may
- [agree] class=0 w=+0.8848 :: age > 30 & contact == unknown & duration < 426 & housing == yes & month == may & previous < 1
- [agree] class=0 w=+0.7008 :: contact == unknown & day < 10 & duration < 507 & previous < 1

| 1024 | 1024 | 0 | 0 | 13 | 0.0000 | duration < 160.0 AND month == 8.0 AND day < 9.0 AND housing == 1.0 AND contact == 2.0 AND previous < 1.0 AND poutcome == 3.0 AND age > 38.0 AND balance < 2533.5999999999985 AND education == 0.0 |

Top contributing rules:
- [agree] class=0 w=+0.8942 :: day < 16 & duration < 279 & housing == yes & month == may
- [agree] class=0 w=+0.8848 :: age > 30 & contact == unknown & duration < 426 & housing == yes & month == may & previous < 1
- [agree] class=0 w=+0.7008 :: contact == unknown & day < 10 & duration < 507 & previous < 1


#### bank-full.csv — FOIL

- Rules: 417; mean literals/rule: 5.31; p90: 7
- Fusion depth (rules fired/sample): mean 5.81; p90 9; max 19; holes 0.2%

| sample | row | true | pred_idx | fired | Ω | combined (top literals) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 12 | 0.0000 | duration < 279.0 AND month == 8.0 AND housing == 1.0 AND previous < 1.0 AND contact == 2.0 AND day < 9.0 AND age > 37.0 AND campaign < 3.0 AND poutcome == 3.0 AND balance > 1125.0 |

Top contributing rules:
- [agree] class=0 w=+0.9669 :: age < 60 & balance > 1125 & campaign < 3 & day < 16 & duration < 437 & housing == yes & month == may & previous < 1
- [agree] class=0 w=+0.9572 :: age > 30 & campaign < 4 & day < 16 & duration < 279 & housing == yes & month == may & previous < 1
- [agree] class=0 w=+0.9429 :: age > 30 & contact == unknown & day < 26 & duration < 437 & housing == yes & month == may & previous < 1

| 123 | 123 | 0 | 0 | 18 | 0.0000 | duration < 178.0 AND day < 6.0 AND housing == 1.0 AND month == 8.0 AND previous < 1.0 AND contact == 2.0 AND age > 39.0 AND balance < 178.0 AND campaign > 1.0 AND marital == 1.0 |

Top contributing rules:
- [agree] class=0 w=+0.9745 :: balance < 178 & campaign > 1 & contact == unknown & duration < 333 & housing == yes & previous < 1
- [agree] class=0 w=+0.9572 :: age > 30 & campaign < 4 & day < 16 & duration < 279 & housing == yes & month == may & previous < 1
- [agree] class=0 w=+0.9501 :: day < 20 & duration < 178 & month == may & pdays < 5

| 1024 | 1024 | 0 | 0 | 10 | 0.0000 | duration < 178.0 AND month == 8.0 AND day < 13.0 AND housing == 1.0 AND age > 39.0 AND previous < 1.0 AND contact == 2.0 AND pdays < 5.0 AND campaign < 4.0 AND day > 6.0 |

Top contributing rules:
- [agree] class=0 w=+0.9572 :: age > 30 & campaign < 4 & day < 16 & duration < 279 & housing == yes & month == may & previous < 1
- [agree] class=0 w=+0.9501 :: day < 20 & duration < 178 & month == may & pdays < 5
- [agree] class=0 w=+0.9429 :: age > 30 & contact == unknown & day < 26 & duration < 437 & housing == yes & month == may & previous < 1


### BrainTumor.csv

#### BrainTumor.csv — STATIC

- Rules: 853; mean literals/rule: 2.52; p90: 3
- Fusion depth (rules fired/sample): mean 514.53; p90 752; max 769; holes 0.0%

| sample | row | true | pred_idx | fired | Ω | combined (top literals) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 751 | 0.0000 | Variance > 365.8471145629883 AND Variance < 628.587646484375 AND ASM > 0.07025238871574402 AND Contrast > 72.54677391052246 AND Kurtosis < 22.622046947479248 AND Entropy > 0.09039528109133244 AND Homogeneity > 0.5123224258422852 AND Contrast < 107.4302864074707 AND Mean > 5.048439025878906 AND Energy > 0.2650516629219055 |

Top contributing rules:
- [agree] class=0 w=+0.4228 :: ASM > 0.0510134 → class 0
- [agree] class=0 w=+0.3521 :: Entropy > 0.0668942 → class 0
- [agree] class=0 w=+0.1971 :: Homogeneity > 0.512322 → class 0

| 123 | 123 | 1 | 1 | 604 | 0.0000 | Kurtosis > 22.622046947479248 AND Variance > 365.8471145629883 AND Energy < 0.04240240156650543 AND ASM < 0.0017979639815166593 AND Entropy < 0.002582444460131228 AND Skewness > 4.6506667137146 AND Homogeneity < 0.36626534163951874 AND Contrast > 88.18684101104736 AND Dissimilarity > 5.0777013301849365 AND Standard Deviation > 19.12712812423706 |

Top contributing rules:
- [agree] class=1 w=+0.6364 :: Energy < 0.131878 → class 1
- [agree] class=1 w=+0.6303 :: ASM < 0.0173918 → class 1
- [agree] class=1 w=+0.3943 :: Entropy < 0.0668942 → class 1

| 1024 | 1024 | 0 | 0 | 505 | 0.0000 | Variance > 365.8471145629883 AND Kurtosis < 5.777422845363617 AND Variance < 1262.3135070800781 AND Contrast > 72.54677391052246 AND Mean > 16.610374450683594 AND Contrast < 162.16564559936523 AND Skewness < 2.322861284017563 AND Standard Deviation > 19.12712812423706 AND Homogeneity > 0.4377995394170284 AND Entropy > 0.023764305049553514 |

Top contributing rules:
- [agree] class=0 w=+0.2414 :: Kurtosis < 9.58856 → class 0
- [conflict] class=1 w=-0.1972 :: Entropy < 0.0668942 → class 1
- [conflict] class=1 w=-0.1869 :: Energy < 0.225861 → class 1


#### BrainTumor.csv — RIPPER

- Rules: 23; mean literals/rule: 2.83; p90: 4
- Fusion depth (rules fired/sample): mean 4.80; p90 7; max 10; holes 0.0%

| sample | row | true | pred_idx | fired | Ω | combined (top literals) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 3 | 0.0918 | Entropy > 0.06500032842159272 AND Contrast < 158.78328552246091 AND Energy > 0.21110250800848007 AND Mean < 16.155856323242187 |

Top contributing rules:
- [agree] class=0 w=+0.5929 :: Contrast < 158.8 & Energy > 0.2111 & Entropy > 0.04169 & Mean < 16.16
- [agree] class=0 w=+0.3232 :: Entropy > 0.065
- [conflict] class=1 w=-0.0533 :: Homogeneity < 0.5629 & Kurtosis > 12.38

| 123 | 123 | 1 | 1 | 6 | 0.0019 | Kurtosis > 19.486724662780748 AND Entropy < 0.046459904685616456 AND Correlation > 0.969596517086029 AND Energy < 0.1936064302921295 AND Skewness > 2.290888452529907 AND Homogeneity < 0.5629000842571258 |

Top contributing rules:
- [agree] class=1 w=+0.7784 :: Correlation > 0.9487 & Energy < 0.2148 & Kurtosis > 5.992 & Skewness > 2.291
- [agree] class=1 w=+0.7782 :: Entropy < 0.04646 & Kurtosis > 5.992
- [agree] class=1 w=+0.3619 :: Correlation > 0.9696 & Energy < 0.1936 & Entropy < 0.08664

| 1024 | 1024 | 0 | 0 | 4 | 0.0792 | Entropy > 0.06500032842159272 AND Mean > 12.204295349121088 AND Variance < 1624.6609069824215 |

Top contributing rules:
- [agree] class=0 w=+0.6422 :: Entropy > 0.05136 & Mean > 12.2 & Variance < 1625
- [agree] class=0 w=+0.3232 :: Entropy > 0.065
- [conflict] class=1 w=-0.0075 :: Entropy < 0.06689


#### BrainTumor.csv — FOIL

- Rules: 60; mean literals/rule: 4.32; p90: 6
- Fusion depth (rules fired/sample): mean 4.10; p90 7; max 10; holes 0.0%

| sample | row | true | pred_idx | fired | Ω | combined (top literals) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 3 | 0.0053 | Contrast < 115.30792999267578 AND Entropy > 0.06026964634656906 AND Homogeneity > 0.5123224258422852 AND Kurtosis < 19.486724662780748 AND Energy > 0.28748713135719295 AND Dissimilarity > 3.6157349586486816 AND Correlation > 0.9599296450614929 AND Variance < 628.587646484375 AND Mean < 8.461685180664062 AND Skewness < 4.39258918762207 |

Top contributing rules:
- [agree] class=0 w=+0.7484 :: Contrast < 115.3 & Correlation > 0.9599 & Energy > 0.2196 & Entropy > 0.05451 & Homogeneity > 0.5029 & Kurtosis < 32.63 & Variance < 628.6
- [agree] class=0 w=+0.7405 :: Contrast < 161.3 & Dissimilarity > 3.616 & Entropy > 0.05641 & Homogeneity > 0.5123 & Kurtosis < 19.49
- [agree] class=0 w=+0.5136 :: Contrast < 156.2 & Energy > 0.2875 & Entropy > 0.06027 & Mean < 8.462 & Skewness < 4.393

| 123 | 123 | 1 | 1 | 5 | 0.0030 | Entropy < 0.048704979196190754 AND Kurtosis > 19.486724662780748 AND Skewness > 3.0491999864578245 AND Homogeneity < 0.5123224258422852 AND Correlation > 0.9644977152347565 AND Energy < 0.20886734277009963 |

Top contributing rules:
- [agree] class=1 w=+0.7988 :: Entropy < 0.0487 & Kurtosis > 5.992
- [agree] class=1 w=+0.5384 :: Entropy < 0.06034 & Homogeneity < 0.5123 & Kurtosis > 19.49 & Skewness > 2.37
- [agree] class=1 w=+0.4012 :: Correlation > 0.9507 & Energy < 0.2105 & Entropy < 0.06689 & Kurtosis > 5.614 & Skewness > 2.37

| 1024 | 1024 | 0 | 0 | 6 | 0.0000 | Entropy > 0.05978953093290329 AND Kurtosis < 4.811543130874634 AND Mean > 20.12857360839842 AND Correlation < 0.9586191773414612 AND Contrast < 159.11398620605468 AND Dissimilarity > 5.073681354522705 AND Homogeneity > 0.49149959087371825 AND Variance < 1624.6609069824215 AND Contrast > 118.69818115234375 AND Energy > 0.21353885531425476 |

Top contributing rules:
- [agree] class=0 w=+0.8038 :: Contrast > 118.7 & Correlation < 0.9586 & Energy > 0.2135 & Entropy > 0.04669 & Kurtosis < 8.123
- [agree] class=0 w=+0.7808 :: Contrast < 159.1 & Dissimilarity > 4.688 & Homogeneity > 0.4532 & Mean > 15.8 & Skewness < 2.37
- [agree] class=0 w=+0.7241 :: Entropy > 0.05168 & Kurtosis < 5.992 & Mean > 20.13 & Variance < 1625


### breast-cancer-wisconsin.csv

#### breast-cancer-wisconsin.csv — STATIC

- Rules: 559; mean literals/rule: 2.56; p90: 3
- Fusion depth (rules fired/sample): mean 26.64; p90 41; max 41; holes 0.0%

| sample | row | true | pred_idx | fired | Ω | combined (top literals) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 41 | 0.0010 | bare_nucleoli == 1.0 AND size_uniformity == 1.0 AND shape_uniformity == 1.0 AND marginal_adhesion == 1.0 AND bland_chromatin == 3.0 AND clump_thickness == 5.0 AND normal_nucleoli == 1.0 AND epithelial_size == 2.0 AND mitoses == 1.0 |

Top contributing rules:
- [agree] class=0 w=+0.0670 :: bare_nucleoli == 1 → class 0
- [agree] class=0 w=+0.0669 :: bare_nucleoli == 1 & marginal_adhesion == 1 & shape_uniformity == 1 → class 0
- [agree] class=0 w=+0.0644 :: bare_nucleoli == 1 & clump_thickness == 5 → class 0

| 123 | 123 | 0 | 0 | 41 | 0.0012 | bare_nucleoli == 1.0 AND shape_uniformity == 1.0 AND size_uniformity == 1.0 AND marginal_adhesion == 1.0 AND clump_thickness == 1.0 AND bland_chromatin == 2.0 AND normal_nucleoli == 1.0 AND epithelial_size == 2.0 AND mitoses == 1.0 |

Top contributing rules:
- [agree] class=0 w=+0.0670 :: bare_nucleoli == 1 → class 0
- [agree] class=0 w=+0.0669 :: bare_nucleoli == 1 & marginal_adhesion == 1 & shape_uniformity == 1 → class 0
- [agree] class=0 w=+0.0636 :: bare_nucleoli == 1 & epithelial_size == 2 & size_uniformity == 1 → class 0

| 341 | 341 | 0 | 0 | 26 | 0.0083 | bare_nucleoli == 1.0 AND shape_uniformity == 1.0 AND marginal_adhesion == 1.0 AND size_uniformity == 1.0 AND bland_chromatin == 2.0 AND epithelial_size == 3.0 AND mitoses == 1.0 AND normal_nucleoli == 2.0 AND clump_thickness == 4.0 |

Top contributing rules:
- [agree] class=0 w=+0.0670 :: bare_nucleoli == 1 → class 0
- [agree] class=0 w=+0.0669 :: bare_nucleoli == 1 & marginal_adhesion == 1 & shape_uniformity == 1 → class 0
- [agree] class=0 w=+0.0629 :: bare_nucleoli == 1 & epithelial_size == 3 & shape_uniformity == 1 → class 0


#### breast-cancer-wisconsin.csv — RIPPER

- Rules: 18; mean literals/rule: 1.33; p90: 2
- Fusion depth (rules fired/sample): mean 3.27; p90 5; max 5; holes 1.6%

| sample | row | true | pred_idx | fired | Ω | combined (top literals) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 5 | 0.2172 | shape_uniformity == 1.0 AND bare_nucleoli == 1.0 AND marginal_adhesion == 1.0 AND epithelial_size == 2.0 AND size_uniformity == 1.0 AND normal_nucleoli == 1.0 |

Top contributing rules:
- [agree] class=0 w=+0.0635 :: shape_uniformity == 1
- [agree] class=0 w=+0.0613 :: bare_nucleoli == 1 & epithelial_size == 2
- [agree] class=0 w=+0.0610 :: bare_nucleoli == 1 & size_uniformity == 1

| 123 | 123 | 0 | 0 | 5 | 0.2172 | shape_uniformity == 1.0 AND bare_nucleoli == 1.0 AND marginal_adhesion == 1.0 AND epithelial_size == 2.0 AND size_uniformity == 1.0 AND normal_nucleoli == 1.0 |

Top contributing rules:
- [agree] class=0 w=+0.0635 :: shape_uniformity == 1
- [agree] class=0 w=+0.0613 :: bare_nucleoli == 1 & epithelial_size == 2
- [agree] class=0 w=+0.0610 :: bare_nucleoli == 1 & size_uniformity == 1

| 341 | 341 | 0 | 0 | 3 | 0.3734 | shape_uniformity == 1.0 AND marginal_adhesion == 1.0 AND bare_nucleoli == 1.0 AND size_uniformity == 1.0 |

Top contributing rules:
- [agree] class=0 w=+0.0635 :: shape_uniformity == 1
- [agree] class=0 w=+0.0610 :: bare_nucleoli == 1 & size_uniformity == 1
- [agree] class=0 w=+0.0552 :: marginal_adhesion == 1


#### breast-cancer-wisconsin.csv — FOIL

- Rules: 42; mean literals/rule: 2.17; p90: 3
- Fusion depth (rules fired/sample): mean 2.90; p90 5; max 6; holes 0.8%

| sample | row | true | pred_idx | fired | Ω | combined (top literals) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 5 | 0.3388 | normal_nucleoli == 1.0 AND shape_uniformity == 1.0 AND marginal_adhesion == 1.0 AND epithelial_size == 2.0 AND bland_chromatin == 3.0 AND clump_thickness == 5.0 AND mitoses == 1.0 AND bare_nucleoli == 1.0 AND size_uniformity == 1.0 |

Top contributing rules:
- [agree] class=0 w=+0.0858 :: bland_chromatin == 3 & clump_thickness == 5 & marginal_adhesion == 1 & mitoses == 1
- [agree] class=0 w=+0.0664 :: epithelial_size == 2 & normal_nucleoli == 1 & shape_uniformity == 1
- [agree] class=0 w=+0.0249 :: marginal_adhesion == 1 & normal_nucleoli == 1 & shape_uniformity == 1

| 123 | 123 | 0 | 1 | 4 | 0.4531 | <empty> |

Top contributing rules:
- [conflict] class=0 w=-0.0332 :: epithelial_size == 2 & normal_nucleoli == 1 & shape_uniformity == 1
- [conflict] class=0 w=-0.0124 :: marginal_adhesion == 1 & normal_nucleoli == 1 & shape_uniformity == 1
- [conflict] class=0 w=-0.0030 :: bare_nucleoli == 1 & epithelial_size == 2

| 341 | 341 | 0 | 1 | 1 | 0.8799 | <empty> |

Top contributing rules:
- [conflict] class=0 w=-0.0010 :: bare_nucleoli == 1 & size_uniformity == 1

---

## Rule Inspector Report (DSGD-Auto)
Источник: `RULE_INSPECTOR_REPORT_SAHEART.md`

This report shows, per dataset and rule-induction method, (1) rule-base scale, (2) fusion depth proxy (rules fired/sample), and (3) combined-rule examples produced by `DSModelMultiQ.get_combined_rule` (explanation-only).

Combined condition is built from positive rule contributions, aggregated per feature/op into a single conjunctive string; it is not used for inference.

### SAheart.csv

#### SAheart.csv — STATIC

- Rules: 796; mean literals/rule: 2.63; p90: 3
- Fusion depth (rules fired/sample): mean 421.74; p90 642; max 705; holes 0.0%

| sample | row | true | pred_idx | fired | Ω | combined (top literals) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 1 | 1 | 454 | 0.0000 | famhist == Present AND alcohol > 0.5775000154972076 AND age > 50.0 AND tobacco > 7.962500035762787 AND adiposity > 19.739999771118164 AND sbp > 124.0 |

Top contributing rules:
- [agree] class=1 w=+0.0442 :: tobacco > 7.9625 → class 1
- [agree] class=1 w=+0.0361 :: adiposity > 19.74 & alcohol > 0 & famhist == Present → class 1
- [agree] class=1 w=+0.0359 :: adiposity > 19.74 & alcohol > 0.5775 & famhist == Present → class 1

| 123 | 123 | 1 | 0 | 657 | 0.0000 | alcohol > 22.832499504089355 AND alcohol < 42.9150013923645 AND sbp > 134.0 AND adiposity > 28.68625044822693 AND typea < 56.0 AND age > 44.0 AND tobacco > 3.5999999046325684 AND obesity > 24.464999198913574 AND ldl < 3.237500011920929 AND sbp < 139.75 |

Top contributing rules:
- [agree] class=0 w=+0.0575 :: alcohol < 42.915 & tobacco < 7.9625 & typea < 60 → class 0
- [agree] class=0 w=+0.0552 :: alcohol < 42.915 & sbp < 160.625 & typea < 64 → class 0
- [agree] class=0 w=+0.0521 :: alcohol < 42.915 & sbp < 160.625 & tobacco < 7.9625 → class 0

| 100 | 100 | 0 | 1 | 556 | 0.0000 | famhist == Present AND alcohol > 0.5775000154972076 AND ldl > 7.0774999260902405 AND adiposity > 19.739999771118164 AND age > 31.0 AND tobacco > 7.962500035762787 AND alcohol < 42.9150013923645 AND sbp > 124.0 |

Top contributing rules:
- [agree] class=1 w=+0.0442 :: tobacco > 7.9625 → class 1
- [agree] class=1 w=+0.0371 :: age > 31 & alcohol < 42.915 & famhist == Present → class 1
- [agree] class=1 w=+0.0361 :: adiposity > 19.74 & alcohol > 0 & famhist == Present → class 1


#### SAheart.csv — RIPPER

- Rules: 25; mean literals/rule: 3.12; p90: 4
- Fusion depth (rules fired/sample): mean 3.50; p90 5; max 7; holes 0.0%

| sample | row | true | pred_idx | fired | Ω | combined (top literals) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 1 | 1 | 3 | 0.2351 | tobacco > 1.85999995470047 AND adiposity < 29.864000701904295 AND age > 51.0 AND ldl > 3.1024999618530273 AND typea < 53.0 |

Top contributing rules:
- [agree] class=1 w=+0.3122 :: adiposity < 29.86 & age > 51 & ldl > 3.102 & tobacco > 1.86 & typea < 53
- [conflict] class=0 w=-0.1513 :: adiposity < 26.1 & age < 53 & tobacco > 4.03 & typea < 60.7
- [agree] class=1 w=+0.0000 :: tobacco > 1.86

| 123 | 123 | 1 | 0 | 3 | 0.6514 | adiposity > 26.104999542236328 AND ldl < 5.406999874114989 AND typea > 53.0 |

Top contributing rules:
- [agree] class=0 w=+0.0300 :: adiposity > 26.1 & ldl < 5.407
- [agree] class=0 w=+0.0075 :: typea > 53
- [conflict] class=1 w=-0.0000 :: tobacco > 1.86

| 100 | 100 | 0 | 0 | 2 | 0.3792 | adiposity < 26.104999542236328 AND age < 53.0 AND tobacco > 4.030000114440918 AND typea < 60.699999999999996 |

Top contributing rules:
- [agree] class=0 w=+0.3026 :: adiposity < 26.1 & age < 53 & tobacco > 4.03 & typea < 60.7
- [conflict] class=1 w=-0.0000 :: tobacco > 1.86


#### SAheart.csv — FOIL

- Rules: 78; mean literals/rule: 4.91; p90: 7
- Fusion depth (rules fired/sample): mean 1.63; p90 3; max 7; holes 1.7%

| sample | row | true | pred_idx | fired | Ω | combined (top literals) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 1 | 1 | 1 | 0.2530 | age > 38.95 AND alcohol > 0.0 AND famhist == 1.0 AND ldl > 5.0400002002716064 AND obesity < 34.60199890136718 AND sbp > 152.0 |

Top contributing rules:
- [agree] class=1 w=+0.5579 :: age > 38.95 & alcohol > 0 & famhist == Present & ldl > 5.04 & obesity < 34.6 & sbp > 152

| 123 | 123 | 1 | 1 | 1 | 0.0694 | adiposity > 26.104999542236328 AND ldl < 2.90199990272522 AND sbp > 136.0 AND tobacco > 3.5999999999999996 AND typea > 53.0 |

Top contributing rules:
- [agree] class=1 w=+0.8660 :: adiposity > 26.1 & ldl < 2.902 & sbp > 136 & tobacco > 3.6 & typea > 53

| 100 | 100 | 0 | 0 | 1 | 0.1565 | adiposity > 12.130000114440918 AND age < 54.5 AND famhist == 1.0 AND ldl > 4.579999923706055 AND obesity < 23.630999183654787 AND sbp > 126.1 AND typea > 49.0 |

Top contributing rules:
- [agree] class=0 w=+0.7116 :: adiposity > 12.13 & age < 54.5 & famhist == Present & ldl > 4.58 & obesity < 23.63 & sbp > 126.1 & typea > 49


### df_wine.csv

#### df_wine.csv — STATIC

- Rules: 542; mean literals/rule: 2.64; p90: 3
- Fusion depth (rules fired/sample): mean 200.32; p90 297; max 298; holes 0.0%

| sample | row | true | pred_idx | fired | Ω | combined (top literals) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 207 | 0.0000 | sulphates > 0.550000011920929 AND pH > 3.4000000953674316 AND alcohol > 9.199999809265137 AND good == 0.0 AND quality == 5.0 AND density > 0.9909999966621399 AND density < 0.9981799721717834 |

Top contributing rules:
- [agree] class=0 w=+0.0405 :: alcohol > 9.2 & sulphates > 0.43 → class 0
- [agree] class=0 w=+0.0362 :: alcohol > 9.2 & good == 0 & sulphates > 0.43 → class 0
- [agree] class=0 w=+0.0345 :: density > 0.991 & pH > 3.11 & sulphates > 0.43 → class 0

| 123 | 123 | 0 | 0 | 206 | 0.0000 | pH > 3.4000000953674316 AND sulphates > 0.5099999904632568 AND alcohol > 9.199999809265137 AND good == 0.0 AND quality == 5.0 AND density > 0.9909999966621399 AND density < 0.9981799721717834 |

Top contributing rules:
- [agree] class=0 w=+0.0405 :: alcohol > 9.2 & sulphates > 0.43 → class 0
- [agree] class=0 w=+0.0362 :: alcohol > 9.2 & good == 0 & sulphates > 0.43 → class 0
- [agree] class=0 w=+0.0345 :: density > 0.991 & pH > 3.11 & sulphates > 0.43 → class 0

| 1024 | 1024 | 0 | 1 | 297 | 0.0000 | alcohol < 10.800000190734863 AND density < 0.9958599805831909 AND pH > 3.1600000858306885 AND sulphates > 0.4300000071525574 AND quality == 7.0 AND good == 1.0 AND sulphates < 0.6000000238418579 AND alcohol > 9.5 AND density > 0.9948599934577942 AND pH < 3.4000000953674316 |

Top contributing rules:
- [agree] class=1 w=+0.0480 :: alcohol < 10.8 → class 1
- [agree] class=1 w=+0.0454 :: density > 0.991 & pH < 3.4 & sulphates < 0.69 → class 1
- [agree] class=1 w=+0.0422 :: alcohol < 11.3 & density < 0.99699 & pH > 3.04 → class 1


#### df_wine.csv — RIPPER

- Rules: 104; mean literals/rule: 4.29; p90: 5
- Fusion depth (rules fired/sample): mean 4.69; p90 6; max 9; holes 0.0%

| sample | row | true | pred_idx | fired | Ω | combined (top literals) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 4 | 0.2257 | pH > 3.380000114440918 AND quality == 5.0 AND sulphates > 0.5299999713897705 AND alcohol < 9.600000381469727 AND alcohol > 9.100000381469727 AND density > 0.9948599934577942 |

Top contributing rules:
- [agree] class=0 w=+0.2072 :: alcohol < 10.4 & pH > 3.38 & quality == 5 & sulphates > 0.53
- [agree] class=0 w=+0.1204 :: alcohol > 9.1 & density > 0.9949 & pH > 3.14 & quality == 5 & sulphates > 0.45
- [conflict] class=1 w=-0.0190 :: density > 0.9966

| 123 | 123 | 0 | 0 | 4 | 0.2067 | alcohol > 9.399999618530273 AND density > 0.9960990071296691 AND pH > 3.1500000953674316 AND sulphates > 0.5099999904632568 AND quality == 5.0 AND alcohol < 9.600000381469727 |

Top contributing rules:
- [agree] class=0 w=+0.2365 :: alcohol > 9.4 & density > 0.9961 & pH > 3.15 & sulphates > 0.51
- [agree] class=0 w=+0.1204 :: alcohol > 9.1 & density > 0.9949 & pH > 3.14 & quality == 5 & sulphates > 0.45
- [conflict] class=1 w=-0.0190 :: density > 0.9966

| 1024 | 1024 | 0 | 1 | 3 | 0.3555 | alcohol < 11.870000123977661 AND density < 0.9965599775314331 AND alcohol > 9.600000381469727 AND density > 0.9948599934577942 AND sulphates < 0.609000015258789 |

Top contributing rules:
- [agree] class=1 w=+0.0698 :: alcohol > 9.6 & density > 0.9949 & sulphates < 0.609
- [agree] class=1 w=+0.0607 :: alcohol < 11.87 & density < 0.9966
- [conflict] class=0 w=-0.0341 :: density < 0.9966 & pH > 3.13 & sulphates > 0.5


#### df_wine.csv — FOIL

- Rules: 164; mean literals/rule: 4.78; p90: 6
- Fusion depth (rules fired/sample): mean 2.92; p90 4; max 6; holes 0.0%

| sample | row | true | pred_idx | fired | Ω | combined (top literals) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 3 | 0.3412 | alcohol < 9.600000381469727 AND pH > 3.2300000190734863 AND sulphates > 0.5099999904632568 AND density < 0.9979000091552734 AND quality == 5.0 AND density > 0.9939719915390015 |

Top contributing rules:
- [agree] class=0 w=+0.1125 :: alcohol < 9.6 & density < 0.9979 & pH > 3.23 & quality == 5 & sulphates > 0.46
- [agree] class=0 w=+0.0379 :: alcohol < 9.6 & density > 0.994 & pH > 3.09 & sulphates > 0.51
- [conflict] class=1 w=-0.0000 :: alcohol < 9.95 & density > 0.9964 & pH > 3.21 & quality == 5

| 123 | 123 | 0 | 0 | 4 | 0.2036 | pH > 3.2300000190734863 AND quality == 5.0 AND alcohol < 9.600000381469727 AND sulphates > 0.5099999904632568 AND density > 0.9939719915390015 AND density < 0.9979000091552734 AND alcohol > 9.399999618530273 AND sulphates < 0.6600000262260437 |

Top contributing rules:
- [agree] class=0 w=+0.1125 :: alcohol < 9.6 & density < 0.9979 & pH > 3.23 & quality == 5 & sulphates > 0.46
- [agree] class=0 w=+0.1039 :: alcohol > 9.4 & density > 0.9924 & pH > 3.22 & quality == 5 & sulphates < 0.66
- [agree] class=0 w=+0.0379 :: alcohol < 9.6 & density > 0.994 & pH > 3.09 & sulphates > 0.51

| 1024 | 1024 | 0 | 1 | 2 | 0.4469 | alcohol < 10.600000381469727 AND good == 1.0 AND pH < 3.380000114440918 AND sulphates > 0.5099999904632568 AND density > 0.9948599934577942 AND pH > 3.299999952316284 AND sulphates < 0.6600000262260437 |

Top contributing rules:
- [agree] class=1 w=+0.0860 :: alcohol < 10.6 & good == 1 & pH < 3.38 & sulphates > 0.51
- [agree] class=1 w=+0.0739 :: alcohol < 11.4 & density > 0.9949 & good == 1 & pH > 3.3 & sulphates < 0.66

---

## complete_rule_table.tex
Источник: `complete_rule_table.tex`

```tex
% Complete rule density table with all 8 datasets
% Data extracted from Common_code/dsb_rules/*_dst.dsb files
% Verified against REPORT.md Section 10.4 and latex_repo_analysis.tex

\begin{table}[t]
\centering
\caption{Rule-base density snapshot from \texttt{Common\_code} runs (train-induced rules). 
Counts may vary slightly with the seed and duplicate-filter threshold. 
Rule length $\mathbb{E}[|r|]$ is the average number of literals (conjuncts) per rule body.
Dense rule bases (STATIC) imply deeper evidence fusion and can trigger fusion-driven ignorance collapse under normalized DS fusion.}
\label{tab:rule-density-snapshot}
\begin{tabular}{l S[table-format=4.0] S[table-format=3.2] S[table-format=3.0] S[table-format=3.2] S[table-format=3.0] S[table-format=3.2]}
\toprule
Dataset & \multicolumn{2}{c}{STATIC} & \multicolumn{2}{c}{RIPPER} & \multicolumn{2}{c}{FOIL} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
 & {$R$} & {$\mathbb{E}[|r|]$} & {$R$} & {$\mathbb{E}[|r|]$} & {$R$} & {$\mathbb{E}[|r|]$} \\
\midrule
adult & 392 & 2.27 & 304 & 4.66 & 614 & 5.45 \\
bank-full & 800 & 2.63 & 267 & 4.85 & 417 & 5.31 \\
BrainTumor & 853 & 2.52 & 23 & 2.83 & 60 & 4.32 \\
breast-cancer-wisconsin & 559 & 2.56 & 18 & 1.33 & 42 & 2.17 \\
df\_wine & 542 & 2.64 & 104 & 4.29 & 164 & 4.78 \\
german & 714 & 2.61 & 61 & 3.49 & 141 & 4.55 \\
SAheart & 796 & 2.63 & 25 & 3.12 & 78 & 4.91 \\
gas\_drift & 2000 & 1.20 & 114 & 4.22 & 169 & 4.30 \\
\bottomrule
\end{tabular}
\end{table}

% Key observations:
% - STATIC: 392-2000 rules, avg length 1.20-2.64 (short, dense)
% - RIPPER: 18-304 rules, avg length 1.33-4.85 (compact, moderate length)
% - FOIL: 42-614 rules, avg length 2.17-5.45 (moderate density, longer rules)
% - gas_drift STATIC has extreme density (2000 rules) but very short rules (1.20 avg)
% - breast-cancer-wisconsin RIPPER has unusually short rules (1.33 avg) - likely due to simple feature space
```

---

## create_rule_visualization.py
Источник: `create_rule_visualization.py`

```python
#!/usr/bin/env python3
"""
Create a professional visualization of the rule generation and combination process
for the DSGD-Auto paper introduction.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

# Set style for professional academic paper
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Create figure with subplots
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3, 
                      left=0.05, right=0.95, top=0.95, bottom=0.05)

# Color scheme (pastel, professional)
colors = {
    'data': '#E8F4F8',  # Light blue
    'static': '#FFB6C1',  # Light pink
    'ripper': '#98FB98',  # Pale green
    'foil': '#DDA0DD',  # Plum
    'rule': '#FFE4B5',  # Moccasin
    'active': '#90EE90',  # Light green
    'mass': '#B0E0E6',  # Powder blue
    'fused': '#FFA07A',  # Light salmon
    'pred': '#87CEEB',  # Sky blue
}

# ===== TOP ROW: Rule Generation =====
ax1 = fig.add_subplot(gs[0, 0])
ax1.text(0.5, 0.9, 'Raw Data', ha='center', va='top', fontsize=12, weight='bold')
rect1 = FancyBboxPatch((0.1, 0.1), 0.8, 0.7, 
                       boxstyle="round,pad=0.02", 
                       facecolor=colors['data'], 
                       edgecolor='black', linewidth=1.5)
ax1.add_patch(rect1)
ax1.text(0.5, 0.45, 'Tabular Dataset\n$(x_i, y_i)$', 
         ha='center', va='center', fontsize=10)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
ax2.text(0.5, 0.9, 'Rule Induction', ha='center', va='top', fontsize=12, weight='bold')
# Three boxes for STATIC, RIPPER, FOIL
for i, (name, color) in enumerate([('STATIC', colors['static']), 
                                    ('RIPPER', colors['ripper']), 
                                    ('FOIL', colors['foil'])]):
    y_pos = 0.7 - i * 0.25
    rect = FancyBboxPatch((0.1, y_pos), 0.8, 0.2, 
                         boxstyle="round,pad=0.01", 
                         facecolor=color, 
                         edgecolor='black', linewidth=1)
    ax2.add_patch(rect)
    ax2.text(0.5, y_pos + 0.1, name, ha='center', va='center', fontsize=9, weight='bold')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
ax3.text(0.5, 0.9, 'Rule Set $\\mathcal{R}$', ha='center', va='top', fontsize=12, weight='bold')
# Show multiple rules
for i in range(5):
    y_pos = 0.75 - i * 0.15
    rect = FancyBboxPatch((0.1, y_pos), 0.8, 0.12, 
                         boxstyle="round,pad=0.01", 
                         facecolor=colors['rule'], 
                         edgecolor='black', linewidth=0.8)
    ax3.add_patch(rect)
    ax3.text(0.5, y_pos + 0.06, f'$r_{i+1}$: IF ... THEN $c_{i+1}$', 
             ha='center', va='center', fontsize=8)
ax3.text(0.5, 0.05, '...', ha='center', va='center', fontsize=10)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')

# Arrows
arrow1 = FancyArrowPatch((0.95, 0.5), (0.05, 0.5), 
                         arrowstyle='->', lw=2, color='black')
ax1.add_patch(arrow1)
arrow2 = FancyArrowPatch((0.95, 0.5), (0.05, 0.5), 
                         arrowstyle='->', lw=2, color='black')
ax2.add_patch(arrow2)

# ===== MIDDLE ROW: Activation and Mass Assignment =====
ax4 = fig.add_subplot(gs[1, 0])
ax4.text(0.5, 0.9, 'Input Sample $x$', ha='center', va='top', fontsize=12, weight='bold')
rect4 = FancyBboxPatch((0.1, 0.1), 0.8, 0.7, 
                       boxstyle="round,pad=0.02", 
                       facecolor=colors['data'], 
                       edgecolor='black', linewidth=1.5)
ax4.add_patch(rect4)
ax4.text(0.5, 0.45, '$x = (f_1, ..., f_d)$', 
         ha='center', va='center', fontsize=10)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

ax5 = fig.add_subplot(gs[1, 1])
ax5.text(0.5, 0.9, 'Rule Activation', ha='center', va='top', fontsize=12, weight='bold')
# Show active rules
active_rules = [1, 3, 4]  # Example: rules 1, 3, 4 fire
for i, rule_idx in enumerate([1, 2, 3, 4, 5]):
    y_pos = 0.75 - i * 0.15
    is_active = rule_idx in active_rules
    color = colors['active'] if is_active else '#E0E0E0'
    rect = FancyBboxPatch((0.1, y_pos), 0.8, 0.12, 
                         boxstyle="round,pad=0.01", 
                         facecolor=color, 
                         edgecolor='black', linewidth=1 if is_active else 0.5)
    ax5.add_patch(rect)
    status = '✓ ACTIVE' if is_active else '✗ inactive'
    ax5.text(0.5, y_pos + 0.06, f'$r_{rule_idx}$: {status}', 
             ha='center', va='center', fontsize=8, 
             weight='bold' if is_active else 'normal')
ax5.text(0.5, 0.05, '$A(x) = \\{1, 3, 4\\}$', ha='center', va='center', fontsize=9, weight='bold')
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.axis('off')

ax6 = fig.add_subplot(gs[1, 2])
ax6.text(0.5, 0.9, 'Mass Vectors', ha='center', va='top', fontsize=12, weight='bold')
# Show mass vectors for active rules
for i, rule_idx in enumerate(active_rules):
    y_pos = 0.75 - i * 0.25
    rect = FancyBboxPatch((0.1, y_pos), 0.8, 0.2, 
                         boxstyle="round,pad=0.01", 
                         facecolor=colors['mass'], 
                         edgecolor='black', linewidth=1)
    ax6.add_patch(rect)
    ax6.text(0.5, y_pos + 0.15, f'$\\mathbf{{m}}_{rule_idx}$', 
             ha='center', va='center', fontsize=10, weight='bold')
    ax6.text(0.5, y_pos + 0.05, '$(m_1, ..., m_k, m_\\Theta)$', 
             ha='center', va='center', fontsize=8)
ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)
ax6.axis('off')

# Arrows
arrow3 = FancyArrowPatch((0.95, 0.5), (0.05, 0.5), 
                         arrowstyle='->', lw=2, color='black')
ax4.add_patch(arrow3)
arrow4 = FancyArrowPatch((0.95, 0.5), (0.05, 0.5), 
                         arrowstyle='->', lw=2, color='black')
ax5.add_patch(arrow4)

# ===== BOTTOM ROW: Fusion and Prediction =====
ax7 = fig.add_subplot(gs[2, 0])
ax7.text(0.5, 0.9, 'DS Fusion', ha='center', va='top', fontsize=12, weight='bold')
# Show fusion process
rect7 = FancyBboxPatch((0.1, 0.1), 0.8, 0.7, 
                       boxstyle="round,pad=0.02", 
                       facecolor=colors['fused'], 
                       edgecolor='black', linewidth=1.5)
ax7.add_patch(rect7)
ax7.text(0.5, 0.6, 'Dempster\\textquotesingle s Rule', 
         ha='center', va='center', fontsize=11, weight='bold')
ax7.text(0.5, 0.45, '$\\mathbf{m}(x) = \\bigoplus_{j \\in A(x)} \\mathbf{m}_j$', 
         ha='center', va='center', fontsize=10)
ax7.text(0.5, 0.3, 'or Yager transfer', 
         ha='center', va='center', fontsize=9, style='italic')
ax7.text(0.5, 0.15, '(conflict $\\kappa \\to \\Theta$)', 
         ha='center', va='center', fontsize=8)
ax7.set_xlim(0, 1)
ax7.set_ylim(0, 1)
ax7.axis('off')

ax8 = fig.add_subplot(gs[2, 1])
ax8.text(0.5, 0.9, 'Pignistic Transform', ha='center', va='top', fontsize=12, weight='bold')
rect8 = FancyBboxPatch((0.1, 0.1), 0.8, 0.7, 
                       boxstyle="round,pad=0.02", 
                       facecolor=colors['pred'], 
                       edgecolor='black', linewidth=1.5)
ax8.add_patch(rect8)
ax8.text(0.5, 0.5, '$\\hat{p}_c = m_c + m_\\Theta \\cdot \\pi_c$', 
         ha='center', va='center', fontsize=11)
ax8.set_xlim(0, 1)
ax8.set_ylim(0, 1)
ax8.axis('off')

ax9 = fig.add_subplot(gs[2, 2])
ax9.text(0.5, 0.9, 'Prediction', ha='center', va='top', fontsize=12, weight='bold')
rect9 = FancyBboxPatch((0.1, 0.1), 0.8, 0.7, 
                       boxstyle="round,pad=0.02", 
                       facecolor=colors['pred'], 
                       edgecolor='black', linewidth=1.5)
ax9.add_patch(rect9)
ax9.text(0.5, 0.6, '$\\hat{y} = \\arg\\max_c \\hat{p}_c$', 
         ha='center', va='center', fontsize=11, weight='bold')
ax9.text(0.5, 0.4, 'Probability: $\\hat{\\mathbf{p}}$', 
         ha='center', va='center', fontsize=9)
ax9.text(0.5, 0.25, 'Uncertainty: $m_\\Theta$', 
         ha='center', va='center', fontsize=9)
ax9.set_xlim(0, 1)
ax9.set_ylim(0, 1)
ax9.axis('off')

# Arrows
arrow5 = FancyArrowPatch((0.95, 0.5), (0.05, 0.5), 
                         arrowstyle='->', lw=2, color='black')
ax7.add_patch(arrow5)
arrow6 = FancyArrowPatch((0.95, 0.5), (0.05, 0.5), 
                         arrowstyle='->', lw=2, color='black')
ax8.add_patch(arrow6)

# Add main title
fig.suptitle('DSGD-Auto: From Rule Induction to Dempster--Shafer Fusion', 
             fontsize=18, weight='bold', y=0.98)

# Save
plt.savefig('dsgd_auto_pipeline_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig('dsgd_auto_pipeline_visualization.pdf', dpi=300, bbox_inches='tight')
print("Created: dsgd_auto_pipeline_visualization.png and .pdf")
```

---

## generate_rule_inspection_report.py
Источник: `Common_code/generate_rule_inspection_report.py`

```python
from __future__ import annotations

"""
Generate a compact Markdown report with rule-inspector examples across datasets.

Designed to work in minimal environments: no pandas / sklearn required.
It loads the saved rule models from `Common_code/pkl_rules/`, encodes the CSV files
using the model schema (feature_names + value_decoders), and then reports:
  - rule-base size and typical rule length
  - "fusion depth" proxy: how many rules fire per sample
  - per-dataset examples with the combined (explanation) condition
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

THIS = Path(__file__).resolve()
COMMON = THIS.parent
ROOT = COMMON.parent

import sys

if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from DSModelMultiQ import DSModelMultiQ


def _guess_delimiter(csv_path: Path) -> str:
    line = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    return "," if line.count(",") >= line.count(";") else ";"


def _invert_value_decoders(value_decoders: dict[str, dict]) -> dict[str, dict[str, float]]:
    inv: dict[str, dict[str, float]] = {}
    for feat, d in (value_decoders or {}).items():
        m: dict[str, float] = {}
        for k, v in d.items():
            if not isinstance(v, str):
                continue
            try:
                if isinstance(k, str):
                    kk = k.strip()
                    code = float(int(kk)) if kk.isdigit() else float(kk)
                else:
                    code = float(k)
            except Exception:
                continue
            m[str(v)] = float(code)
        if m:
            inv[str(feat)] = m
    return inv


def _pick_label_column(columns: list[str]) -> str:
    preferred = {"label", "labels", "class", "target", "outcome", "diagnosis", "y", "result", "income", "chd"}
    for c in columns:
        if c.strip().lower() in preferred:
            return c
    return columns[-1]


@dataclass(frozen=True)
class LoadedData:
    X: np.ndarray  # float32
    y: np.ndarray  # int
    row_index: np.ndarray  # int: index in the original CSV (excluding header)
    classes: list[str]


def load_csv_for_model(
    csv_path: Path,
    *,
    feature_names: list[str],
    value_decoders: dict[str, dict],
    label_column: str | None = None,
) -> LoadedData:
    sep = _guess_delimiter(csv_path)

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=sep)
        columns = reader.fieldnames or []
        y_col = label_column or _pick_label_column(columns)

        inv = _invert_value_decoders(value_decoders)
        missing_tokens = {"", "?", "NA", "nan", "NaN", "None"}

        X_rows: list[list[float]] = []
        y_vals: list[str] = []
        row_ids: list[int] = []

        for i, row in enumerate(reader):
            ok = True
            for c in list(feature_names) + [y_col]:
                v = row.get(c)
                if v is None or v.strip() in missing_tokens:
                    ok = False
                    break
            if not ok:
                continue

            feats: list[float] = []
            for c in feature_names:
                raw = row[c].strip()
                if c in inv:
                    code = inv[c].get(raw)
                    if code is None:
                        ok = False
                        break
                    feats.append(float(code))
                else:
                    try:
                        feats.append(float(raw))
                    except Exception:
                        ok = False
                        break
            if not ok:
                continue

            X_rows.append(feats)
            y_vals.append(row[y_col].strip())
            row_ids.append(i)

    X = np.asarray(X_rows, dtype=np.float32)
    classes = sorted(set(y_vals))
    y_map = {c: j for j, c in enumerate(classes)}
    y = np.asarray([y_map[v] for v in y_vals], dtype=int)
    return LoadedData(X=X, y=y, row_index=np.asarray(row_ids, dtype=int), classes=classes)


def _set_simple_class_prior(model: DSModelMultiQ, y: np.ndarray) -> None:
    k = int(model.num_classes)
    cnt = np.bincount(np.asarray(y, dtype=int), minlength=k).astype(np.float32)
    cnt[cnt <= 0.0] = 1.0
    inv = (1.0 / cnt).astype(np.float32)
    inv = inv / (inv.sum() + 1e-12)
    model.class_prior = torch.tensor(inv, device=model.device, dtype=torch.float32)


def _percentile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.quantile(x, q))


def _format_literal(lit: tuple, *, max_value_len: int = 24) -> str:
    name, op, val = lit
    v = str(val)
    if len(v) > max_value_len:
        v = v[: max_value_len - 3] + "..."
    return f"{name} {op} {v}"


def _markdown_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def summarize_dataset(
    dataset_csv: Path,
    *,
    algo: str,
    sample_ids: Iterable[int],
    probe_n: int,
    seed: int,
    top_literals: int,
    top_rules: int,
) -> str:
    model_pkl = COMMON / "pkl_rules" / f"{algo.lower()}_{dataset_csv.stem}_dst.pkl"
    model = DSModelMultiQ(k=2, algo=algo, device="cpu")
    model.load_rules_bin(str(model_pkl))

    data = load_csv_for_model(
        dataset_csv,
        feature_names=list(model.feature_names or []),
        value_decoders=model.value_names or {},
    )
    _set_simple_class_prior(model, data.y)

    rule_lens = np.asarray([len(r.get("specs") or ()) for r in model.rules], dtype=int)

    rng = np.random.default_rng(int(seed))
    n = min(int(probe_n), int(data.X.shape[0]))
    if n > 0:
        idx = rng.choice(int(data.X.shape[0]), size=n, replace=False)
        Xp = data.X[idx]
        act = model._activation_matrix(model._prepare_numeric_tensor(Xp)).detach().cpu().numpy().astype(bool)
        fired = act.sum(axis=1).astype(int)
        holes = float((fired == 0).mean())
        depth_mean = float(fired.mean())
        depth_p90 = _percentile(fired, 0.90)
        depth_max = int(fired.max())
    else:
        holes = 1.0
        depth_mean = 0.0
        depth_p90 = 0.0
        depth_max = 0

    lines: list[str] = []
    lines.append(f"### {dataset_csv.name} — {algo}")
    lines.append("")
    lines.append(
        f"- Rules: {int(rule_lens.size)}; mean literals/rule: {rule_lens.mean():.2f}; p90: {int(_percentile(rule_lens, 0.90))}"
    )
    lines.append(
        f"- Fusion depth (rules fired/sample): mean {depth_mean:.2f}; p90 {int(depth_p90)}; max {depth_max}; holes {holes*100:.1f}%"
    )
    lines.append("")
    lines.append("| sample | row | true | pred_idx | fired | Ω | combined (top literals) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")

    for sid in sample_ids:
        if data.X.shape[0] == 0:
            continue
        i = int(sid) % int(data.X.shape[0])
        d = model.get_combined_rule(
            data.X[i],
            return_details=True,
            combination_rule="dempster",
        )
        comb_lits = d.get("combined_literals") or []
        shown = []
        for it in comb_lits[: int(top_literals)]:
            lit = it.get("literal")
            if not lit:
                continue
            shown.append(_format_literal(tuple(lit)))
        comb_str = " AND ".join(shown) if shown else "<empty>"

        lines.append(
            "| {sample} | {row} | {true} | {pred} | {fired} | {omega:.4f} | {comb} |".format(
                sample=i,
                row=int(data.row_index[i]),
                true=_markdown_escape(str(data.classes[int(data.y[i])])),
                pred=int(d.get("predicted_class", -1)),
                fired=int(d.get("n_rules_fired", 0)),
                omega=float(d.get("combined_omega", float("nan"))),
                comb=_markdown_escape(comb_str),
            )
        )

        # Optional: top contributing rules (useful to see real human-readable rules).
        contrib = d.get("rule_contributions") or []
        if int(top_rules) > 0 and contrib:
            lines.append("")
            lines.append("Top contributing rules:")
            for rc in contrib[: int(top_rules)]:
                cap = rc.get("caption", "")
                w = rc.get("weight", 0.0)
                cls = rc.get("rule_class", None)
                agree = "agree" if rc.get("agrees") else "conflict"
                lines.append(f"- [{agree}] class={cls} w={float(w):+.4f} :: {_markdown_escape(str(cap))}")
            lines.append("")

    lines.append("")
    return "\n".join(lines)


def resolve_dataset(path_or_name: str) -> Path:
    path = Path(path_or_name)
    if path.suffix.lower() == ".csv" and path.is_file():
        return path
    for base in (ROOT, COMMON, Path.cwd()):
        candidate = base / f"{path_or_name}.csv"
        if candidate.is_file():
            return candidate
    return path


def main() -> None:
    p = argparse.ArgumentParser("Generate a Markdown report with combined-rule examples.")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["adult.csv", "bank-full.csv", "BrainTumor.csv", "breast-cancer-wisconsin.csv"],
        help="Dataset paths or names (CSV).",
    )
    p.add_argument("--algos", nargs="+", default=["STATIC", "RIPPER", "FOIL"], help="Any of: STATIC RIPPER FOIL")
    p.add_argument("--samples", nargs="+", type=int, default=[0, 123, 1024], help="Sample indices (mod N)")
    p.add_argument("--probe-n", type=int, default=500, help="How many random samples to estimate depth stats")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for depth stats sampling")
    p.add_argument("--top-literals", type=int, default=10, help="How many literals to show in combined condition")
    p.add_argument("--top-rules", type=int, default=3, help="How many rule contributions to list per example")
    p.add_argument("--out", default="RULE_INSPECTOR_REPORT.md", help="Output Markdown path")
    args = p.parse_args()

    datasets = [resolve_dataset(d) for d in args.datasets]
    for ds in datasets:
        if not ds.exists():
            raise SystemExit(f"dataset not found: {ds}")

    algos = [str(a).upper() for a in args.algos]
    for a in algos:
        if a not in {"STATIC", "RIPPER", "FOIL"}:
            raise SystemExit(f"unsupported algo: {a}")

    out = Path(args.out)
    lines: list[str] = []
    lines.append("# Rule Inspector Report (DSGD-Auto)")
    lines.append("")
    lines.append("This report shows, per dataset and rule-induction method, (1) rule-base scale, (2) fusion depth proxy (rules fired/sample), and (3) combined-rule examples produced by `DSModelMultiQ.get_combined_rule` (explanation-only).")
    lines.append("")
    lines.append("Combined condition is built from positive rule contributions, aggregated per feature/op into a single conjunctive string; it is not used for inference.")
    lines.append("")

    for ds in datasets:
        lines.append(f"## {ds.name}")
        lines.append("")
        for algo in algos:
            lines.append(
                summarize_dataset(
                    ds,
                    algo=algo,
                    sample_ids=args.samples,
                    probe_n=int(args.probe_n),
                    seed=int(args.seed),
                    top_literals=int(args.top_literals),
                    top_rules=int(args.top_rules),
                )
            )

    out.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
```

---

## generate_deep_rule_quality_report.py
Источник: `Common_code/generate_deep_rule_quality_report.py`

```python
from __future__ import annotations

"""
Generate a deep, evaluation-focused Markdown report about rule quality and model behavior.

Goals (per dataset + algorithm):
- predictive quality (accuracy, macro/weighted F1, NLL, Brier, ECE)
- generalization (train vs test for combined-rule masks)
- DS-specific behavior (unc_rule vs unc_comb, fusion depth, holes)
- interpretability proxies (rule length, combined-rule length, feature diversity, decoded literals)
- rule quality (coverage/precision of most-used rules)
- examples with decoded combined rules + coverage/precision

Designed to work without pandas/sklearn: uses CSV module + numpy/torch only.
"""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

THIS = Path(__file__).resolve()
COMMON = THIS.parent
ROOT = COMMON.parent

import sys

if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from DSModelMultiQ import DSModelMultiQ


# ----------------------------- CSV loading (no pandas) -----------------------------
def _guess_delimiter(csv_path: Path) -> str:
    line = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    return "," if line.count(",") >= line.count(";") else ";"


def _pick_label_column(columns: list[str]) -> str:
    preferred = {"label", "labels", "class", "target", "outcome", "diagnosis", "y", "result", "income", "chd"}
    for c in columns:
        if c.strip().lower() in preferred:
            return c
    return columns[-1]


def _invert_value_decoders(value_decoders: dict[str, dict]) -> dict[str, dict[str, float]]:
    inv: dict[str, dict[str, float]] = {}
    for feat, d in (value_decoders or {}).items():
        m: dict[str, float] = {}
        for k, v in d.items():
            if not isinstance(v, str):
                continue
            try:
                if isinstance(k, str):
                    kk = k.strip()
                    code = float(int(kk)) if kk.isdigit() else float(kk)
                else:
                    code = float(k)
            except Exception:
                continue
            m[str(v)] = float(code)
        if m:
            inv[str(feat)] = m
    return inv


@dataclass(frozen=True)
class LoadedData:
    X: np.ndarray  # float32
    y: np.ndarray  # int [0..K-1]
    row_index: np.ndarray  # int: original CSV row index (excluding header)
    classes: list[str]  # sorted unique labels (as strings)
    label_column: str


def load_csv_for_model(
    csv_path: Path,
    *,
    feature_names: list[str],
    value_decoders: dict[str, dict],
    label_column: str | None = None,
) -> LoadedData:
    sep = _guess_delimiter(csv_path)
    missing_tokens = {"", "?", "NA", "nan", "NaN", "None"}
    inv = _invert_value_decoders(value_decoders)

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=sep)
        columns = reader.fieldnames or []
        y_col = label_column or _pick_label_column(columns)

        X_rows: list[list[float]] = []
        y_vals: list[str] = []
        row_ids: list[int] = []

        for i, row in enumerate(reader):
            ok = True
            for c in list(feature_names) + [y_col]:
                v = row.get(c)
                if v is None or v.strip() in missing_tokens:
                    ok = False
                    break
            if not ok:
                continue

            feats: list[float] = []
            for c in feature_names:
                raw = row[c].strip()
                if c in inv:
                    code = inv[c].get(raw)
                    if code is None:
                        ok = False
                        break
                    feats.append(float(code))
                else:
                    try:
                        feats.append(float(raw))
                    except Exception:
                        ok = False
                        break
            if not ok:
                continue

            X_rows.append(feats)
            y_vals.append(row[y_col].strip())
            row_ids.append(i)

    X = np.asarray(X_rows, dtype=np.float32)
    classes = sorted(set(y_vals))
    y_map = {c: j for j, c in enumerate(classes)}
    y = np.asarray([y_map[v] for v in y_vals], dtype=int)
    return LoadedData(
        X=X,
        y=y,
        row_index=np.asarray(row_ids, dtype=int),
        classes=classes,
        label_column=y_col,
    )


def split_train_test_stratified(
    X: np.ndarray,
    y: np.ndarray,
    row_index: np.ndarray,
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    test_idx = []
    train_idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = int(round(float(test_size) * len(cls_idx)))
        if len(cls_idx) > 1:
            n_test = max(1, min(len(cls_idx) - 1, n_test))
        else:
            n_test = 1
        test_idx.append(cls_idx[:n_test])
        train_idx.append(cls_idx[n_test:])
    test_idx = np.concatenate(test_idx) if test_idx else np.array([], dtype=int)
    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], row_index[train_idx], row_index[test_idx]


# ----------------------------- Metrics -----------------------------
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def f1_scores(y_true: np.ndarray, y_pred: np.ndarray, *, k: int) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    k = int(k)
    if y_true.size == 0:
        return {"f1_macro": float("nan"), "f1_weighted": float("nan")}

    f1s = []
    weights = []
    for c in range(k):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 0.0 if (prec + rec) < 1e-12 else (2.0 * prec * rec) / (prec + rec)
        f1s.append(float(f1))
        weights.append(int((y_true == c).sum()))

    weights_np = np.asarray(weights, dtype=float)
    wsum = float(weights_np.sum()) if weights_np.size else 0.0
    f1_macro = float(np.mean(f1s))
    f1_weighted = float(np.dot(weights_np / max(wsum, 1e-12), np.asarray(f1s, dtype=float))) if wsum > 0 else float("nan")
    return {"f1_macro": f1_macro, "f1_weighted": f1_weighted}


def nll(probs: np.ndarray, y_true: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if probs.size == 0:
        return float("nan")
    p = probs[np.arange(len(y_true)), y_true]
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def brier(probs: np.ndarray, y_true: np.ndarray, *, k: int) -> float:
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    k = int(k)
    if probs.size == 0:
        return float("nan")
    y_oh = np.zeros_like(probs)
    y_oh[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)))


def ece(probs: np.ndarray, y_true: np.ndarray, *, n_bins: int = 15) -> float:
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if probs.size == 0:
        return float("nan")
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    e = 0.0
    for i in range(int(n_bins)):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        acc_bin = float(correct[m].mean())
        conf_bin = float(conf[m].mean())
        e += float(m.mean()) * abs(acc_bin - conf_bin)
    return float(e)


# ----------------------------- Decoding / readability -----------------------------
def decode_literal(model: DSModelMultiQ, lit: tuple[str, str, Any]) -> str:
    name, op, val = lit
    if op == "==" and isinstance(val, (float, int, np.floating, np.integer)):
        dec = (model.value_names.get(name) or {})
        # try float -> int matching
        try:
            iv = int(round(float(val)))
        except Exception:
            iv = None
        if iv is not None and iv in dec:
            return f"{name} == {dec[iv]}"
    return f"{name} {op} {val}"


def format_combined_rule(model: DSModelMultiQ, combined_literals: list[dict], *, top_k: int = 12) -> str:
    if not combined_literals:
        return "<empty>"
    parts = []
    for it in combined_literals[: int(top_k)]:
        lit = it.get("literal")
        if not lit:
            continue
        parts.append(decode_literal(model, tuple(lit)))
    return " AND ".join(parts) if parts else "<empty>"


def _lit_to_mask(
    X: np.ndarray,
    feat_to_idx: dict[str, int],
    model: DSModelMultiQ,
    lit: tuple[str, str, Any],
) -> np.ndarray:
    name, op, val = lit
    j = feat_to_idx.get(str(name))
    if j is None:
        return np.zeros(X.shape[0], dtype=bool)
    col = X[:, int(j)]
    if op == "==":
        # allow string values from some STATIC rules
        if isinstance(val, str):
            inv = _invert_value_decoders(model.value_names).get(str(name), {})
            v = inv.get(val)
            if v is None:
                return np.zeros(X.shape[0], dtype=bool)
            val = v
        try:
            fv = float(val)
        except Exception:
            return np.zeros(X.shape[0], dtype=bool)
        return np.isfinite(col) & np.isclose(col, fv, atol=1e-5)
    try:
        fv = float(val)
    except Exception:
        return np.zeros(X.shape[0], dtype=bool)
    if op == "<":
        return np.isfinite(col) & (col < fv)
    if op == ">":
        return np.isfinite(col) & (col > fv)
    return np.zeros(X.shape[0], dtype=bool)


def combined_rule_quality(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: DSModelMultiQ,
    combined_literals: list[dict],
    pred_class: int,
    max_literals: int,
) -> dict[str, float]:
    """Evaluate the combined (explanation) rule as a hard filter on (X,y)."""
    if X.shape[0] == 0:
        return {"coverage": 0.0, "precision": float("nan"), "matched": 0, "correct": 0}
    lits = []
    for it in (combined_literals or [])[: int(max_literals)]:
        lit = it.get("literal")
        if lit:
            lits.append(tuple(lit))
    if not lits:
        return {"coverage": 0.0, "precision": float("nan"), "matched": 0, "correct": 0}

    feat_to_idx = {n: i for i, n in enumerate(model.feature_names or [])}
    mask = np.ones(X.shape[0], dtype=bool)
    for lit in lits:
        mask &= _lit_to_mask(X, feat_to_idx, model, lit)
        if not mask.any():
            break
    cov = float(mask.mean())
    matched = int(mask.sum())
    if matched > 0:
        correct = int((y[mask] == int(pred_class)).sum())
        prec = float(correct / matched)
    else:
        correct = 0
        prec = float("nan")
    return {"coverage": cov, "precision": prec, "matched": matched, "correct": correct}


# ----------------------------- Model execution helpers -----------------------------
def set_class_prior_inverse_freq(model: DSModelMultiQ, y_train: np.ndarray) -> None:
    k = int(model.num_classes)
    cnt = np.bincount(np.asarray(y_train, dtype=int), minlength=k).astype(np.float32)
    cnt[cnt <= 0.0] = 1.0
    inv = (1.0 / cnt).astype(np.float32)
    inv = inv / (inv.sum() + 1e-12)
    model.class_prior = torch.tensor(inv, device=model.device, dtype=torch.float32)


@torch.inference_mode()
def forward_probs(model: DSModelMultiQ, X: np.ndarray, *, batch_size: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    out = []
    for i in range(0, X.shape[0], int(batch_size)):
        xb = X[i : i + int(batch_size)]
        pb = model.forward(xb, combination_rule="dempster").detach().cpu().numpy()
        out.append(pb)
    return np.concatenate(out, axis=0) if out else np.zeros((0, int(model.num_classes)), dtype=np.float32)


@torch.inference_mode()
def activation_counts(model: DSModelMultiQ, X: np.ndarray, *, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (fired_per_sample, fired_per_rule)."""
    R = int(len(model.rules))
    fired_rule = np.zeros(R, dtype=np.int64)
    fired_sample = []
    for i in range(0, X.shape[0], int(batch_size)):
        xb = X[i : i + int(batch_size)]
        act = model._activation_matrix(model._prepare_numeric_tensor(xb)).detach().cpu().numpy().astype(bool)
        fired_sample.append(act.sum(axis=1).astype(np.int32))
        fired_rule += act.sum(axis=0).astype(np.int64)
    return (np.concatenate(fired_sample, axis=0) if fired_sample else np.zeros(0, dtype=np.int32)), fired_rule


def summarize_rule_quality(
    model: DSModelMultiQ,
    *,
    fired_per_rule: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Compute coverage/precision for top fired rules on test."""
    R = int(len(model.rules))
    top_n = min(int(top_n), R)
    if top_n <= 0:
        return []
    order = np.argsort(-np.asarray(fired_per_rule, dtype=np.int64))[:top_n]

    feat_to_idx = {n: i for i, n in enumerate(model.feature_names or [])}

    masses = None
    if getattr(model, "rule_mass_params", None) is not None:
        try:
            masses = model.get_rule_masses().detach().cpu().numpy()  # (R, K+1)
        except Exception:
            masses = None

    res = []
    for rid in order.tolist():
        r = model.rules[int(rid)]
        specs = r.get("specs") or ()
        if not specs:
            continue
        mask = np.ones(X_test.shape[0], dtype=bool)
        for lit in specs:
            mask &= _lit_to_mask(X_test, feat_to_idx, model, tuple(lit))
            if not mask.any():
                break
        matched = int(mask.sum())
        cov = float(mask.mean())
        lbl = r.get("label")
        if lbl is None or matched == 0:
            prec = float("nan")
        else:
            correct = int((y_test[mask] == int(lbl)).sum())
            prec = float(correct / matched)
        omega = None
        if masses is not None and int(rid) < masses.shape[0]:
            omega = float(masses[int(rid), -1])
        res.append(
            {
                "rule_id": int(rid),
                "caption": str(r.get("caption", "")),
                "label": int(lbl) if lbl is not None else None,
                "coverage": cov,
                "matched": matched,
                "precision": prec,
                "omega": omega,
            }
        )
    return res


def percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)) if x.size else float("nan")


def safe_int(x: float) -> int:
    if not np.isfinite(x):
        return 0
    return int(x)


def md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


# ----------------------------- Report generation -----------------------------
def analyze_one(
    dataset_csv: Path,
    *,
    algo: str,
    test_size: float,
    seed: int,
    batch_size: int,
    explain_n: int,
    combined_top_literals: int,
) -> dict[str, Any]:
    model_pkl = COMMON / "pkl_rules" / f"{algo.lower()}_{dataset_csv.stem}_dst.pkl"
    model = DSModelMultiQ(k=2, algo=algo, device="cpu")
    model.load_rules_bin(str(model_pkl))

    data = load_csv_for_model(
        dataset_csv,
        feature_names=list(model.feature_names or []),
        value_decoders=model.value_names or {},
    )

    K = int(model.num_classes)
    if len(data.classes) != K:
        # Not fatal, but affects readability/metrics; keep the analysis honest.
        pass

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = split_train_test_stratified(
        data.X, data.y, data.row_index, test_size=float(test_size), seed=int(seed)
    )

    # Match pignistic prior behavior to something deterministic (inverse-frequency).
    set_class_prior_inverse_freq(model, y_tr)

    # Predictive metrics (test)
    probs_te = forward_probs(model, X_te, batch_size=int(batch_size))
    pred_te = probs_te.argmax(axis=1).astype(int) if probs_te.size else np.zeros(0, dtype=int)

    acc_te = accuracy(y_te, pred_te)
    f1 = f1_scores(y_te, pred_te, k=K)
    nll_te = nll(probs_te, y_te)
    brier_te = brier(probs_te, y_te, k=K)
    ece_te = ece(probs_te, y_te, n_bins=15)

    # DS uncertainty stats (test)
    unc = model.uncertainty_stats(X_te, combination_rule="dempster")
    unc_rule_mean = float(np.mean(unc["unc_rule"])) if len(unc["unc_rule"]) else float("nan")
    unc_comb_mean = float(np.mean(unc["unc_comb"])) if len(unc["unc_comb"]) else float("nan")

    # Activation depth stats (test) + per-rule firing frequency
    fired_per_sample, fired_per_rule = activation_counts(model, X_te, batch_size=int(batch_size))
    holes = float((fired_per_sample == 0).mean()) if fired_per_sample.size else 1.0

    # Top rules by usage + their generalization (precision on test)
    top_rules = summarize_rule_quality(
        model,
        fired_per_rule=fired_per_rule,
        X_test=X_te,
        y_test=y_te,
        batch_size=int(batch_size),
        top_n=12,
    )

    # Explain subset: conflict rate, combined rule length + combined-rule precision/coverage (train/test)
    rng = np.random.default_rng(int(seed))
    n_ex = min(int(explain_n), int(X_te.shape[0]))
    ex_idx = rng.choice(int(X_te.shape[0]), size=n_ex, replace=False) if n_ex > 0 else np.array([], dtype=int)

    conflict_flags = []
    conf_counts = []
    agree_counts = []
    comb_len = []
    comb_feat_count = []
    comb_empty = []
    comb_cov_te = []
    comb_prec_te = []
    comb_match_te = []
    comb_cov_tr = []
    comb_prec_tr = []
    comb_match_tr = []
    comb_decode_ok = []
    comb_decode_total = []

    examples = []
    for i in ex_idx.tolist():
        x = X_te[int(i)]
        y_true = int(y_te[int(i)])
        d = model.get_combined_rule(x, return_details=True, combination_rule="dempster")
        pred_cls = int(d.get("predicted_class", -1))
        fired = int(d.get("n_rules_fired", 0))
        omega = float(d.get("combined_omega", float("nan")))

        n_conf = int(d.get("n_conflicting", 0))
        n_agree = int(d.get("n_agreeing", 0))
        conflict_flags.append(n_conf > 0)
        conf_counts.append(n_conf)
        agree_counts.append(n_agree)

        cl = d.get("combined_literals") or []
        comb_empty.append(len(cl) == 0)
        comb_len.append(int(len(cl)))
        feats = {str(it.get("literal", ("", "", ""))[0]) for it in cl if it.get("literal")}
        feats.discard("")
        comb_feat_count.append(int(len(feats)))

        q_te = combined_rule_quality(
            X_te,
            y_te,
            model=model,
            combined_literals=cl,
            pred_class=pred_cls,
            max_literals=int(combined_top_literals),
        )
        q_tr = combined_rule_quality(
            X_tr,
            y_tr,
            model=model,
            combined_literals=cl,
            pred_class=pred_cls,
            max_literals=int(combined_top_literals),
        )
        comb_cov_te.append(float(q_te["coverage"]))
        comb_prec_te.append(float(q_te["precision"]) if np.isfinite(q_te["precision"]) else float("nan"))
        comb_match_te.append(int(q_te.get("matched", 0)))
        comb_cov_tr.append(float(q_tr["coverage"]))
        comb_prec_tr.append(float(q_tr["precision"]) if np.isfinite(q_tr["precision"]) else float("nan"))
        comb_match_tr.append(int(q_tr.get("matched", 0)))

        # Decoding rate for categorical equalities in combined literals
        ok_cnt = 0
        tot_cnt = 0
        for it in cl[: int(combined_top_literals)]:
            lit = it.get("literal")
            if not lit or len(lit) != 3:
                continue
            name, op, val = lit
            if op != "==":
                continue
            dec = (model.value_names.get(str(name)) or {})
            if not dec:
                continue
            tot_cnt += 1
            # If the value is already a human label string (STATIC sometimes stores decoded strings),
            # count it as "decoded".
            if isinstance(val, str):
                ok_cnt += 1
                continue
            try:
                iv = int(round(float(val)))
            except Exception:
                iv = None
            if iv is not None and iv in dec:
                ok_cnt += 1
        comb_decode_ok.append(ok_cnt)
        comb_decode_total.append(tot_cnt)

        # Keep a small curated set of examples: wrong predictions, high omega, and typical.
        # We'll choose them later from this pool.
        examples.append(
            {
                "test_index": int(i),
                "row": int(idx_te[int(i)]),
                "true_idx": y_true,
                "true_label": data.classes[y_true] if y_true < len(data.classes) else str(y_true),
                "pred_idx": pred_cls,
                "fired": fired,
                "omega": omega,
                "combined": format_combined_rule(model, cl, top_k=int(combined_top_literals)),
                "contrib": (d.get("rule_contributions") or [])[:5],
                "rule_cov_te": float(q_te["coverage"]),
                "rule_prec_te": float(q_te["precision"]) if np.isfinite(q_te["precision"]) else float("nan"),
                "rule_cov_tr": float(q_tr["coverage"]),
                "rule_prec_tr": float(q_tr["precision"]) if np.isfinite(q_tr["precision"]) else float("nan"),
                "rule_matched_te": int(q_te.get("matched", 0)),
                "rule_matched_tr": int(q_tr.get("matched", 0)),
                "n_train": int(X_tr.shape[0]),
                "n_test": int(X_te.shape[0]),
                "correct": bool(pred_cls == y_true),
            }
        )

    rule_lens = np.asarray([len(r.get("specs") or ()) for r in model.rules], dtype=int)

    # Curate examples deterministically from the explain pool
    ex_wrong = next((e for e in examples if not e["correct"]), None)
    ex_high_omega = max(examples, key=lambda e: float(e["omega"]) if np.isfinite(e["omega"]) else -1.0) if examples else None
    ex_typ = min(examples, key=lambda e: abs(float(e["omega"]) - unc_comb_mean) + abs(float(e["fired"]) - safe_mean(fired_per_sample))) if examples else None
    curated = [e for e in [ex_typ, ex_high_omega, ex_wrong] if e is not None]

    cov_te_arr = np.asarray(comb_cov_te, dtype=float)
    prec_te_arr = np.asarray(comb_prec_te, dtype=float)
    cov_tr_arr = np.asarray(comb_cov_tr, dtype=float)
    prec_tr_arr = np.asarray(comb_prec_tr, dtype=float)

    dec_ok = int(np.sum(np.asarray(comb_decode_ok, dtype=int))) if comb_decode_ok else 0
    dec_tot = int(np.sum(np.asarray(comb_decode_total, dtype=int))) if comb_decode_total else 0
    dec_rate = float(dec_ok / dec_tot) if dec_tot > 0 else float("nan")

    return {
        "dataset": dataset_csv.name,
        "algo": str(algo).upper(),
        "label_column": data.label_column,
        "n_total": int(data.X.shape[0]),
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "classes": list(data.classes),
        "class_counts_total": np.bincount(data.y, minlength=max(1, int(model.num_classes))).astype(int).tolist(),
        "num_features": int(data.X.shape[1]),
        "num_rules": int(len(model.rules)),
        "rule_len_mean": float(rule_lens.mean()) if rule_lens.size else 0.0,
        "rule_len_p90": float(percentile(rule_lens, 0.90)) if rule_lens.size else 0.0,
        "rule_len_max": int(rule_lens.max()) if rule_lens.size else 0,
        "depth_mean": float(fired_per_sample.mean()) if fired_per_sample.size else 0.0,
        "depth_p90": float(percentile(fired_per_sample, 0.90)) if fired_per_sample.size else 0.0,
        "depth_max": int(fired_per_sample.max()) if fired_per_sample.size else 0,
        "holes": holes,
        "acc_test": acc_te,
        "f1_macro_test": float(f1["f1_macro"]),
        "f1_weighted_test": float(f1["f1_weighted"]),
        "nll_test": nll_te,
        "brier_test": brier_te,
        "ece_test": ece_te,
        "unc_rule_mean": unc_rule_mean,
        "unc_comb_mean": unc_comb_mean,
        "conf_rate_explain": float(np.mean(conflict_flags)) if conflict_flags else float("nan"),
        "agree_mean_explain": float(np.mean(agree_counts)) if agree_counts else float("nan"),
        "conf_mean_explain": float(np.mean(conf_counts)) if conf_counts else float("nan"),
        "combined_empty_rate": float(np.mean(comb_empty)) if comb_empty else float("nan"),
        "combined_len_mean": float(np.mean(comb_len)) if comb_len else float("nan"),
        "combined_len_p90": float(percentile(np.asarray(comb_len, dtype=float), 0.90)) if comb_len else float("nan"),
        "combined_feat_mean": float(np.mean(comb_feat_count)) if comb_feat_count else float("nan"),
        "combined_cov_test_mean": float(np.nanmean(np.asarray(comb_cov_te, dtype=float))) if comb_cov_te else float("nan"),
        "combined_prec_test_mean": float(np.nanmean(np.asarray(comb_prec_te, dtype=float))) if comb_prec_te else float("nan"),
        "combined_cov_train_mean": float(np.nanmean(np.asarray(comb_cov_tr, dtype=float))) if comb_cov_tr else float("nan"),
        "combined_prec_train_mean": float(np.nanmean(np.asarray(comb_prec_tr, dtype=float))) if comb_prec_tr else float("nan"),
        "combined_cov_test_p50": float(np.nanmedian(cov_te_arr)) if cov_te_arr.size else float("nan"),
        "combined_cov_test_p90": float(np.nanquantile(cov_te_arr, 0.90)) if cov_te_arr.size else float("nan"),
        "combined_prec_test_p50": float(np.nanmedian(prec_te_arr)) if prec_te_arr.size else float("nan"),
        "combined_prec_test_p90": float(np.nanquantile(prec_te_arr, 0.90)) if prec_te_arr.size else float("nan"),
        "combined_cat_decode_rate": dec_rate,
        "top_rules": top_rules,
        "examples": curated,
    }


def md_section_for_result(r: dict[str, Any]) -> str:
    ds = r["dataset"]
    algo = r["algo"]
    K = max(1, len(r.get("classes") or []))

    lines: list[str] = []
    lines.append(f"### {algo}")
    lines.append("")
    lines.append(f"- Data: n_total={r['n_total']} (train={r['n_train']}, test={r['n_test']}), d={r['num_features']}, label_col=`{r['label_column']}`")
    lines.append(f"- Classes (K={K}): {', '.join(map(str, r.get('classes') or []))} | counts={r.get('class_counts_total')}")
    lines.append(f"- Rule base: R={r['num_rules']}; literals/rule mean={r['rule_len_mean']:.2f}, p90={r['rule_len_p90']:.1f}, max={r['rule_len_max']}")
    lines.append(
        f"- Fusion depth (rules fired/sample): mean={r['depth_mean']:.2f}, p90={r['depth_p90']:.1f}, max={r['depth_max']}; holes={r['holes']*100:.2f}%"
    )
    lines.append("")

    lines.append("Predictive quality (test):")
    lines.append(f"- Accuracy={r['acc_test']:.4f} | F1_macro={r['f1_macro_test']:.4f} | F1_weighted={r['f1_weighted_test']:.4f}")
    lines.append(f"- NLL={r['nll_test']:.4f} | Brier={r['brier_test']:.4f} | ECE(15)={r['ece_test']:.4f}")
    lines.append("")

    lines.append("Uncertainty / DS behavior:")
    lines.append(f"- Ω rule-avg (unc_rule) mean={r['unc_rule_mean']:.4f}")
    lines.append(f"- Ω combined (unc_comb) mean={r['unc_comb_mean']:.4f}")
    lines.append("")

    lines.append("Combined-rule interpretability (explain subset; explanation-only):")
    lines.append(
        f"- Combined empty rate={r['combined_empty_rate']*100:.1f}% | literals mean={r['combined_len_mean']:.2f} (p90={r['combined_len_p90']:.1f}) | distinct features mean={r['combined_feat_mean']:.2f}"
    )
    lines.append(
        f"- Combined rule as filter (mean over explain subset): test coverage={r['combined_cov_test_mean']*100:.3f}% precision={r['combined_prec_test_mean']*100:.2f}%"
    )
    lines.append(
        f"- Generalization (mean train→test): coverage {r['combined_cov_train_mean']*100:.3f}%→{r['combined_cov_test_mean']*100:.3f}% | precision {r['combined_prec_train_mean']*100:.2f}%→{r['combined_prec_test_mean']*100:.2f}%"
    )
    if "combined_cov_test_p50" in r:
        lines.append(
            f"- Combined rule distribution (test): coverage p50={r['combined_cov_test_p50']*100:.3f}% p90={r['combined_cov_test_p90']*100:.3f}% | precision p50={r['combined_prec_test_p50']*100:.2f}% p90={r['combined_prec_test_p90']*100:.2f}%"
        )
        lines.append(
            f"- Literal decoding (combined rules): decoded categorical share={r.get('combined_cat_decode_rate', float('nan'))*100:.1f}% (lower means more code-like, less human-readable)"
        )
    lines.append("")

    if np.isfinite(r.get("conf_rate_explain", float("nan"))):
        lines.append("Conflict diagnostics (explain subset):")
        lines.append(
            f"- conflict rate={r['conf_rate_explain']*100:.1f}% | mean agreeing rules={r['agree_mean_explain']:.2f} | mean conflicting rules={r['conf_mean_explain']:.2f}"
        )
        lines.append("")

    # Top rules
    lines.append("Most-used rules on test (top coverage):")
    lines.append("| rule_id | label | coverage | precision | Ω | caption |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for tr in r.get("top_rules") or []:
        omega = tr.get("omega")
        omega_s = f"{float(omega):.3f}" if omega is not None and np.isfinite(omega) else ""
        prec = tr.get("precision")
        prec_s = f"{float(prec)*100:.1f}%" if prec is not None and np.isfinite(prec) else ""
        lines.append(
            f"| {tr['rule_id']} | {tr.get('label','')} | {tr['coverage']*100:.2f}% ({tr.get('matched','')}) | {prec_s} | {omega_s} | {md_escape(tr.get('caption',''))} |"
        )
    lines.append("")

    # Examples
    lines.append("Examples (decoded combined rule + generalization of that combined rule):")
    for ex in r.get("examples") or []:
        status = "OK" if ex["correct"] else "WRONG"
        lines.append(
            f"- sample(test_idx={ex['test_index']}, row={ex['row']}) true=`{md_escape(ex['true_label'])}` pred_idx={ex['pred_idx']} fired={ex['fired']} Ω={ex['omega']:.4f} [{status}]"
        )
        lines.append(f"  - combined: {md_escape(ex['combined'])}")
        lines.append(
            f"  - combined-as-filter: "
            f"train matches={ex.get('rule_matched_tr',0)}/{ex.get('n_train',0)} cov={ex['rule_cov_tr']*100:.3f}% prec={ex['rule_prec_tr']*100:.2f}% | "
            f"test matches={ex.get('rule_matched_te',0)}/{ex.get('n_test',0)} cov={ex['rule_cov_te']*100:.3f}% prec={ex['rule_prec_te']*100:.2f}%"
        )
        contrib = ex.get("contrib") or []
        if contrib:
            lines.append("  - top contributions:")
            for rc in contrib[:3]:
                agree = "agree" if rc.get("agrees") else "conflict"
                lines.append(f"    - [{agree}] w={float(rc.get('weight',0.0)):+.4f} :: {md_escape(str(rc.get('caption','')))}")
    lines.append("")

    return "\n".join(lines)


def resolve_dataset(path_or_name: str) -> Path:
    path = Path(path_or_name)
    if path.suffix.lower() == ".csv" and path.is_file():
        return path
    for base in (ROOT, COMMON, Path.cwd()):
        candidate = base / f"{path_or_name}.csv"
        if candidate.is_file():
            return candidate
    return path


def main() -> None:
    p = argparse.ArgumentParser("Generate a deep rule-quality analysis report.")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "adult.csv",
            "bank-full.csv",
            "BrainTumor.csv",
            "breast-cancer-wisconsin.csv",
            "SAheart.csv",
            "df_wine.csv",
        ],
        help="Dataset CSV paths or names.",
    )
    p.add_argument("--algos", nargs="+", default=["STATIC", "RIPPER", "FOIL"], help="Any of: STATIC RIPPER FOIL")
    p.add_argument("--out", default="DEEP_RULE_QUALITY_REPORT.md", help="Output Markdown path")
    p.add_argument("--test-size", type=float, default=0.16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--explain-n", type=int, default=400, help="How many test samples to run get_combined_rule() on")
    p.add_argument("--combined-top-literals", type=int, default=12, help="How many combined literals to evaluate/print")
    args = p.parse_args()

    datasets = [resolve_dataset(d) for d in args.datasets]
    for ds in datasets:
        if not ds.exists():
            raise SystemExit(f"dataset not found: {ds}")

    algos = [str(a).upper() for a in args.algos]
    for a in algos:
        if a not in {"STATIC", "RIPPER", "FOIL"}:
            raise SystemExit(f"unsupported algo: {a}")

    lines: list[str] = []
    lines.append("# Deep Rule Quality Report (DSGD-Auto)")
    lines.append("")
    lines.append("This report evaluates rule-induced DSGD models across datasets, focusing on predictive quality, DST uncertainty behavior, fusion depth, and interpretability of the combined (explanation-only) rule.")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Train/test split is stratified with `test_size` and `seed`.")
    lines.append("- Probabilities are pignistic; for reproducibility the pignistic prior is set to inverse class frequency on the train split.")
    lines.append("- “Combined rule” is built from positive contributions only and is not used for inference; we still evaluate it as a *filter* to measure how sharply it describes the predicted situation and how well it generalizes.")
    lines.append("")

    # Group by dataset
    for ds in datasets:
        lines.append(f"## {ds.name}")
        lines.append("")
        for algo in algos:
            r = analyze_one(
                ds,
                algo=algo,
                test_size=float(args.test_size),
                seed=int(args.seed),
                batch_size=int(args.batch_size),
                explain_n=int(args.explain_n),
                combined_top_literals=int(args.combined_top_literals),
            )
            lines.append(md_section_for_result(r))

    Path(args.out).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
```
