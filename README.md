# Predicting Future E-Commerce Purchase Behaviour
A leakage-audited temporal machine learning study of 1.85M Amazon purchases

or

# Temporal ML for E-Commerce Purchase Prediction — 1.85M records, leakage-audited, SHAP-explained


**A time-based analysis of historical purchasing patterns and future purchase activity**

Predicting whether a customer will make at least one purchase in the next 90
days, using only behaviour observable before a fixed prediction cutoff. Built
on 1.85M real Amazon purchase records from 5,027 U.S. consumers.

> New to the project? Read [`readme_beginner.md`](readme_beginner.md) for a
> 30-second plain-English version.

**Headline results**

| | |
|---|---|
| Best ROC-AUC | **0.9046** (random = 0.500) |
| Best PR-AUC, minority class | **0.5280** (baseline 0.1258 — **4.20x**) |
| Best configuration | Behavioural features + Random Forest |
| Do demographics help? | **No** — ROC-AUC gain of +0.0003 to +0.0033 |
| Strongest single predictor | Days since last purchase |

---

## Overview

E-commerce retention teams need to know who is about to go quiet. This project
tests whether that is predictable from purchase history alone, and — the part
most projects skip — whether knowing a customer's demographics adds anything
once you already know how they shop.

It is framed as a **temporal prediction problem**, not a generic
classification task. Features come only from before a cutoff date; the target
comes only from after it; the test period falls chronologically after all
training periods. Leakage prevention is verified empirically, not asserted.

## Research Questions

> **RQ1.** Can a customer's future purchase activity be predicted from their
> historical purchasing behaviour?

> **RQ2.** Which historical purchasing patterns are most predictive of future
> purchase activity?

> **RQ3.** Does demographic information improve future-purchase prediction
> beyond historical purchasing behaviour?

## Dataset

**Open e-commerce 1.0 — Five years of crowdsourced U.S. Amazon purchase
histories with user demographics**
Berke, Calacci, Mahari, Yabe, Larson & Pentland (2023), Harvard Dataverse.
[doi:10.7910/DVN/YGLYDY](https://doi.org/10.7910/DVN/YGLYDY)

| Property | Value (measured, not quoted from documentation) |
|---|---|
| Customers | 5,027 |
| Purchase records | 1,850,717 |
| Records in analysis window | 1,778,350 (96.09%) |
| Nominal date range | 2018-01-01 to 2024-08-15 |
| Analysis window used | 2018-01-01 to 2022-11-30 |
| Files | `amazon-purchases.csv`, `survey.csv`, `fields.csv` |

The documentation's headline figures were treated as claims and verified
against the files. Row and customer counts matched; **the stated date range did
not** — data extends to 2024, not 2022.

### Privacy and variable selection

The survey contains health and sensitive lifestyle items. All were excluded:
substance use (cigarettes, marijuana, alcohol), diabetes, wheelchair use, and
sexual orientation. Also excluded were the study's own data-sharing opinion
questions, which are not customer attributes.

Race and Hispanic origin **were** included, deliberately. Measuring whether
ethnicity carries predictive signal is a legitimate empirical question that
directly serves RQ3. Deploying a model that allocates offers by ethnicity is a
different act with real legal exposure (ECOA/FHA) — a production version would
exclude them. See [Limitations](#limitations).

Two survey columns were excluded as **temporal leakage**, not privacy risk:

- `Q-amazon-use-how-oft` — self-reported Amazon purchase frequency, collected
  in 2022–23, *after* every prediction window. It directly encodes the target.
- `Q-life-changes` — recent life events, also post-cutoff.

## Prediction Task

```
        180 days of history          |         90 days of outcome
   ------------------------------------|------------------------------------
   recency, frequency, spending,       |    future_purchase_90d
   diversity, repeat, trend            |    1 = bought at least once
                                       |    0 = bought nothing
                            prediction cutoff
```

**Temporal design:**

```
TRAIN                                       TEST
2019-12-01  ─┐                              2021-06-01
2020-06-01  ─┤  180d features / 90d target  180d features (Dec 2020–May 2021)
2020-12-01  ─┘                              90d target (Jun–Aug 2021)

last training target closes 2021-03-01   →   92-day gap   →   test target opens
```

| | Rows | Positive rate |
|---|---|---|
| Train (3 snapshots) | 14,258 | 86.41% |
| Test (1 snapshot) | 4,928 | 87.42% |

Note the imbalance runs **toward the positive class**. The minority class is
*non-purchasers* — which is also the commercially interesting group, since
those are the re-engagement targets. All primary metrics are reported for that
class.

### Why the cutoff is mid-2021 and not late 2022

Each participant's purchase record terminates on the day they enrolled in the
study, not at a common endpoint. A prediction window overlapping the enrolment
period therefore labels unobserved customers as non-buyers — **fabricated
negatives**, concentrated entirely in the minority class.

We quantified this. For each candidate cutoff, a negative label counts as
*confirmed* if that customer has any purchase recorded after the window closes:

| Cutoff | Eligible | Negatives | Confirmed | Ambiguous share of all labels |
|---|---|---|---|---|
| 2019-12-01 | 4,618 | 680 | **97.2%** | 0.41% |
| 2020-06-01 | 4,771 | 705 | 95.7% | 0.63% |
| 2020-12-01 | 4,869 | 553 | 92.4% | 0.86% |
| **2021-06-01** | **4,928** | **620** | **86.9%** | **1.64%** |
| 2021-12-01 | 4,978 | 534 | 77.5% | 2.41% |
| 2022-06-01 | 5,007 | 633 | 61.0% | 4.93% |

Reliability declines continuously with no clean boundary, so there is no
"correct" cutoff — only a trade of recency against label validity. We chose
2021-06-01 at 1.64% ambiguous. Post-window data was used **only** for this
audit and never as a feature.

This censoring is not documented in the dataset paper.

## Methodology

**Data cleaning.** Missing `Title`/`Category`/`Shipping Address State` (~4.8%)
co-occur in 88,832 of 89,458 cases — one shared cause. Core fields (date,
price, quantity, customer) are complete on every affected row, so the rows are
**retained**; only category-diversity features skip them. 11,624 fully
duplicated rows (0.63%) are also **retained**: the file has no order ID, so
identical rows plausibly represent genuine same-day repeat orders. Both
decisions are documented rather than defaulted.

**Feature engineering.** 17 behavioural features from a single
`build_snapshot(cutoff)` function, computed strictly from the 180-day history
window:

| Group | Features |
|---|---|
| Recency | `recency_days`, `has_history` |
| Frequency | `n_items`, `n_transactions`, `n_active_days`, `tx_per_month` |
| Spending | `total_spend`, `avg_tx_value`, `median_item_price`, `max_item_price` |
| Diversity | `n_unique_products`, `n_unique_categories`, `category_concentration` |
| Repeat | `n_repeat_products`, `repeat_ratio` |
| Trend | `tx_trend`, `spend_trend` |

Customers with no activity in the history window are **kept**, not dropped —
they are the most likely non-buyers, and removing them would delete much of the
minority class. Recency is capped at 180 with `has_history = 0`.

**Deliberately excluded:** customer tenure. Records begin on 2018-01-01 for
everyone, so tenure measures the data export boundary, not the customer.
`avg_days_between_tx` was also dropped — undefined for 11% of rows for two
different reasons (never purchased vs purchased once), which no single imputed
value can represent honestly.

**Leakage prevention, verified.** Rather than relying on code review, the
snapshot function was rerun on data physically truncated at the cutoff and the
outputs compared:

```
features identical with and without post-cutoff data : True
target on truncated data : 0.00%
```

**Preprocessing** lives inside a scikit-learn `Pipeline` — median imputation
and scaling fitted on training data only, so test-set statistics cannot
influence training.

**Class imbalance** handled with `class_weight="balanced"` and XGBoost's
`scale_pos_weight`. SMOTE was **considered and rejected**: 1,938 minority
examples is a workable absolute count, and synthesising customers by
interpolating counts and monetary sums is hard to justify.

**Thresholds** were selected by maximising minority-class F1 via time-ordered
cross-validation **on the training snapshots only** — never on the test set.

## Results

### Full 3×3 comparison — minority class (non-purchasers), tuned thresholds

| Feature Group | Model | Thr | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| Behavioural | Logistic Regression | 0.31 | 0.8780 | 0.5131 | 0.5984 | 0.5525 | 0.9000 | 0.5218 |
| Behavioural | **Random Forest** | 0.57 | 0.8726 | 0.4951 | 0.6500 | 0.5621 | **0.9046** | **0.5280** |
| Behavioural | XGBoost | 0.34 | 0.8782 | 0.5139 | 0.5984 | 0.5529 | 0.9017 | 0.5302 |
| Demographic | Logistic Regression | 0.54 | 0.5081 | 0.1652 | 0.7177 | 0.2686 | 0.6293 | 0.1780 |
| Demographic | Random Forest | 0.48 | 0.7839 | 0.3084 | 0.5774 | 0.4020 | 0.7779 | 0.3416 |
| Demographic | XGBoost | 0.46 | 0.7116 | 0.2377 | 0.5855 | 0.3381 | 0.7254 | 0.2637 |
| Behavioural + Demographic | Logistic Regression | 0.32 | 0.8756 | 0.5045 | 0.6387 | 0.5637 | 0.9003 | 0.5088 |
| Behavioural + Demographic | Random Forest | 0.50 | 0.8784 | 0.5139 | 0.6242 | 0.5637 | 0.9078 | 0.5349 |
| Behavioural + Demographic | XGBoost | 0.38 | 0.8748 | 0.5019 | 0.6532 | 0.5676 | 0.9023 | 0.5157 |

Baselines: accuracy 0.8742 (majority class), ROC-AUC 0.5000, PR-AUC 0.1258.

**Accuracy is not used for judgement.** A model predicting "everyone buys"
scores 0.8742 accuracy while identifying zero at-risk customers — higher than
our best model's 0.8726.

### Operational performance

Behavioural + Random Forest at threshold 0.57, on 4,928 test customers:

|  | Predicted: no purchase | Predicted: purchase |
|---|---|---|
| **Actually no purchase** | 403 | 217 |
| **Actually purchased** | 411 | 3,897 |

Flag 814 customers → 403 genuinely go quiet. **50% precision, 65% recall.**

## Key Findings

### RQ1 — Yes, and the signal is strong

ROC-AUC 0.9000–0.9046 across three model families, PR-AUC 4.20x over baseline.
Consistency across Logistic Regression, Random Forest and XGBoost — all within
0.005 ROC-AUC of each other — indicates the relationships are largely
linear-monotonic. Gradient boosting bought almost nothing over a linear model.

That consistency also partly addresses a known caveat: because the same
customers appear at multiple cutoffs, tree models *could* memorise individuals.
If they were doing so, they would have pulled clearly ahead of logistic
regression. They did not.

### RQ2 — Recency individually, activity volume collectively — and both saturate

SHAP decision weight by behaviour type:

| Behaviour type | Share |
|---|---|
| Activity volume | 57.5% |
| Breadth / diversity | 15.0% |
| Recency | 12.1% |
| Basket size | 7.2% |
| Momentum (trend) | 5.8% |
| Repeat buying | 2.3% |

Permutation importance ranks `recency_days` **first** at 0.0319 — roughly 3x
the next feature. SHAP ranks the volume cluster higher. Both are correct:
volume is a cluster of collinear features sharing credit, recency is one. Per
feature, **recency is the strongest single signal**; as a group, volume
dominates. Together they account for ~70% of decision weight.

#### Redundancy structure

The 17 behavioural features reduce to **9 independent groups**, one containing
eight of them: `n_items`, `n_transactions`, `n_active_days`, `tx_per_month`,
`total_spend`, `n_unique_products`, `n_unique_categories` and
`category_concentration`. Within it, `n_transactions`, `n_active_days` and
`tx_per_month` correlate at **1.000** — they are the same quantity, since two
are derived from the third. `n_items` and `n_unique_products` correlate 0.994.

Two consequences for the table above:

1. **The "Breadth / diversity" share overstates diversity as a distinct
   behaviour.** `n_unique_products` and `n_unique_categories` correlate
   0.939–0.984 with the volume cluster — a customer who buys more items
   necessarily touches more products and categories. Diversity here is largely
   volume in disguise, not an independent preference signal.
2. **`category_concentration` sits inside the volume cluster** at rho = −0.294,
   which explains the sign reversal described below.

`recency_days` is **independent**, in a cluster of its own. That is mechanically
why permutation importance ranks it first: no duplicate exists to mask it when
it is shuffled.

**The most useful finding is that effects saturate.** Average SHAP contribution
by real-unit customer group:

| Purchase days in 180 | 0–4 | 4–11 | 11–20 | 20–36 | 36–133 |
|---|---|---|---|---|---|
| Contribution | −0.044 | +0.043 | +0.082 | +0.083 | +0.083 |

| Total spend | ≤$170 | $170–515 | $515–1,021 | $1,021–1,991 | $1,991+ |
|---|---|---|---|---|---|
| Contribution | +0.004 | +0.030 | +0.036 | +0.036 | +0.036 |

Beyond ~11 purchase days per 180, or ~$500 spent, more activity adds nothing.
The model separates **barely active from active**, not moderate from heavy.
Recency is the only feature that goes properly negative (−0.056 in the 43–180
day band) and is therefore what distinguishes among already-active customers.

One reversal worth noting: `category_concentration` correlated −0.29 with the
target marginally, but SHAP shows it weakly *positive* across all bins. The
negative marginal relationship was activity volume acting through it — focused
shoppers simply buy less overall. A good illustration of why marginal
correlations mislead.

### RQ3 — No. Demographics add nothing beyond behaviour

Gain from adding 8 demographic features to the behavioural set:

| Metric | Logistic | Random Forest | XGBoost |
|---|---|---|---|
| ROC-AUC | +0.0003 | +0.0033 | +0.0006 |
| PR-AUC (minority) | −0.0130 | +0.0069 | −0.0146 |
| F1 (minority) | +0.0112 | +0.0017 | +0.0147 |

Demographics *alone* do beat chance (ROC-AUC 0.629–0.778), so they carry real
signal. But it is **redundant** with what purchase behaviour already reveals.
Gains flip sign depending on which metric you choose, which is the signature of
noise rather than effect.

EDA predicted this before any model ran: recency alone spanned 53 percentage
points of purchase rate (98% → 45%), while the widest demographic — income —
spanned 12.

This is a negative result and is reported as a primary finding.

## Business Implications

Clearly separating **model finding** from **business recommendation**:

**Finding:** the model identifies 65% of customers who will go quiet, at 50%
precision, flagging 814 of 4,928 customers.

**Possible applications, not proven interventions:**

- *Low-cost outreach* (email reminders, re-engagement content) is well
  supported. Reaching two-thirds of lapsing customers at 50% precision is
  strong value when the per-contact cost is cents.
- *High-cost incentives* (discount codes, free shipping) are marginal. Half of
  the 814 flagged customers would have purchased anyway, so you would give away
  411 unnecessary discounts to retain 403 customers.
- *Recency beats sophistication for triggering.* Since effects saturate, a
  simple recency threshold captures much of the model's value. Deploy the model
  where the extra 10–15% of discrimination justifies the complexity.
- *Do not over-segment by spend tier.* The model shows a $2,000 customer is no
  more predictable than a $500 one.

**No causal claims.** Higher historical purchase frequency was *associated with*
a higher predicted probability of future purchase. This does not establish that
increasing purchase frequency would cause future purchases. SHAP explains model
behaviour, not real-world causation.

## Limitations

- **Volunteer panel.** 5,027 self-selected participants who consented to share
  Amazon data. Skews young (37% aged 25–34) and is not representative of U.S.
  consumers.
- **Right-censoring is unresolvable, only mitigated.** With no enrolment-date
  field, "stopped buying" and "stopped being observed" are fundamentally
  indistinguishable for any customer whose record simply ends. At our chosen
  cutoff, 1.64% of labels remain ambiguous.
- **Repeated customers across snapshots.** The same 5,027 people appear at
  multiple cutoffs, so the model can partly memorise individual idiosyncrasy
  and performance is optimistic for genuinely new customers. This is not
  temporal leakage — no feature crosses its own cutoff — and it mirrors
  production retention modelling, where you retrain on an existing base. A
  customer-disjoint variant would quantify the gap.
- **Demographic stability assumed.** The survey ran in 2022–23, after every
  cutoff. Age, income and state can change over 2–3 years; we treat them as
  stable. Given RQ3's null result, this assumption does not affect conclusions.
- **Race as a possible proxy.** Ethnicity correlates with income, education and
  geography, all already in the feature set. Its importance cannot distinguish
  independent signal from proxy effects. A deployed model should exclude it for
  legal reasons regardless.
- **Transaction is a proxy.** No order ID exists, so a transaction is defined as
  a distinct `(customer, date)` pair. Two separate same-day orders count as one.
- **A structural blind spot.** The model's most confident mistake and its most
  confident correct rejection had *identical* feature values — both customers
  had zero purchases in the 180-day window. One returned, one did not. ~350
  customers per snapshot fall in this region, and roughly 45% of them purchase
  anyway. No modelling change fixes this; the information is absent.
- **Amazon only.** Behaviour on a marketplace with Prime subscriptions and
  one-click reordering may not transfer to other platforms.
- **One prediction window.** Results are specific to 90 days. A 30-day window
  showed a 72.9% base rate and would likely behave differently.
- **Two features are duplicates.** `n_transactions` and `n_active_days` are the
  same quantity under two names. Both were retained and flagged rather than
  removed mid-project.

## Debugging notes

Three bugs were caught and are documented rather than quietly fixed, because
each is instructive:

1. **`scale_pos_weight` inverted.** The convention `n_neg/n_pos` up-weights the
   positive class — but our minority class is negative. The wrong value (6.357)
   made XGBoost predict "will purchase" for 99.9% of the test set, giving
   minority recall of 0.0065 while ROC-AUC stayed at 0.9008. The ranking was
   fine; only the threshold behaviour was broken.
2. **`Pipeline` does not clone its final step.** Because one estimator instance
   per model was reused across loop iterations, all nine stored pipelines ended
   up pointing at classifiers fitted on the last feature group. Results were
   unaffected (predictions were taken inside the loop) but post-hoc explanation
   failed. Fixed with `clone()`.
3. **SHAP baseline misread.** `TreeExplainer.expected_value` returned ~0.50, and
   the first explanation offered — that balanced class weights shift the mean
   output — was disproved by measurement (mean predicted probability was 0.77
   on train, 0.82 on test). The additivity check confirmed max error 0.000000,
   so all SHAP values were valid regardless.

## Technologies

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, SciPy,
Jupyter / Google Colab.

## How to Run

```bash
pip install -r requirements.txt
```

1. Download the dataset from
   [Harvard Dataverse](https://doi.org/10.7910/DVN/YGLYDY) (Access Dataset →
   Download ZIP).
2. Place `amazon-purchases.csv`, `survey.csv` and `fields.csv` in `data/raw/`.
3. Open `notebooks/Ecommerce_future_purchase_prediction.ipynb` and set `RAW_DIR` to your
   data path.
4. Run all cells. Runtime is roughly 10 minutes; the threshold search and SHAP
   are the slow steps.

## Repository Structure

```
ecommerce-future-purchase-prediction/
├── data/
│   └── README.md          # download instructions, not the data
├── notebooks/
│   └── Ecommerce_future_purchase_prediction.ipynb
├── figures/
├── README.md
├── readme_beginner.md
├── requirements.txt
└── .gitignore
```

## Citation

```
Berke, A., Calacci, D., Mahari, R., Yabe, T., Larson, K., & Pentland, S. (2024).
Open e-commerce 1.0, five years of crowdsourced U.S. Amazon purchase histories
with user demographics. Scientific Data, 11, 491.
https://doi.org/10.1038/s41597-024-03329-6
```
