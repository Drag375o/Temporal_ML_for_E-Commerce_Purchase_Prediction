# Dataset

**Open e-commerce 1.0 — Five years of crowdsourced U.S. Amazon purchase
histories with user demographics**

- Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YGLYDY
- DOI: 10.7910/DVN/YGLYDY
- Authors: Berke, Calacci, Mahari, Yabe, Larson & Pentland (2023)
- Paper: https://doi.org/10.1038/s41597-024-03329-6

## How to obtain

1. Open the Dataverse link and choose Access Dataset → Download ZIP.
2. Unzip and place these three files in `data/raw/`:
   - `amazon-purchases.csv`  (~313 MB)
   - `survey.csv`            (~1.3 MB)
   - `fields.csv`            (survey codebook)
3. Set `RAW_DIR` in the notebook's first cell to point at `data/raw/`.

The data is not committed to this repository.

## Notes on the published metadata

Verified against the actual files:

| Claim | Measured | Match |
|---|---|---|
| ~5,027 consumers | 5,027 | yes |
| >1.8M purchases | 1,850,717 | yes |
| Spans 2018–2022 | 2018-01-01 to **2024-08-15** | **no** |

Records extend well past 2022. More importantly, each participant's history
terminates on their **enrolment date** rather than a common endpoint, which
right-censors the data from roughly September 2022 onward. The notebook
quantifies this (Section 6) and restricts all modelling to 2018-01-01 through
2022-11-30. This censoring is not described in the dataset documentation.

## Excluded variables

Health and sensitive lifestyle items are excluded by design: substance use
(cigarettes, marijuana, alcohol), diabetes, wheelchair use, sexual orientation.

Two further columns are excluded as **temporal leakage**, not privacy risk:
`Q-amazon-use-how-oft` (self-reported purchase frequency, collected after every
prediction window — it encodes the target) and `Q-life-changes` (recent events,
also post-cutoff).
