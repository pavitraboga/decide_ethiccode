# Model card — Decide v0.1-baseline

**Authors:** TAGISWILD, pavitraboga

## What we built
A single CPU model serving three decisions behind `POST /v1/decide`:

- **Shared backbone:** a single TF-IDF `FeatureUnion` (word 1–2 grams + char_wb 3–5 grams, 80k features) fitted over all text fields: ticket subject+body, draft replies and KB articles.
- **route head:** a 5-way logistic regression on the backbone embedding of subject+body.
- **approve head:** a hard policy gate plus a classifier. Any amount above the tier cap (free $31.10, pro $121.29, enterprise $499.45) is held with p=0. In train, none of the 2,563 over-cap tickets were approved. Under-cap tickets go to a logistic regression on the backbone embedding plus amount/tier features. We approve only when p ≥ 0.75, the cost-optimal threshold for the 3:1 loss.
- **grounded head:** a logistic regression on the backbone embedding of the *draft words absent from the KB article*, which catches swapped entities. It also uses contradiction features: durations normalised to hours ("a week" vs "30 days", "half an hour" vs "60 minutes"), unmatched numbers, and token overlap.
- **Calibration:** each head uses sigmoid (Platt) calibration with 5-fold CV (`CalibratedClassifierCV`).
- **Abstention:** each head has a confidence cut set from out-of-fold predictions so that about 83% of requests are answered.

Each request runs exactly one head. No pretrained weights are used. Artifacts total 3.5 MB.

## Training data
`data/train.jsonl` (6,500 records). `data/dev.jsonl` (1,500) is used only for evaluation.

## Measured on dev (selfcheck.py, 4,500 calls)
| | route | approve | grounded |
|---|---|---|---|
| quality | macro-F1 (answered) 0.92* | cost 0.220 vs always-hold 0.369 | macro-F1 0.920 |
| full-coverage error | 0.156 | 0.113 | 0.114 |
| ECE | 0.082 | 0.037 | 0.027 |
| Brier skill | 0.696 | 0.629 | 0.632 |
| coverage | 81.1% | 82.9% | 85.5% |
| p95 latency | 2.1 ms | 1.7 ms | 1.7 ms |

Total self-check score: **75.93 / 100**, with all gates passing. *Route macro-F1 is 0.840 at full coverage. The full report is in `results.json`.

## Where it fails
- **route:** confuses billing and account, and technical and account, when a ticket mixes topics. Labels are noisy. Route ECE is the weakest calibration of the three heads.
- **approve:** it cannot see intent beyond surface words. Security-sensitive requests (2FA reset, deletion) are learned only from TF-IDF, not from an explicit rule.
- **grounded:** duration parsing misses unusual phrasings. Contradictions that are not numeric or lexical, such as a changed eligibility condition worded with KB vocabulary, slip through.

## When it abstains
It abstains when the top-class probability (route) or max(p, 1−p) (binary heads) falls below the per-head cut: route 0.61, approve 0.73, grounded 0.72. Over-cap approvals never abstain because they are a deterministic policy hold.
