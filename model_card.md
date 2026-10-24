# Model card — Decide

**Authors:** TAGISWILD, pavitraboga

## What we built
A single CPU model serving three decisions behind `POST /v1/decide`. Each request runs exactly one head.

- **Shared backbone:** one TF-IDF `FeatureUnion` fitted over all ticket, draft and KB text. It combines word 1–2 grams, char 3–5 grams, and canonical-quantity tokens in which every duration and count is rewritten to a normal form, so "a week", "7 days" and "seven days" become the same token.
- **Primary-intent extraction:** multi-request tickets flag the request that matters ("what I actually need help with is this:") and the asides ("for background,"). Labels follow the flagged request, so the route and approve heads read only that part.
- **route head:** multinomial logistic regression on the primary-intent text, with temperature and uniform smoothing fitted on out-of-fold log-loss. The labels carry noise, so confident errors are expected.
- **approve head:** deterministic policy holds come first, followed by a calibrated logistic regression.
  - *Tier cap:* any amount above the cap (free $31.25, pro $125, enterprise $500) is held with p=0.
  - *Policy holds:* requests to disable or reset 2FA, recover access through an authenticator or recovery codes, delete an account or data (including GDPR), or make partnership/reseller enquiries are held. So are tickets asking for a policy exception, disclosing a prior bank dispute, or being repeat or very new requests. None of these were approved in the data. They get p=0.003.
  - *Classifier:* primary-intent text plus amount and tier features, including bands at 60% and 80% of the tier cap, where approval odds step down. The approve threshold is tuned on out-of-fold cost under the 3:1 loss.
- **grounded head:** calibrated logistic regression on three views of the draft–KB difference after quantity normalisation: words only in the draft, words only in the KB article, and hashed cross-pairs of the two. The cross-pairs let the model learn which substitutions are harmless (synonyms) and which are contradictions (audience, method or quantity swaps). Quantity-mismatch counts are included as well.
- **Abstention:** per-head confidence cuts are set from out-of-fold predictions so that about 84.5% of requests are answered (90% for route). Route gets the wider margin above the 80% target because five-way confidence varies most between data sets.

No pretrained weights are used. Artifacts total about 3 MB.

## Training data
`data/train.jsonl` (6,500 records). `data/dev.jsonl` (1,500) is used only for evaluation.

## Measured on dev (1,500 records, selfcheck.py)
| | route | approve | grounded |
|---|---|---|---|
| accuracy (full coverage) | 0.892 | 0.902 | 0.990 |
| quality (answered) | macro-F1 0.947 | cost 0.202 | macro-F1 1.000 |
| ECE | 0.029 | 0.019 | 0.011 |
| Brier skill | 0.791 | 0.685 | 0.968 |
| coverage | 89.4% | 83.3% | 85.3% |

Total self-check score: **84.37 / 100**, with all gates passing and a mean p95 of about 2 ms. The full report is in `results.json`.

## Where it fails
- **route:** about 8% of labels disagree on identical tickets, which caps accuracy. Vague tickets ("I have an issue, please get back to me") have no real signal.
- **approve:** within the tier cap, outcomes are random within amount bands (about 85%, 60% and 30% approved below 60%, at 60–80%, and above 80% of the cap). The model reports those odds honestly, but it cannot resolve them.
- **grounded:** quantity phrasings outside the normaliser's coverage, and new synonym pairs not seen in training.

## When it abstains
It abstains when the top-class probability (route) or max(p, 1−p) (binary heads) falls below the per-head cut chosen for about 84.5% coverage (90% for route). Policy holds never abstain.
