# Model card — Decide v0.2-gate2

**Authors:** TAGISWILD, pavitraboga

## What we built
A single CPU model serving three decisions behind `POST /v1/decide`:

- **Shared backbone:** a single TF-IDF `FeatureUnion` (word 1–2 grams + char_wb 3–5 grams, 80k features) fitted over all text fields: ticket subject+body, draft replies and KB articles.
- **route head:** a 5-way multinomial logistic regression on the backbone embedding of subject+body. It needs no extra calibration because it was already better calibrated than a Platt-wrapped version.
- **approve head:** a hard policy gate plus a classifier. Any amount above the tier cap (free $31.25, pro $125, enterprise $500) is held with p=0. None of the over-cap tickets in train were approved. $0 requests mentioning 2FA, authenticator, recovery codes or account deletion are held for an identity check with p=0.02, their approval rate in train. Under-cap tickets go to a logistic regression on the backbone embedding plus amount/tier features. We approve when p ≥ 0.65. In theory the optimum for the 3:1 loss is 0.75, but sweeping out-of-fold cost on train put the minimum at 0.65.
- **grounded head:** a logistic regression on the backbone embedding of the *draft words absent from the KB article*, which catches swapped entities. It also uses contradiction features: durations normalised to hours ("a week" vs "30 days", "half an hour" vs "60 minutes"), unmatched numbers, mismatched condition words (only/except/annual/monthly/domestic, and so on), and token overlap.
- **Calibration:** the approve and grounded heads use sigmoid (Platt) calibration with 5-fold CV. The route head uses the multinomial softmax directly.
- **Abstention:** each head has a confidence cut set from out-of-fold predictions so that about 84.5% of requests are answered, leaving margin above the 80% target.

Each request runs exactly one head. No pretrained weights are used. Artifacts total 3.5 MB.

## Training data
`data/train.jsonl` (6,500 records). `data/dev.jsonl` (1,500) is used only for evaluation.

## Measured on dev (selfcheck.py, 4,500 calls)
| | route | approve | grounded |
|---|---|---|---|
| quality | macro-F1 (answered) 0.920 | cost 0.217 vs always-hold 0.364 | macro-F1 0.917 |
| full-coverage error | 0.156 | 0.113 | 0.113 |
| ECE | 0.044 | 0.035 | 0.024 |
| Brier skill | 0.709 | 0.634 | 0.633 |
| coverage | 82.5% | 85.1% | 87.2% |

Total self-check score: **76.93 / 100**, with all gates passing and a mean p95 of 1.5 ms. Route macro-F1 is 0.840 at full coverage. The full report is in `results.json`.

## Where it fails
- **route:** confuses billing and account, and technical and account, when a ticket mixes topics. Labels are noisy. 
- **approve:** it cannot see intent beyond surface words. The security rule only fires on exact phrases, so paraphrased requests fall through to the classifier.
- **grounded:** duration parsing misses unusual phrasings. Contradictions that are not numeric or lexical, such as a changed eligibility condition worded with KB vocabulary, slip through.

## When it abstains
It abstains when the top-class probability (route) or max(p, 1−p) (binary heads) falls below the per-head cut: route 0.67, approve 0.71, grounded 0.71. Over-cap and security holds never abstain because they are deterministic policy holds.
