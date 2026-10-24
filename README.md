# decide_ethiccode

A CPU-only decision service for customer-support tickets. A single endpoint, `POST /v1/decide`, answers one of three questions per request:

- `route`: which of five queues (billing, technical, account, shipping, general) should handle the ticket
- `approve`: whether the request can be auto-approved or needs a human
- `grounded`: whether a drafted reply is supported by the referenced knowledge-base article

Every response carries a calibrated probability and an abstain flag.

## Quickstart

Requires Python 3.9 or later. A single command installs the dependencies and starts the service:

```bash
pip install -r requirements.txt && uvicorn app:app --host 0.0.0.0 --port 8000
```

The trained weights are committed in `artifacts/decide.joblib` (about 3 MB), so no training or download step is needed at startup.

Health check:

```bash
curl -s localhost:8000/health
```

Example request:

```bash
curl -s -X POST localhost:8000/v1/decide -H 'Content-Type: application/json' -d '{"request_id":"r1","decision":"route","state":{"subject":"Charged twice","body":"I was billed twice this month.","customer_tier":"pro","amount":20.0}}'
```

## API

Request body:

| Field | Type | Description |
|---|---|---|
| `request_id` | string | Echoed back in the response |
| `decision` | `route` \| `approve` \| `grounded` | The single decision requested |
| `state` | object | `subject`, `body`, `customer_tier`, `amount`, `draft_reply`, `kb_article` |

Responses follow the contract in `app.py`. `route` returns a `choice` with a probability for every queue. `approve` and `grounded` return a `binary` with `p_true`. Both forms include `abstain`, `request_id`, `decision` and `latency_ms`.

## Where the sharing happens

All three decisions share one text representation, the backbone. It is built by `build_backbone` in `train.py`, saved with the heads in `artifacts/decide.joblib`, and loaded once in `DecisionModel.__init__` in `model.py`.

The backbone is a scikit-learn `FeatureUnion` of three TF-IDF vocabularies (32,337 features in total), fitted jointly on ticket text, draft replies and knowledge-base articles:

- word unigrams and bigrams
- character 3–5 grams, which make it robust to misspellings
- canonical-quantity tokens, in which durations and counts are normalised so that equivalent expressions ("a week", "7 days") map to the same feature

Each request runs exactly one lightweight head on top of the shared backbone. The other two heads are never evaluated. Feature construction lives in `features.py`.

## Decisions

**route.** A multinomial logistic regression over the backbone representation of the ticket. When a ticket contains several requests, the head reads the request the customer marks as the main one. The probabilities are temperature-scaled and smoothed. Both parameters were fitted on out-of-fold predictions.

**approve.** Requests that the training data shows are never approved, such as amounts above the customer's tier limit, are held deterministically. All other requests go to a logistic regression with Platt (sigmoid) calibration over the backbone plus amount and tier features. The decision threshold reflects the asymmetric cost of errors: a wrong approval costs three times as much as a wrong hold, which puts the break-even probability at 0.75. The deployed threshold is τ = 0.79, chosen by minimising out-of-fold cost on the training set.

**grounded.** A logistic regression with Platt calibration over the difference between the draft and the article, after quantity normalisation. Its inputs are the terms found only in the draft, the terms found only in the article, the cross-pairs of the two, and quantity-mismatch counts.

## Calibration and abstention

- The `approve` and `grounded` heads use 5-fold Platt scaling. The `route` head uses temperature scaling with uniform smoothing.
- Each head has its own abstention threshold, set from out-of-fold confidence so that declared coverage stays above the 80% target (about 90% for `route` and 84.5% for the other two).
- The service abstains when the top-class probability (`route`) or max(p, 1 − p) (binary heads) falls below that threshold.
- Deterministic holds never abstain.

## Results

Measured with `selfcheck.py` on `data/dev.jsonl` (1,500 records, 4,500 calls, one request at a time). The full report is in `results.json`.

| Decision | Quality | Accuracy (full coverage) | ECE | Declared coverage |
|---|---|---|---|---|
| route | macro-F1 0.947 | 0.892 | 0.029 | 89.4% |
| approve | cost 0.202 (always-hold baseline 0.365) | 0.902 | 0.019 | 83.3% |
| grounded | macro-F1 1.000 | 0.990 | 0.011 | 85.3% |

| Axis | Score |
|---|---|
| Quality | 0.843 |
| Calibration | 0.858 |
| Latency (mean p95 1.84 ms) | 1.000 |
| Abstention | 0.614 |
| **Total** | **84.37 / 100** |

All gates pass: schema valid on 4,500 of 4,500 responses, latency under the gate, and the service reachable. Latency was measured on the development machine and will vary with hardware.

## Reproducing the model

Place `train.jsonl` and `dev.jsonl` in `data/`, then:

```bash
python train.py
```

This rebuilds `artifacts/decide.joblib` and prints development-set metrics. To score a running service:

```bash
python selfcheck.py --url http://localhost:8000/v1/decide
```

## Repository layout

| Path | Contents |
|---|---|
| `app.py` | HTTP layer and response contract |
| `model.py` | `DecisionModel`: loads the artifact and dispatches to one head per request |
| `features.py` | Feature extraction shared by training and serving |
| `train.py` | Backbone construction, head training, calibration and threshold selection |
| `artifacts/decide.joblib` | Trained backbone and heads |
| `selfcheck.py` | Evaluation harness |
| `results.json` | Development-set evaluation report |
| `model_card.md` | Model card |

## Pretrained components and licences

No pretrained models or external weights are used. Every component was trained from scratch on `data/train.jsonl`, and no network calls are made at inference time.

| Dependency | Version used | Licence |
|---|---|---|
| scikit-learn | 1.6.1 | BSD-3-Clause |
| NumPy | 2.0.2 | BSD-3-Clause |
| SciPy | 1.13.1 | BSD-3-Clause |
| joblib | 1.5.3 | BSD-3-Clause |
| FastAPI | 0.128.8 | MIT |
| Uvicorn | 0.39.0 | BSD-3-Clause |
| Pydantic | 2.13.5 | MIT |

Tested with Python 3.9.6.

## Authors

TAGISWILD, pavitraboga
