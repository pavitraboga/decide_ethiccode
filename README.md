# decide_ethiccode

## Run

```bash
pip install -r requirements.txt && uvicorn app:app --host 0.0.0.0 --port 8000
```

Model weights are included in `artifacts/`.

## Shared model

A single text encoder is loaded once at startup in `model.py` and shared by all three decision types (`route`, `approve`, `grounded`).

## Pretrained models

None.
