"""
Train the Decide model and write artifacts/decide.joblib.

    python train.py            # train on data/train.jsonl, report on data/dev.jsonl

One shared TF-IDF backbone (word + char n-grams over every text field), three
logistic-regression heads with sigmoid (Platt) calibration fitted by
cross-validation, and per-head abstention thresholds chosen from out-of-fold
confidences so that about 84.5% of requests are answered.
"""

import json
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)  # spurious BLAS matmul warnings on macOS
import os

import joblib
import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import FeatureUnion

import features as F

COVERAGE = 0.845
APPROVE_TAU = 0.75  # 3:1 cost asymmetry -> approve only when p >= 3/4


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def build_backbone(rows):
    corpus = []
    for r in rows:
        s = r["state"]
        corpus += [F.ticket_text(s), s.get("draft_reply") or "", s.get("kb_article") or ""]
    vec = FeatureUnion([
        ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000,
                              sublinear_tf=True)),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                              max_features=40000, sublinear_tf=True)),
    ])
    vec.fit(corpus)
    return vec


def route_X(vec, states):
    return vec.transform([F.ticket_text(s) for s in states])


def approve_X(vec, states):
    dense = np.vstack([F.approve_dense(s) for s in states])
    return sparse.hstack([vec.transform([F.ticket_text(s) for s in states]),
                          sparse.csr_matrix(dense)]).tocsr()


def grounded_X(vec, states):
    drafts = [s.get("draft_reply") or "" for s in states]
    kbs = [s.get("kb_article") or "" for s in states]
    novel = vec.transform([F.novel_text(d, k) for d, k in zip(drafts, kbs)])
    dense = np.vstack([F.grounded_dense(d, k) for d, k in zip(drafts, kbs)])
    return sparse.hstack([novel, sparse.csr_matrix(dense * 3.0)]).tocsr()


def calibrated(C):
    return CalibratedClassifierCV(LogisticRegression(C=C, max_iter=3000), method="sigmoid", cv=5)


def abstain_cut(conf):
    """Confidence below which we abstain, so ~COVERAGE of cases are answered."""
    return float(np.quantile(conf, 1.0 - COVERAGE))


def ece(p, y, bins=15):
    p, y = np.asarray(p), np.asarray(y)
    e = 0.0
    for b in range(bins):
        m = (p >= b / bins) & (p < (b + 1) / bins) if b < bins - 1 else (p >= b / bins)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def main():
    tr, dv = load("data/train.jsonl"), load("data/dev.jsonl")
    S = [r["state"] for r in tr]
    D = [r["state"] for r in dv]

    # Sanity-check the tier-cap rule before relying on it.
    over = [r["labels"]["approve"] for r in tr if F.over_cap(r["state"])]
    print("over-cap tickets: %d, approved: %d" % (len(over), sum(over)))

    vec = build_backbone(tr)

    # route
    yr = np.array([F.QUEUES.index(r["labels"]["route"]) for r in tr])
    Xr = route_X(vec, S)
    # Plain multinomial LR is already well calibrated here; Platt one-vs-rest
    # made it worse (dev ECE 0.082 -> 0.044 without it).
    route = LogisticRegression(C=10.0, max_iter=3000)
    oof = cross_val_predict(route, Xr, yr, cv=5, method="predict_proba")
    route.fit(Xr, yr)
    route_cut = abstain_cut(oof.max(1))

    # approve: train only on under-cap tickets; over-cap is a hard hold.
    under = np.array([not F.over_cap(s) for s in S])
    secure = np.array([F.security_risk(s) for s in S])
    ya = np.array([int(r["labels"]["approve"]) for r in tr])
    Xa = approve_X(vec, S)
    approve = calibrated(2.0)
    oof_a = cross_val_predict(approve, Xa[under], ya[under], cv=5, method="predict_proba")[:, 1]
    approve.fit(Xa[under], ya[under])
    pa_all = np.zeros(len(S))
    pa_all[under] = oof_a
    pa_all[secure & under] = F.SECURITY_P
    # Pick tau on out-of-fold cost; 0.75 is the theoretical optimum for 3:1.
    taus = np.round(np.arange(0.55, 0.801, 0.01), 2)
    costs = [np.mean(np.where((pa_all >= t) & (ya == 0), 3.0,
                              np.where((pa_all < t) & (ya == 1), 1.0, 0.0))) for t in taus]
    tau = float(taus[int(np.argmin(costs))])
    print("tau sweep (oof cost):", dict(zip(taus.tolist(), np.round(costs, 4).tolist())), "->", tau)
    approve_cut = abstain_cut(np.maximum(pa_all, 1 - pa_all))

    # grounded
    yg = np.array([int(r["labels"]["grounded"]) for r in tr])
    Xg = grounded_X(vec, S)
    grounded = calibrated(2.0)
    oof_g = cross_val_predict(grounded, Xg, yg, cv=5, method="predict_proba")[:, 1]
    grounded.fit(Xg, yg)
    grounded_cut = abstain_cut(np.maximum(oof_g, 1 - oof_g))

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump({
        "vec": vec, "route": route, "approve": approve, "grounded": grounded,
        "cuts": {"route": route_cut, "approve": approve_cut, "grounded": grounded_cut},
        "approve_tau": tau,
    }, "artifacts/decide.joblib", compress=3)

    # ---- dev report ----
    pr = route.predict_proba(route_X(vec, D))
    yrd = [F.QUEUES.index(r["labels"]["route"]) for r in dv]
    print("route    acc %.4f  macroF1 %.4f" % (accuracy_score(yrd, pr.argmax(1)),
                                               f1_score(yrd, pr.argmax(1), average="macro")))
    pa = approve.predict_proba(approve_X(vec, D))[:, 1]
    pa[[F.security_risk(s) for s in D]] = F.SECURITY_P
    pa[[F.over_cap(s) for s in D]] = 0.0
    yad = np.array([int(r["labels"]["approve"]) for r in dv])
    pred = pa >= tau
    cost = np.mean(np.where(pred & (yad == 0), 3.0, np.where(~pred & (yad == 1), 1.0, 0.0)))
    print("approve  cost %.4f (always-hold %.4f)  ece %.4f" % (cost, yad.mean(), ece(pa, yad)))
    pg = grounded.predict_proba(grounded_X(vec, D))[:, 1]
    ygd = np.array([int(r["labels"]["grounded"]) for r in dv])
    print("grounded acc %.4f  macroF1 %.4f  ece %.4f" % (
        accuracy_score(ygd, pg > 0.5), f1_score(ygd, pg > 0.5, average="macro"), ece(pg, ygd)))
    print("abstain cuts", route_cut, approve_cut, grounded_cut)


if __name__ == "__main__":
    main()
