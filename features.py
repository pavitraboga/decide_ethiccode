"""
Shared feature code used by both train.py and model.py.

The shared backbone is a single TF-IDF vocabulary (word 1-2 grams + char 3-5
grams) fitted over every text field in the training set. All three heads read
from the same vectoriser; each head adds a small set of task-specific features.
"""

import re
from typing import Dict, List, Set, Tuple

import numpy as np

QUEUES: List[str] = ["billing", "technical", "account", "shipping", "general"]
TIER_CAPS = {"free": 31.10, "pro": 121.29, "enterprise": 499.45}
TIERS = ["free", "pro", "enterprise"]

WORD_NUMS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30,
    "forty": 40, "forty-five": 45, "sixty": 60, "ninety": 90, "half": 0.5,
    "quarter": 0.25,
}
# Duration units in hours.
UNITS = {
    "minute": 1 / 60, "minutes": 1 / 60, "hour": 1, "hours": 1,
    "day": 24, "days": 24, "business day": 24, "business days": 24,
    "week": 168, "weeks": 168, "fortnight": 336, "month": 720, "months": 720,
    "year": 8760, "years": 8760,
}
_NUM = r"(\d+(?:\.\d+)?|" + "|".join(sorted(WORD_NUMS, key=len, reverse=True)) + r")"
_UNIT = r"(business days?|minutes?|hours?|days?|weeks?|fortnight|months?|years?)"
DUR_RE = re.compile(r"\b" + _NUM + r"(?:\s+an?)?\s+" + _UNIT + r"\b")
BARE_DUR_RE = re.compile(r"\b(?:a|an|the)\s+(fortnight|week|month|year|day|hour)\b")
INT_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = set("""a an the and or of to in on for with is are be by at it this that
your you we our i as from will can may any all up so there has have was its
which these into after before about until within per""".split())


def _num(tok: str) -> float:
    try:
        return float(tok)
    except ValueError:
        return float(WORD_NUMS.get(tok, 0))


def durations(text: str) -> Set[float]:
    t = text.lower()
    out = set()
    for n, u in DUR_RE.findall(t):
        out.add(round(_num(n) * UNITS[u], 4))
    for u in BARE_DUR_RE.findall(t):
        out.add(round(UNITS[u], 4))
    return out


def numbers(text: str) -> Set[float]:
    t = text.lower()
    out = {float(x) for x in INT_RE.findall(t)}
    for w in TOKEN_RE.findall(t):
        if w in WORD_NUMS and w not in ("a", "an", "half", "quarter"):
            out.add(float(WORD_NUMS[w]))
    return out


def content_tokens(text: str) -> Set[str]:
    return {w for w in TOKEN_RE.findall(text.lower()) if w not in STOP and len(w) > 2}


def ticket_text(state: Dict) -> str:
    return "%s . %s" % (state.get("subject", "") or "", state.get("body", "") or "")


def grounded_dense(draft: str, kb: str) -> np.ndarray:
    dd, kd = durations(draft), durations(kb)
    dn, kn = numbers(draft), numbers(kb)
    dt, kt = content_tokens(draft), content_tokens(kb)
    extra_dur = len(dd - kd)
    extra_num = len(dn - kn)
    novel = dt - kt
    return np.array([
        float(extra_dur > 0), extra_dur, len(dd), float(extra_num > 0), extra_num,
        len(novel) / (len(dt) + 1.0), len(dt & kt) / (len(dt) + 1.0),
        len(draft) / (len(kb) + 1.0),
    ], dtype=np.float64)


def novel_text(draft: str, kb: str) -> str:
    """Draft words that do not appear in the KB article - swapped entities show up here."""
    kt = content_tokens(kb)
    return " ".join(w for w in TOKEN_RE.findall(draft.lower()) if w not in kt)


def approve_dense(state: Dict) -> np.ndarray:
    tier = state.get("customer_tier") or "free"
    amount = float(state.get("amount") or 0.0)
    cap = TIER_CAPS.get(tier, TIER_CAPS["free"])
    return np.array([
        np.log1p(amount), amount / cap, float(amount > cap), float(amount == 0.0),
    ] + [float(tier == t) for t in TIERS], dtype=np.float64)


def over_cap(state: Dict) -> bool:
    tier = state.get("customer_tier") or "free"
    return float(state.get("amount") or 0.0) > TIER_CAPS.get(tier, TIER_CAPS["free"])
