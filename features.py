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
TIER_CAPS = {"free": 31.25, "pro": 125.00, "enterprise": 500.00}
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
# Approval policy holds. Each pattern had zero approvals in train+dev.
SECURITY_RE = re.compile(
    r"(?i)(disable two[- ]factor|reset (my )?(2fa|two[- ]factor)|\b2fa\b|authenticator|"
    r"recovery codes?|permanently delet\w*|\bdelet\w* (my )?account|data permanently|erasure|"
    r"\bgdpr\b|erase all my data|close (my )?account permanently)")
PARTNER_RE = re.compile(r"(?i)partnership|reseller|affiliate|joint offering")
HOLD_RE = re.compile(r"(?i)only signed up recently|outside your usual policy|appreciate an exception|"
                     r"raised something similar|disputed? a charge")
POLICY_P = 0.003  # 0 approvals in ~1,000 matching tickets
CONDITIONS = {"only", "except", "excluding", "annual", "monthly", "domestic",
              "international", "business"}
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
    cond = (dt & CONDITIONS) - (kt & CONDITIONS)
    return np.array([
        float(len(cond) > 0),
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
    r = amount / cap
    # Approval odds step down at ~60% and ~80% of the tier cap.
    return np.array([
        np.log1p(amount), r, float(amount > cap), float(amount == 0.0),
        float(0 < r <= 0.6), float(0.6 < r <= 0.8), float(0.8 < r <= 1.0),
    ] + [float(tier == t) for t in TIERS], dtype=np.float64) * 2.0


def over_cap(state: Dict) -> bool:
    tier = state.get("customer_tier") or "free"
    return float(state.get("amount") or 0.0) > TIER_CAPS.get(tier, TIER_CAPS["free"])


def policy_hold(state: Dict) -> bool:
    """Requests that are never auto-approved: identity/deletion actions, partnerships, policy exceptions."""
    prim = primary_ticket(state)
    return bool(SECURITY_RE.search(prim) or PARTNER_RE.search(prim)
                or HOLD_RE.search(ticket_text(state)))


# ---- canonical quantity normalisation for grounded -------------------------
_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
         "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_NW = "|".join(list(_ONES) + list(_TENS) + ["hundred", "thousand"])
_NUMSEQ_RE = re.compile(r"\b(?:%s)(?:(?:\s+and\s+|[\s-]+)(?:%s))*\b" % (_NW, _NW))
_UNIT_M = {"minute": 1, "hour": 60, "day": 1440, "week": 10080, "fortnight": 20160,
           "month": 43200, "year": 525600}
_CDUR_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s+(?:business\s+|calendar\s+|working\s+)?"
                      r"(minute|hour|day|week|fortnight|month|year)s?\b")
_ADUR_RE = re.compile(r"\b(?:a|an|one|per)\s+(minute|hour|day|week|fortnight|month|year)\b")


def _words_to_int(seq: str) -> int:
    total, cur = 0, 0
    for w in re.split(r"[\s-]+", seq):
        if w == "and":
            continue
        if w in _ONES:
            cur += _ONES[w]
        elif w in _TENS:
            cur += _TENS[w]
        elif w == "hundred":
            cur = max(cur, 1) * 100
        elif w == "thousand":
            total += max(cur, 1) * 1000
            cur = 0
    return total + cur


def canon(text: str) -> str:
    """Rewrite quantities into canonical tokens: durations -> dur<minutes>, counts -> n<value>."""
    t = text.lower()
    t = re.sub(r"\ba quarter of an hour\b|\bquarter of an hour\b", "15 minutes", t)
    t = re.sub(r"\bhalf an hour\b|\bhalf hour\b", "30 minutes", t)
    t = re.sub(r"\bhalf a day\b", "12 hours", t)
    t = _NUMSEQ_RE.sub(lambda m: str(_words_to_int(m.group(0))), t)
    t = re.sub(r"\b(\d+)\s*(?:-|to)\s*(\d+)\s+(business |working )?days\b",
               lambda m: " range%s_%s%s " % (m.group(1), m.group(2), "b" if m.group(3) else ""), t)
    t = _CDUR_RE.sub(lambda m: " dur%d " % round(float(m.group(1)) * _UNIT_M[m.group(2)]), t)
    t = _ADUR_RE.sub(lambda m: " dur%d " % _UNIT_M[m.group(1)], t)
    t = re.sub(r"\b(\d+(?:\.\d+)?)\b", lambda m: "n%g" % float(m.group(1)), t)
    return t


CTOK_RE = re.compile(r"[a-z][a-z0-9.]*")


def canon_tokens(text: str) -> List[str]:
    return [w.rstrip(".") for w in CTOK_RE.findall(canon(text))]


def grounded_parts(draft: str, kb: str) -> Tuple[str, str, str, np.ndarray]:
    """(draft-only text, kb-only text, cross pairs, dense) over canonical tokens."""
    dt, kt = canon_tokens(draft), canon_tokens(kb)
    ds, ks = set(dt), set(kt)
    nov = [w for w in dt if w not in ks and w not in STOP]
    mis = [w for w in kt if w not in ds and w not in STOP]
    dq = {w for w in ds if re.match(r"(dur|n|range)\d", w)}
    kq = {w for w in ks if re.match(r"(dur|n|range)\d", w)}
    pairs = " ".join("%s__%s" % (a, b) for a in set(nov) for b in set(mis))
    dense = np.array([
        float(bool(dq - kq)), len(dq - kq), len({w for w in dq - kq if w.startswith("dur")}),
        len({w for w in dq - kq if w[0] == "n"}), len(dq), len(nov) / (len(dt) + 1.0),
        len(mis) / (len(kt) + 1.0),
    ], dtype=np.float64)
    return " ".join(nov), " ".join(mis), pairs, dense


# ---- primary-intent extraction ---------------------------------------------
# Multi-request tickets flag the request that matters ("what I actually need
# help with is this:") and the asides ("for background,"). Labels follow the
# flagged request, so heads read it separately.
_PRIMARY = r"priority for me is|main thing i need|real question is|actually need help|most importantly|the part i need"
_SECONDARY = r"separately|less important|on a related note|for background|also worth mentioning|some context first|small thing"
_MARK_RE = re.compile(r"(?i)\b(%s|%s)\b[^:,]{0,30}[:,]" % (_PRIMARY, _SECONDARY))
_PRIMARY_RE = re.compile(r"(?i)" + _PRIMARY)


def primary_text(state: Dict) -> str:
    body = state.get("body") or ""
    marks = list(_MARK_RE.finditer(body))
    if not marks:
        return body
    segs = [(None, body[:marks[0].start()])]
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        segs.append((m.group(1), body[m.end():end]))
    prim = [t for k, t in segs if k and _PRIMARY_RE.match(k)]
    if prim:
        return " ".join(prim)
    return " ".join(t for k, t in segs if k is None)


def primary_ticket(state: Dict) -> str:
    return "%s . %s" % (state.get("subject", "") or "", primary_text(state))


def has_aside(state: Dict) -> bool:
    return bool(_MARK_RE.search(state.get("body") or ""))
