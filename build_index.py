"""
Build the signed salience index from the cached Guardian quarters.

    python build_index.py

Needs data/raw/*.json.gz from guardian_collect.py. No API calls, so the filter
can be re-tuned freely.

    API match (body-level)              broad recall net, guardian_collect.py
      -> anchor in headline+standfirst  article is about UK consumer prices
      -> not excluded                   drops foreign-economy and asset prices
      = numerator N_t
    salience_t = N_t / n_total_t
    sign_t     = (n_rising - n_falling) / (n_rising + n_falling)   in [-1, 1]
    s_t        = 100 * (salience_t / mean salience) * sign_t

Selection uses headline and standfirst. Both are editorial summary devices:
they state what a piece is about, which is what a household skimming a page
absorbs (Carroll 2003). Adding the lead admits incidental mentions, since a
novel review and a photo essay both carry "inflation" in their opening lines.
Signing uses the wider unit, because direction takes more words to establish
than topic; on the narrow unit alone, 48% of a quiet quarter's articles carry
no directional word and the quarterly sign rests on about eleven articles.

The unit was fixed before the contrasts were computed. For the record the
crisis-to-quiet ratio is 2.95x on headline alone, 2.63x on headline and
standfirst, 4.12x adding the trail and 3.33x adding the lead; the chosen unit
is not the maximum.

Exclusions target two contaminants that are correlated with the regime rather
than off-topic content generally. Foreign-economy inflation peaks in crisis
quarters and asset-price inflation in quiet ones, so each biases a different
end of the range. An article is excluded only if it carries no UK or household
marker.

Outputs:
    data/salience_index.csv        the index, one row per quarter
    data/salience_index_exboe.csv  same, dropping articles whose headline or
                                   standfirst names the Bank of England or MPC

The ex-BoE variant answers whether the salience shock is central-bank
communication reaching households through the press. It is derived from the
same cached text, so it needs no second collection.

Reading the output: |s_t| is salience relative to the 2001-2025 average, where
100 is an average quarter; the sign is the framing, positive for rising prices.
"""

import csv
import gzip
import json
import re
import sys
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data")


ANCHOR = re.compile(r"""
    cost[-\s]of[-\s]living
  | \binflation\b | \binflationary\b | \bdeflation\b | \bdisinflation\b
  | (energy|gas|electricity|fuel|heating|household|utility|water)\s+bills?
  | (food|grocery|groceries|supermarket|petrol|diesel|consumer|shop)\s+price
  | price\s+cap
  | (household|family)\s+(budget|finance|spending)
  | weekly\s+shop | shopping\s+bill | food\s+bank
  | \brpi\b | \bcpi\b | \bcpih\b
  | retail\s+price\s+index | consumer\s+price\s+index
""", re.I | re.X)


FOREIGN = re.compile(r"""
    zimbabwe | venezuela | argentin | weimar | hyperinflation
  | zambia | mugabe | harare | darfur | yemen | somalia | ethiopia
  | \bafrica\b | african
  | eurozone | euro\s+zone | \bgreece\b | greek | \bgermany\b | german
  | \bturkey\b | turkish | \bjapan\b | japanese | \bchina\b | chinese
  | \bindia\b | \brussia\b | russian | \bbrazil\b | \bmexico\b
  | united\s+states | \bus\s+(inflation|economy|prices|fed)
  | federal\s+reserve | \bopec\b | middle\s+east
""", re.I | re.X)


UK_MARKER = re.compile(r"""
    \buk\b | \bbritain\b | \bbritish\b | \bengland\b | \bscotland\b | \bwales\b
  | \bofgem\b | \bons\b | bank\s+of\s+england | \bmpc\b | treasury | chancellor
  | \bhousehold | \bfamil(y|ies)\b | \bshoppers?\b | \bconsumers?\b
  | westminster | whitehall | downing\s+street
""", re.I | re.X)


ASSET = re.compile(r"""
    (house|housing|property|home|land|rent|asset|share|stock|equity|wage|pay|grade)
    [-\s]price[-\s]inflation
  | (house|housing|property)\s+prices?
  | \bftse\b | share\s+price | stock\s+market | buy[-\s]to[-\s]let
  | grade\s+inflation | wage\s+inflation | pay\s+inflation
""", re.I | re.X)


RISING = re.compile(r"""
    \bris(e|es|ing|en)\b | \brose\b | \bsoar(s|ing|ed)?\b | \bsurg(e|es|ing|ed)\b
  | \bjump(s|ing|ed)?\b | \bclimb(s|ing|ed)?\b | \bspiral(s|ling|led)?\b
  | \bsqueez(e|es|ing|ed)\b | \bhik(e|es|ing|ed)\b | \bspike(s|d)?\b
  | \brais(e|es|ing|ed)\b | \bpush(es|ing|ed)?\s+up\b | \bpump(s|ing|ed)?\s+up\b
  | \badd(s|ing|ed)?\b | \bhigher\b | \bhigh\b | \brecord\b | \bpeak(s|ing|ed)?\b
  | \bincreas(e|es|ing|ed)\b | \baccelerat(e|es|ing|ed)\b | \bupward\b
  | \bmount(s|ing)?\b | \bsteep(er)?\b | \bsharp(ly|er)?\b | \bdoubl(e|es|ing|ed)\b
  | \bcrisis\b | \bpressure\b | \bpain\b | \bburden\b | \bstruggl(e|es|ing)\b
  | \bworse\b | \bhit\b | \bwarn(s|ing|ed)?\b | \bexpensive\b | \bdearer\b
  | \bcostly\b | \bstubborn\b | \bsting\b | \bbites?\b
""", re.I | re.X)


FALLING = re.compile(r"""
    \bfall(s|ing|en)?\b | \bfell\b | \beas(e|es|ing|ed)\b | \bdrop(s|ping|ped)?\b
  | \bdeclin(e|es|ing|ed)\b | \bdeflation\b | \bdisinflation\b
  | \bcool(s|ing|ed)?\b | \bslow(s|ing|ed|er)?\b | \bcheaper\b | \bcheap\b
  | \blower\b | \blow\b | \bdownward\b | \bcut(s|ting)?\b | \breliefs?\b
  | \brespite\b | \bsubdued\b | \bweaker\b | \bretreat(s|ing|ed)?\b
  | \bbetter\s+off\b | \bdiscount(s|ing|ed)?\b | \bslump(s|ing|ed)?\b
  | \btumbl(e|es|ing|ed)\b | \bplung(e|es|ing|ed)\b
""", re.I | re.X)


NEUTRAL = re.compile(r"""
    \bunchanged\b | \bstable\b | \bsteady\b | \bflat\b | \bholds?\s+at\b
  | \bremains?\b | \bstays?\b
""", re.I | re.X)


def select_unit(a):
    """Unit for SELECTION: is this article ABOUT UK household consumer prices?

    Headline + standfirst only. Both are editorial summary devices - they state what
    the piece is about. Adding the lead (400 chars) admits incidental mentions:
    "The Mandibles - digested read" and "London tube maps - in pictures" both carry
    'inflation' in their lead and are not cost-of-living coverage.
    """
    return f"{a.get('headline','')} {a.get('standfirst','')}"


def sign_unit(a):
    """Unit for SIGNING: which direction does an on-topic article frame prices?

    Headline + standfirst + lead. Deliberately WIDER than the selection unit, because
    selection and signing answer different questions:
      - selection needs PROMINENCE (is it about this?) -> narrow unit
      - signing needs ENOUGH WORDS (which way?)        -> wider unit
    Topic is already established by the time we sign, so the wider unit cannot pull in
    off-topic framing - which was the original body-select/headline-sign bug. On the
    selection unit alone, 48% of a quiet quarter's articles carry no directional word
    at all, leaving the quarterly sign resting on ~11 articles and flipping on noise.

    The unit is the Carroll (2003) one: what a household skimming a page absorbs is the
    headline and the opening. Full body is reported as a robustness check.
    """
    return f"{a.get('headline','')} {a.get('standfirst','')} {a.get('lead','')}"


unit_text = select_unit


def classify(a):
    """Return (kept, reason). reason is None if kept."""
    t = select_unit(a)
    if not ANCHOR.search(t):
        return False, "no_anchor"
    if ASSET.search(t):
        return False, "asset_price"
    if FOREIGN.search(t) and not UK_MARKER.search(t):
        return False, "foreign"
    return True, None


def article_sign(a, unit=None):
    """+1 rising, -1 falling, 0 ambiguous. Signs on the WIDER sign_unit."""
    t = unit if unit is not None else sign_unit(a)
    r = len(RISING.findall(t))
    f = len(FALLING.findall(t))
    if r > f:
        return 1
    if f > r:
        return -1
    return 0


def process_quarter(path):
    d = json.load(gzip.open(path, "rt", encoding="utf-8"))
    kept, reasons = [], {}
    for a in d["articles"]:
        ok, why = classify(a)
        if ok:
            kept.append(a)
        else:
            reasons[why] = reasons.get(why, 0) + 1

    signs = [article_sign(a) for a in kept]
    n_rise = sum(1 for s in signs if s > 0)
    n_fall = sum(1 for s in signs if s < 0)
    n_amb = sum(1 for s in signs if s == 0)
    denom = n_rise + n_fall
    sign_t = (n_rise - n_fall) / denom if denom else 0.0

    return {
        "quarter": d["quarter"],
        "n_api": len(d["articles"]),
        "n_kept": len(kept),
        "n_total": d["n_total"],
        "salience_raw": len(kept) / d["n_total"] if d["n_total"] else float("nan"),
        "n_rising": n_rise,
        "n_falling": n_fall,
        "n_ambiguous": n_amb,
        "sign": sign_t,
        "dropped": reasons,
        "kept_articles": kept,
    }


def build_index(rows):
    """BBD normalisation: scale MAGNITUDE to mean 100, then apply sign.

    Note we normalise the UNSIGNED component. Normalising the signed series (as the
    proposal originally specified) divides by a mean that sits near zero, which makes
    the scaling factor explode and the units meaningless. |s_t| = salience relative to
    the sample average; sign(s_t) = framing.
    """
    sal = [r["salience_raw"] for r in rows]
    mean_sal = sum(sal) / len(sal)
    for r in rows:
        r["salience_norm"] = 100 * r["salience_raw"] / mean_sal
        r["s_t"] = r["salience_norm"] * r["sign"]
    return rows


BOE = re.compile(
    r"bank\s+of\s+england|monetary\s+policy\s+committee|\bmpc\b|threadneedle"
    r"|\bcarney\b|\bbailey\b|\bking\b\s+(said|warns)|rate[-\s]setters?",
    re.I,
)


COLS = ["quarter", "n_api", "n_kept", "n_total", "salience_raw", "salience_norm",
        "n_rising", "n_falling", "n_ambiguous", "sign", "s_t",
        "drop_no_anchor", "drop_foreign", "drop_asset"]


def build(exclude_boe=False):
    paths = sorted(RAW_DIR.glob("*.json.gz"))
    if not paths:
        sys.exit(f"No cached quarters found in {RAW_DIR}/")

    rows = []
    for p in paths:
        r = process_quarter(p)

        if exclude_boe:
            # Re-derive the numerator and sign excluding Bank-attributed coverage.
            kept = [a for a in r["kept_articles"] if not BOE.search(select_unit(a))]
            signs = [article_sign(a) for a in kept]
            n_rise = sum(1 for s in signs if s > 0)
            n_fall = sum(1 for s in signs if s < 0)
            denom = n_rise + n_fall
            r["n_kept"] = len(kept)
            r["n_rising"] = n_rise
            r["n_falling"] = n_fall
            r["n_ambiguous"] = sum(1 for s in signs if s == 0)
            r["sign"] = (n_rise - n_fall) / denom if denom else 0.0
            r["salience_raw"] = len(kept) / r["n_total"] if r["n_total"] else float("nan")

        d = r.pop("dropped")
        r["drop_no_anchor"] = d.get("no_anchor", 0)
        r["drop_foreign"] = d.get("foreign", 0)
        r["drop_asset"] = d.get("asset_price", 0)
        r.pop("kept_articles", None)
        rows.append(r)

    rows = build_index(rows)   # normalise magnitude to mean 100, then apply sign
    return rows


def write(rows, path):
    OUT_DIR.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"Wrote {path}  ({len(rows)} rows)")


def summarise(rows):
    print()
    print("  quarter    N_t   sal      sign     s_t")
    print("  " + "-" * 44)
    for r in rows:
        print(f"  {r['quarter']}  {r['n_kept']:>5}  {r['salience_raw']:.5f}  "
              f"{r['sign']:+.3f}  {r['s_t']:+8.1f}")

    print()
    hi = sorted(rows, key=lambda r: -r["s_t"])[:6]
    lo = sorted(rows, key=lambda r: r["s_t"])[:6]
    print("  Most positive s_t (expect 2008 and 2022):")
    for r in hi:
        print(f"    {r['quarter']}  {r['s_t']:+8.1f}")
    print("  Most negative s_t (expect 2009 and 2015):")
    for r in lo:
        print(f"    {r['quarter']}  {r['s_t']:+8.1f}")

    thin = [r for r in rows if r["n_kept"] < 15]
    if thin:
        print()
        print(f"  WARNING: {len(thin)} quarters have N_t < 15. The sign in these rests")
        print("  on very few articles and may be noisy. Quarters:")
        print("    " + ", ".join(r["quarter"] for r in thin))


if __name__ == "__main__":
    print("Building main index...")
    rows = build(exclude_boe=False)
    write(rows, OUT_DIR / "salience_index.csv")
    summarise(rows)

    print()
    print("Building ex-BoE robustness variant...")
    rows_ex = build(exclude_boe=True)
    write(rows_ex, OUT_DIR / "salience_index_exboe.csv")

    import statistics
    a = [r["s_t"] for r in rows]
    b = [r["s_t"] for r in rows_ex]
    if len(a) > 2:
        try:
            print(f"  correlation(main, ex-BoE) = {statistics.correlation(a, b):.4f}")
            print("  (High correlation => the salience measure is not merely")
            print("   central-bank communication reported second-hand.)")
        except Exception:
            pass
