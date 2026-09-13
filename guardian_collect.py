"""
Collect UK cost-of-living coverage from the Guardian Open Platform API.

    python guardian_collect.py --check-key      one call, confirms the key works
    python guardian_collect.py --counts-only    ~2 calls per quarter
    python guardian_collect.py                  full text, ~5-30 calls per quarter
    python guardian_collect.py --dry-run 2022Q3

Outputs:
    data/raw/<YYYYQn>.json.gz   matched articles with body text, one file per quarter
    data/quarterly_panel.csv    quarter, n_matched, n_total, salience_raw

The query is a broad recall net matched on the full article body. Precision
filtering happens downstream in sign_pass.py, on the cached text, so the
numerator can be re-derived without re-collecting.

Collection is resumable: each quarter is written as it completes and a re-run
skips what is already on disk. Both `production-office=uk` and `type=article`
are applied to the matched count and the total, so salience_raw is a share
within one consistently defined universe. Liveblogs and video did not exist in
2004 and expanded after 2020; leaving them in would put a trend in the
denominator.

The API key is read from GUARDIAN_API_KEY or from guardian_key.txt next to the
script, never from the source. There is no fallback to the Guardian test key:
it is rate-limited to the point of failing several minutes into a run, which
looks like a network fault rather than a missing key.
"""

import argparse
import gzip
import html
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

# CONFIG - everything you might want to change lives in this block

# The key file sits next to THIS script, not next to the working directory, so the
# collector works regardless of where you launch it from.
KEY_FILE = Path(__file__).resolve().parent / "guardian_key.txt"


def load_api_key():
    """Return the Guardian key, or None if no key is configured.

    Resolution order, first hit wins:
      1. GUARDIAN_API_KEY environment variable
      2. guardian_key.txt sitting next to this script

    Deliberately returns None rather than raising, and never falls back to the
    "test" key. Two reasons:
      * Importing this module must not require a key - test_collect.py imports it
        and runs entirely on mock data.
      * The old default of "test" failed SILENTLY: a mistyped or invisible env var
        left you with a heavily rate-limited key and a 429 several minutes into a
        run, which looks like a network fault rather than a missing key. main()
        now refuses to start without a real key instead.

    The key stays out of this file on purpose: guardian_collect.py
    goes into the dissertation appendix verbatim, and guardian_key.txt does not.
    """
    env = os.environ.get("GUARDIAN_API_KEY", "").strip()
    if env:
        return env

    if KEY_FILE.exists():
        # Tolerate a trailing newline, stray quotes, or a pasted KEY=value line -
        # all three are easy to do by hand and none should cost you a run.
        text = KEY_FILE.read_text(encoding="utf-8-sig").strip()
        for line in text.splitlines():
            line = line.strip().strip('"').strip("'")
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                line = line.split("=", 1)[1].strip().strip('"').strip("'")
            if line:
                return line

    return None


def key_source():
    """Where the key came from, for diagnostics."""
    if os.environ.get("GUARDIAN_API_KEY", "").strip():
        return "GUARDIAN_API_KEY environment variable"
    if KEY_FILE.exists():
        return str(KEY_FILE)
    return "nowhere - no key configured"


def mask_key(key):
    """Show enough of the key to spot a typo, never enough to leak it into a log."""
    if not key:
        return "(none)"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]} ({len(key)} chars)"


API_KEY = load_api_key()

BASE_URL = "https://content.guardianapis.com/search"

START_YEAR, START_Q = 2001, 1
END_YEAR, END_Q = 2025, 4

# The cost-of-living basket. This is the NUMERATOR query.
# Guardian's `q` supports AND / OR / NOT, parentheses, and "exact phrases".
# It does NOT support Nexis-style truncation (price!) or proximity (w/10), so word
# forms are enumerated explicitly. Keep terms lowercase; the API is case-insensitive.
COST_OF_LIVING_QUERY = " OR ".join([
    '"cost of living"',
    '"cost-of-living"',
    'inflation',
    '"energy bills"',
    '"household bills"',
    '"gas bills"',
    '"electricity bills"',
    '"fuel bills"',
    '"food prices"',
    '"grocery prices"',
    '"supermarket prices"',
    '"fuel prices"',
    '"petrol prices"',
    '"energy price cap"',
    '"household budgets"',
    '"family finances"',
])

# Robustness variant answering the supervisor's question: is this shock really just
# central-bank communication picked up by the media? Set BASKET = "ex_boe" to exclude
# coverage attributed to the Bank. Collected as a separate panel, not a replacement.
BOE_TERMS = '"Bank of England" OR "Monetary Policy Committee" OR MPC OR "Threadneedle Street"'

# Fields pulled for the sign. headline + standfirst + trailText + the opening of body
# is what a household skimming a front page actually absorbs. Carroll
# (2003) epidemiological mechanism stated at the level of the query.
SHOW_FIELDS = "headline,standfirst,trailText,body,wordcount,firstPublicationDate"

PAGE_SIZE = 200          # Guardian max for the content endpoint
SLEEP_BETWEEN_CALLS = 1.2  # seconds. Free-tier ceiling is ~1 call/sec; 0.3s trips 429.
MAX_RETRIES = 5

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"

# Quarter helpers

def quarters(start_year, start_q, end_year, end_q):
    """Yield ('2004Q1', '2004-01-01', '2004-03-31') tuples inclusive of both ends."""
    y, q = start_year, start_q
    while (y, q) <= (end_year, end_q):
        first_month = 3 * (q - 1) + 1
        last_month = first_month + 2
        last_day = {3: 31, 6: 30, 9: 30, 12: 31}[last_month]
        yield (f"{y}Q{q}",
               f"{y}-{first_month:02d}-01",
               f"{y}-{last_month:02d}-{last_day:02d}")
        q += 1
        if q == 5:
            q, y = 1, y + 1


# API layer

class RateLimited(Exception):
    pass


def api_get(params, session=None):
    """Single GET with retry/backoff. Returns the parsed 'response' object."""
    session = session or requests
    params = dict(params)
    params["api-key"] = API_KEY

    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(BASE_URL, params=params, timeout=30)
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"    network error ({e}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue

        if r.status_code == 200:
            time.sleep(SLEEP_BETWEEN_CALLS)
            return r.json()["response"]

        if r.status_code == 429:
            raise RateLimited(
                "Guardian API returned 429 (rate limit / daily quota exhausted).\n"
                "Your progress is saved. Re-run this script tomorrow and it will "
                "resume from the first uncollected quarter."
            )

        if r.status_code in (401, 403):
            raise SystemExit(
                f"HTTP {r.status_code}: the Guardian rejected this API key.\n"
                f"  key in use : {mask_key(API_KEY)}\n"
                f"  source     : {key_source()}\n"
                "Check for a typo or a stale key. Register a free one at\n"
                "https://open-platform.theguardian.com/access/"
            )

        wait = 2 ** attempt
        print(f"    HTTP {r.status_code}; retrying in {wait}s", file=sys.stderr)
        time.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {params}")


def base_params(from_date, to_date):
    return {
        "from-date": from_date,
        "to-date": to_date,
        "production-office": "uk",
        "type": "article",
        "lang": "en",
        "page-size": 1,
        "format": "json",
    }


def get_total(from_date, to_date, query=None, session=None):
    """Return the number of matching articles, using a single 1-result call."""
    params = base_params(from_date, to_date)
    if query:
        params["q"] = query
    resp = api_get(params, session=session)
    return resp["total"]


def get_articles(from_date, to_date, query, session=None, verbose=True):
    """Page through every matching article, returning a list of dicts with text."""
    out = []
    page = 1
    while True:
        params = base_params(from_date, to_date)
        params.update({
            "q": query,
            "page-size": PAGE_SIZE,
            "page": page,
            "show-fields": SHOW_FIELDS,
            "order-by": "oldest",
        })
        resp = api_get(params, session=session)
        results = resp.get("results", [])
        out.extend(clean_article(a) for a in results)

        pages = resp.get("pages", 1)
        if verbose:
            print(f"    page {page}/{pages} ({len(out)} articles)")
        if page >= pages or not results:
            break
        page += 1
    return out


# Text cleaning

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(text):
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def clean_article(a):
    """Flatten one API result into the fields the sign pass needs."""
    f = a.get("fields", {}) or {}
    body = strip_html(f.get("body", ""))
    return {
        "id": a.get("id"),
        "date": f.get("firstPublicationDate") or a.get("webPublicationDate"),
        "section": a.get("sectionId"),
        "headline": strip_html(f.get("headline", "")),
        "standfirst": strip_html(f.get("standfirst", "")),
        "trail": strip_html(f.get("trailText", "")),
        "body": body,
        # lead = the first ~400 chars of body, i.e. roughly the opening paragraphs.
        # This is the unit the sign is applied to in the baseline specification.
        "lead": body[:400],
        "wordcount": f.get("wordcount"),
        "url": a.get("webUrl"),
    }


# Cache

def cache_path(qlabel, tag=""):
    suffix = f"_{tag}" if tag else ""
    return RAW_DIR / f"{qlabel}{suffix}.json.gz"


def write_cache(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(obj, fh)
    tmp.replace(path)  # atomic - a killed process never leaves a half-written quarter


def read_cache(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


# Main routines

def run_counts_only(args):
    """~2 calls per quarter. Gives n_matched and n_total for the whole sample."""
    query = build_query(args.basket)
    rows = []
    session = requests.Session()
    for qlabel, d0, d1 in quarters(START_YEAR, START_Q, END_YEAR, END_Q):
        n_matched = get_total(d0, d1, query, session=session)
        n_total = get_total(d0, d1, None, session=session)
        sal = n_matched / n_total if n_total else float("nan")
        rows.append({
            "quarter": qlabel,
            "n_matched": n_matched,
            "n_total": n_total,
            "salience_raw": sal,
        })
        print(f"  {qlabel}: {n_matched:>5} / {n_total:>6}  =  {sal:.4f}")

    out = DATA_DIR / f"quarterly_panel_counts{'_' + args.basket if args.basket != 'main' else ''}.csv"
    write_csv(out, rows, ["quarter", "n_matched", "n_total", "salience_raw"])
    print(f"\nWrote {out}")
    est = sum(max(1, r["n_matched"] // PAGE_SIZE + 1) for r in rows)
    print(f"Full text collection would need roughly {est + len(rows)} calls "
          f"({sum(r['n_matched'] for r in rows):,} articles).")


def run_full(args):
    """Counts + body text, cached per quarter, resumable."""
    query = build_query(args.basket)
    tag = "" if args.basket == "main" else args.basket
    rows = []
    session = requests.Session()

    for qlabel, d0, d1 in quarters(START_YEAR, START_Q, END_YEAR, END_Q):
        path = cache_path(qlabel, tag)
        if path.exists() and not args.force:
            payload = read_cache(path)
            print(f"  {qlabel}: cached ({payload['n_matched']} articles)")
        else:
            print(f"  {qlabel}: collecting...")
            n_total = get_total(d0, d1, None, session=session)
            articles = get_articles(d0, d1, query, session=session)
            payload = {
                "quarter": qlabel,
                "from": d0,
                "to": d1,
                "query": query,
                "n_matched": len(articles),
                "n_total": n_total,
                "articles": articles,
            }
            write_cache(path, payload)

        rows.append({
            "quarter": payload["quarter"],
            "n_matched": payload["n_matched"],
            "n_total": payload["n_total"],
            "salience_raw": (payload["n_matched"] / payload["n_total"]
                             if payload["n_total"] else float("nan")),
        })

    out = DATA_DIR / f"quarterly_panel{'_' + tag if tag else ''}.csv"
    write_csv(out, rows, ["quarter", "n_matched", "n_total", "salience_raw"])
    print(f"\nWrote {out}")
    print(f"Raw article text cached in {RAW_DIR}/ - this is the input to the sign pass.")


def run_dry(args):
    """Inspect a single quarter before committing to 88 of them."""
    query = build_query(args.basket)
    target = args.dry_run.upper()
    for qlabel, d0, d1 in quarters(START_YEAR, START_Q, END_YEAR, END_Q):
        if qlabel != target:
            continue
        print(f"Quarter {qlabel} ({d0} to {d1})")
        print(f"Query: {query}\n")
        n_total = get_total(d0, d1, None)
        n_matched = get_total(d0, d1, query)
        print(f"  matched : {n_matched}")
        print(f"  total   : {n_total}")
        print(f"  salience: {n_matched / n_total:.4f}\n" if n_total else "")

        params = base_params(d0, d1)
        params.update({"q": query, "page-size": 10, "show-fields": SHOW_FIELDS})
        resp = api_get(params)
        print("First 10 matches - EYEBALL THESE. If they are not about UK consumer")
        print("prices, fix the basket before collecting 88 quarters.\n")
        for a in resp["results"]:
            art = clean_article(a)
            print(f"  [{art['date'][:10]}] {art['headline']}")
            print(f"      {art['lead'][:120]}...")
        return
    raise SystemExit(f"{target} is not in the sample range.")


def build_query(basket):
    if basket == "main":
        return f"({COST_OF_LIVING_QUERY})"
    if basket == "ex_boe":
        return f"({COST_OF_LIVING_QUERY}) AND NOT ({BOE_TERMS})"
    raise SystemExit(f"Unknown basket: {basket}")


def write_csv(path, rows, cols):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check-key", action="store_true",
                   help="Verify the API key with one call, then exit. Run this first.")
    p.add_argument("--counts-only", action="store_true",
                   help="Collect counts only (~176 calls). Run this first.")
    p.add_argument("--dry-run", metavar="QUARTER",
                   help="Inspect one quarter's matches, e.g. --dry-run 2022Q3")
    p.add_argument("--basket", default="main", choices=["main", "ex_boe"],
                   help="'main' or 'ex_boe' (excludes Bank of England coverage)")
    p.add_argument("--force", action="store_true",
                   help="Re-collect quarters even if already cached")
    args = p.parse_args()

    # Fail loudly and immediately rather than several minutes into a collection.
    if not API_KEY:
        raise SystemExit(
            "No Guardian API key found.\n"
            f"  looked for : GUARDIAN_API_KEY environment variable\n"
            f"  then       : {KEY_FILE}\n\n"
            "Fix by writing your key into that file, one line, nothing else:\n"
            f'    Set-Content -Path "{KEY_FILE}" -Value "your-key-here" -Encoding utf8\n\n'
            "Register a free key at https://open-platform.theguardian.com/access/"
        )

    if API_KEY == "test":
        raise SystemExit(
            "Refusing to run with the Guardian 'test' key: it is heavily rate-limited\n"
            "and will 429 partway through, which looks like a network fault. Put a real\n"
            f"key in {KEY_FILE} or in GUARDIAN_API_KEY."
        )

    try:
        if args.check_key:
            print(f"Key       : {mask_key(API_KEY)}")
            print(f"Source    : {key_source()}")
            n = get_total("2022-07-01", "2022-09-30", build_query(args.basket))
            print(f"Test call : OK - {n} matches in 2022Q3")
            print("\nThe key works. You are clear to run --counts-only.")
            return

        if args.dry_run:
            run_dry(args)
        elif args.counts_only:
            run_counts_only(args)
        else:
            run_full(args)
    except RateLimited as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
