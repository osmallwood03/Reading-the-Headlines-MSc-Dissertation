# Reading the Headlines

Code for *Reading the Headlines: Cost-of-Living Media Salience and the
Household–Professional Inflation Expectations Gap in the UK* (MSc Economics
dissertation, University of Warwick, 2026).

There are two halves in executing the code. The first builds three series from primary sources: a signed index
of UK cost-of-living news coverage, de-censored moments of the Bank of England
Inflation Attitudes Survey, and a constant-horizon professional forecast
consensus. The second estimates a six-variable SVAR on those series, identified
by sign restrictions, exact zeros on the global block, small-open-economy block
exogeneity and a narrative restriction, with wild bootstrap inference.

## Files

**Salience index**

| | |
| --- | --- |
| `guardian_collect.py` | collects coverage from the Guardian Open Platform API |
| `build_index.py` | precision filter, direction dictionary, and the index itself |

**Survey and forecast series**

| | |
| --- | --- |
| `build_ias_moments_combined.py` | de-censors the IAS response distribution |
| `build_panel.py` | cleans the HM Treasury forecaster panel |
| `validate_aggregation.py` | recovers HMT's aggregation rule from its own published average |
| `build_professional.py` | constant-horizon professional consensus |
| `validate_professional.py` | four checks on that series |
| `build_gap.py` | merges both sides into the expectations gap |
| `build_data_v3.py` | adds the sterling exchange rate, writes the estimation panel |

**Estimation**

| | |
| --- | --- |
| `config_v3.py` | data loading, bridging, restrictions, specification grid |
| `svar_lib_v3.py` | reduced form, rotation sampling, impulse responses |
| `identify_v3.py` | rotation sampler, narrative check, scheme ladder |
| `run_v3.py` | command entry points |
| `bootstrap_r16.py` | wild bootstrap for the split system |

**Output and tests**

| | |
| --- | --- |
| `figures.py` | every figure |
| `gen_tables.py` | LaTeX table bodies straight from the results workbook |
| `tests.py` | mock tests for the collector, synthetic-data tests for block exogeneity |

`salience_index.csv` and `salience_index_exboe.csv` are the constructed index,
included because it is the dissertation's own contribution.

## Requirements

Python 3.13.6, and the versions in `requirements.txt`, which are the ones the
reported results were produced on.

```
pip install -r requirements.txt
```

## Running it

The collector needs a Guardian Open Platform key, read from the
`GUARDIAN_API_KEY` environment variable or from `guardian_key.txt` next to the
script. The key is never written into the source and the file is gitignored.

```
python guardian_collect.py --check-key      # one call, confirms the key works
python guardian_collect.py --counts-only    # ~176 calls, count-based index
python guardian_collect.py                  # full text collection, resumable
python build_index.py
```

Collection caches each quarter as it completes, so a rate limit or a crash just
means re-running.

The survey and forecast series are built from data not redistributed here, then
assembled into the estimation panel:

```
python build_ias_moments_combined.py
python build_professional.py
python build_gap.py
python build_data_v3.py
```

Estimation, in order. Block exogeneity changes the reduced form, so `soe` has
to be settled before the scheme ladder means anything:

```
python run_v3.py diagnostics
python run_v3.py soe
python run_v3.py baseline
python run_v3.py grid
python run_v3.py bootstrap 500
python bootstrap_r16.py run 500
python run_v3.py combine
python figures.py
python gen_tables.py
```

Seeds are fixed throughout.

## Data

Not redistributed, for licensing reasons.

| Series | Source |
| --- | --- |
| Commodity prices | IMF Primary Commodity Price System, All Commodity Index |
| Global real activity | Federal Reserve Bank of Dallas, index following Kilian (2009) |
| RPI, CPI | ONS |
| News coverage | Guardian Open Platform Content API |
| Household expectations | Bank of England / Ipsos Inflation Attitudes Survey microdata |
| Professional consensus | HM Treasury, *Forecasts for the UK Economy* |
| Sterling ERI | ONS, CDID BK67, dataset MRET |
| Validation | Google Trends, UK search interest in "cost of living" |

## Licence

MIT for the code. The data belong to their respective sources.
