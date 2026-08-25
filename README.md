# HARP — Hurricane Adaptation and Recovery Parcel Dataset (Ian, Ida & Harvey)

The **Hurricane Adaptation and Recovery Parcel (HARP)** dataset is a harmonized,
household-level parcel dataset for studying post-hurricane housing recovery
across three major U.S. hurricanes — Harvey (Texas, 2017), Ida (Louisiana,
2021), and Ian (Florida, 2022). Built around
geocoded single-family residential parcels linked to Census Block Groups (CBGs),
HARP harmonizes parcel attributes, damage assessments, building-permit and
assistance-based repair records, and property transactions into monthly recovery
sequences at both household and community scales, and augments them with
model-derived neighborhood influence networks inferred from repair and sale
event histories. Together these enable post-disaster housing recovery to be
analyzed as a temporally sequenced and spatially interdependent diffusion
process.

| File | Hurricane | Region | Year | Communities | Parcels |
|------|-----------|--------|------|-------------|---------|
| `harp_ian.npz` | Ian | Lee County, FL | 2022 | 187 | 69,145 |
| `harp_ida.npz` | Ida | Orleans Parish, LA | 2021 | 382 | 121,130 |
| `harp_harvey.npz` | Harvey | Harris County, TX | 2017 | 1,992 | 1,016,927 |

Each file is a single NumPy archive. Records are organized by **community**
(a Census block group); within a community, each row is a single **parcel**
(a household / residential building).

**License:** Creative Commons Attribution-NoDerivatives 4.0 International
(CC BY-ND 4.0).

---

## Loading

```python
import numpy as np

d = np.load("harp_ian.npz", allow_pickle=True)      # allow_pickle required
meta   = d["meta"].item()                            # metadata dictionary
months = d["time_index"]                             # e.g. ['2021-01', ..., '2023-10']
t0 = meta["t_hurricane"]                             # months[:t0] pre-landfall, months[t0:] post

for comm in d["communities"]:
    cbg    = comm["cbg"]          # Census block-group GEOID, e.g. '120710205021'
    H      = comm["H"]            # (N, 128) household feature embedding
    Y      = comm["Y"]            # (T, N, 2) monthly events: [:, :, 0]=sale, [:, :, 1]=repair
    acs    = comm["acs"]          # (31,)  community socioeconomic covariates
    damage = comm["damage"]       # (N, D) parcel-level hurricane damage
    buyer  = comm.get("buyer_type")  # (N,) individual/investor/unknown (Harvey only)
    A      = comm.get("A")        # (2, N, N) neighborhood influence network, or None

    # exact-date event timelines, grouped into two dicts of (N,) 'YYYY-MM-DD' strings
    # ('' if none); which keys exist is storm-specific — see meta['sale_fields'] / ['repair_fields']
    sale   = comm["sale"]      # e.g. sale["first_sale_date"], sale["listing_date"]
    repair = comm["repair"]    # e.g. repair["lee_statuses"], repair["fema_applied_dates"]
```

`d["communities"]` is an object array of dicts and `meta = d["meta"].item()`
recovers the metadata dictionary; both require `allow_pickle=True`. Within a
community, `N` is the number of parcels and `T` the number of months. All
per-parcel arrays (`H`, `damage`, and both axes of `A`) share a common
row order, so index `j` denotes the same parcel throughout.

---

## Record structure

Each archive exposes three top-level entries — `communities` (an object array of
per-community dicts), `meta` (metadata), and `time_index` (a length-`T` array of
`YYYY-MM` strings). Each community record contains the following fields:

| field | shape | description |
|-------|-------|-------------|
| `cbg` | string | Census **block-group GEOID**; the community identifier and the join key to external Census/ACS data |
| `H` | `(N, 128)` | Learned **feature embedding** of each parcel's attributes (location, size, valuation, age, damage, …). **This is the only form in which household attributes ship — the raw attribute values are *not* released as columns and cannot be recovered from `H`** |
| `Y` | `(T, N, 2)` | Monthly **event sequences**: `Y[t, j, 0]=1` if parcel `j` recorded a **sale** in month `t`; `Y[t, j, 1]=1` for a **repair**. Sparse (predominantly zero) |
| `acs` | `(31,)` | Community-level **socioeconomic covariates** (population, age, income, tenure, housing) from the American Community Survey; one vector per community |
| `damage` | `(N, D)` | Parcel-level **damage** (flood depth, estimated loss, damage level, aerial roof-damage flag, …); `D` and column names are storm-specific |
| `buyer_type` | `(N,)` | **Buyer classification** of each parcel's sale — `individual` / `investor` / `unknown` (`''` if the parcel did not sell in the window). **Harvey only** |
| `A` | `(2, N, N)` | Inferred **neighborhood influence network**: `A[0]` for sales, `A[1]` for repairs; `A[i, j]` is the influence of parcel `j` on parcel `i`. Provided for communities included in model training |
| `sale` | dict | Per-parcel **sale-event exact dates** (`first_sale_date`, `listing_date`, `sale_date_history`), each an `(N,)` `YYYY-MM-DD` string array; keys in `meta['sale_fields']` |
| `repair` | dict | Per-parcel **repair-event exact dates & statuses** (permit / FEMA), each an `(N,)` string array; keys in `meta['repair_fields']` |

Column names and orderings are recorded in `meta`: `acs_fields` (31 names),
`damage_fields` (storm-specific), and `sale_fields` / `repair_fields`
(the exact-date event keys inside `comm['sale']` / `comm['repair']` for this storm).
A complete field-by-field dictionary and a description of every `meta` key are given
in the **Appendix** at the end of this file.

---

## Derived quantities: embeddings (`H`) and influence networks (`A`)

- **Feature embedding `H`.** To preserve confidentiality of licensed and
  identifiable parcel attributes, raw features are not distributed; each parcel
  is instead represented by `H`, a fixed 128-dimensional vector produced by a
  trained neural encoder. The embedding retains structure relevant to downstream
  analysis (clustering, similarity, prediction) but is non-invertible: the
  original address, valuation, or geometry cannot be recovered from it because
  the encoder parameters are withheld.
- **Influence network `A`.** `A` is a learned, directed, weighted graph over the
  parcels of a community, quantifying how households' sale and repair decisions
  influence one another; it is inferred from event histories under the decision
  diffusion framework of the companion study. Networks are provided for the
  communities used in model training (178/187 for Ian, 348/382 for Ida,
  1,652/1,992 for Harvey); other communities omit the `A` field but retain all
  remaining fields.

---

## Damage fields

`damage` is an `(N, D)` array; column `i` is `meta['damage_fields'][i]`. Columns
are **storm-specific**: the FEMA columns come from public FEMA / DesignSafe damage
assessments, the `nhc_*` columns from National Hurricane Center storm-surge
products (Ida), and `*_AerialDamaged` is an in-house blue-tarp detection on public
NAIP imagery. Fully-empty columns are dropped, so `D` differs by storm. For
categorical / ordinal codes (`*_DamageLevel`, `*_Occupancy`, `nhc_*_category`) a
value of `0` is a baseline / unknown class.

| column | storms | description |
|--------|--------|-------------|
| `<storm>_FloodDepth` | Ian, Ida, Harvey | Flood depth at the parcel during the hurricane |
| `<storm>_BldgValue` | Ian, Harvey | Building replacement value at time of assessment (USD) |
| `<storm>_EstLoss` | Ian, Ida, Harvey | Estimated loss — **Ian: real USD**; **Ida / Harvey: ordinal severity code** |
| `<storm>_Occupancy` | Ian, Ida | Occupancy type (`single_family` / `multi_family` / `manufactured_home`), categorical code |
| `<storm>_DamageLevel` | Ian, Ida, Harvey | Damage severity level; event-specific ordinal code |
| `<storm>_AerialDamaged` | Ian, Ida | `0/1` damage flag from aerial blue-tarp detection on NAIP imagery |
| `nhc_surge_category` | Ida | NHC storm-surge category (ordinal) |
| `nhc_surge_depth_ft` | Ida | NHC modeled storm-surge depth (feet) |
| `nhc_tidalmask_category` | Ida | NHC tidal-zone category (ordinal) |

Exact per-storm column order (`meta['damage_fields']`):

- **Ian (6):** `ian_FloodDepth, ian_BldgValue, ian_EstLoss, ian_Occupancy, ian_DamageLevel, Ian_AerialDamaged`
- **Ida (8):** `ida_FloodDepth, ida_EstLoss, nhc_surge_category, nhc_surge_depth_ft, nhc_tidalmask_category, ida_Occupancy, ida_DamageLevel, Ida_AerialDamaged`
- **Harvey (4):** `harvey_FloodDepth, harvey_BldgValue, harvey_EstLoss, harvey_DamageLevel`

---

## Temporal structure

`time_index` is a continuous monthly axis spanning both the **pre-landfall** and
**post-landfall** periods, and `Y` is indexed against it. `meta['t_hurricane']`
gives the index at which the post-landfall period begins, so any sequence may be
partitioned as `Y[:t0]` (pre-landfall) and `Y[t0:]` (post-landfall).

| Hurricane | Pre-landfall | Post-landfall | Months (pre + post) |
|-----------|--------------|---------------|---------------------|
| Ian    | 2021-01 – 2022-08 | 2022-09 – 2023-10 | 34 (20 + 14) |
| Ida    | 2020-09 – 2021-07 | 2021-08 – 2022-07 | 23 (11 + 12) |
| Harvey | 2016-05 – 2017-07 | 2017-08 – 2018-11 | 31 (15 + 16) |

---

## Event timeline fields

In addition to the monthly `Y`, each parcel carries a
set of **exact-date** event fields drawn from the underlying permit, sale, and
FEMA assistance records. These are grouped into two per-community dicts,
`comm['sale']` and `comm['repair']` (keys listed in `meta['sale_fields']` and
`meta['repair_fields']`). Each field is an `(N,)` array of strings, row-aligned
with `H`; a value is `''` where the parcel has no such record. Fields with more
than one record per parcel are **semicolon-joined `YYYY-MM-DD` sequences**
(e.g. `'2003-04-22;2004-06-15;2006-10-17'`). The exact set of fields is
storm-specific.

| field | group | storms | meaning |
|-------|-------|--------|---------|
| `first_sale_date` | sale | Ian, Ida, Harvey | Parcel's sale close date (single date) |
| `listing_date` | sale | Ian, Ida, Harvey | Estimated listing date = close − Redfin days-on-market (see below) |
| `sale_date_history` | sale | Ian, Harvey | All recorded sale close dates, semicolon-joined |
| `lee_first_completion_date` | repair | Ian | First Lee County permit whose status indicates completion (CC/CO/etc.) |
| `lee_record_dates` | repair | Ian | All Lee County permit record dates, semicolon-joined (aligned with `lee_statuses`) |
| `lee_statuses` | repair | Ian | Lee County permit status labels (e.g. `Closed-CC Issued`), aligned with `lee_record_dates` |
| `nola_app_submit_dates` | repair | Ida | New Orleans permit application-submitted dates |
| `nola_app_review_dates` | repair | Ida | Application-review dates (aligned) |
| `nola_plan_review_dates` | repair | Ida | Plan-review dates (aligned) |
| `nola_zoning_review_dates` | repair | Ida | Zoning-review dates (aligned) |
| `nola_approved_dates` | repair | Ida | Permit-approved dates (aligned) |
| `nola_issued_dates` | repair | Ida | Permit-issued dates (aligned) |
| `nola_inspection_dates` | repair | Ida | First-inspection dates (aligned) |
| `nola_co_dates` | repair | Ida | Certificate-of-Occupancy dates (aligned) |
| `hcad_issue_dates` | repair | Harvey | All HCAD permit issue dates, semicolon-joined |
| `fema_applied_dates` | repair | Ian, Ida, Harvey | FEMA IHP application dates (zip-level assignment) |

**Alignment of the `nola_*` fields (Ida).** The eight `nola_*_dates` fields are
positionally aligned: the *k*-th slot in every `nola_*` field refers to the same
permit. A stage that a permit never reached appears as an empty slot, so a value
like `'2022-11-09;;2024-10-21'` has three permits with the second missing this
stage. Use the semicolon layout to pair a permit's lifecycle across fields.

**`fema_applied_dates` caveat.** FEMA IHP records are attached to parcels by
**zip-level assignment** among damaged parcels, so these dates are real but the
specific parcel they attach to is approximate; treat them as neighborhood-level
signal, not a certified per-household timestamp. (The FEMA `inspnIssued` field in
the source is a 0/1 flag, not a date, so no FEMA inspection *date* is released.)

**`listing_date` is an estimate.** The source records only sale *close* dates, so
`listing_date` is derived as `first_sale_date − median_days_on_market(zip, month)`
using public Redfin zip-level DOM (with county-level fallback). For Ian and Ida the
monthly totals are further calibrated to Redfin county new-listing counts; Harvey uses
the plain DOM shift. It is therefore an approximation of when the parcel was listed,
not an independently observed date, and it is always `≤ first_sale_date`.

---

## Usage notes

- **Network availability.** An `A` field is provided only for communities used
  in model training — those whose parcel count `N` falls within the per-storm
  bounds below. Other communities omit `A` but retain all other fields
  (test with `"A" in comm`).

  | Hurricane | min parcels | max parcels |
  |-----------|-------------|-------------|
  | Ian    | 20  | 2000 |
  | Ida    | 100 | 900  |
  | Harvey | 100 | 900  |

- **Community identifier.** `cbg` is a valid Census block-group GEOID; use it to
  join external Census / ACS / spatial data.
- **Exact event timelines.** The `sale` and `repair` dicts carry raw `YYYY-MM-DD`
  permit, sale, and FEMA dates (see *Event timeline fields*). Unlike `Y`
  these are **not month-coarsened** and are **not clipped to the study window** —
  sale and permit histories may predate landfall by years. Multi-record fields are
  semicolon-joined.
- **Buyer type (Harvey only).** `buyer_type` classifies each sold parcel's buyer
  as `individual` / `investor` / `unknown` (`''` if it did not sell in the
  window). Present only in `harp_harvey.npz`; Ian/Ida have no buyer field.
- **Storm-specific fields.** `damage` columns vary by storm; the four `median_*`
  `acs` fields are `NaN` where the Census suppressed small-sample estimates.
  Consult `meta['damage_fields']` and `meta['acs_fields']`.

---

# Appendix — complete field dictionary

This appendix documents **every community field in the released `.npz`** and the
`meta` keys needed to interpret them (a few internal-provenance `meta` keys are
omitted — see §C). Fields marked *(storm)* appear only for the storms listed.

## A. Top-level archive keys

| key | type | description |
|-----|------|-------------|
| `communities` | object array of dicts | one dict per community (Census block group); requires `allow_pickle=True` |
| `meta` | dict (0-d object array) | dataset metadata; recover with `d['meta'].item()` |
| `time_index` | `(T,)` string array | monthly axis as `YYYY-MM`, spanning pre- and post-landfall |

## B. Per-community fields (inside each `communities` dict)

`N` = number of parcels in the community; `T` = number of months. Every
per-parcel array shares the same row order, so row `j` is the same parcel across
all of them.

| field | shape | dtype | present for | description |
|-------|-------|-------|-------------|-------------|
| `cbg` | scalar | str | all | 12-digit Census **block-group GEOID**; community id and join key to ACS |
| `H` | `(N, 128)` | float32 | all | non-invertible **feature embedding** of the parcel's raw attributes |
| `Y` | `(T, N, 2)` | int16 | all | monthly **event indicators**; `[:,:,0]`=sale, `[:,:,1]`=repair; 0/1 |
| `acs` | `(31,)` | float32 | all | community **ACS covariates** (order = `meta['acs_fields']`); `NaN` in suppressed `median_*` |
| `damage` | `(N, D)` | float32 | all | per-parcel **damage** (order = `meta['damage_fields']`); `D` storm-specific |
| `buyer_type` | `(N,)` | str | Harvey | buyer class `individual`/`investor`/`unknown` (`''` if not sold in window) |
| `A` | `(2, N, N)` | float32 | trained communities | influence network; `A[0]`=sale, `A[1]`=repair; absent key if untrained |
| `sale` | dict | all | **sale-event exact dates** (keys = `meta['sale_fields']`); see B.1 |
| `repair` | dict | all | **repair-event exact dates & statuses** (keys = `meta['repair_fields']`); see B.2 |

Every value inside `sale` / `repair` is an `(N,)` string array, row-aligned with `H`
(`''` where the parcel has no such record). Multi-record fields are semicolon-joined
`YYYY-MM-DD` sequences.

### B.1 `sale` (inside `comm['sale']`)

| field | present for | description |
|-------|-------------|-------------|
| `first_sale_date` | all | sale **close date** `YYYY-MM-DD` (`''` if none) |
| `listing_date` | all | **estimated listing date** = close − Redfin DOM (`≤ first_sale_date`) |
| `sale_date_history` | Ian, Harvey | all sale close dates, `;`-joined |

### B.2 `repair` (inside `comm['repair']`)

| field | present for | description |
|-------|-------------|-------------|
| `lee_first_completion_date` | Ian | first Lee County permit with a completion status |
| `lee_record_dates` | Ian | all Lee County permit record dates, `;`-joined (aligned with `lee_statuses`) |
| `lee_statuses` | Ian | Lee County permit status labels, `;`-joined (aligned with `lee_record_dates`) |
| `nola_app_submit_dates` | Ida | NOLA permit application-submitted dates, `;`-joined |
| `nola_app_review_dates` | Ida | NOLA application-review dates, `;`-joined (aligned) |
| `nola_plan_review_dates` | Ida | NOLA plan-review dates, `;`-joined (aligned) |
| `nola_zoning_review_dates` | Ida | NOLA zoning-review dates, `;`-joined (aligned) |
| `nola_approved_dates` | Ida | NOLA permit-approved dates, `;`-joined (aligned) |
| `nola_issued_dates` | Ida | NOLA permit-issued dates, `;`-joined (aligned) |
| `nola_inspection_dates` | Ida | NOLA first-inspection dates, `;`-joined (aligned) |
| `nola_co_dates` | Ida | NOLA Certificate-of-Occupancy dates, `;`-joined (aligned) |
| `hcad_issue_dates` | Harvey | all HCAD permit issue dates, `;`-joined |
| `fema_applied_dates` | all | FEMA IHP application dates (zip-level assignment), `;`-joined |

## C. `meta` keys

The keys needed to read and interpret the arrays. `meta` also carries a few
internal-provenance keys — `source_npz`, `pre_npz`, `checkpoint`, `encoder_type`,
`d_hid`, `K`, `includes_coords`, `includes_node_ids` — that are not needed to use
the data and are not documented here.

| key | example | description |
|-----|---------|-------------|
| `hurricane` | `'ian'` | storm identifier |
| `T` / `T_pre` / `T_post` | `34 / 20 / 14` | total / pre-landfall / post-landfall month counts |
| `t_hurricane` | `20` | index into `time_index` where the post-landfall window begins |
| `acs_fields` | 31 names | column order of `acs` |
| `damage_fields` | storm-specific | column order of `damage` |
| `sale_fields` | list | keys present inside each `comm['sale']` |
| `repair_fields` | list | keys present inside each `comm['repair']` |
| `has_buyer_type` | `True`/`False` | whether `buyer_type` is present (Harvey only) |
| `n_communities` | `187` | number of community dicts |
| `n_communities_with_network` | `178` | number carrying an `A` field |
| `note` | text | free-text summary of the release conventions |

## D. Community attributes (`acs`)

`acs` is a length-31 vector per community; `acs[i]` corresponds to
`meta['acs_fields'][i]` (the order below). Source: U.S. Census Bureau American
Community Survey 5-Year estimates (Ian 2022 / Ida 2021 / Harvey 2017), at the
block-group level, joined to each parcel through its `cbg`. Percentages are
0–100. The four `median_*` fields can be `NaN` where the Census suppressed
small-sample estimates; all count / percentage fields are complete.

| # | field | description |
|---|-------|-------------|
| 1 | `total_population` | Total resident population |
| 2 | `pct_age_under_18` | Share of residents under 18 (%) |
| 3 | `pct_age_65_plus` | Share of residents aged 65 and over (%) |
| 4 | `median_age_total` | Median age of all residents (years) |
| 5 | `median_age_male` | Median age of male residents (years) |
| 6 | `median_age_female` | Median age of female residents (years) |
| 7 | `pct_white_alone` | Share reporting White alone (%) |
| 8 | `pct_black_alone` | Share reporting Black or African American alone (%) |
| 9 | `pct_asian_alone` | Share reporting Asian alone (%) |
| 10 | `pct_two_or_more_race` | Share reporting two or more races (%) |
| 11 | `total_households` | Total number of households |
| 12 | `pct_family_households` | Share of family households (%) |
| 13 | `pct_single_person_hh` | Share of single-person households (%) |
| 14 | `pct_hh_with_children` | Share of households with children under 18 (%) |
| 15 | `median_household_income` | Median household income, previous year (USD) |
| 16 | `pct_hh_income_under_25k` | Share of households with income below $25,000 (%) |
| 17 | `pct_in_labor_force` | Share of population aged 16+ in the labor force (%) |
| 18 | `unemployment_rate` | Unemployed as a share of the civilian labor force (%) |
| 19 | `pct_bachelor_or_higher` | Share of adults 25+ with a bachelor's degree or higher (%) |
| 20 | `pct_hs_or_higher` | Share of adults 25+ with at least a high-school diploma (%) |
| 21 | `pct_owner_occupied` | Share of occupied units that are owner-occupied (%) |
| 22 | `pct_renter_occupied` | Share of occupied units that are renter-occupied (%) |
| 23 | `median_home_value` | Median value of owner-occupied housing units (USD) |
| 24 | `median_gross_rent` | Median gross rent for renter-occupied units (USD/month) |
| 25 | `median_year_built` | Median year the structure was built |
| 26 | `total_housing_units` | Total housing units, occupied and vacant |
| 27 | `pct_vacant_units` | Share of housing units that are vacant (%) |
| 28 | `pct_single_family` | Share of units in 1-unit detached/attached structures (%) |
| 29 | `pct_mobile_home` | Share of units classified as mobile homes (%) |
| 30 | `pct_commute_60min_plus` | Share of commuters with 60+ min one-way travel (%) |
| 31 | `pct_65plus_living_alone` | Share of residents 65+ who are a householder living alone (%) |

## E. Source household attributes encoded in `H`

> **The fields listed in this section are *not* columns in the released `.npz`.**
> They are the raw inputs that were folded into the non-invertible embedding `H`,
> listed here to document what information `H` summarizes.
> None of them can be read back or reconstructed from the released data.

Raw household attributes are **not** released — each parcel is represented only
by its 128-dimensional embedding `H` (see *Derived quantities*). The tables below
document the **source attributes that `H` encodes** — that is, the information the
embedding summarizes. Ian and Ida come from the Regrid schema;
Harvey from the HCAD schema. `lat` / `lon` are encoded in `H` (the embedding
carries location) but exact coordinates are not released as raw values, and the
`parcelnumb` / `census_blockgroup` identifiers are not distributed
(`census_blockgroup` is released as `cbg`).

### E.1 Regrid schema (Ian, Ida)

These are the Regrid source attributes **actually encoded into `H`** after
preprocessing. Ian and Ida encode overlapping but different feature sets, marked:
**†** = encoded for **Ida only**, **‡** = encoded for **Ian only**, unmarked =
both. Valuation / area / distance fields are log-scaled and year fields are
converted to age before encoding.

| field | description |
|-------|-------------|
| `lat` | Latitude of parcel centroid (WGS-84); location is encoded in `H`, but exact coordinates are not released as a raw value |
| `lon` | Longitude of parcel centroid (WGS-84) |
| `yearbuilt` | Structure year built (encoded as building age) |
| `landval` | Land value (assessed value of the land) |
| `parval` | Total parcel value |
| `improvval` ‡ | Improvement value (assessed value of structures) |
| `agval` ‡ | Agricultural value |
| `sqft` † | Building square footage |
| `totalarea` ‡ | Total building area (sq ft) of the existing structure |
| `existing_heatedarea` ‡ | Heated living area (sq ft) of the current structure |
| `existing_bathrooms` ‡ | Number of bathrooms in the current structure |
| `existing_bedrooms` ‡ | Number of bedrooms in the current structure |
| `new_heatedarea` ‡ | Heated area (sq ft) of new permitted construction |
| `new_totalarea` ‡ | Total area (sq ft) of new permitted construction |
| `new_bathrooms` ‡ | Number of bathrooms in new permitted construction |
| `new_bedrooms` ‡ | Number of bedrooms in new permitted construction |
| `usecode` ‡ | Land use code assigned by the county assessor |
| `zoning` † | Local zoning code (possible-use category) |
| `lbcs_activity` † | LBCS Activity code — actual activity on the land |
| `lbcs_function` † | LBCS Function code — economic function |
| `lbcs_structure` † | LBCS Structure code — type of structure |
| `highest_parcel_elevation` | Highest elevation (m) intersecting the parcel (USGS 10 m DEM) |
| `lowest_parcel_elevation` | Lowest elevation (m) intersecting the parcel (USGS 10 m DEM) |
| `roughness_rating` | Within-parcel elevation variability class (USGS 10 m DEM) |
| `transmission_line_distance` | Distance (m) to nearest 69–765 kV transmission line |
| `fema_nri_risk_rating` † | FEMA National Risk Index (NRI) rating |
| `nhc_in_tidal_zone` † | Whether the parcel lies in the NHC tidal zone (0/1) |
| `population_density` | Population density (per sq mi), CBG level |
| `population_growth_past_5_years` | CAGR of population over the past 5 years, CBG level |
| `population_growth_next_5_years` | Projected CAGR of population over the next 5 years, CBG level |
| `housing_growth_past_5_years` | CAGR of housing units over the past 5 years, CBG level |
| `housing_growth_next_5_years` | Projected CAGR of housing units over the next 5 years, CBG level |
| `household_income_growth_next_5_years` | Projected CAGR of median household income over the next 5 years, CBG level |
| `median_household_income` | Median household income, previous year, CBG level |
| `housing_affordability_index` | Housing Affordability Index (HAI), CBG level |

`H` additionally encodes `pre_hurricane_repair_count` and the storm damage fields
(see *Damage fields*). Regrid columns present in the source but not encoded in
either embedding (`numstories`, `usedesc`, `owntype`, and the `parcelnumb` /
`census_blockgroup` identifiers) are omitted.

### E.2 HCAD schema (Harvey)

These are the HCAD source attributes **actually encoded into Harvey's `H`** after
preprocessing (raw HCAD columns that the pipeline drops as redundant — e.g.
`hcad_improvval`, `hcad_tot_mkt_val`, `hcad_year_built`, `hcad_acreage`,
`hcad_im_sqft`, `hcad_heated_area`, `hcad_bedrooms`, `hcad_full_baths`,
`hcad_neighborhood`, `existing_heatedarea` — are **not** listed). Valuation and
area fields are log-scaled and year fields are converted to age before encoding.

| field | description |
|-------|-------------|
| `lat` | Latitude of parcel centroid (WGS-84) — location is encoded in `H`; exact coordinates are not released as a raw value |
| `lon` | Longitude of parcel centroid (WGS-84) — as above |
| `yearbuilt` | Year the structure was originally built (encoded as building age) |
| `improvval` | Improvement value (assessed value of structures) |
| `landval` | Market land value |
| `parval` | Total parcel value |
| `hcad_assessed_val` | Assessed value |
| `hcad_tot_appr_val` | Appraised value |
| `hcad_tot_rcn_val` | Total building replacement-cost value |
| `hcad_x_features_val` | Extra-feature value (e.g. detached garage, pool) |
| `hcad_taxable_val` | Taxable value |
| `sqft` | Building square footage |
| `totalarea` | Total parcel area (land and improvements), sq ft |
| `hcad_bld_ar` | Total building area |
| `hcad_land_ar` | Total land area (sq ft) |
| `existing_bathrooms` | Number of bathrooms |
| `existing_bedrooms` | Number of bedrooms |
| `hcad_half_baths` | Number of half bathrooms |
| `hcad_total_rooms` | Total number of rooms |
| `hcad_story_height` | Number of stories |
| `hcad_fireplaces` | Number of fireplaces |
| `hcad_yr_remodel` | Year remodeled (encoded as years since remodel) |
| `hcad_state_class` | HCAD state / property class code |
| `hcad_market_area` | HCAD primary market-area code (valuation grouping) |
| `hcad_school_dist` | Independent School District (ISD) code |
| `population_density` | Population density (per sq mi), CBG level |
| `median_household_income` | Median household income, previous year, CBG level |

Beyond these, `H` also encodes `pre_hurricane_repair_count` (count of the parcel's
pre-landfall repair permits), three engineered interaction features
(value-per-sqft, land-value-to-area ratio, and age × improvement value), and the
storm damage fields (documented in *Damage fields*).
