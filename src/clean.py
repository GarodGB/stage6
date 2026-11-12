"""Clean the raw transactions Excel -> data/processed/transactions_clean.csv.

The pandas copy of the Stage 3 Spark job (spark_clean_job.py): the SAME rules, the SAME sections,
the SAME rules book (config/clean_config.json = Stage 3's pipeline_config.json).
What changed:
  - reads the Excel directly        (Stage 3: load_raw.py -> Postgres raw table -> Spark reads it)
  - writes a CSV                    (Stage 3: upsert into Postgres transactions_cleaned)
  - the run report goes to a JSON   (Stage 3: pipeline_runs + audit_log tables, + a Kafka message)
  - section 5b: the Stage 2 "x100" fix (billing 100x too small -> TXN x FX), which Stage 5 was trained on

Run from the repo root:  python -m src.clean
Exit code 0 = written, 1 = quality gate failed (nothing written).
"""
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

CONFIG_FILE = os.getenv("CLEAN_CONFIG", "config/clean_config.json")

EXPECTED_COLUMNS = [                                   # from load_raw.py
    "ACCOUNT_ID", "TXN_ID", "TXN_DATE_TIME", "SETTLE_DATE", "TXN_TYPE", "TXN_AMOUNT",
    "TXN_CCY", "BILLING_AMOUNT", "BILLING_CURRENCY", "FX_RATE", "MERCHANT_NAME",
    "MCC_CODE", "MERCHANT_COUNTRY", "MERCHANT_CITY", "PROCESSING_CODE",
    "PROCESSING_TYPE", "TERMINAL_ID", "AUTH_CODE", "MATCHES_STATUS",
]


# ---------- small helpers: Spark's null rules, in pandas ----------
def upper_squeeze(s):
    """upper + trim + many spaces -> one space (Spark: regexp_replace(upper(trim(x)), '\\s+', ' '))."""
    return s.map(lambda v: re.sub(r"\s+", " ", str(v).strip(" ").upper(), flags=re.ASCII),
                 na_action="ignore").astype(object)


def boolean(values):
    """pandas 'boolean' = True / False / <NA>, like a Spark boolean with NULL."""
    return pd.array(values, dtype="boolean")


def spark_to_strptime(fmt):
    """Spark date pattern -> Python strptime pattern (yyyy-MM-dd HH:mm -> %Y-%m-%d %H:%M)."""
    for spark, py in [("yyyy", "%Y"), ("MMM", "%b"), ("yy", "%y"), ("MM", "%m"),
                      ("dd", "%d"), ("HH", "%H"), ("mm", "%M"), ("ss", "%S")]:
        fmt = fmt.replace(spark, py)
    return fmt


def main():
    # ---------- 1. open the rules book ----------
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)

    # ---------- 2 + 3. read the raw Excel, as TEXT, untouched (was: load_raw.py + JDBC read) ----------
    src = cfg["source"]
    excel = src["excel_file"]
    raw = pd.read_excel(excel, sheet_name=src["transactions_sheet"], dtype=str)
    mcc_ref = pd.read_excel(excel, sheet_name=src["mcc_sheet"], dtype=str)

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in raw.columns]
    if missing_cols:
        sys.exit(f"STOP: missing columns {missing_cols}")
    raw = raw[EXPECTED_COLUMNS].astype(object).where(raw[EXPECTED_COLUMNS].notna(), None)
    raw.columns = [c.lower() for c in EXPECTED_COLUMNS]
    raw = raw.drop_duplicates("txn_id", keep="first")       # load_raw: ON CONFLICT (txn_id) DO NOTHING

    with open(excel, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    source_file = os.path.basename(excel)
    rows_read = len(raw)
    df = raw.copy()

    print("config version:", cfg["config_version"])
    print("rows read:", rows_read)

    # ---------- 4. clean the dates ----------
    d = cfg["dates"]
    formats = [spark_to_strptime(f) for f in d["formats"]]

    def parse_date(v):
        """Epoch millis first, then every format; a year before min_year -> None."""
        if v is None:
            return None
        t = str(v).strip(" ")
        ts = None
        if re.fullmatch(r"[0-9]{13}", t):
            ts = datetime.fromtimestamp(int(t) / 1000, tz=timezone.utc).replace(tzinfo=None)
        else:
            for fmt in formats:
                try:
                    ts = datetime.strptime(t, fmt)
                    break
                except ValueError:
                    continue
        return ts if ts is not None and ts.year >= d["min_year"] else None

    df["txn_date_clean"] = df["txn_date_time"].map(parse_date).astype(object)
    df["settle_date_clean"] = df["settle_date"].map(parse_date).astype(object)

    def unparseable(raw_col, clean_col):
        """had a value, but we could not read it"""
        return int((df[raw_col].notna() & df[clean_col].isna()).sum())

    print("txn_date unparseable:", unparseable("txn_date_time", "txn_date_clean"))
    print("settle_date unparseable:", unparseable("settle_date", "settle_date_clean"))
    print("settle_date empty:", int(df["settle_date"].isna().sum()))

    # ---------- 5. clean the amounts ----------
    FOUR = Decimal("0.0001")
    TWELVE = Decimal("0.000000000001")

    def parse_amount(v):
        """Dirty money text -> Decimal with 4 decimals, or None (Spark: cast to decimal(18,4))."""
        if v is None:
            return None
        t = str(v).strip(" ")
        negative = t.startswith("(") and t.endswith(")")
        digits = re.sub(r"[^0-9.,-]", "", t)
        if re.search(r",[0-9]*$", digits):                 # European: 176.711,00
            fixed = digits.replace(".", "").replace(",", ".")
        else:                                              # US: 1,250.00
            fixed = digits.replace(",", "")
        try:
            number = Decimal(fixed).quantize(FOUR, rounding=ROUND_HALF_UP)
        except InvalidOperation:
            return None
        if abs(number) >= Decimal("1e14"):                # does not fit decimal(18,4) -> NULL
            return None
        if number == 0:                                    # Spark has no "-0"
            return abs(number)
        return -abs(number) if negative else number

    def parse_fx(v):
        """Spark: trim -> double -> decimal(24,12)."""
        if v is None:
            return None
        t = str(v).strip()
        if "_" in t:                                       # Python float() allows 1_000, Spark does not
            return None
        try:
            x = float(t)
        except ValueError:
            return None
        if x != x or abs(x) >= 1e12:                       # NaN, infinity, or too big for decimal(24,12)
            return None
        return Decimal(repr(x)).quantize(TWELVE, rounding=ROUND_HALF_UP)

    df["txn_amount_clean"] = df["txn_amount"].map(parse_amount).astype(object)
    df["billing_amount_clean"] = df["billing_amount"].map(parse_amount).astype(object)
    df["fx_rate_clean"] = df["fx_rate"].map(parse_fx).astype(object)

    print("txn_amount unparseable:", unparseable("txn_amount", "txn_amount_clean"))
    print("billing_amount unparseable:", unparseable("billing_amount", "billing_amount_clean"))
    print("fx_rate unparseable:", unparseable("fx_rate", "fx_rate_clean"))

    # 5b. Stage 2 rule (not in the Spark job): billing 100x too small -> recalculate it as TXN x FX
    x = cfg["amounts"]["billing_x100"]
    CENTS = Decimal("0.01")

    def is_x100(t, f, b):
        """billing x 100 is (almost) TXN x FX: the billing was divided by 100 by mistake."""
        if t is None or f is None or b is None:
            return False
        expected = t * f
        if expected == 0:
            return False
        return abs(b * 100 - expected) <= Decimal(str(x["tolerance_abs"])) + Decimal(str(x["tolerance_pct"])) * abs(expected)

    x100 = [is_x100(t, f, b) for t, f, b in zip(df["txn_amount_clean"], df["fx_rate_clean"], df["billing_amount_clean"])]
    df["billing_fixed"] = boolean(x100)
    df.loc[df["billing_fixed"].to_numpy(dtype=bool), "billing_amount_clean"] = [
        (t * f).quantize(CENTS, rounding=ROUND_HALF_EVEN).quantize(FOUR)
        for t, f, fixed in zip(df["txn_amount_clean"], df["fx_rate_clean"], x100) if fixed]
    print("billing 100x too small, recalculated as TXN x FX:", int(df["billing_fixed"].sum()))

    # ---------- 6. clean the merchant name ----------
    m = cfg["merchant"]
    with open(m["map_file"]) as f:
        merchant_map = json.load(f)
    JUNK = [r"\s*\*\w+$", r"\s+\d{4,}$", r"\s*#\d+$", r"\s+BR(\s*\d+)?$", r"\s*/\w+$"]

    def clean_merchant(name):
        """-> (merchant_clean, payment_via)"""
        if name is None:
            return None, "DIRECT"
        before_star = name.split("*", 1)[0].strip(" ")
        has_star = "*" in name
        is_prefix = has_star and before_star in m["known_prefixes"]
        via = m["prefix_rename"].get(before_star, before_star) if is_prefix else "DIRECT"
        if is_prefix:
            merchant = name.split("*", 1)[1].strip(" ")
        elif has_star:
            merchant = before_star
        else:
            merchant = name
        for _ in range(2):
            for pattern in JUNK:
                merchant = re.sub(pattern, "", merchant, flags=re.ASCII)
        short = merchant.strip(" ")
        return merchant_map.get(short, short), via

    names = upper_squeeze(df["merchant_name"])
    cleaned = [clean_merchant(None if pd.isna(n) else n) for n in names]
    df["merchant_clean"] = pd.Series([c[0] for c in cleaned], index=df.index, dtype=object)
    df["payment_via"] = pd.Series([c[1] for c in cleaned], index=df.index, dtype=object)

    print("distinct merchants: raw", df["merchant_name"].nunique(dropna=False),
          "-> clean", df["merchant_clean"].nunique(dropna=False))
    print(df["payment_via"].value_counts().to_string())

    # ---------- 7. clean the city and the country ----------
    c = cfg["city"]
    city = upper_squeeze(df["merchant_city"])
    df["is_online"] = boolean(city.isin(c["online_words"]))          # NULL city -> False (coalesce)

    def clean_city(v, online):
        if v is None or pd.isna(v) or v == "":
            return c["unknown_value"]
        if online:
            return c["online_value"]
        return c["map"].get(v, v)

    df["merchant_city_clean"] = [clean_city(v, o) for v, o in zip(city, df["is_online"])]

    country = upper_squeeze(df["merchant_country"])
    df["merchant_country_clean"] = [c["unknown_value"] if (v is None or pd.isna(v) or v == "") else v
                                    for v in country]

    with open(c["country_file"]) as f:
        city_country = json.load(f)
    expected = df["merchant_city_clean"].map(city_country)            # city not in the file -> NaN
    df["flag_country_mismatch"] = boolean([bool(e == e and cc != e)   # coalesce(..., False)
                                           for cc, e in zip(df["merchant_country_clean"], expected)])

    print("online:", int(df["is_online"].sum()),
          "| unknown city:", int((df["merchant_city_clean"] == c["unknown_value"]).sum()),
          "| country mismatch:", int(df["flag_country_mismatch"].sum()))

    # ---------- 8. terminal, auth code, MCC category ----------
    terminal = df["terminal_id"].map(lambda v: str(v).strip(" ").upper(), na_action="ignore")
    missing_terminal = terminal.isna() | (terminal == "") | terminal.isin(cfg["terminal"]["missing_values"])
    df["has_terminal"] = boolean(~missing_terminal)
    df["terminal_id_clean"] = terminal.where(~missing_terminal, None).astype(object)

    auth = df["auth_code"].map(lambda v: str(v).strip(" ").upper(), na_action="ignore")
    fake_auth = auth.isna() | (auth == "") | auth.isin(cfg["auth"]["fake_codes"])
    df["flag_auth_suspicious"] = boolean(fake_auth)
    df["auth_code_clean"] = auth.where(~fake_auth, None).astype(object)

    mcc_map = dict(zip(mcc_ref["MCC_CODE"], mcc_ref["CATEGORY"]))
    df["mcc_category"] = df["mcc_code"].map(lambda v: mcc_map.get(str(v).strip(" ")), na_action="ignore").astype(object)

    print("no terminal:", int((~df["has_terminal"]).sum()),
          "| fake auth:", int(df["flag_auth_suspicious"].sum()),
          "| mcc without category:", int(df["mcc_category"].isna().sum()))

    # ---------- 9. the 13 flags ----------
    r = cfg["rules"]
    a = cfg["amounts"]
    txn_day = df["txn_date_clean"].map(lambda t: t.date(), na_action="ignore")
    settle_day = df["settle_date_clean"].map(lambda t: t.date(), na_action="ignore")
    txn, bill, fx = df["txn_amount_clean"], df["billing_amount_clean"], df["fx_rate_clean"]

    def both(x, y, fn):
        """fn(x, y) row by row; NULL if either side is NULL (Spark's rule)."""
        return boolean([None if (xi is None or yi is None or pd.isna(xi) or pd.isna(yi)) else fn(xi, yi)
                        for xi, yi in zip(x, y)])

    def coalesce_false(arr):
        return arr.fillna(False)

    def sign(v):
        return (v > 0) - (v < 0)

    pct, absolute = Decimal(str(a["fx_tolerance_pct"])), Decimal(str(a["fx_tolerance_abs"]))
    amount_mismatch = [None if (t is None or f is None or b is None)
                       else abs(t * f - b) > pct * abs(b) + absolute
                       for t, f, b in zip(txn, fx, bill)]

    # the "usual" MCC of each merchant = the MCC it uses most often (ties -> smallest code, NULL first)
    counts = (df.groupby(["merchant_clean", "mcc_code"], dropna=False).size()
                .reset_index(name="n")
                .sort_values(["merchant_clean", "n", "mcc_code"], ascending=[True, False, True],
                             na_position="first"))
    usual = counts.drop_duplicates("merchant_clean").dropna(subset=["merchant_clean"])
    usual_mcc = df["merchant_clean"].map(dict(zip(usual["merchant_clean"], usual["mcc_code"])))

    df["flag_settle_missing"] = boolean(df["settle_date_clean"].isna())
    df["flag_settle_before_txn"] = coalesce_false(both(settle_day, txn_day, lambda s, t: s < t))
    df["flag_settle_lag"] = coalesce_false(both(settle_day, txn_day,
                                                lambda s, t: (s - t).days > d["max_settle_lag_days"]))
    df["flag_sign_mismatch"] = both(txn, bill, lambda t, b: sign(t) != sign(b))
    df["flag_amount_mismatch"] = coalesce_false(boolean(amount_mismatch))
    same_ccy = both(df["txn_ccy"], df["billing_currency"], lambda x, y: x == y)
    df["flag_same_ccy_diff"] = same_ccy & both(txn, bill, lambda t, b: t != b)
    df["flag_status_not_matched"] = both(df["matches_status"], pd.Series(r["matched_status"], index=df.index),
                                         lambda s, ok: s != ok)
    is_atm = both(df["processing_type"], pd.Series(r["atm_processing_type"], index=df.index), lambda p, x: p == x)
    df["flag_atm_mcc_mismatch"] = is_atm & both(df["mcc_code"], pd.Series(r["atm_mcc"], index=df.index),
                                               lambda m_, x: m_ != x)
    df["flag_mcc_inconsistent"] = both(df["mcc_code"], usual_mcc, lambda m_, u: m_ != u)
    df["flag_city_unknown"] = boolean(df["merchant_city_clean"] == c["unknown_value"])
    df["flag_amount_outlier"] = both(bill, pd.Series(Decimal(str(a["outlier_usd"])), index=df.index),
                                     lambda b, limit: abs(b) > limit)

    FLAGS = [col for col in df.columns if col.startswith("flag_")]
    df["flags_count"] = df[FLAGS].astype("Int64").sum(axis=1, skipna=False)   # any NULL flag -> NULL

    flag_counts = {f: int(df[f].sum()) for f in FLAGS}
    for name, n in flag_counts.items():
        print(f"{name:26} {n}")
    print("rows with at least 1 flag:", int((df["flags_count"] > 0).sum()), "of", len(df))

    # ---------- 10. input checks + quality gate ----------
    q = cfg["quality_gate"]
    before_dedupe = len(df)
    df = df.drop_duplicates("txn_id", keep="first")       # repeats of the same txn_id (none today)
    rows_cleaned = len(df)
    rows_dropped = before_dedupe - rows_cleaned

    def bad(cond):
        """count rows where the condition is true; NULL counts as bad (Spark: coalesce(cond, True))."""
        return int(pd.array(cond, dtype="boolean").fillna(True).sum())

    txn_day = df["txn_date_clean"].map(lambda t: t.date(), na_action="ignore")
    start = datetime.strptime(d["window_start"], "%Y-%m-%d").date()
    end = datetime.strptime(d["window_end"], "%Y-%m-%d").date()
    checks = {
        "null_txn_id": bad(df["txn_id"].isna()),
        "bad_account_id": bad([None if v is None or pd.isna(v) else not re.fullmatch(r"[0-9]{9}", str(v))
                               for v in df["account_id"]]),
        "date_outside_window": bad([None if t is None or pd.isna(t) else (t < start or t > end) for t in txn_day]),
        "missing_txn_date": bad(df["txn_date_clean"].isna()),
        "missing_txn_amount": bad(df["txn_amount_clean"].isna()),
        "missing_billing_amount": bad(df["billing_amount_clean"].isna()),
        "missing_fx_rate": bad(df["fx_rate_clean"].isna()),
        "missing_currency": bad(df["txn_ccy"].isna()),
    }
    rows_flagged = int((df["flags_count"] > 0).sum())

    # ---- the gate: collect every problem, then decide ----
    problems = []
    if rows_read != rows_cleaned + rows_dropped:
        problems.append("counts do not add up")
    for name, n in checks.items():
        if n > q["max_unparseable_required"]:
            problems.append(f"{name}: {n} rows")
    for name, n in flag_counts.items():
        rate = n / rows_cleaned
        if rate > q["max_flag_rate"]:
            problems.append(f"{name}: {rate:.1%} of rows (limit {q['max_flag_rate']:.0%})")

    report = {
        "rows_read": rows_read, "rows_cleaned": rows_cleaned,
        "rows_dropped": rows_dropped, "rows_flagged": rows_flagged,
        "billing_fixed": int(df["billing_fixed"].sum()),
        "checks": checks, "flag_counts": flag_counts, "problems": problems,
    }
    print(json.dumps(report, indent=2))

    if problems:
        print("QUALITY GATE FAILED -> nothing is written")
        sys.exit(1)
    print("QUALITY GATE PASSED")

    # ---------- 11. save: write the clean CSV, then the bookkeeping ----------
    # 11a. the run id: same file + same config  ->  same run id
    run_id = hashlib.sha256(f"{file_hash}|{cfg['config_version']}".encode()).hexdigest()[:16]
    print("run_id:", run_id)

    # 11b. the final table: keep only clean values, with the final column names
    def stamp(t):
        return None if t is None or pd.isna(t) else t.strftime("%Y-%m-%d %H:%M:%S")

    cleaned_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    final = pd.DataFrame({
        "txn_id": df["txn_id"], "account_id": df["account_id"],
        "txn_date_time": df["txn_date_clean"].map(stamp),
        "settle_date": df["settle_date_clean"].map(stamp),
        "txn_type": df["txn_type"],
        "txn_amount": df["txn_amount_clean"],
        "txn_ccy": df["txn_ccy"],
        "billing_amount": df["billing_amount_clean"],
        "billing_currency": df["billing_currency"],
        "fx_rate": df["fx_rate_clean"],
        "merchant_name": df["merchant_clean"],
        "payment_via": df["payment_via"], "mcc_code": df["mcc_code"], "mcc_category": df["mcc_category"],
        "merchant_country": df["merchant_country_clean"],
        "merchant_city": df["merchant_city_clean"],
        "processing_code": df["processing_code"], "processing_type": df["processing_type"],
        "terminal_id": df["terminal_id_clean"],
        "auth_code": df["auth_code_clean"],
        "matches_status": df["matches_status"], "is_online": df["is_online"], "has_terminal": df["has_terminal"],
        "billing_fixed": df["billing_fixed"],
        **{f: df[f] for f in FLAGS}, "flags_count": df["flags_count"],
        "run_id": run_id, "cleaned_at": cleaned_at, "config_version": cfg["config_version"],
    })

    # 11c. write the CSV safely: write a temp file, then swap it in (a crash never leaves half a file)
    out = Path(cfg["output"]["clean_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".csv.tmp")
    final.to_csv(tmp, index=False)
    os.replace(tmp, out)
    print("written to", out, ":", rows_cleaned, "rows")

    # 11d. reconcile: does the file now hold exactly what we produced?
    expected_sum = sum(final["txn_amount"])
    back = pd.read_csv(out, dtype={"txn_amount": str})
    file_count, file_sum = len(back), sum(Decimal(v) for v in back["txn_amount"])
    if file_count != rows_cleaned or file_sum != expected_sum:
        raise RuntimeError(f"reconcile failed: file {file_count}/{file_sum} vs pandas {rows_cleaned}/{expected_sum}")

    # 11e. the run report (was: pipeline_runs + audit_log)
    run = {
        "run_id": run_id, "status": "SUCCESS", "source_file": source_file, "file_hash": file_hash,
        "config_version": cfg["config_version"], "target_file": str(out),
        **report, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    rep = Path(cfg["output"]["report"])
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(run, indent=2))

    print(f"file has {file_count} rows for this run | sum {file_sum}")
    print("SAVED OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
