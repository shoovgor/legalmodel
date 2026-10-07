import numpy as np
import pandas as pd

FEATURES_NUM = [
    "claim_amount_log", "claim_amount_missing", "judge_case_count",
    "narrative_missing", "narrative_length_log",
    "has_interest_rate", "has_institutional_lender", "has_app_lending",
    "has_collateral", "has_guarantor", "has_gov_fund",
    "has_uncontested", "has_settlement", "has_withdrawal",
    "has_opposition", "has_evidence_dispute", "has_default",
    "has_multiple_claims",
]
FEATURES_CAT = ["plaintiff_type", "defendant_type", "court_type", "party_type_interaction", "claim_amount_bucket"]
TEXT_COL = "narrative_filled"
ALL_FEATURE_COLS = FEATURES_NUM + FEATURES_CAT + [TEXT_COL]


def infer_court_type(court):
    return "urban" if "дүүргийн" in str(court) else "rural"


def build_narrative_flags(narrative_filled):
    """narrative_filled: a pandas Series of strings (already .fillna(''))."""
    return pd.DataFrame({
        "has_interest_rate": narrative_filled.str.contains(r"хүү|хувь.{0,10}тэй", regex=True).astype(int),
        "has_institutional_lender": narrative_filled.str.contains(r"ХХК|ББСБ", regex=True).astype(int),
        "has_app_lending": narrative_filled.str.contains(r"аппликейшн|апп\b", regex=True).astype(int),
        "has_collateral": narrative_filled.str.contains(r"барьцаа", regex=True).astype(int),
        "has_guarantor": narrative_filled.str.contains(r"батлан", regex=True).astype(int),
        "has_gov_fund": narrative_filled.str.contains(r"хөгжүүлэх сан|Засаг дарга", regex=True).astype(int),
        "has_uncontested": narrative_filled.str.contains(r"зөвшөөрч байна|хүлээн зөвшөөр", regex=True).astype(int),
        "has_settlement": narrative_filled.str.contains(r"эвлэрэх|эвлэрлийн|эвлэрсэн", regex=True).astype(int),
        "has_withdrawal": narrative_filled.str.contains(r"татгалз|хүсэлтээ буцаа", regex=True).astype(int),
        "has_opposition": narrative_filled.str.contains(r"эсэргүүцэ|маргаж|татгалзаж байна", regex=True).astype(int),
        "has_evidence_dispute": narrative_filled.str.contains(r"нотлох баримт.*хангалтгүй|нотолгоогүй", regex=True).astype(int),
        "has_default": narrative_filled.str.contains(r"ирээгүй|хариу тайлбар ирүүлээгүй", regex=True).astype(int),
        "has_multiple_claims": narrative_filled.str.contains(r"нэгдүгээрт.*хоёрдугаарт", regex=True).astype(int),
    })


def build_training_features(fi, median_claim=None, judge_counts=None):
    fi = fi.copy()
    if median_claim is None:
        median_claim = fi.claim_amount_requested.median()
    if judge_counts is None:
        judge_counts = fi.judge.value_counts()

    fi["court_type"] = fi.court.apply(infer_court_type)
    fi["judge_case_count"] = fi.judge.map(judge_counts).fillna(judge_counts.median())
    fi["claim_amount_missing"] = fi.claim_amount_requested.isna().astype(int)
    fi["claim_amount_log"] = np.log1p(fi.claim_amount_requested.fillna(median_claim))
    fi["plaintiff_type"] = fi.plaintiff_type.fillna("unknown")
    fi["defendant_type"] = fi.defendant_type.fillna("unknown")
    fi["party_type_interaction"] = fi["plaintiff_type"] + "_vs_" + fi["defendant_type"]
    fi["claim_amount_bucket"] = pd.cut(fi.claim_amount_requested.fillna(median_claim), bins=[-np.inf, 1000000, 10000000, np.inf], labels=["low", "medium", "high"]).astype(str)
    fi["narrative_missing"] = fi.case_narrative.isna().astype(int)
    narrative = fi.case_narrative.fillna("")
    fi["narrative_length_log"] = np.log1p(narrative.str.len())
    fi[TEXT_COL] = narrative
    fi = pd.concat([fi, build_narrative_flags(narrative)], axis=1)

    return fi[ALL_FEATURE_COLS], median_claim, judge_counts


def build_case_features(claim_amount, plaintiff_type, defendant_type, court_type,
                         narrative_text, judge_name=None, median_claim=None, judge_counts=None):
    narrative_filled = pd.Series([narrative_text or ""])
    if judge_name is not None and judge_counts is not None and judge_name in judge_counts.index:
        jcc = judge_counts[judge_name]
    elif judge_counts is not None:
        jcc = judge_counts.median()
    else:
        jcc = 24.0 

    row = pd.DataFrame({
        "claim_amount_log": [np.log1p(claim_amount if claim_amount is not None else (median_claim or 0))],
        "claim_amount_missing": [1 if claim_amount is None else 0],
        "judge_case_count": [jcc],
        "narrative_missing": [1 if not narrative_text else 0],
        "narrative_length_log": [np.log1p(len(narrative_text or ""))],
        "plaintiff_type": [plaintiff_type or "unknown"],
        "defendant_type": [defendant_type or "unknown"],
        "court_type": [court_type],
        "party_type_interaction": [f"{plaintiff_type or 'unknown'}_vs_{defendant_type or 'unknown'}"],
        "claim_amount_bucket": [pd.cut(pd.Series([claim_amount if claim_amount is not None else (median_claim or 0)]), bins=[-np.inf, 1000000, 10000000, np.inf], labels=["low", "medium", "high"]).astype(str).iloc[0]],
        TEXT_COL: [narrative_text or ""],
    })
    row = pd.concat([row, build_narrative_flags(narrative_filled)], axis=1)
    return row[ALL_FEATURE_COLS]
