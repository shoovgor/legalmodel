#!/usr/bin/env python3
"""
shuukh_parser_v2.py — parser for the live.shuukh.mn plain-text export
format (different from the HTML-as-.doc format shuukh_parser.py handles).

Usage:
    python3 shuukh_parser_v2.py <input_folder> <output_csv>

KEY DIFFERENCE FROM v1: this format has a labeled metadata block at the
top (Шүүх / Шүүгч / Дугаар / Огноо / Маргааны төрөл) — case_id and date
come straight from there, no regex-on-prose needed.

BIG FINDING FROM THE FIRST 11 SAMPLES: 8 of 11 are not actually
adjudicated disputes at all. "Маргааны төрөл: Нэхэмжлэгчээс гаргасан
хүсэлтийн дагуу хариуцагчийг эрэн сурвалжлах" is a procedural
locate-the-defendant request (court orders police to find an address so
the real case can proceed) — there's no claim being granted/denied, it's
a formality. The other 3 are genuine debt/contract disputes resolved by
SETTLEMENT ("эвлэрэл"), a third outcome type alongside granted/denied,
using a different operative marker (ЗАХИРАМЖЛАХ нь:, not ТОГТООХ нь:).

The 'Маргааны төрөл' field is the cheap, reliable filter for this —
it's already labeled by the source, no heuristic needed.

VERSION 3 UPDATE: Now handles ALL court levels:
  - first_instance: original extraction (outcome, award, petition, etc.)
  - appellate (давж заалдах шатны): extracts appeal_outcome (upheld /
    overturned / modified / settled / dismissed), the referenced
    first-instance decision number, the appealing party, and claim
    amounts where available in the header text.
  - supervisory (Улсын дээд шүүх / хяналтын): same as appellate but
    for the Supreme Court review level.

Data from all 3 levels is written to a single CSV. First-instance
civil_outcome uses the original granted/denied/settled/partial taxonomy;
appellate and supervisory get appeal_outcome in a dedicated column.
"""

import re
import csv
import sys
from pathlib import Path

PROCEDURAL_DISPUTE_TYPES = [
    "Нэхэмжлэгчээс гаргасан хүсэлтийн дагуу хариуцагчийг эрэн сурвалжлах",  # locate-the-defendant request - not a claim outcome
]


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_metadata(text, warnings):
    """
    Anchored to the metadata block specifically (after 'Шийдвэрийн
    мэдээлэл', with a tab between label and value) - NOT a bare
    label-anywhere-in-text search. Found a real bug testing this: the
    page's nav menu has a line 'Шүүх хуралдаан' (Court session) that a
    loose search for 'Шүүх' matches before ever reaching the real
    metadata field, silently returning the wrong value.
    """
    block_start = text.find("Шийдвэрийн мэдээлэл")
    search_text = text[block_start:] if block_start != -1 else text

    fields = {}
    for label, key in [
        ("Шүүх", "court"), ("Шүүгч", "judge"), ("Хэргийн индекс", "case_index"),
        ("Дугаар", "case_id"), ("Огноо", "decision_date"), ("Маргааны төрөл", "dispute_type"),
    ]:
        m = re.search(rf"{label}[ \t]+(.+)", search_text)
        fields[key] = m.group(1).strip().rstrip(",") if m else None
        if fields[key] is None:
            warnings.append(f"meta_{key}_missing")
            
    # If case_id is just a number (e.g. '1019' or '01584'), it loses uniqueness 
    # and gets mangled by Excel. Fallback to case_index if available.
    if fields.get("case_id") and re.match(r"^\d+$", fields["case_id"]):
        if fields.get("case_index"):
            fields["case_id"] = fields["case_index"]
        else:
            # Prefix with a tick to force Excel to treat it as text if we must keep it
            fields["case_id"] = f"'{fields['case_id']}"
            
    return fields


def normalize_mongolian_number(s):
    """
    Two number formats found in real data, varying by judge/court, not
    by any rule we can predict: '3,000,000' (commas=thousands) from one
    judge, '3.252.626,03' (periods=thousands, comma=decimal) from
    another. Naive comma-stripping breaks the second format outright
    (turns it into '3.252.626.03', not a valid number). Detects which
    format a given string is in before converting.
    """
    s = s.strip()
    if re.match(r"^\d{1,3}(\.\d{3})+,\d+$", s):
        return float(s.replace(".", "").replace(",", "."))
    if re.match(r"^\d{1,3}(\.\d{3})+$", s):
        return float(s.replace(".", ""))
    if re.match(r"^\d{1,3}(,\d{3})+(\.\d+)?$", s):
        return float(s.replace(",", ""))
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def is_procedural(dispute_type):
    if not dispute_type:
        return False
    return any(p in dispute_type for p in PROCEDURAL_DISPUTE_TYPES)


def get_operative_section(text):
    """Two possible markers: ТОГТООХ нь: (contested rulings) or
    ЗАХИРАМЖЛАХ нь: (single-judge orders, e.g. settlement approval).
    'нь' and the colon are both sometimes missing from the scraped
    text, and literal asterisks (redaction/scraping artifact) sometimes
    stand in for whitespace between the marker word and 'нь' - confirmed
    against 19 real cases. Doesn't catch the rarer case of asterisks
    spliced into the middle of the marker word itself (e.g.
    'ЗАХИРА***ЖЛАХ') - left as a known residual.

    First attempt made BOTH нь and : optional at once, which let
    re.search match the bare marker word wherever it appeared FIRST -
    including earlier reasoning text or law citations unrelated to the
    actual ruling. That broke 56 previously-correct cases (confirmed by
    re-running against the real corpus and diffing case-by-case against
    the prior output). Fix: require at least one of нь/: to still be
    present per candidate match (not both simultaneously optional), and
    take the LAST matching occurrence in the document rather than the
    first - the true operative marker is the final one before the
    ruling; any earlier bare-word appearances are false positives from
    reasoning text."""
    best_end, best_settlement = None, False
    for marker_word, is_settlement in [(r"ТОГТООХ", False), (r"ЗАХИРАМЖЛАХ", True)]:
        patterns = [
            marker_word + r"[\s*]*(?:нь[\s*]*:?|:)",
            # bare word, neither нь nor : - only safe when anchored by
            # this specific lead-in immediately before it (confirmed as
            # court 137's consistent template across 4 real cases).
            # Without this anchor it's the exact pattern that caused the
            # regression above.
            r"удирдлага\s+болгон\s+" + marker_word,
        ]
        for pattern in patterns:
            for m in re.finditer(pattern, text, re.I):
                if best_end is None or m.end() > best_end:
                    best_end, best_settlement = m.end(), is_settlement
    if best_end is None:
        return None, False
    return text[best_end:], best_settlement


def get_point1_text(op_text):
    if op_text is None:
        return None
    start = re.search(r"^\s*1\s*\.\s*", op_text)
    if start:
        rest = op_text[start.end():]
    else:
        # Confirmed against 55 real cases: the literal '1.' is sometimes
        # just not in the extracted text at all (scraping artifact - a
        # stray \xa0 non-breaking space shows up right where it should
        # be, and text goes straight into ruling content). Not a
        # formatting choice either - '2.' still appears completely
        # normally later in the SAME documents, so the numbering scheme
        # is in use, point 1 specifically is just missing its marker.
        # The operative marker was already found to get here, so
        # whatever comes next has nowhere else to be but point 1.
        rest = op_text
    end = re.search(r"\n\s*\d+\s*\.\s*[А-ЯӨҮ0-9]", rest)
    return rest[:end.start()] if end else rest


def classify_outcome(point1_text, dispute_type, is_settlement_doc, warnings):
    """
    'хэрэгсэхгүй болгосугай' (case dismissed) shows up in genuinely
    DENIED cases, but also in claims the defendant ACCEPTED in full
    (no need to litigate further) and in SETTLED cases - confirmed
    across real examples of all three. Checking that phrase alone would
    misclassify acceptance/settlement as denial. So: check the specific
    phrase describing WHY first, fall back to the ambiguous dismiss
    phrase only when nothing more specific is found.
    """
    if is_procedural(dispute_type):
        return "out_of_scope_procedural"
    if point1_text is None:
        warnings.append("outcome_no_point1")
        return None
    if re.search(r"\bэвлэр\w*", point1_text):
        return "settled"

    # A counter-claim (сөрөг нэхэмжлэл) resolved in the SAME point 1 as
    # the main claim can carry a different outcome than the main claim -
    # confirmed in a real case where a main claim was granted (cash
    # award + an order to deliver goods) while a separate counter-claim
    # for damages was denied, all in one paragraph. A single outcome
    # value per row can't safely represent two different claims with
    # two different results - flagging for manual read, not guessing
    # which one to record.
    if "сөрөг нэхэмжлэл" in point1_text:
        warnings.append("outcome_counterclaim_mixed_not_modeled")
        return None

    # Partial grant: part of a claim awarded, a separate smaller
    # remainder of the SAME claim dismissed - distinct from a full
    # denial. Signal is tight: 'үлдэх <amount> ... хэрэгсэхгүй
    # болгосугай' with the dismiss verb sitting close to the leftover
    # amount, not separated by another claim's worth of text - confirmed
    # against a real case, and confirmed NOT to false-fire on the
    # counter-claim case above (there the gap between the leftover
    # figure and the dismiss verb is 100+ characters of unrelated
    # clauses, so the tight pattern correctly skips it).
    if re.search(r"үлдэх\s*[\d][\d,\.]*\d\s*төгрөг\w*\s*хэрэгсэхгүй болгосугай", point1_text):
        return "partial"

    # Partial grant, looser form: part of a claim awarded (олгож/олгон)
    # and a separate remainder of the SAME claim dismissed, without the
    # tight 'үлдэх ... хэрэгсэхгүй' shorthand above. Confirmed against
    # 17 real cases with an award clause plus a separate dismiss clause
    # for the leftover amount. Checked before the granted checks below
    # for the same reason the tight partial check is checked first: the
    # awarded portion often ALSO contains granted-trigger words.
    if re.search(r"(олгож|олгон)", point1_text) and "хэрэгсэхгүй" in point1_text:
        return "partial"

    # Full acceptance/performance - matched with a bounded character gap
    # rather than fixed literal phrases, after finding a third real
    # variant ('хүлээн зөвшөөрсөн ЗӨВШӨӨРЛИЙГ баталж' - approving the
    # consent that was given - vs. the original 'хүлээн зөвшөөрснийг
    # баталж'). Bounding the gap should generalize better to the next
    # phrasing variant than hardcoding every one seen so far.
    if re.search(r"хүлээн зөвшөөрс\w*.{0,30}баталж", point1_text):
        return "granted"
    if re.search(r"биелүүлс\w*.{0,30}баталж", point1_text):
        return "granted"

    # Defendant's consent/acceptance confirmed and case closed, WITHOUT
    # the 'хүлээн' qualifier above ('хариуцагчийн зөвшөөрлийг баталж' /
    # '...зөвшөөрснийг баталж'). Confirmed against 82 real cases that
    # were all falling through to the dismissed-unspecified catch-all.
    # Labeled separately from plain 'granted' since the defendant never
    # actually contested here - merge into 'granted' downstream if the
    # contested/uncontested distinction doesn't matter for your framing.
    if re.search(r"зөвшөөр\w*.{0,30}баталж", point1_text):
        return "granted_uncontested"

    if re.search(r"\b(хангаж|хүлээн зөвшөөрсүгэй|даалгасугай|олгосугай|хангасугай|сурвалжлуулсугай)\b", point1_text):
        return "granted"
    if "татгалзсугай" in point1_text:
        return "denied"
    if re.search(r"\b(цуцалсугай|тэжээн тэтгүүлсүгэй|тэтгүүлсүгэй|тогтоосугай)\b", point1_text):
        # petition itself granted, just phrased per dispute type:
        # цуцалсугай = divorce dissolved, тэтгүүлсүгэй = support ordered,
        # тогтоосугай = a requested fact (e.g. death/years-employed)
        # confirmed established. None of these carry a lump-sum award the
        # way debt cases do - extract_award_amount correctly returns None
        # for these since no 'X төгрөг ...' pattern will match.
        return "granted"

    # Plaintiff voluntarily withdraws/renounces their OWN claim, court
    # just confirms it and never rules on the merits - a genuinely
    # different outcome from denial. Confirmed against 53 real cases,
    # all 'нэхэмжлэгч ... татгалз(сан/сны/аны)-ыг баталж'. Doesn't
    # collide with the court's own 'татгалзсугай' ruling-verb above -
    # that's the court's own imperative and doesn't co-occur with a
    # nearby 'баталж' confirming a party's action.
    if re.search(r"татгалз\w*.{0,60}баталж", point1_text):
        return "withdrawn"

    # Full denial on the merits: entire claim dismissed citing
    # substantive Civil Code grounds, none of the admission/withdrawal/
    # partial-award language above present. Confirmed against 8 real
    # cases, all citing a Иргэний хууль article number alongside
    # 'бүхэлд нь'.
    if "бүхэлд нь" in point1_text and "хэрэгсэхгүй" in point1_text:
        return "denied"

    if "хэрэгсэхгүй болгосугай" in point1_text:
        # dismissed for a reason we don't have a specific pattern for yet -
        # NOT automatically 'denied', that's a guess. Flag it instead.
        warnings.append("outcome_dismissed_unspecified_reason")
        return None
    warnings.append("outcome_unknown")
    return None


NON_MONETARY_DISPUTE_TYPES = [
    "Гэрлэлт цуцалсан",
    "Хүүхдийн тэтгэлэг гаргуулах",
    "Хүнийг сураггүй алга болсонд тооцох",
    "Хүүхэд үрчилснийг хүчингүйд тооцох",
    "Ажилласан жил тогтоолгох",
]


def is_non_monetary_dispute(dispute_type):
    if not dispute_type:
        return False
    return any(p in dispute_type for p in NON_MONETARY_DISPUTE_TYPES)


def extract_award_amount(point1_text, outcome, warnings, dispute_type=None):
    """
    For 'settled': the final agreed/reduced claim amount.
    For 'granted': the claim amount - which for an accepted-in-full or
    already-paid claim equals the originally requested amount, restated
    in point 1 via '<N> төгрөг гаргуулах тухай' (confirmed across
    multiple real templates, including ones with no separate award
    figure anywhere else in point 1).

    NOT a 'take the first/last number' heuristic - tested that against
    real settlement text and it picked a payment-INSTALLMENT figure
    instead of the actual settled total (a real bug, caught before
    shipping).

    Redacted documents (amount shown as '*') correctly return None too -
    that's a real data gap, not an extraction failure.
    """
    if point1_text is None or outcome not in ("settled", "granted", "partial"):
        return None
    if is_non_monetary_dispute(dispute_type):
        # divorce/support/declaratory-fact cases genuinely have no lump
        # sum (support is an ongoing formula, divorce/declaration carry
        # no amount at all) - this is the correct state, not a gap.
        return None
    if "тэмдэгтийн хураамж" in point1_text:
        warnings.append("award_fee_language_in_point1")
        return None

    if outcome == "partial":
        # The AWARDED portion specifically, not the dismissed remainder -
        # anchored to '<amount> ... гаргуулж ... олго(ж/сугай)', confirmed
        # against one real case so far. Separate from the generic
        # patterns below since those anchor on different phrasing
        # ('шаардлаг', 'багасгаж') that doesn't appear in this structure.
        m = re.search(r"([\d][\d,\.]*\d)\s*төгрөгийг\s*гаргуулж.{0,80}олго\w*", point1_text)
        if m:
            val = normalize_mongolian_number(m.group(1))
            if val is not None:
                return val
        warnings.append("partial_award_amount_not_found")
        return None

    for pattern in [
        r"([\d][\d,\.]*\d)\s*төгрөг\w*\s*шаардлаг",
        r"([\d][\d,\.]*\d)\s*төгрөг\s*болгон багасгаж",
        r"([\d][\d,\.]*\d)\s*төгрөг\s*гаргуулах тухай",
        r"([\d][\d,\.]*\d)\s*төгрөгийг\s*гаргуулж.{0,60}олгосугай",
        r"шаардлага\w*\s*болох\s*([\d][\d,\.]*\d)\s*төгрөг",
    ]:
        m = re.search(pattern, point1_text)
        if m:
            val = normalize_mongolian_number(m.group(1))
            if val is not None:
                return val

    warnings.append("award_amount_not_found" + ("_possible_redaction" if "*" in point1_text else ""))
    return None

def get_case_level(court_field, warnings=None):
    """
    Appellate and supervisory courts have exactly two consistent naming
    patterns nationwide - check those first, they're reliable. First-
    instance courts do NOT have one consistent pattern: Ulaanbaatar
    district courts say '...хэргийн анхан шатны тойргийн шүүх', but
    rural courts are named '...сум дундын шүүх' (inter-soum court) and
    never say 'анхан шатны' at all - confirmed on a real rural case that
    silently fell through as case_level=missing under the old logic.
    So: check for the two specific non-first-instance patterns, and
    default to first_instance otherwise rather than requiring a match.
    """
    if not court_field:
        if warnings is not None:
            warnings.append("case_level_no_court_field")
        return None
    if "давж заалдах шатны" in court_field:
        return "appellate"
    if "Улсын дээд шүүх" in court_field or "хяналтын" in court_field:
        return "supervisory"
    return "first_instance"


# ──────────────────────────────────────────────────────────────────────
# APPELLATE / SUPERVISORY EXTRACTION FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def classify_appeal_outcome(point1_text, warnings):
    """Classify the outcome of an appellate or supervisory decision.

    Unlike first-instance outcomes (granted/denied/settled), appeals
    review a LOWER court's decision and produce one of:
      - upheld:     lower court decision left as-is
                    ("хэвээр үлдээ...")
      - overturned: lower court decision nullified
                    ("хүчингүй болго...")
      - modified:   lower court decision partially changed
                    ("өөрчлөлт оруул..." or "өөрчлөн...")
      - settled:    parties settled during the appeal
                    ("эвлэр...")
      - dismissed:  appeal itself dismissed procedurally
                    ("хэрэгсэхгүй болго..." without upheld/overturned)

    Validated against 200 appellate + 113 supervisory cases: these five
    patterns cover all observed decisions. Checked in priority order so
    that e.g. 'upheld + dismissed the remainder' correctly classifies
    as upheld (the dominant action), not dismissed.
    """
    if point1_text is None:
        warnings.append("appeal_outcome_no_point1")
        return None

    # Upheld: the lower court decision stands. Most common pattern.
    if "хэвээр үлдээ" in point1_text:
        return "upheld"

    # Overturned: the lower court decision is voided, often with a new
    # ruling substituted in the same point or the next.
    if "хүчингүй болго" in point1_text:
        return "overturned"

    # Modified: lower court decision changed in part but not voided.
    if "өөрчлөлт оруул" in point1_text or "өөрчлөн" in point1_text:
        return "modified"

    # Settlement during appeal process
    if re.search(r"\bэвлэр\w*", point1_text):
        return "settled"

    # Dismissed (the appeal itself, not the original claim)
    if "хэрэгсэхгүй болго" in point1_text:
        return "dismissed"

    warnings.append("appeal_outcome_unknown")
    return None


def extract_referenced_decisions(text):
    """Extract case numbers from the 'Холбоотой шийдвэрүүд' (Related
    decisions) section that appears at the top of every appellate and
    supervisory case. These are the first-instance (and appellate, for
    supervisory cases) decision numbers being reviewed.

    Returns a pipe-separated list of decision numbers, e.g.
    '192/ШШ2025/05426|210/МА2025/01618'.
    """
    block = re.search(r"Холбоотой шийдвэрүүд(.*?)(?:МАГАДЛАЛ|ТОГТООЛ|ЗАХИРАМЖ|ХЯНАВАЛ|ТОДОРХОЙЛОХ)",
                      text, re.S)
    if not block:
        return None
    # Match case number patterns: NNN/XX20YY/NNNNN
    nums = re.findall(r"(\d{3}/[А-ЯӨҮа-яөүA-Za-z]{2}\d{4}/\d{4,5})", block.group(1))
    if not nums:
        return None
    seen = []
    for n in nums:
        if n not in seen:
            seen.append(n)
    return "|".join(seen)


def extract_appealing_party(text, level, warnings):
    """Identify WHO filed the appeal/complaint.

    Appellate: 'давж заалдах гомдол' filed by either нэхэмжлэгч
    (plaintiff), хариуцагч (defendant), or their representatives.
    Supervisory: 'хяналтын журмаар гаргасан ... гомдол'.

    Returns 'plaintiff', 'defendant', 'both', or None.
    """
    if level == "appellate":
        pattern = r"давж заалдах\s+(?:журмаар\s+)?гомдл"
    elif level == "supervisory":
        pattern = r"хяналтын\s+журмаар\s+гаргасан.*?гомдл"
    else:
        return None

    m = re.search(pattern, text, re.S)
    if not m:
        # Try to find it in the broader text
        m = re.search(r"гомдлыг.*?(?:хянан хэлэлц|илтгэснээр)", text, re.S)

    if not m:
        warnings.append("appealing_party_not_found")
        return None

    # Look in a window around the match for party keywords
    start = max(0, m.start() - 300)
    end = min(len(text), m.end() + 200)
    window = text[start:end]

    has_plaintiff = bool(re.search(r"нэхэмжлэгч\w*", window, re.I))
    has_defendant = bool(re.search(r"хариуцагч\w*", window, re.I))

    if has_plaintiff and has_defendant:
        return "both"
    if has_plaintiff:
        return "plaintiff"
    if has_defendant:
        return "defendant"
    warnings.append("appealing_party_ambiguous")
    return None


def extract_appeal_claim_amount(text, warnings):
    """Try to extract the disputed monetary amount from appellate /
    supervisory case header text. These cases often state the original
    claim amount in the description line near the top, using the same
    '<N> төгрөг гаргуулах тухай' pattern as first-instance cases.

    Not all appellate cases have this (e.g. non-monetary disputes like
    divorce, or when the text is heavily redacted), so failure to find
    an amount here is NOT a warning — it's expected.
    """
    m = re.search(r"([\d][\d,\.]*\d)\s*төгрөг\s*гаргуулах тухай", text)
    if m:
        val = normalize_mongolian_number(m.group(1))
        if val is not None:
            return val
    # Also try: 'нийт <N> төгрөг гаргуулах' (common in appellate summaries)
    m = re.search(r"нийт\s*([\d][\d,\.]*\d)\s*төгрөг\w*\s*гаргуулах", text)
    if m:
        val = normalize_mongolian_number(m.group(1))
        if val is not None:
            return val
    return None


def get_petition_section(text):
    """
    Primary: ТОДОРХОЙЛОХ нь: -> up to ҺYНДЭСЛЭХ/ТОГТООХ/ЗАХИРАМЖЛАХ.

    Fallback: many real templates skip ТОДОРХОЙЛОХ entirely and embed
    the petition straight inside ҺYНДЭСЛЭХ - but ҺYНДЭСЛЭХ ALSO contains
    the court's own outcome-preview sentence (e.g. '...хэрэгсэхгүй
    болгох нь зүйтэй гэж үзлээ.') BEFORE the formal ruling section.
    Using all of ҺYНДЭСЛЭХ as an input feature would leak that preview.
    So the fallback extracts only the quoted petition span itself -
    'нэхэмжлэлийн шаардлагадаа: ... гэжээ.' - which is just the
    plaintiff's own stated facts and claim, confirmed clean of outcome
    language across real examples.
    """
    start = re.search(r"ТОДОРХОЙЛОХ\s*нь\s*:", text, re.I)
    if start:
        end_candidates = []
        for pattern in [r"ҮНДЭСЛЭХ\s*нь\s*:", r"ТОГТООХ\s*нь\s*:", r"ЗАХИРАМЖЛАХ\s*нь\s*:", r"ХЯНАВАЛ\s*:"]:
            m = re.search(pattern, text, re.I)
            if m and m.start() > start.end():
                end_candidates.append(m.start())
        end = min(end_candidates) if end_candidates else len(text)
        return text[start.end():end].strip()

    quotes = re.findall(r"шаардлагадаа\s*:(.*?)гэжээ\.?", text, re.S)
    if quotes:
        return " ".join(q.strip() for q in quotes)
    return None


def extract_claim_amount_requested(text, warnings, dispute_type=None):
    """
    Searched against the FULL document text (not just the petition
    span) - the case description line near the top
    ('...холбогдох ... үүрэгт <N> төгрөг гаргуулах тухай нэхэмжлэлийг')
    states this reliably across every template seen, including ones
    where the formal petition section is missing entirely.

    Still attempted even for typically-non-monetary dispute types - a
    real divorce case turned out to carry a legitimate 106M property-
    division claim alongside the divorce petition, found by this exact
    pattern. Only the WARNING is suppressed for those types when nothing
    is found, since absence there is the normal case, not a gap.
    """
    m = re.search(r"([\d][\d,\.]*\d)\s*төгрөг\s*гаргуулах тухай", text)
    if m:
        val = normalize_mongolian_number(m.group(1))
        if val is not None:
            return val
    if not is_non_monetary_dispute(dispute_type) and not is_procedural(dispute_type):
        warnings.append("claim_amount_requested_not_found")
    return None


def extract_law_articles(text, warnings):
    """
    Civil rulings cite governing law articles in the lead-in clause
    right before the operative marker (e.g. '...хуулийн 74 дүгээр
    зүйлийн 74.2 ... удирдлага болгон ЗАХИРАМЖЛАХ нь:'), not inside
    point 1 itself - validated against 4 real cases. Format differs
    from the criminal-case pattern: chapter number, then the full
    dotted clause number (e.g. '74 дүгээр зүйлийн 74.2'), not a
    separate paragraph number.
    """
    m = re.search(
        r"(Иргэний хэрэг шүүхэд хянан шийдвэрлэх тухай хуулийн.{0,250}?)"
        r"(?:ЗАХИРАМЖЛАХ|ТОГТООХ)\s*нь\s*:",
        text, re.S,
    )
    if not m:
        warnings.append("law_articles_lead_in_not_found")
        return None
    arts = re.findall(r"\d+\s*д(?:угаар|үгээр) зүйлийн\s*(\d+\.\d+)", m.group(1))
    if not arts:
        warnings.append("law_articles_not_found")
        return None
    seen = []
    for a in arts:
        if a not in seen:
            seen.append(a)
    return "|".join(seen)


def extract_party_type(text, label):
    """label is 'Нэхэмжлэгч' (plaintiff) or 'Хариуцагч' (defendant).
    Looks at the text right after that label for a company-suffix
    keyword vs a personal name pattern. Window widened from 150 to 300
    chars - real addresses before the party name/suffix routinely run
    150+ characters on their own, which could push a company suffix
    past the old window entirely."""
    m = re.search(rf"{label}:?\s*(.{{0,300}})", text, re.S)
    if not m:
        return None
    window = m.group(1)
    if re.search(r"ХХК|ХК|ББСБ|ТӨХК|компани", window):
        return "company"
    return "individual"


def parse_file(path: Path) -> dict:
    text = load_text(path)
    warnings = []
    meta = extract_metadata(text, warnings)
    level = get_case_level(meta["court"], warnings)

    # Common fields for ALL court levels
    base = {
        "source_file": path.name,
        "case_id": meta["case_id"],
        "case_index": meta["case_index"],
        "decision_date": meta["decision_date"],
        "court": meta["court"],
        "judge": meta["judge"],
        "case_level": level,
        "dispute_type": meta["dispute_type"],
        "plaintiff_type": extract_party_type(text, "Нэхэмжлэгч"),
        "defendant_type": extract_party_type(text, "Хариуцагч"),
    }

    if level in ("appellate", "supervisory"):
        # ── APPELLATE / SUPERVISORY ──────────────────────────────
        op_text, _ = get_operative_section(text)
        point1 = get_point1_text(op_text)
        appeal_outcome = classify_appeal_outcome(point1, warnings)
        referenced = extract_referenced_decisions(text)
        appealing_party = extract_appealing_party(text, level, warnings)
        claim_amount = extract_appeal_claim_amount(text, warnings)
        law_articles = extract_law_articles(text, warnings)

        base.update({
            "claim_amount_requested": claim_amount,
            "law_articles": law_articles,
            "case_narrative": None,
            "civil_outcome": None,
            "civil_award_amount": None,
            "appeal_outcome": appeal_outcome,
            "referenced_decisions": referenced,
            "appealing_party": appealing_party,
            "parse_warnings": "|".join(warnings) if warnings else "",
        })
        return base

    # ── FIRST INSTANCE ───────────────────────────────────────────
    op_text, is_settlement_doc = get_operative_section(text)
    point1 = get_point1_text(op_text)
    outcome = classify_outcome(point1, meta["dispute_type"], is_settlement_doc, warnings)
    award = extract_award_amount(point1, outcome, warnings, meta["dispute_type"])

    petition = get_petition_section(text)
    claim_requested = extract_claim_amount_requested(text, warnings, meta["dispute_type"])
    law_articles = extract_law_articles(text, warnings)

    base.update({
        "claim_amount_requested": claim_requested,
        "law_articles": law_articles,
        "case_narrative": petition,
        "civil_outcome": outcome,
        "civil_award_amount": award,
        "appeal_outcome": None,
        "referenced_decisions": None,
        "appealing_party": None,
        "parse_warnings": "|".join(warnings) if warnings else "",
    })
    return base


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 shuukh_parser_v2.py <input_folder> <output_csv>")
        sys.exit(1)
    input_folder, output_csv = Path(sys.argv[1]), Path(sys.argv[2])
    files = sorted(
        p for p in input_folder.rglob("*.txt")
        if p.is_file() and not p.name.startswith(".")
    )
    rows = []
    for f in files:
        parsed = parse_file(f)
        d_type = parsed.get("dispute_type") or ""
        if "Бусад зээл" in d_type:
            rows.append(parsed)

    # Count by level
    level_counts = {}
    for r in rows:
        lv = r.get("case_level") or "unknown"
        level_counts[lv] = level_counts.get(lv, 0) + 1

    # Use a stable column order — all columns from first-instance +
    # the three new appellate columns.
    fieldnames = [
        "source_file", "case_id", "case_index", "decision_date",
        "court", "judge", "case_level", "dispute_type",
        "plaintiff_type", "defendant_type", "claim_amount_requested",
        "law_articles", "case_narrative",
        "civil_outcome", "civil_award_amount",
        "appeal_outcome", "referenced_decisions", "appealing_party",
        "parse_warnings",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as out:
        w = csv.DictWriter(out, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Parsed {len(rows)} file(s) -> {output_csv}")
    for lv in ["first_instance", "appellate", "supervisory"]:
        if lv in level_counts:
            print(f"  {level_counts[lv]:>5} {lv}")


if __name__ == "__main__":
    main()