import csv
import sys
from pathlib import Path

import joblib
import numpy as np
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

import features
import shuukh_parser_v2 as parser

MODEL_PATH = "trained_model.joblib"
INPUT_FOLDER = "new_cases"
OUTPUT_CSV = "predictions_output.csv"

console = Console()

OUTCOME_COLORS = {
    "granted": "green",
    "granted_uncontested": "bright_green",
    "settled": "cyan",
    "partial": "yellow",
    "denied": "red",
    "withdrawn": "grey58",
    "uncertain": "white",
}
OUTCOME_LABELS = {
    "granted": "НЭХЭМЖЛЭЛИЙГ ХАНГАСАН",
    "granted_uncontested": "ХҮЛЭЭН ЗӨВШӨӨРСНИЙГ БАТАЛСАН",
    "settled": "ЭВЛЭРЛИЙГ БАТАЛСАН",
    "partial": "ХЭСЭГЧЛЭН ХАНГАСАН",
    "denied": "ХЭРЭГСЭХГҮЙ БОЛГОСОН",
    "withdrawn": "ТАТГАЛЗСАНЫГ БАТАЛСАН",
    "uncertain": "ТОДОРХОЙГҮЙ",
}

SIGNAL_LABELS = {
    "has_interest_rate": ("", "Хүү, алданги дурдагдсан"),
    "has_institutional_lender": ("", "Байгууллагаас зээлсэн (ХХК / ББСБ)"),
    "has_app_lending": ("", "Аппликейшн ашигласан зээл"),
    "has_collateral": ("", "Барьцаа хөрөнгөтэй"),
    "has_guarantor": ("", "Батлан даагчтай"),
    "has_gov_fund": ("", "Төрийн сан, төрийн байгууллага"),
    "has_uncontested": ("", "Зөвшөөрсөн хандлага"),
    "has_settlement": ("", "Эвлэрсэн хандлага"),
    "has_withdrawal": ("", "Татгалзсан хандлага"),
    "has_opposition": ("", "Эсэргүүцсэн хандлага"),
    "has_evidence_dispute": ("", "Нотлох баримтын маргаантай"),
    "has_default": ("", "Хариу тайлбаргүй / Ирээгүй"),
    "has_multiple_claims": ("", "Олон шаардлагатай"),
}

KEY_TERM_GROUPS = {
    "settlement_signals": {
        "terms": ["эвлэрлийн", "эвлэрч", "эвлэрсэн", "зөвшөөрч байна",
                  "зөвшөөрч", "барагдуулсан", "хялбаршуулсан"],
        "icon": "",
        "label": "Эвлэрэх, зөвшөөрөх хандлагатай",
        "color": "cyan",
    },
    "opposition_signals": {
        "terms": ["татгалзаж", "татгалзаж байна",
                  "эсэргүүцэж", "буцааж"],
        "icon": "",
        "label": "Хариуцагч эсэргүүцсэн, маргасан",
        "color": "yellow",
    },
    "procedural_signals": {
        "terms": ["шүүх хуралдаанд", "хуралдаанд", "шүүхэд гаргасан",
                  "нэхэмжлэлдээ", "тайлбартаа", "хариу тайлбартаа", "хүсэлтдээ",
                  "гаргасан хүсэлтдээ"],
        "icon": "",
        "label": "Шүүхийн процесс, ажиллагаа",
        "color": "blue",
    },
    "claim_signals": {
        "terms": ["төлбөр", "төгрөг", "хүү", "алданги", "зээл",
                  "гэрээ", "нэхэмжлэгч"],
        "icon": "",
        "label": "Нэхэмжлэлийн мөнгөн дүн, шаардлага",
        "color": "white",
    },
}

SIGNAL_BOOSTS = {
    "has_settlement":   {"settled": 1.5, "granted_uncontested": 1.2},
    "has_uncontested":  {"granted_uncontested": 1.8, "settled": 1.2},
    "has_withdrawal":   {"withdrawn": 1.5},
    "has_default":      {"granted": 1.3},
    "has_opposition":   {"partial": 1.3, "denied": 1.2},
    "has_evidence_dispute": {"partial": 1.2, "denied": 1.3},
}


CLASS_PRECISION_THRESHOLDS = {
    "denied":              0.50,
    "partial":             0.40,
    "granted_uncontested": 0.40,
    "withdrawn":           0.35,
    "granted":             0.0,
    "settled":             0.0,
}


def confidence_label(ranked):
    top_p = ranked[0][1]
    second_p = ranked[1][1] if len(ranked) > 1 else 0
    margin = top_p - second_p
    if margin >= 0.20 and top_p >= 0.50:
        return "ӨНДӨР ИТГЭЛТЭЙ", "green"
    elif margin >= 0.08 or top_p >= 0.30:
        return "ДУНД ЗЭРГИЙН ИТГЭЛТЭЙ", "yellow"
    return "БАГА ИТГЭЛТЭЙ — ойролцоо магадлалууд", "red"


def load_model():
    saved = joblib.load(MODEL_PATH)
    return saved["pipeline"], saved["median_claim"], saved["judge_counts"]


def predict_case(claim_amount, plaintiff_type, defendant_type, court_type,
                  narrative_text, judge_name=None, model=None):
    if model is None:
        pipe, median_claim, judge_counts = load_model()
    else:
        pipe, median_claim, judge_counts = model

    row = features.build_case_features(
        claim_amount=claim_amount,
        plaintiff_type=plaintiff_type,
        defendant_type=defendant_type,
        court_type=court_type,
        narrative_text=narrative_text,
        judge_name=judge_name,
        median_claim=median_claim,
        judge_counts=judge_counts,
    )
    proba = pipe.predict_proba(row)[0]
    classes = pipe.classes_
    proba_dict = dict(zip(classes, proba))

    for flag, boosts in SIGNAL_BOOSTS.items():
        if flag in row.columns and row[flag].iloc[0] == 1:
            for outcome, multiplier in boosts.items():
                if outcome in proba_dict:
                    proba_dict[outcome] *= multiplier

    total = sum(proba_dict.values())
    proba_dict = {k: v / total for k, v in proba_dict.items()}

    ranked = sorted(proba_dict.items(), key=lambda x: -x[1])
    final = None
    for outcome, prob in ranked:
        threshold = CLASS_PRECISION_THRESHOLDS.get(outcome, 0.0)
        if prob >= threshold:
            final = (outcome, prob)
            break
    if final is None:
        final = ranked[0] 

    if final[0] != ranked[0][0]:
        ranked = [final] + [(o, p) for o, p in ranked if o != final[0]]

    if ranked[0][1] < 0.35:
        ranked.insert(0, ("uncertain", ranked[0][1]))

    return ranked, row


def parse_and_predict_folder(input_folder, model=None):
    folder = Path(input_folder)
    if not folder.is_dir():
        print(f"ERROR: Input folder '{folder}' does not exist.")
        print(f"Create it and drop .txt case files there, then re-run.")
        sys.exit(1)

    txt_files = sorted(
        p for p in folder.glob("*")
        if p.is_file() and not p.name.startswith(".") and p.name != "README.md" and p.name != OUTPUT_CSV
    )
    if not txt_files:
        print(f"No valid case files found in '{folder}/'.")
        print(f"Drop case narrative files there and re-run.")
        sys.exit(0)

    if model is None:
        model = load_model()

    results = []
    for filepath in txt_files:
        parsed = parser.parse_file(filepath)
        dispute_type = parsed.get("dispute_type") or ""
        case_level = parsed.get("case_level")

        skip_reason = None
        if not dispute_type:
            skip_reason = "dispute_type could not be extracted - can't confirm this is even a loan dispute"
        elif "Бусад зээл" not in dispute_type:
            skip_reason = f"dispute_type is '{dispute_type}', not 'Бусад зээл' - out of scope for this model (e.g. a bank loan, lease, or other dispute type wasn't part of training)"
        elif case_level != "first_instance":
            skip_reason = f"case_level is '{case_level}', not 'first_instance' - this model only predicts first-instance outcomes"

        if skip_reason:
            results.append({
                "source_file": filepath.name,
                "case_id": parsed.get("case_id"),
                "case_level": case_level,
                "dispute_type": parsed.get("dispute_type"),
                "court": parsed.get("court"),
                "judge": parsed.get("judge"),
                "plaintiff_type": parsed.get("plaintiff_type"),
                "defendant_type": parsed.get("defendant_type"),
                "court_type": None,
                "claim_amount": parsed.get("claim_amount_requested"),
                "has_narrative": bool(parsed.get("case_narrative")),
                "predicted_outcome": None,
                "predicted_probability": None,
                "all_probabilities": {},
                "parse_warnings": parsed.get("parse_warnings", ""),
                "skip_reason": skip_reason,
            })
            continue

        # Determine court_type from the court name
        court_type = features.infer_court_type(parsed.get("court") or "")

        # Build the prediction input from parsed fields
        case_args = dict(
            claim_amount=parsed.get("claim_amount_requested"),
            plaintiff_type=parsed.get("plaintiff_type"),
            defendant_type=parsed.get("defendant_type"),
            court_type=court_type,
            narrative_text=parsed.get("case_narrative"),
            judge_name=parsed.get("judge"),
            model=model,
        )

        ranked, feature_row = predict_case(**case_args)
        top_outcome, top_prob = ranked[0]

        results.append({
            "source_file": filepath.name,
            "case_id": parsed.get("case_id"),
            "case_level": parsed.get("case_level"),
            "dispute_type": parsed.get("dispute_type"),
            "court": parsed.get("court"),
            "judge": parsed.get("judge"),
            "plaintiff_type": parsed.get("plaintiff_type"),
            "defendant_type": parsed.get("defendant_type"),
            "court_type": court_type,
            "claim_amount": parsed.get("claim_amount_requested"),
            "has_narrative": bool(parsed.get("case_narrative")),
            "predicted_outcome": top_outcome,
            "predicted_probability": round(top_prob * 100, 1),
            "all_probabilities": {o: round(p * 100, 1) for o, p in ranked},
            "parse_warnings": parsed.get("parse_warnings", ""),
            "skip_reason": None,
            "narrative_text": parsed.get("case_narrative") or "",
            "feature_row": feature_row,
        })

    return results


def predict_narrative_only_folder(input_folder, model=None, court_type="urban"):
    folder = Path(input_folder)
    if not folder.is_dir():
        print(f"ERROR: Input folder '{folder}' does not exist.")
        print(f"Create it and drop plain narrative .txt files there, then re-run.")
        sys.exit(1)

    txt_files = sorted(p for p in folder.glob("*.txt") if p.is_file())
    if not txt_files:
        print(f"No .txt files found in '{folder}/'.")
        sys.exit(0)

    if model is None:
        model = load_model()

    results = []
    for filepath in txt_files:
        narrative_text = filepath.read_text(encoding="utf-8", errors="replace").strip()
        ranked, feature_row = predict_case(
            claim_amount=None,
            plaintiff_type=None,
            defendant_type=None,
            court_type=court_type,
            narrative_text=narrative_text,
            judge_name=None,
            model=model,
        )
        top_outcome, top_prob = ranked[0]
        results.append({
            "source_file": filepath.name,
            "case_id": None,
            "case_level": None,
            "dispute_type": None,
            "court": None,
            "judge": None,
            "plaintiff_type": None,
            "defendant_type": None,
            "court_type": court_type,
            "claim_amount": None,
            "has_narrative": bool(narrative_text),
            "predicted_outcome": top_outcome,
            "predicted_probability": round(top_prob * 100, 1),
            "all_probabilities": {o: round(p * 100, 1) for o, p in ranked},
            "parse_warnings": "",
            "skip_reason": None,
            "narrative_text": narrative_text,
            "feature_row": feature_row,
        })

    return results


def explain_prediction(r, model):
    """Generate a structured explanation of WHY the model predicted this
    outcome. Returns a dict with reasoning components that display_result
    can render.
    """
    pipe, _, _ = model
    feature_row = r.get("feature_row")
    narrative = r.get("narrative_text", "")
    if feature_row is None:
        return None

    tree_vote_summary = [] # Removed for GBM compatibility

    narrative_lower = narrative.lower()
    detected_signals = []
    for group_key, group in KEY_TERM_GROUPS.items():
        found_terms = []
        contexts = []
        for t in group["terms"]:
            idx = narrative_lower.find(t)
            if idx != -1:
                found_terms.append(t)
                start = max(0, idx - 40)
                end = min(len(narrative), idx + len(t) + 40)
                snippet = narrative[start:end].strip()
                # Clean up newlines in snippet
                snippet = " ".join(snippet.split())
                if start > 0: snippet = "..." + snippet
                if end < len(narrative): snippet = snippet + "..."
                
                # Highlight the term in the snippet (case-insensitive replace for display)
                import re
                escaped_t = re.escape(t)
                pattern = re.compile(escaped_t, re.IGNORECASE)
                snippet = pattern.sub(f"[bold white][u]\g<0>[/u][/bold white]", snippet)
                
                if snippet not in contexts:
                    contexts.append(snippet)
                
        if found_terms:
            detected_signals.append({
                "group": group_key,
                "icon": group["icon"],
                "label": group["label"],
                "color": group["color"],
                "terms": found_terms,
                "contexts": contexts
            })

    active_factors = []
    for flag_col, (icon, label) in SIGNAL_LABELS.items():
        if flag_col in feature_row.columns and feature_row[flag_col].iloc[0] == 1:
            active_factors.append((icon, label))

    top_outcome = r["predicted_outcome"]
    top_label = OUTCOME_LABELS.get(top_outcome, top_outcome)
    reasons = []

    settlement_found = any(s["group"] == "settlement_signals" for s in detected_signals)
    opposition_found = any(s["group"] == "opposition_signals" for s in detected_signals)

    if settlement_found and top_outcome in ("settled", "granted_uncontested"):
        reasons.append("Нэхэмжлэлд эвлэрэх/зөвшөөрөх хандлага илэрсэн — талууд зөвшилцөлд хүрсэн байх магадлалтай")
    elif settlement_found and top_outcome == "granted":
        reasons.append("Эвлэрэх хандлага илэрсэн хэдий ч шүүхээр эцэслэн шийдвэрлэсэн байх магадлалтай")
    if opposition_found and top_outcome == "partial":
        reasons.append("Хариуцагч идэвхтэй эсэргүүцэж байгаа нь шүүх нэхэмжлэлийг хэсэгчлэн хангах магадлалыг харуулж байна")
    elif opposition_found and top_outcome == "denied":
        reasons.append("Хүчтэй эсэргүүцсэн хандлага — хариуцагчийн үндэслэл давамгайлсан байх магадлалтай")
    elif opposition_found:
        reasons.append("Хариуцагч нэхэмжлэлийг эсэргүүцсэн тайлбар гаргасан")
    if not opposition_found and top_outcome == "granted":
        reasons.append("Эсэргүүцсэн хандлага илрээгүй — хариуцагч маргаагүй байх магадлалтай")

    if any(f[1] == "Барьцаа хөрөнгөтэй" for f in active_factors):
        reasons.append("Барьцаа хөрөнгө дурдагдсан — нэхэмжлэгчийн шаардлагыг бэхжүүлж байна")
    if any(f[1] == "Батлан даагчтай" for f in active_factors):
        reasons.append("Батлан даагчтай — нэхэмжлэгчийн төлбөр барагдуулах боломжийг нэмэгдүүлж байна")

    # Confidence-based caveat
    ranked_list = sorted(r["all_probabilities"].items(), key=lambda x: -x[1])
    if len(ranked_list) >= 2:
        margin = ranked_list[0][1] - ranked_list[1][1]
        if margin < 10:
            runner_up = OUTCOME_LABELS.get(ranked_list[1][0], ranked_list[1][0])
            reasons.append(f"Ойролцоо магадлалтай — {runner_up} байх хүчтэй магадлал байна ({ranked_list[1][1]:.1f}%)")

    if not reasons:
        reasons.append("Нэхэмжлэлийн ерөнхий агуулгад үндэслэн таамаглал гаргасан")

    return {
        "detected_signals": detected_signals,
        "active_factors": active_factors,
        "reasons": reasons,
    }


def display_result(r, model=None):
    """One rich panel per case - colored probability bars, a confidence
    read, bordered in the predicted outcome's color so a quick glance
    down a long batch run tells you a lot before reading any text.
    Now includes explainability reasoning when a model is provided."""
    if r.get("skip_reason"):
        console.print(Panel(
            f"[bold red]No prediction made[/bold red]\n{r['skip_reason']}",
            title=f"[dim]{r['source_file']}[/dim]",
            border_style="red",
        ))
        return

    ranked = sorted(r["all_probabilities"].items(), key=lambda x: -x[1])
    top_outcome = r["predicted_outcome"]
    top_color = OUTCOME_COLORS.get(top_outcome, "white")
    conf_label, conf_color = confidence_label([(o, p / 100) for o, p in ranked])

    facts = f"{r.get('plaintiff_type') or '?'} vs {r.get('defendant_type') or '?'}"
    if r.get("claim_amount"):
        facts += f"  ·  ₮{r['claim_amount']:,.0f}"
    if r.get("court_type"):
        facts += f"  ·  {'Улаанбаатар' if r['court_type'] == 'urban' else 'Орон нутгийн'} шүүх"
    if not r.get("has_narrative"):
        facts += "  ·  [red]нэхэмжлэл байхгүй — найдвар багатай[/red]"

    body = Text.from_markup(f"[dim]{facts}[/dim]\n\n")
    # Show only the top probability to avoid repeating the 6 graphs
    top_p = ranked[0][1]
    body.append(f"Таамагласан магадлал: {top_p:5.1f}%\n\n", style=f"bold {top_color}")
    body.append(f"{conf_label}", style=f"bold {conf_color}")

    console.print(f"[bold]{r['source_file']}[/bold]")
    console.print(body)
    console.print(f"[bold {top_color}]{OUTCOME_LABELS.get(top_outcome, top_outcome)}[/]")

    if model is None:
        return

    explanation = explain_prediction(r, model)
    if explanation is None:
        return

    console.print()

    console.print(
        f"  [bold {top_color}]ШҮҮХИЙН ҮНДЭСЛЭЛ — ЯАГААД {OUTCOME_LABELS.get(top_outcome, top_outcome)}?[/]"
    )
    console.print(f"  [dim]{'─' * 60}[/dim]")

    if explanation["detected_signals"] or explanation["active_factors"]:
        console.print(f"  [bold white]ИЛРҮҮЛСЭН :[/bold white]")
        console.print()
        for sig in explanation["detected_signals"]:
            terms_str = ", ".join(f'[bold]{t}[/bold]' for t in sig["terms"])
            icon_str = f"{sig['icon']}  " if sig['icon'] else ""
            console.print(
                f"    {icon_str}[bold white]{sig['label']}[/bold white]"
            )
            console.print(
                f"       [dim]олдсон:[/dim] {terms_str}"
            )
            # Show the actual sentence context from the text
            for ctx in sig["contexts"][:3]:  # Show up to 3 context snippets
                console.print(f"       [dim]\"[/dim]{ctx}[dim]\"[/dim]")
        for icon, label in explanation["active_factors"]:
            icon_str = f"{icon}  " if icon else ""
            console.print(f"    {icon_str}[bold white]{label}[/bold white]  [dim]✓ текстээс олдсон[/dim]")
        console.print()



    # ━━ REASONING SUMMARY ━━
    console.print(f"  [bold white]ҮНДЭСЛЭЛ[/bold white]")
    console.print()
    for i, reason in enumerate(explanation["reasons"]):
        bullet = "›" if i > 0 else "▸"
        console.print(f"    [bold {top_color}]{bullet}[/bold {top_color}]  {reason}")
    console.print()
    console.print(f"  [dim]{'─' * 60}[/dim]")


def display_summary(results):
    predicted = [r for r in results if not r.get("skip_reason")]
    skipped = [r for r in results if r.get("skip_reason")]

    table = Table(title=f"{len(results)} хэрэг боловсруулав", show_lines=False)
    table.add_column("Шийдвэр")
    table.add_column("Тоо", justify="right")
    from collections import Counter
    counts = Counter(r["predicted_outcome"] for r in predicted)
    for outcome, label in OUTCOME_LABELS.items():
        if counts[outcome]:
            table.add_row(f"[{OUTCOME_COLORS[outcome]}]{label}[/{OUTCOME_COLORS[outcome]}]", str(counts[outcome]))
    if skipped:
        table.add_row("[red]Алгассан (загварт хамааралгүй)[/red]", str(len(skipped)))
    console.print(table)


def save_results_csv(results, output_path):
    """Write predictions to a CSV file."""
    if not results:
        return
    fieldnames = [
        "source_file", "case_id", "case_level", "dispute_type",
        "court", "judge", "plaintiff_type", "defendant_type",
        "court_type", "claim_amount", "has_narrative",
        "predicted_outcome", "predicted_probability",
        "all_probabilities", "parse_warnings", "skip_reason",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in results:
            row_copy = dict(row)
            # Flatten the probabilities dict to a readable string
            row_copy["all_probabilities"] = " | ".join(
                f"{o}: {p}%" for o, p in row["all_probabilities"].items()
            )
            w.writerow(row_copy)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    # In Jupyter, sys.argv might contain connection files like '-f /run/user/.../kernel.json'
    # We want to ignore those, plus pull out our own flags before treating
    # whatever's left as the folder path.
    raw_args = sys.argv[1:]
    narrative_only = "--narrative-only" in raw_args
    court_type_override = "rural" if "--court-type" in raw_args and "rural" in raw_args else None

    positional = [
        a for a in raw_args
        if not a.startswith("-") and a not in ("urban", "rural")
    ]
    input_dir = positional[0] if positional else INPUT_FOLDER

    model = load_model()

    if narrative_only:
        court_type = court_type_override or "urban"
        console.print(f"[dim]'{input_dir}/' сангаас зөвхөн нэхэмжлэл бүхий файлуудыг уншиж байна (court_type={court_type})...[/dim]\n")
        results = predict_narrative_only_folder(input_dir, model=model, court_type=court_type)
    else:
        console.print(f"[dim]'{input_dir}/' сангаас хэргийн файлуудыг уншиж байна...[/dim]\n")
        results = parse_and_predict_folder(input_dir, model=model)

    for r in results:
        display_result(r, model=model)
        console.print()

    display_summary(results)

    # Save CSV
    save_results_csv(results, OUTPUT_CSV)  