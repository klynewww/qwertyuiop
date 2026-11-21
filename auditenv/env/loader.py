"""Загрузка ОТКРЫТЫХ файлов компании.

Читает только белый список имён. truth.json, provenance.json и всё из
hidden/ загрузчик не открывает никогда: он получает путь только к open/.
"""
import csv
import json
from pathlib import Path

OPEN_FILES = (
    "manifest.json",
    "ledger.csv",
    "invoices.jsonl",
    "payments.jsonl",
    "fx.csv",
    "policy.json",
)

# таблица -> поле-идентификатор записи
ID_FIELDS = {
    "ledger": "row_id",
    "invoices": "invoice_id",
    "payments": "payment_id",
    "fx": "row_id",
}


def record_id(table: str, rec: dict) -> str:
    """ID записи. У курсов из генератора A колонки id нет: он вычисляется по полям
    (policy.json: FX:<date>:<source_currency>:<target_currency>)."""
    if table == "fx" and ID_FIELDS["fx"] not in rec:
        return f"FX:{rec['date']}:{rec['source_currency']}:{rec['target_currency']}"
    return rec[ID_FIELDS[table]]


def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_csv(path: Path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


_LEDGER_INT = ("debit_minor", "credit_minor")


def _typed_row(table: str, row: dict) -> dict:
    """Строки CSV -> JSON-значения: суммы целыми, «null» -> None (явный null вместо угаданной связи)."""
    out = dict(row)
    if table == "ledger":
        for key in _LEDGER_INT:
            if key in out and isinstance(out[key], str) and out[key].lstrip("-").isdigit():
                out[key] = int(out[key])
        for key in ("invoice_id", "payment_id"):
            if out.get(key) in ("null", ""):
                out[key] = None
    return out


def load_open_company(open_dir, typed: bool = False) -> dict:
    """Возвращает {'manifest', 'policy', 'tables', 'documents'}.

    Записи каждой таблицы отсортированы по id: порядок не зависит от
    порядка строк в файле, поэтому курсор стабилен.
    """
    root = Path(open_dir)
    missing = [n for n in OPEN_FILES if not (root / n).is_file()]
    if missing:
        raise FileNotFoundError(f"В {root} нет файлов: {missing}")

    tables = {
        "ledger": _read_csv(root / "ledger.csv"),
        "invoices": _read_jsonl(root / "invoices.jsonl"),
        "payments": _read_jsonl(root / "payments.jsonl"),
        "fx": _read_csv(root / "fx.csv"),
    }
    if typed:
        tables = {n: [_typed_row(n, r) for r in rows] for n, rows in tables.items()}
    for name, records in tables.items():
        tables[name] = sorted(records, key=lambda r, n=name: record_id(n, r))

    documents = {}
    docs_dir = root / "docs"
    if docs_dir.is_dir():
        for p in sorted(docs_dir.glob("*.txt")):
            documents[p.stem] = {"doc_id": p.stem, "text": p.read_text(encoding="utf-8")}
    # документ = сама запись счёта или платежа с полем doc_id (формат генератора A)
    for name in ("invoices", "payments"):
        for rec in tables[name]:
            doc_id = rec.get("doc_id")
            if isinstance(doc_id, str) and doc_id not in documents:
                documents[doc_id] = dict(rec)

    return {
        "manifest": _read_json(root / "manifest.json"),
        "policy": _read_json(root / "policy.json"),
        "tables": tables,
        "documents": documents,
    }
