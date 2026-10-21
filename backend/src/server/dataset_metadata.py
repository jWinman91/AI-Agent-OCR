"""
Describes a dataset for the analyser's system prompt.

The analyser writes code against `df` exactly as pandas.read_csv loads the file,
so the description reports the loaded dtypes plus what the values really are:
numbers and dates stored as text (with every format, separator and unit in use),
values that don't fit their column, all values of small text categories, and
sample rows in file order. It only describes the data and leaves it to the
analyser's own prompt how to handle it. It exists so generated code fits the
whole dataset instead of just its first rows; backend/tests/test_dataset_metadata.py
shows the resulting text.
"""

import re
from pathlib import Path

import pandas as pd

# Limits that trade prompt length against detail.
# Text columns with at most this many distinct values list all of them, so
# typos and truncated categories (e.g. 'Überweisun') stand out.
MAX_LISTED_VALUES = 20
MAX_LISTED_FORMATS = 10
MAX_LISTED_UNEXPECTED = 5
MAX_EXAMPLES = 3
# Datasets with at most MAX_FULL_ROWS rows are shown whole; larger ones as the
# first and last SAMPLE_EDGE_ROWS rows plus SAMPLE_RANDOM_ROWS random rows.
MAX_FULL_ROWS = 200
SAMPLE_EDGE_ROWS = 50
SAMPLE_RANDOM_ROWS = 100
# Fixed seed, so every run (and every judge) sees the same sample rows.
SAMPLE_SEED = 0
# Share of a text column's values that must look like dates/numbers for the
# column to be described as such; the remaining values are reported as
# unexpected.
TEXT_KIND_SHARE = 0.8
# Columns left out of the description. The data extractor adds image_url (the
# source image of each row) to every dataset; it carries nothing to analyse.
HIDDEN_COLUMNS = ["image_url"]

# A month name or abbreviation, English or German (Jan, März, Mai, Oct, Okt...).
_MONTH = r"(?:jan|feb|m[aä]r|apr|ma[iy]|jun|jul|aug|sep|o[ck]t|nov|de[cz])[a-zä]*\.?"
DATE_TEXT = re.compile(
    rf"""
    (?:
        \d{{1,4}}[./-]\d{{1,2}}[./-]\d{{1,4}}     # 2026-01-31, 31.01.26, 1/31/2026
      | \d{{1,2}}\.?\s*{_MONTH}\s*\d{{2,4}}       # 31. Jan 26, 1 März 2026
      | {_MONTH}\s+\d{{1,2}},?\s+\d{{2,4}}        # Jan 31, 2026
    )
    (?:                                           # optional time of day
        [ T]\d{{1,2}}:\d{{2}}(?::\d{{2}}(?:\.\d+)?)?
        (?:\s?[AP]M)?
        (?:Z|\s?[+-]\d{{2}}:?\d{{2}})?
    )?
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Digits with optional sign and separators ('.', ',', apostrophe, space).
NUMBER_TEXT = re.compile(r"[+-]?\d[\d.,' ]*")
# Currency and percent signs that number columns often carry.
UNITS = re.compile(r"€|\$|£|%|\b(?:EUR|USD|CHF|GBP)\b", re.IGNORECASE)


def get_dataset_metadata(data_file_path: Path | None) -> str:
    """
    Describes a CSV file for the analyser's system prompt: shape, exact column
    names, one line per column (see _describe_column) and sample rows (see
    _sample_rows). HIDDEN_COLUMNS are left out of all of it.

    :param data_file_path: The CSV file the analyser gets as a DataFrame, or
    None if there is none.
    :return: The description, or an empty string without a file.
    """
    if data_file_path is None:
        return ""

    # Default options, exactly like the executor loads `df`.
    df = pd.read_csv(data_file_path).drop(columns=HIDDEN_COLUMNS, errors="ignore")

    lines = [
        f"DATASET INFOS for file {data_file_path}:",
        "========================================",
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"Exact column names: {list(df.columns)}",
        "",
        "COLUMNS (pandas dtype of `df` as loaded, then what the values are):",
        *(_describe_column(str(name), df[name]) for name in df.columns),
        "",
        _sample_rows(df),
    ]
    return "\n".join(lines)


def _describe_column(name: str, column: pd.Series) -> str:
    """
    Describes one column in a single line: its dtype in `df`, the missing and
    distinct counts, and what its values are.

    :param name: The column name.
    :param column: The column as loaded by pandas.read_csv.
    :return: E.g. "  - Saldo: dtype object, 0 missing, 242 distinct; numbers
    stored as text, ...".
    """
    values = column.dropna()
    missing = len(column) - len(values)

    if values.empty:
        details = "no values"
    elif pd.api.types.is_bool_dtype(column):
        details = "boolean"
    elif pd.api.types.is_numeric_dtype(column):
        details = _describe_numbers(values, has_missing=missing > 0)
    else:
        details = _describe_text(values.astype(str))

    return (
        f"  - {name}: dtype {column.dtype}, {missing} missing, "
        f"{values.nunique()} distinct; {details}"
    )


def _describe_numbers(values: pd.Series, has_missing: bool) -> str:
    """
    Describes a numeric column: its range, and whether a float column only
    holds whole numbers (pandas loads integer columns with gaps as float).

    :param values: The column's non-missing values.
    :param has_missing: Whether the column has missing values.
    :return: E.g. "numbers from 1.0 to 5.0 (whole numbers, float only because
    of missing values)".
    """
    description = f"numbers from {values.min()} to {values.max()}"
    if pd.api.types.is_float_dtype(values) and bool(values.eq(values.round()).all()):
        reason = "float only because of missing values" if has_missing else "as float"
        description += f" (whole numbers, {reason})"
    return description


def _describe_text(values: pd.Series) -> str:
    """
    Describes a text (object) column. If most values look like dates or
    numbers, the column is described as such (see _describe_typed_text);
    otherwise small sets of categories are listed in full, so typos and
    variants stand out.

    Values are matched after stripping surrounding whitespace, but shown
    unchanged, so e.g. a stray space in ' Zinsen' stays visible.

    :param values: The column's non-missing values as strings.
    :return: The description of the values.
    """
    stripped = values.str.strip()
    is_date = stripped.str.fullmatch(DATE_TEXT)
    without_units = stripped.str.replace(UNITS, "", regex=True).str.strip()
    is_number = ~is_date & without_units.str.fullmatch(NUMBER_TEXT)

    if is_date.mean() >= TEXT_KIND_SHARE:
        return _describe_typed_text("dates/times stored as text", values, is_date)
    if is_number.mean() >= TEXT_KIND_SHARE:
        return _describe_typed_text(
            "numbers stored as text",
            values,
            is_number,
            details=_number_details(values[is_number]),
        )
    if values.nunique() <= MAX_LISTED_VALUES:
        return "text, all values: " + _format_counts(values.value_counts())
    return f"text, e.g. {_examples(values)}"


def _describe_typed_text(
    kind: str, values: pd.Series, fits: pd.Series, details: str | None = None
) -> str:
    """
    Describes a text column whose values are mostly dates or numbers: examples,
    kind-specific details, every format in use and the values that don't fit.

    Formats show every digit as 9, so mixed formats in one column become
    visible, e.g. '99. Jan 99' next to '99. März 9999'.

    :param kind: What the fitting values are, e.g. "numbers stored as text".
    :param values: The column's non-missing values as strings.
    :param fits: Boolean mask of the values that match the kind.
    :param details: Further facts about the values, e.g. their separators.
    :return: The parts joined with "; ".
    """
    formats = values[fits].str.replace(r"\d", "9", regex=True).value_counts()
    parts = [f"{kind}, e.g. {_examples(values[fits])}"]
    if details:
        parts.append(details)
    parts.append("formats (digit = 9): " + _format_counts(formats, MAX_LISTED_FORMATS))
    unexpected = values[~fits].value_counts()
    if not unexpected.empty:
        parts.append(
            "UNEXPECTED values that don't match this format: "
            + _format_counts(unexpected, MAX_LISTED_UNEXPECTED)
        )
    return "; ".join(parts)


def _number_details(numbers: pd.Series) -> str:
    """
    Describes the separators and units of numbers stored as text.

    :param numbers: The values that look like numbers, e.g. '60.394,59 €'.
    :return: E.g. "decimal separator ',', thousands separator '.'; contains
    units/symbols ['€']".
    """
    units = sorted(
        {unit.upper() for unit in numbers.str.findall(UNITS).explode().dropna()}
    )
    details = _separators(numbers.str.replace(UNITS, "", regex=True).str.strip())
    if units:
        details += f"; contains units/symbols {units}"
    return details


def _separators(numbers: pd.Series) -> str:
    """
    Works out the decimal and thousands separators of numbers stored as text.

    A value with both '.' and ',' settles it: the later one is the decimal
    separator ('1.234,56'). With only one kind of separator, it is likely a
    thousands separator if it is always followed by groups of exactly three
    digits ('12.500'), otherwise it is the decimal separator ('12,50').

    :param numbers: Numbers as text, without units.
    :return: The separators, e.g. "decimal separator ','".
    """
    has_dot = numbers.str.contains(".", regex=False)
    has_comma = numbers.str.contains(",", regex=False)

    with_both = numbers[has_dot & has_comma]
    if not with_both.empty:
        value = str(with_both.iloc[0])
        decimal, thousands = (
            (",", ".") if value.rfind(",") > value.rfind(".") else (".", ",")
        )
        return f"decimal separator {decimal!r}, thousands separator {thousands!r}"

    for separator, has_separator in ((",", has_comma), (".", has_dot)):
        with_separator = numbers[has_separator]
        if with_separator.empty:
            continue
        groups_of_three = rf"[+-]?\d{{1,3}}(?:{re.escape(separator)}\d{{3}})+"
        if with_separator.str.fullmatch(groups_of_three).all():
            return (
                f"{separator!r} is always followed by 3 digits, so it is likely a "
                f"thousands separator (e.g. {with_separator.iloc[0]!r})"
            )
        return f"decimal separator {separator!r}"
    return "no decimal or thousands separators"


def _sample_rows(df: pd.DataFrame) -> str:
    """
    Renders the rows to show the analyser: all of them for small datasets,
    otherwise the first and last SAMPLE_EDGE_ROWS rows plus SAMPLE_RANDOM_ROWS
    random rows in between (drawn with SAMPLE_SEED, so every run shows the same
    rows). Rows stay in file order and keep their 0-based position as row
    label, so it's visible whether the data runs e.g. newest first.

    :param df: The dataset as loaded by pandas.read_csv (RangeIndex).
    :return: The heading and the rows as a text table.
    """
    if len(df) <= MAX_FULL_ROWS:
        return "ALL ROWS (row = 0-based position in the file):\n" + df.to_string()

    first = range(min(SAMPLE_EDGE_ROWS, len(df)))
    last = range(max(len(df) - SAMPLE_EDGE_ROWS, 0), len(df))
    middle = pd.Series(range(first.stop, last.start), dtype=int)
    random = middle.sample(
        n=min(SAMPLE_RANDOM_ROWS, len(middle)), random_state=SAMPLE_SEED
    )
    positions = sorted({*first, *random, *last})
    return (
        f"SAMPLE ROWS: first {len(first)}, {len(random)} random and last "
        f"{len(last)}, in file order (row = 0-based position in the file):\n"
        + df.iloc[positions].to_string()
    )


def _format_counts(counts: pd.Series, limit: int | None = None) -> str:
    """
    Renders value counts as "'value' (count), ...".

    :param counts: Counts indexed by value, most frequent first (value_counts()).
    :param limit: Maximum number of values to show; the rest is summarised as
    "and N more".
    :return: The rendered counts.
    """
    shown = counts if limit is None else counts.head(limit)
    text = ", ".join(f"{value!r} ({count})" for value, count in shown.items())
    if len(counts) > len(shown):
        text += f" and {len(counts) - len(shown)} more"
    return text


def _examples(values: pd.Series) -> str:
    """
    Renders the first MAX_EXAMPLES distinct values, in file order.

    :param values: The values to pick examples from.
    :return: E.g. "'01. Jan 26', '02. Jan 26', '03. Jan 26'".
    """
    return ", ".join(
        repr(value) for value in values.drop_duplicates().head(MAX_EXAMPLES)
    )
