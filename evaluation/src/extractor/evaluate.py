"""A module for evaluating the data extractor agent by comparing extracted data with
ground truth data."""

import re
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from src.experiments import GROUND_TRUTH_DIR
from src.extractor import EXTRACTED_DATA_DIR

FieldValue = str | int | float | None
FieldMetrics = dict[str, float | None]

# Metrics for multi-value fields (many values per page, e.g. a table column):
# classification metrics computed from a per-page bag match.
CLASSIFICATION_METRIC_NAMES = ["accuracy", "precision", "recall", "f1_score"]
# Metrics for single-value fields (exactly one value per page, e.g. a scale
# reading): mean absolute error, since precision/recall/accuracy collapse when
# a wrong value is both a false positive and a false negative.
ERROR_METRIC_NAMES = ["mae"]
# Superset of every metric that can appear in the "overall" summary - used to
# log a consistent set of (NaN) overall columns for skipped experiments.
ALL_OVERALL_METRIC_NAMES = CLASSIFICATION_METRIC_NAMES + ERROR_METRIC_NAMES

# Metadata columns attached to every row by the pipeline, never real extracted
# values - compared implicitly via the per-image_url grouping, not as fields.
NON_FIELD_COLUMNS = {"image_url", "timestamp"}

_CURRENCY_CHARS = re.compile(r"[€$£\s]")


class DataExtractorEvaluator:
    IMAGE_URL_COL = "image_url"

    def __init__(
        self,
        extracted_data_dir: Path = EXTRACTED_DATA_DIR,
        ground_truth_data_dir: Path = GROUND_TRUTH_DIR,
    ) -> None:
        self._EXTRACTED_DATA_DIR = extracted_data_dir
        self._GROUND_TRUTH_DATA_DIR = ground_truth_data_dir

    @staticmethod
    def _parse_numeric(text: str) -> float | None:
        """Parses a numeric string that may use German/European or US thousands
        and decimal separators, with an optional currency symbol.

        E.g. "60.394,59 €" and "1,234.56" both parse correctly: whichever of
        "," or "." appears LAST in the string is treated as the decimal
        separator, and the other is stripped as a thousands separator.

        Args:
            text (str): The raw string to parse.
        Returns:
            float | None: The parsed number, or None if it isn't numeric.
        """
        cleaned = _CURRENCY_CHARS.sub("", text)
        if not cleaned:
            return None

        if "," in cleaned and "." in cleaned:
            if cleaned.rfind(",") > cleaned.rfind("."):
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        elif "," in cleaned:
            # Single separator: treat as a decimal comma only when it looks
            # like one (e.g. "12,50"), otherwise as a thousands separator.
            whole, _, fraction = cleaned.rpartition(",")
            if whole and len(fraction) in (1, 2):
                cleaned = cleaned.replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")

        try:
            return float(cleaned)
        except ValueError:
            return None

    @classmethod
    def _normalize_value(cls, value: FieldValue) -> FieldValue:
        """Normalizes a field value so that equivalent values compare equal.

        Args:
            value: Raw field value coming from either the extracted data or the
                ground truth CSV.
        Returns:
            FieldValue: None for missing/NaN values, a float rounded to 6 decimal
            places for anything numeric - including currency-formatted strings
            like "60.394,59 €" - otherwise a stripped string.
        """
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return round(float(value), 6)
        # Try numeric parsing first (handles plain floats, "70.5" vs 70.50, and
        # currency-formatted strings), fall back to a plain string compare for
        # genuinely text fields.
        numeric = cls._parse_numeric(str(value).strip())
        if numeric is not None:
            return round(numeric, 6)
        return str(value).strip()

    def _load_data(
        self,
        experiment_name: str,
        run_name: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
        """Loads the extracted data and ground truth data for a given experiment/run
        and aligns their columns.

        Args:
            experiment_name (str): Name of the experiment to load data for.
            run_name (str): Name of the run to load extracted data for.
        Returns:
            tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]: The extracted
            dataframe, the ground truth dataframe, the full list of ground truth
            field names (every ground truth column except pipeline metadata
            columns), and the subset of those that also exist in the extracted
            data (case-insensitive) and can therefore actually be scored.
        """
        extracted_path = self._EXTRACTED_DATA_DIR / f"{experiment_name}_{run_name}.csv"
        ground_truth_path = self._GROUND_TRUTH_DATA_DIR / f"{experiment_name}.csv"

        extracted_df = pd.read_csv(extracted_path)
        ground_truth_df = pd.read_csv(ground_truth_path)

        # The data extractor's own field-name casing isn't guaranteed to match
        # ground truth (e.g. "SALDO" vs "Saldo"), so match case-insensitively
        # and rename the extracted column to the ground truth's casing.
        extracted_columns_by_lower = {
            column.lower(): column for column in extracted_df.columns
        }
        rename_map = {
            extracted_columns_by_lower[column.lower()]: column
            for column in ground_truth_df.columns
            if column.lower() in extracted_columns_by_lower
        }
        extracted_df = extracted_df.rename(columns=rename_map)

        ground_truth_fields = [
            column
            for column in ground_truth_df.columns
            if column.lower() not in NON_FIELD_COLUMNS
        ]
        # A ground truth field the extractor was never asked to produce can
        # never match - scored as None rather than attempted, see
        # evaluate_extracted_data.
        matchable_fields = [
            field for field in ground_truth_fields if field in extracted_df.columns
        ]

        return extracted_df, ground_truth_df, ground_truth_fields, matchable_fields

    def _match_field(
        self,
        field: str,
        extracted_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Matches one field's extracted values against its ground truth values,
        independently per source image.

        Within each image_url group, extracted and ground truth values for this
        one field are compared as an unordered multiset (bag match): an
        extracted value is a true positive if a not-yet-matched ground truth
        value equal to it (after normalization) exists for the same image,
        otherwise it's a false positive. Ground truth values left unmatched
        after this are false negatives. Values are compared independently of
        which row they originally appeared in and independently of every other
        field, so e.g. a correctly extracted Saldo still counts even if that
        row's Datum was wrong (or never extracted at all).

        Args:
            field (str): Name of the field to match.
            extracted_df (pd.DataFrame): Extracted data, already column-aligned
                with ground truth by _load_data.
            ground_truth_df (pd.DataFrame): Ground truth data.
        Returns:
            dict[str, Any]: Dict with "true_positive_count", "false_positive_count",
            "false_negative_count", and "false_positive_records" /
            "false_negative_records" - one record per (image_url, field) page that
            has any mismatch, each holding the list of hallucinated (false
            positive) or missed (false negative) values for that page.
        """
        image_urls = set(extracted_df[self.IMAGE_URL_COL]) | set(
            ground_truth_df[self.IMAGE_URL_COL]
        )

        true_positive_count = 0
        false_positive_count = 0
        false_negative_count = 0
        false_positive_records: list[dict[str, Any]] = []
        false_negative_records: list[dict[str, Any]] = []

        for image_url in sorted(image_urls):
            extracted_values = extracted_df.loc[
                extracted_df[self.IMAGE_URL_COL] == image_url, field
            ].tolist()
            ground_truth_values = ground_truth_df.loc[
                ground_truth_df[self.IMAGE_URL_COL] == image_url, field
            ].tolist()

            # Bag match: each extracted value consumes at most one matching
            # ground truth value, so duplicate values on both sides are
            # handled correctly instead of everything matching everything.
            ground_truth_pool = list(ground_truth_values)
            hallucinated_values = []

            for value in extracted_values:
                key = self._normalize_value(value)
                match_index = next(
                    (
                        i
                        for i, gt_value in enumerate(ground_truth_pool)
                        if self._normalize_value(gt_value) == key
                    ),
                    None,
                )
                if match_index is not None:
                    true_positive_count += 1
                    ground_truth_pool.pop(match_index)
                else:
                    # Extracted but not in the ground truth for this page.
                    hallucinated_values.append(value)

            # Whatever is left unmatched in the ground truth pool was missed
            # (present on the page, never extracted).
            missed_values = ground_truth_pool

            false_positive_count += len(hallucinated_values)
            false_negative_count += len(missed_values)

            # One record per page per field, with the mismatched values as a
            # list - pages with no mismatch produce no record.
            if hallucinated_values:
                false_positive_records.append(
                    {
                        self.IMAGE_URL_COL: image_url,
                        "field": field,
                        "hallucinated_values": hallucinated_values,
                    }
                )
            if missed_values:
                false_negative_records.append(
                    {
                        self.IMAGE_URL_COL: image_url,
                        "field": field,
                        "missed_values": missed_values,
                    }
                )

        return {
            "strategy": "multi_value",
            "true_positive_count": true_positive_count,
            "false_positive_count": false_positive_count,
            "false_negative_count": false_negative_count,
            "false_positive_records": false_positive_records,
            "false_negative_records": false_negative_records,
        }

    def _is_single_value_field(
        self,
        field: str,
        ground_truth_df: pd.DataFrame,
    ) -> bool:
        """Decides whether a field has exactly one value per page in the ground truth.

        Single-value fields (one reading per image, e.g. a scale's body weight)
        are scored with MAE; multi-value fields (many rows per page, e.g. a table
        column) are scored with classification metrics.

        Args:
            field (str): Name of the field to inspect.
            ground_truth_df (pd.DataFrame): Ground truth data.
        Returns:
            bool: True iff every image_url group has exactly one non-null ground
            truth value for this field.
        """
        counts = ground_truth_df.groupby(self.IMAGE_URL_COL)[field].count()
        return bool((counts == 1).all())

    def _match_single_value_field(
        self,
        field: str,
        extracted_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Scores a single-value field (one value per page) by mean absolute error.

        Per image_url, the single ground truth value is compared against the
        single extracted value. The mean absolute error is taken over pages
        where both values parse numeric; pages where the extracted value is
        missing or non-numeric can't contribute an error and are counted as
        unscored. A page is a mismatch when the normalized extracted value
        differs from the normalized ground truth value (including missing
        extractions, and pages where the extractor produced more than one value).

        Args:
            field (str): Name of the field to match.
            extracted_df (pd.DataFrame): Extracted data, already column-aligned
                with ground truth by _load_data.
            ground_truth_df (pd.DataFrame): Ground truth data.
        Returns:
            dict[str, Any]: Dict tagged "strategy": "single_value" with "mae"
            (mean absolute error, or None if no page was numerically scoreable),
            "n_unscored" (pages that couldn't contribute an error), and
            "mismatch_records" - one {image_url, field, extracted_value,
            ground_truth_value} dict per mismatched page.
        """
        image_urls = set(extracted_df[self.IMAGE_URL_COL]) | set(
            ground_truth_df[self.IMAGE_URL_COL]
        )

        absolute_errors: list[float] = []
        n_unscored = 0
        mismatch_records: list[dict[str, Any]] = []

        for image_url in sorted(image_urls):
            extracted_values = extracted_df.loc[
                extracted_df[self.IMAGE_URL_COL] == image_url, field
            ].tolist()
            ground_truth_values = ground_truth_df.loc[
                ground_truth_df[self.IMAGE_URL_COL] == image_url, field
            ].tolist()

            # Ground truth is single-value by construction here; a page with no
            # ground truth value isn't scoreable for this field.
            ground_truth_value = ground_truth_values[0] if ground_truth_values else None
            extracted_value = extracted_values[0] if extracted_values else None

            gt_norm = self._normalize_value(ground_truth_value)
            ex_norm = self._normalize_value(extracted_value)
            # More than one extracted value for a single-value page is itself
            # wrong - treat the page as a mismatch regardless of the first value.
            is_mismatch = ex_norm != gt_norm or len(extracted_values) > 1

            if is_mismatch:
                mismatch_records.append(
                    {
                        self.IMAGE_URL_COL: image_url,
                        "field": field,
                        "extracted_value": extracted_value,
                        "ground_truth_value": ground_truth_value,
                    }
                )

            gt_num = self._parse_numeric(str(ground_truth_value))
            ex_num = (
                self._parse_numeric(str(extracted_value))
                if extracted_value is not None and not pd.isna(extracted_value)
                else None
            )
            if gt_num is not None and ex_num is not None:
                absolute_errors.append(abs(gt_num - ex_num))
            else:
                n_unscored += 1

        mae = sum(absolute_errors) / len(absolute_errors) if absolute_errors else None

        return {
            "strategy": "single_value",
            "mae": mae,
            "n_unscored": n_unscored,
            "mismatch_records": mismatch_records,
        }

    @staticmethod
    def _format_values(values: list[Any]) -> str:
        """Formats a list of values as a readable list string for a CSV cell.

        Args:
            values (list[Any]): The values to format.
        Returns:
            str: The values as a string list, e.g. "[46272.21, 60394.59]".
        """
        return str([str(value) for value in values])

    @staticmethod
    def _compute_metrics(match: dict[str, Any]) -> FieldMetrics:
        """Computes one field's metrics from its match result, per its strategy.

        Args:
            match (dict[str, Any]): A match result as returned by _match_field
                (multi-value) or _match_single_value_field (single-value).
        Returns:
            FieldMetrics: For multi-value fields, {"accuracy", "precision",
            "recall", "f1_score"}. For single-value fields, {"mae"}.
        """
        if match["strategy"] == "single_value":
            return {"mae": match["mae"]}

        true_positives = match["true_positive_count"]
        false_positives = match["false_positive_count"]
        false_negatives = match["false_negative_count"]

        # Guard every ratio against a zero denominator (e.g. no extractions at
        # all yet), rather than letting a ZeroDivisionError bubble up.
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (
            true_positives / (true_positives + false_positives + false_negatives)
            if (true_positives + false_positives + false_negatives) > 0
            else 1.0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def _match_all_fields(
        self,
        experiment_name: str,
        run_name: str,
    ) -> tuple[dict[str, dict[str, Any] | None], dict[str, str]]:
        """Matches every ground truth field independently, choosing the metric
        strategy (single-value MAE vs. multi-value classification) per field from
        the ground truth.

        Args:
            experiment_name (str): Name of the experiment to match fields for.
            run_name (str): Name of the run to match extracted fields for.
        Returns:
            tuple[dict[str, dict[str, Any] | None], dict[str, str]]: A dict
            mapping every ground truth field name to its match result (None for a
            field the extractor was never asked to produce), and a dict mapping
            every ground truth field name to its strategy ("single_value" or
            "multi_value") - known even for non-matchable fields, since the
            strategy is decided from the ground truth alone.
        """
        extracted_df, ground_truth_df, ground_truth_fields, matchable_fields = (
            self._load_data(experiment_name, run_name)
        )

        matches: dict[str, dict[str, Any] | None] = {}
        strategies: dict[str, str] = {}
        for field in ground_truth_fields:
            single_value = self._is_single_value_field(field, ground_truth_df)
            strategies[field] = "single_value" if single_value else "multi_value"

            if field not in matchable_fields:
                matches[field] = None
                continue

            if single_value:
                matches[field] = self._match_single_value_field(
                    field, extracted_df, ground_truth_df
                )
            else:
                matches[field] = self._match_field(field, extracted_df, ground_truth_df)

        return matches, strategies

    def evaluate_extracted_data(
        self,
        experiment_name: str,
        run_name: str,
    ) -> dict[str, Any]:
        """Evaluates the extracted data for a given experiment against the ground truth
        data, with one set of metrics per ground truth field plus a macro-averaged
        overall set.

        Args:
            experiment_name (str): Name of the experiment to evaluate.
            run_name (str): Name of the run to evaluate.
        Returns:
            dict[str, Any]: Dict with:
            - "fields": mapping of field name to its metrics dict. Multi-value
              fields have {accuracy, precision, recall, f1_score}; single-value
              fields have {mae}. A field the extractor was never asked to produce
              has its strategy's metric keys all set to None.
            - "overall": each metric name macro-averaged across the fields that
              produce it and have a non-None value (so mae averages over
              single-value fields, classification metrics over multi-value
              fields). A metric with no contributing field is set to None.
        """
        matches, strategies = self._match_all_fields(experiment_name, run_name)

        field_metrics: dict[str, FieldMetrics] = {}
        for field, match in matches.items():
            if match is None:
                # Non-matchable field: None-fill the metric keys its strategy
                # would have produced, so it still appears in the report.
                metric_names = (
                    ERROR_METRIC_NAMES
                    if strategies[field] == "single_value"
                    else CLASSIFICATION_METRIC_NAMES
                )
                field_metrics[field] = dict.fromkeys(metric_names)
            else:
                field_metrics[field] = self._compute_metrics(match)

        # Average each metric only over the fields that actually have it with a
        # real (non-None) value.
        overall: dict[str, float | None] = {}
        for metric_name in ALL_OVERALL_METRIC_NAMES:
            values = [
                metrics[metric_name]
                for metrics in field_metrics.values()
                if metrics.get(metric_name) is not None
            ]
            overall[metric_name] = sum(values) / len(values) if values else None

        return {"fields": field_metrics, "overall": overall}

    def save_mismatches(
        self,
        experiment_name: str,
        run_name: str,
    ) -> list[Path]:
        """Saves the mismatches from the evaluation into temporary csv files, one
        shape per metric strategy.

        Writes into the system temp directory - the caller is expected to upload
        the returned files as MLflow artifacts and then remove them. Depending on
        which field strategies are present in the experiment, up to three files
        are produced:

        - Multi-value fields: a false positives file
          ({experiment_name}_{run_name}_false_positives.csv) with one row per
          (image_url, field) page holding the hallucinated values (extracted but
          not in the ground truth for that page, as a list), and a false
          negatives file ({experiment_name}_{run_name}_false_negatives.csv) with
          one row per page holding the missed values (in the ground truth for
          that page but never extracted, as a list).
        - Single-value fields: a mismatches file
          ({experiment_name}_{run_name}_mismatches.csv) with one row per
          mismatched page - columns image_url, field, extracted_value,
          ground_truth_value.

        A file is only written (and returned) when it has at least one record, so
        an experiment with only one kind of field produces only the relevant
        file(s).

        Args:
            experiment_name (str): Name of the experiment to save mismatches for.
            run_name (str): Name of the run to save mismatches for.
        Returns:
            list[Path]: Paths to the csv files that were actually written.
        """
        matches, _ = self._match_all_fields(experiment_name, run_name)
        real_matches = [match for match in matches.values() if match is not None]

        false_positive_records = [
            {
                **record,
                "hallucinated_values": self._format_values(
                    record["hallucinated_values"]
                ),
            }
            for match in real_matches
            if match["strategy"] == "multi_value"
            for record in match["false_positive_records"]
        ]
        false_negative_records = [
            {
                **record,
                "missed_values": self._format_values(record["missed_values"]),
            }
            for match in real_matches
            if match["strategy"] == "multi_value"
            for record in match["false_negative_records"]
        ]
        mismatch_records = [
            record
            for match in real_matches
            if match["strategy"] == "single_value"
            for record in match["mismatch_records"]
        ]

        tmp_dir = Path(tempfile.gettempdir())
        written_paths: list[Path] = []
        for records, suffix in (
            (false_positive_records, "false_positives"),
            (false_negative_records, "false_negatives"),
            (mismatch_records, "mismatches"),
        ):
            if not records:
                continue
            path = tmp_dir / f"{experiment_name}_{run_name}_{suffix}.csv"
            pd.DataFrame(records).to_csv(path, index=False)
            written_paths.append(path)

        return written_paths
