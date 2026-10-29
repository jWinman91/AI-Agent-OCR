import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.server import dataset_metadata  # noqa: E402
from src.server.dataset_metadata import get_dataset_metadata  # noqa: E402


class DatasetMetadata(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def metadata(self, csv: str) -> str:
        path = Path(self.directory.name) / "data.csv"
        path.write_text(csv, encoding="utf-8")
        return get_dataset_metadata(path)

    def test_no_file(self) -> None:
        self.assertEqual(get_dataset_metadata(None), "")

    def test_small_dataset_shows_all_rows(self) -> None:
        text = self.metadata("day,kg\n1,80.0\n2,81.5\n")
        self.assertIn("Shape: 2 rows x 2 columns", text)
        self.assertIn("- day: dtype int64, 0 missing, 2 distinct", text)
        self.assertIn("numbers from 80.0 to 81.5", text)
        self.assertIn("ALL ROWS", text)

    def test_formatted_numbers_and_dates_stored_as_text(self) -> None:
        text = self.metadata(
            'Datum,Saldo\n01. Jan 26,"60.394,59 €"\n02. Jan 26,"1.200,00 €"\n'
        )
        self.assertIn("- Datum: dtype object", text)
        self.assertIn("dates/times stored as text, e.g. '01. Jan 26'", text)
        self.assertIn("- Saldo: dtype object", text)
        self.assertIn("numbers stored as text", text)
        self.assertIn("decimal separator ',', thousands separator '.'", text)
        self.assertIn("contains units/symbols ['€']", text)
        # Describes the data only; handling it is up to the analyser's prompt.
        self.assertNotIn("convert", text)
        self.assertNotIn("parse", text)

    def test_image_url_column_is_hidden(self) -> None:
        text = self.metadata("kg,image_url\n80.0,/data/a.png\n81.5,/data/b.png\n")
        self.assertIn("Shape: 2 rows x 1 columns", text)
        self.assertIn("Exact column names: ['kg']", text)
        self.assertNotIn("image_url", text)
        self.assertNotIn("a.png", text)

    def test_mixed_formats_in_one_column_are_listed(self) -> None:
        text = self.metadata(
            "Datum\n01. Jan 26\n02. Jan 26\n01. März 2026\n02. Mai 26\n"
        )
        self.assertIn(
            "formats (digit = 9): '99. Jan 99' (2), '99. März 9999' (1), "
            "'99. Mai 99' (1)",
            text,
        )

    def test_values_that_do_not_fit_the_column_are_flagged(self) -> None:
        rows = "".join(f"2026-01-0{day},{day}.5\n" for day in range(1, 6))
        text = self.metadata("when,amount\n" + rows + "yesterday,-\n")
        self.assertIn(
            "UNEXPECTED values that don't match this format: 'yesterday'", text
        )
        self.assertIn("UNEXPECTED values that don't match this format: '-'", text)

    def test_text_categories_are_all_listed(self) -> None:
        text = self.metadata("Typ\nZinsen\nErtrag\nZinsen\nÜberweisun\n")
        self.assertIn(
            "text, all values: 'Zinsen' (2), 'Ertrag' (1), 'Überweisun' (1)", text
        )

    def test_whole_numbers_with_missing_values(self) -> None:
        text = self.metadata("n,label\n1,a\n,b\n3,c\n")
        self.assertIn("- n: dtype float64, 1 missing", text)
        self.assertIn("whole numbers, float only because of missing values", text)

    def test_large_dataset_shows_a_fixed_sample_in_file_order(self) -> None:
        edge = dataset_metadata.SAMPLE_EDGE_ROWS
        size = dataset_metadata.MAX_FULL_ROWS + 300
        csv = "value\n" + "".join(f"{i}\n" for i in range(size))
        text = self.metadata(csv)
        self.assertEqual(text, self.metadata(csv))
        rows = text.split("SAMPLE ROWS")[1].splitlines()[2:]
        positions = [int(row.split()[0]) for row in rows]
        self.assertEqual(len(positions), 2 * edge + dataset_metadata.SAMPLE_RANDOM_ROWS)
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(positions[:edge], list(range(edge)))
        self.assertEqual(positions[-edge:], list(range(size - edge, size)))

    def test_sample_settings_that_overlap_do_not_break(self) -> None:
        with (
            patch.object(dataset_metadata, "MAX_FULL_ROWS", 3),
            patch.object(dataset_metadata, "SAMPLE_EDGE_ROWS", 3),
            patch.object(dataset_metadata, "SAMPLE_RANDOM_ROWS", 10),
        ):
            text = self.metadata("value\n" + "".join(f"{i}\n" for i in range(5)))
        self.assertIn("SAMPLE ROWS: first 3, 0 random and last 3", text)
        rows = text.split("SAMPLE ROWS")[1].splitlines()[2:]
        self.assertEqual([int(row.split()[0]) for row in rows], [0, 1, 2, 3, 4])

    def test_whole_floats_without_missing_values(self) -> None:
        text = self.metadata("n\n1.0\n2.0\n")
        self.assertIn("numbers from 1.0 to 2.0 (whole numbers, as float)", text)

    def test_empty_column(self) -> None:
        text = self.metadata("n,label\n,a\n,b\n")
        self.assertIn("- n: dtype float64, 2 missing, 0 distinct; no values", text)

    def test_thousands_separator_without_decimals(self) -> None:
        text = self.metadata('Betrag\n"1.234 €"\n"12.500 €"\n"3.000 €"\n')
        self.assertIn("'.' is always followed by 3 digits", text)
        self.assertIn("likely a thousands separator", text)

    def test_decimal_comma_only(self) -> None:
        text = self.metadata('Betrag\n"12,50"\n"3,1"\n')
        self.assertIn("decimal separator ','", text)

    def test_trailing_text_after_a_date_is_unexpected(self) -> None:
        dates = "".join(f"2026-01-0{day}\n" for day in range(1, 6))
        text = self.metadata("when\n" + dates + "2026-01-06 (storniert)\n")
        self.assertIn("UNEXPECTED values that don't match this format", text)
        self.assertIn("'2026-01-06 (storniert)' (1)", text)

    def test_iso_timestamps_are_dates(self) -> None:
        text = self.metadata("when\n2026-01-01T10:00:00Z\n2026-01-02 10:00:00.5\n")
        self.assertIn("dates/times stored as text", text)
        self.assertNotIn("UNEXPECTED", text)


if __name__ == "__main__":
    unittest.main()
