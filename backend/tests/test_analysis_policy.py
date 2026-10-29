import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.server.plot_server import execute_python_code  # noqa: E402
from src.utils.data_models import AnalyserResponse  # noqa: E402

# Ordinary analysis code the policy has to allow.
ALLOWED_PROGRAMS = {
    "style name": "import matplotlib.pyplot as plt\n"
    "plt.style.use('seaborn-v0_8-whitegrid')",
    "style context": "import matplotlib.pyplot as plt\n"
    "with plt.style.context('ggplot'):\n    plt.plot([1, 2])",
    "rc params": "import matplotlib.pyplot as plt\n"
    "plt.rcParams.update({'font.size': 12})",
    "date formatter": "import matplotlib.dates as mdates\n"
    "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\n"
    "ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))",
    "index assignment": "s = df.groupby('Typ')['value'].sum()\ns.index = ['x', 'y']",
    "columns assignment": "d = df.copy()\nd.columns = ['a', 'b']",
    "str.format": "t = '{:.2f} {name}'.format(1.5, name='x')",
    "re": "import re\nm = re.compile(r'\\d+').findall('a1b22')",
    "math and datetime": "import math, datetime\n"
    "x = math.floor(2.5) + datetime.date(2026, 1, 1).year",
    "scipy": "from scipy import stats\nr = stats.linregress([1, 2, 3], [2, 4, 7])",
    "agg by name": "r = df.groupby('Typ').agg({'value': 'sum'})",
    "seaborn": "import seaborn as sns\nsns.set_theme(style='whitegrid')\n"
    "ax = sns.lineplot(data=df, x='value', y='value')",
    "statsmodels": "import statsmodels.api as sm\n"
    "p = sm.OLS(df['value'], sm.add_constant(df['value'])).fit().params",
    "os.path": "import os\np = os.path.join('a', 'b')",
    "print and assert": "print(df.shape)\nassert len(df) == 3",
}

# Obviously harmful code, also when the harmful part is only known at runtime.
DENIED_PROGRAMS = {
    "style url": "import matplotlib.pyplot as plt\n"
    "plt.style.use('https://example.com/x.mplstyle')",
    "style package": "import matplotlib.pyplot as plt\nplt.style.use('os.anything')",
    "matplotlib backend": "import matplotlib\nmatplotlib.use('module://os')",
    "module re-exported by a package": "import matplotlib\nm = matplotlib.subprocess",
    "denied submodule via attribute": "import pandas as pd\nm = pd.io",
    "frame via generator": "g = df.iterrows()\nf = g.gi_frame",
    "method name built at runtime": "name = 'to_' + 'pickle'\ndf.agg(name, 'x.pkl')",
    "patching a class": "import pandas as pd\npd.DataFrame.diff = len",
    "patching a module": "import numpy as np\nnp.mean = len",
    "formula api": "import statsmodels.formula.api as smf",
    "from_formula": "import statsmodels.api as sm\n"
    "m = sm.OLS.from_formula('y ~ x', df)",
    "canvas print": "import matplotlib.pyplot as plt\n"
    "plt.figure().canvas.print_png('x.png')",
    "environment": "import os\nk = os.environ",
    "os alias": "import os as o\no.remove('x')",
    "os at runtime": "import os\nf = [os][0].system",
    "format string built at runtime": "t = '{0.__cl' + 'ass__}'\nx = t.format(df)",
    "format on non-strings": "x = df.style.format('{:.2f}')",
    "dataset download": "import seaborn as sns\nd = sns.load_dataset('iris')",
    "dataset module": "from sklearn.datasets import fetch_openml",
    "file read": "import polars as pl\nd = pl.read_csv('x.csv')",
    "file write": "df.to_excel('x.xlsx')",
    "numpy save": "import numpy as np\nnp.save('x.npy', [1])",
    "operator lookups": "import operator\ng = operator.attrgetter('x')",
    "typing evaluates strings": "import typing",
    "while loop": "while True:\n    pass",
}


class AnalysisPolicy(unittest.TestCase):
    def run_program(self, program: str) -> str | None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "src.server.plot_server.pd.read_csv",
                return_value=pd.DataFrame(
                    {"value": [1.0, 2.0, 3.0], "Typ": ["a", "b", "a"]}
                ),
            ),
        ):
            result = execute_python_code(
                AnalyserResponse(python_code=program, code_summary="summary"),
                Path("data.csv"),
                None,
                output_dir=Path(directory),
            )
        return result.error_message

    def test_ordinary_analysis_code_is_allowed(self) -> None:
        for name, program in ALLOWED_PROGRAMS.items():
            with self.subTest(name):
                self.assertIsNone(self.run_program(program))

    def test_harmful_code_is_rejected(self) -> None:
        for name, program in DENIED_PROGRAMS.items():
            with self.subTest(name):
                error = self.run_program(program)
                self.assertIsNotNone(error)
                self.assertIn("Unsafe analysis code", error or "")

    def test_styles_do_not_leak_into_later_analyses(self) -> None:
        import matplotlib.pyplot as plt

        before = plt.rcParams["axes.facecolor"]
        program = "import matplotlib.pyplot as plt\nplt.style.use('ggplot')"
        self.assertIsNone(self.run_program(program))
        self.assertEqual(plt.rcParams["axes.facecolor"], before)


if __name__ == "__main__":
    unittest.main()
