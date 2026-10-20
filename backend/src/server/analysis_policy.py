"""Conservative AST filtering, not an isolation boundary for hostile Python."""

import ast
import builtins
from collections.abc import Callable, Mapping
from pathlib import Path
from types import CodeType, ModuleType

ALLOWED_MODULES = frozenset(
    """
    pandas polars numpy numpy.linalg numpy.fft numpy.random
    matplotlib matplotlib.pyplot matplotlib.dates matplotlib.ticker
    seaborn sklearn sklearn.linear_model sklearn.preprocessing sklearn.pipeline
    sklearn.compose sklearn.impute sklearn.model_selection sklearn.metrics
    sklearn.cluster sklearn.decomposition sklearn.ensemble sklearn.tree
    sklearn.svm sklearn.neighbors sklearn.naive_bayes sklearn.feature_selection
    statsmodels statsmodels.api statsmodels.formula.api statsmodels.tsa.api
    statsmodels.tsa.stattools statsmodels.tsa.seasonal statsmodels.tsa.arima.model
    statsmodels.tsa.statespace.sarimax statsmodels.tsa.holtwinters
    sktime sktime.forecasting.base sktime.forecasting.arima
    sktime.forecasting.naive sktime.forecasting.exp_smoothing
    sktime.forecasting.theta sktime.forecasting.trend sktime.forecasting.compose
    sktime.forecasting.model_selection sktime.performance_metrics.forecasting
    os os.path
    """.split()
)

ALLOWED_ATTRIBUTES = frozenset(
    """
    DataFrame Series Index DatetimeIndex PeriodIndex Timestamp Timedelta
    Categorical Float64 Int64 Utf8 Date Datetime String
    array asarray arange linspace zeros ones empty full eye concatenate stack
    column_stack vstack hstack where select clip abs sqrt log log1p exp diff
    isnan isfinite isinf nan inf pi mean median std var sum min max percentile
    quantile corrcoef cov polyfit polyval unique sort argsort dot transpose
    reshape flatten ravel astype squeeze round copy to_numpy to_pandas from_pandas
    to_datetime to_numeric date_range period_range concat merge pivot_table crosstab
    cut qcut get_dummies isna notna isnull notnull fillna dropna replace interpolate
    groupby group_by agg aggregate transform apply map applymap filter select
    drop rename reset_index set_index sort_values sort_index sort select_dtypes
    value_counts nunique duplicated drop_duplicates describe info head tail sample
    columns index values shape size ndim dtype dtypes empty loc iloc at iat
    dt str cat year month day hour minute second dayofweek quarter days
    to_period to_timestamp total_seconds normalize floor ceil strftime
    contains startswith endswith lower upper strip lstrip rstrip split join
    rolling expanding ewm resample shift pct_change cumsum cumprod cummin cummax
    count prod sem skew kurt corr cov rank idxmin idxmax nsmallest nlargest
    isin between all any bool item items keys get to_dict to_list tolist
    assign melt pivot unstack stack explode reindex align combine_first
    col lit with_columns alias cast collect to_series len
    pyplot dates ticker linalg fft random api tsa formula
    linear_model preprocessing pipeline compose impute model_selection metrics
    cluster decomposition ensemble tree svm neighbors naive_bayes feature_selection
    forecasting base arima naive exp_smoothing theta trend performance_metrics
    subplots subplot figure plot scatter bar barh hist boxplot pie fill_between
    imshow matshow errorbar axhline axvline xlabel ylabel title legend grid
    xticks yticks xlim ylim tight_layout close gca gcf set_title set_xlabel
    set_ylabel set_xlim set_ylim set_xticks set_yticks tick_params set_xscale
    set_yscale set_major_locator set_major_formatter xaxis yaxis set
    suptitle autofmt_xdate set_size_inches subplots_adjust annotate text
    lineplot scatterplot barplot histplot kdeplot boxplot violinplot heatmap
    pairplot regplot jointplot displot set_theme set_style despine color_palette
    DateFormatter MonthLocator YearLocator AutoDateLocator ConciseDateFormatter
    MaxNLocator FuncFormatter PercentFormatter MultipleLocator
    fit predict fit_predict fit_transform inverse_transform score transform
    get_params set_params coef_ intercept_ labels_ cluster_centers_
    feature_importances_ explained_variance_ratio_ components_
    LinearRegression Ridge Lasso ElasticNet LogisticRegression
    StandardScaler MinMaxScaler RobustScaler OneHotEncoder LabelEncoder
    PolynomialFeatures SimpleImputer Pipeline make_pipeline ColumnTransformer
    train_test_split cross_val_score GridSearchCV TimeSeriesSplit
    mean_squared_error mean_absolute_error r2_score accuracy_score
    classification_report confusion_matrix KMeans DBSCAN PCA
    RandomForestRegressor RandomForestClassifier GradientBoostingRegressor
    DecisionTreeRegressor DecisionTreeClassifier SVR SVC KNeighborsRegressor
    GaussianNB SelectKBest OLS WLS GLM add_constant summary params pvalues
    rsquared residuals resid fittedvalues forecast get_forecast predicted_mean
    conf_int seasonal_decompose STL ARIMA SARIMAX ExponentialSmoothing
    SimpleExpSmoothing Holt acf pacf adfuller plot_acf plot_pacf
    AutoARIMA NaiveForecaster ThetaForecaster PolynomialTrendForecaster
    ForecastingHorizon ForecastingGridSearchCV ForecastingPipeline
    TransformedTargetForecaster temporal_train_test_split
    mean_absolute_percentage_error predict_interval predict_quantiles
    default_rng normal uniform integers choice
    path basename dirname splitext abspath normpath exists isfile isdir makedirs
    """.split()
)

OUTPUT_METHODS = frozenset({"to_csv", "write_csv", "savefig"})
SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs all any bool dict enumerate filter float int isinstance len list map "
        "max min next range reversed round set slice sorted str sum tuple zip "
        "Exception ValueError TypeError KeyError RuntimeError ZeroDivisionError"
    ).split()
}


def check_import(module: str, names: tuple[str, ...] = ()) -> None:
    if module not in ALLOWED_MODULES:
        raise ValueError(f"Unsafe analysis code: import {module!r} is not allowed")
    for name in names:
        if name not in ALLOWED_ATTRIBUTES and f"{module}.{name}" not in ALLOWED_MODULES:
            raise ValueError(f"Unsafe analysis code: import {name!r} is not allowed")


def restricted_import(
    name: str,
    globals: Mapping[str, object] | None = None,
    locals: Mapping[str, object] | None = None,
    fromlist: tuple[str, ...] | None = None,
    level: int = 0,
) -> ModuleType:
    if level:
        raise ValueError("Unsafe analysis code: relative imports are not allowed")
    check_import(name, fromlist or ())
    return builtins.__import__(name, globals, locals, fromlist, level)


class AnalysisValidator(ast.NodeVisitor):
    def __init__(self, tree: ast.Module) -> None:
        self.output_attributes = {
            id(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

    def generic_visit(self, node: ast.AST) -> None:
        allowed = (
            ast.Module,
            ast.Import,
            ast.ImportFrom,
            ast.alias,
            ast.Assign,
            ast.AugAssign,
            ast.Expr,
            ast.If,
            ast.IfExp,
            ast.For,
            ast.Break,
            ast.Continue,
            ast.Pass,
            ast.FunctionDef,
            ast.arguments,
            ast.arg,
            ast.Return,
            ast.Lambda,
            ast.Raise,
            ast.Try,
            ast.ExceptHandler,
            ast.Name,
            ast.Constant,
            ast.Attribute,
            ast.Subscript,
            ast.Slice,
            ast.List,
            ast.Tuple,
            ast.Dict,
            ast.Set,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.comprehension,
            ast.Call,
            ast.keyword,
            ast.BinOp,
            ast.UnaryOp,
            ast.BoolOp,
            ast.Compare,
            ast.JoinedStr,
            ast.FormattedValue,
            ast.Starred,
            ast.operator,
            ast.unaryop,
            ast.boolop,
            ast.cmpop,
            ast.expr_context,
        )
        if not isinstance(node, allowed):
            raise ValueError(
                f"Unsafe analysis code: {type(node).__name__} is not allowed"
            )
        # Keyword argument names (e.g. format=...) only bind parameters of the
        # callee and cannot reach builtins, so they are not name-checked.
        fields = (
            () if isinstance(node, ast.keyword) else ("id", "arg", "asname", "name")
        )
        for field in fields:
            value = getattr(node, field, None)
            if isinstance(value, str) and (
                value.startswith("_")
                or (value in vars(builtins) and value not in SAFE_BUILTINS)
            ):
                raise ValueError(f"Unsafe analysis code: name {value!r} is not allowed")
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            check_import(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or node.module is None:
            raise ValueError("Unsafe analysis code: relative imports are not allowed")
        check_import(node.module, tuple(alias.name for alias in node.names))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Store) or (
            node.attr not in ALLOWED_ATTRIBUTES
            and not (node.attr in OUTPUT_METHODS and id(node) in self.output_attributes)
        ):
            raise ValueError(
                f"Unsafe analysis code: attribute {node.attr!r} is not allowed"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, (ast.Name, ast.Attribute)):
            raise ValueError("Unsafe analysis code: indirect calls are not allowed")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.decorator_list:
            raise ValueError("Unsafe analysis code: decorators are not allowed")
        self.generic_visit(node)


class OutputGuard(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr in OUTPUT_METHODS:
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="_save_output", ctx=ast.Load()),
                    args=[node.func, *node.args],
                    keywords=node.keywords,
                ),
                node,
            )
        return node


def compile_analysis(code: str) -> CodeType:
    tree = ast.parse(code, filename="<analysis>", mode="exec")
    AnalysisValidator(tree).visit(tree)
    guarded_tree = OutputGuard().visit(tree)
    return compile(ast.fix_missing_locations(guarded_tree), "<analysis>", "exec")


def save_output(
    output_dir: Path,
    method: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> object:
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.figure import Figure

    name = getattr(method, "__name__", "")
    owner = getattr(method, "__self__", None)
    if name == "savefig":
        if method is not plt.savefig and not isinstance(owner, Figure):
            raise ValueError("Unsafe analysis code: unsupported plot writer")
        path_key, suffix = "fname", ".png"
        if kwargs.get("format", "png") != "png" or "backend" in kwargs:
            raise ValueError("Unsafe analysis code: only PNG plots are allowed")
    elif name == "to_csv" and isinstance(owner, (pd.DataFrame, pd.Series)):
        path_key, suffix = "path_or_buf", ".csv"
        if "compression" in kwargs or "storage_options" in kwargs:
            raise ValueError(
                "Unsafe analysis code: CSV storage options are not allowed"
            )
    elif name == "write_csv":
        import polars as pl

        if not isinstance(owner, pl.DataFrame):
            raise ValueError("Unsafe analysis code: unsupported CSV writer")
        path_key, suffix = "file", ".csv"
    else:
        raise ValueError("Unsafe analysis code: unsupported output writer")
    path = args[0] if args else kwargs.get(path_key)
    if not isinstance(path, str):
        raise ValueError("Unsafe analysis code: output path must be a string")
    target = Path(path).resolve()
    if not target.is_relative_to(output_dir.resolve()) or target.suffix != suffix:
        raise ValueError(
            "Unsafe analysis code: output must stay in the output directory"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if args:
        args = (str(target), *args[1:])
    else:
        kwargs[path_key] = str(target)
    return method(*args, **kwargs)
