"""Guards for LLM-generated analysis code. Not an isolation boundary for hostile
Python: the code runs in the backend process, and these checks only stop the
obviously harmful things a model might write by mistake or via prompt injection.

Analysis code may use any public API of the allowed packages, except:
- file and network I/O other than the guarded CSV/PNG outputs (read_*, to_pickle,
  np.load, ...), and access to environment variables or processes,
- evaluating code from strings (eval, query, statsmodels formulas) and
  str.format field lookups like "{0.__class__}",
- modules outside the allowed packages, frames/code objects (reflection), private
  (_-prefixed) names, and modifying modules or classes.

Attribute names are checked statically for early, readable errors. Every
attribute access and assignment is additionally rewritten into a runtime check
against the actual object, which is what keeps e.g. `matplotlib.subprocess` or
`df.iterrows().gi_frame` out of reach despite the permissive name rules.
"""

import ast
import builtins
import re
import string
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from types import CodeType, FrameType, ModuleType, TracebackType
from typing import cast

# Third-party packages whose public submodules may be imported.
ALLOWED_PACKAGES = frozenset(
    "pandas polars numpy scipy matplotlib seaborn sklearn statsmodels sktime".split()
)
# Standard-library modules without file, network, process or reflection access.
ALLOWED_STDLIB_MODULES = frozenset(
    """
    math cmath statistics datetime calendar zoneinfo collections collections.abc
    itertools functools operator decimal fractions numbers random re string
    textwrap json copy bisect heapq warnings
    os os.path posixpath ntpath
    """.split()
)
# Submodules that read/write files, download data, evaluate strings or wrap C
# libraries; blocked inside every allowed package (e.g. pandas.io, sklearn.datasets).
DENIED_SUBMODULES = frozenset(
    """
    io iolib datasets testing tests conftest ctypeslib f2py distutils formula
    computation backends animation
    """.split()
)
# `os` itself only for path handling.
OS_ATTRIBUTES = frozenset("path sep linesep makedirs getcwd".split())
# Objects defined in these modules are never handed to analysis code, wherever
# an allowed package re-exports them (e.g. a library that did `from os import *`).
DENIED_ORIGINS = frozenset(
    """
    os posix nt subprocess shutil sys io _io pickle _pickle copyreg socket ssl
    urllib http ftplib smtplib requests urllib3 webbrowser ctypes _ctypes
    importlib pkgutil runpy code codeop inspect pydoc gc marshal shelve dbm
    sqlite3 tempfile glob pathlib zipfile tarfile gzip bz2 lzma multiprocessing
    threading _thread concurrent asyncio signal pty logging joblib cloudpickle
    dill patsy numexpr pyarrow fsspec h5py tables sqlalchemy openpyxl xlsxwriter
    xlrd odf lxml PIL IPython
    """.split()
)
# Attribute names that read/write files, download, evaluate strings, run
# processes or expose frames - whatever object they are looked up on.
DENIED_ATTRIBUTE = re.compile(
    r"""^(
        (read|scan|sink|fetch|print)_.*
        | write_(?!csv$).*
        | to_(pickle|parquet|feather|hdf|sql|stata|excel|json|html|xml|latex
              |markdown|clipboard|orc|gbq|string|ipc|avro|delta|database|iceberg
              |html5_video|jshtml|file)
        | load(s|txt|mat|_.*)? | save(z|z_compressed|txt|mat|_.*)?
        | dumps? | tofile | fromfile | fromregex | genfromtxt | memmap | open_memmap
        | DataSource | HDFStore | ExcelWriter | ExcelFile | imread | imsave
        | serialize | deserialize | get_rdataset | get_dataset_names | webdoc | test
        | register_plugin_function | SQLContext | sql | from_formula
        | from_custom_template | switch_backend
        | eval | query | exec.* | system | popen | spawn.* | startfile
        | environ | environb | getenv | putenv | unsetenv | expandvars
        | vformat | attrgetter | methodcaller
        | gi_.* | cr_.* | ag_.* | tb_.* | co_.*
        | f_(globals|locals|builtins|back|code|trace|lasti|lineno)
    )$""",
    re.VERBOSE,
)
# Methods writing CSV/PNG outputs; only callable directly, through save_output.
OUTPUT_METHODS = frozenset({"to_csv", "write_csv", "savefig"})
# Methods that look up and call a method by name when given a string, e.g.
# df.agg("sum") - their string arguments get the same name checks.
STRING_DISPATCH_METHODS = frozenset({"agg", "aggregate", "apply", "transform"})

SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs all any ascii bin bool callable chr complex dict divmod enumerate "
        "filter float format frozenset hash hex int isinstance issubclass iter "
        "len list map max min next oct ord pow print range repr reversed round "
        "set slice sorted str sum tuple zip "
        "Exception ArithmeticError AssertionError AttributeError IndexError "
        "KeyError LookupError NotImplementedError OverflowError RuntimeError "
        "StopIteration TypeError ValueError ZeroDivisionError Warning UserWarning"
    ).split()
}
# Builtins that must not be reachable through an attribute either.
_DENIED_BUILTINS = frozenset(
    id(getattr(builtins, name))
    for name in (
        "eval exec compile open input breakpoint __import__ getattr setattr "
        "delattr globals locals vars dir help exit quit memoryview type object "
        "super classmethod staticmethod property"
    ).split()
    if hasattr(builtins, name)
)


def _unsafe(message: str) -> ValueError:
    return ValueError(f"Unsafe analysis code: {message}")


def module_allowed(name: str) -> bool:
    if name in ALLOWED_STDLIB_MODULES:
        return True
    parts = name.split(".")
    return (
        parts[0] in ALLOWED_PACKAGES
        and not any(part.startswith("_") for part in parts)
        and not any(part in DENIED_SUBMODULES for part in parts[1:])
    )


def check_attribute_name(name: str) -> None:
    if name.startswith("_") or name == "*" or DENIED_ATTRIBUTE.match(name):
        raise _unsafe(f"attribute {name!r} is not allowed")


def check_import(module: str, names: tuple[str, ...] = ()) -> None:
    if not module_allowed(module):
        raise _unsafe(f"import {module!r} is not allowed")
    for name in names:
        check_attribute_name(name)
        if module == "os" and name not in OS_ATTRIBUTES:
            raise _unsafe(f"import {name!r} from 'os' is not allowed")


def restricted_import(
    name: str,
    globals: Mapping[str, object] | None = None,
    locals: Mapping[str, object] | None = None,
    fromlist: tuple[str, ...] | None = None,
    level: int = 0,
) -> ModuleType:
    if level:
        raise _unsafe("relative imports are not allowed")
    check_import(name, fromlist or ())
    module = builtins.__import__(name, globals, locals, fromlist, level)
    # `from x import y` fetches y without an attribute access the code could
    # guard, so check what it would get here.
    for attribute in fromlist or ():
        guarded_getattr(module, attribute)
    return module


def guarded_getattr(obj: object, name: str) -> object:
    """Runtime check of `obj.name`, on the actual objects involved."""
    if name not in OUTPUT_METHODS:
        check_attribute_name(name)
    is_os = isinstance(obj, ModuleType) and obj.__name__ == "os"
    if is_os and name not in OS_ATTRIBUTES:
        raise _unsafe(f"attribute {name!r} of 'os' is not allowed")
    value = getattr(obj, name)

    if isinstance(value, ModuleType):
        if not module_allowed(value.__name__):
            raise _unsafe(f"module {value.__name__!r} is not allowed")
    elif isinstance(value, (FrameType, CodeType, TracebackType)):
        raise _unsafe(f"attribute {name!r} is not allowed")
    elif id(value) in _DENIED_BUILTINS:
        raise _unsafe(f"attribute {name!r} is not allowed")
    elif not is_os:
        origin = getattr(value, "__module__", None)
        if isinstance(origin, str) and origin.split(".")[0] in DENIED_ORIGINS:
            raise _unsafe(f"attribute {name!r} from {origin!r} is not allowed")

    if name in ("format", "format_map"):
        return _checked_format(obj, value)
    if name in ("use", "context"):
        return _guard_style(value)
    if name in STRING_DISPATCH_METHODS and callable(value):
        return _checked_dispatch(value)
    return value


def guarded_setattr(obj: object, name: str, value: object) -> None:
    """Runtime check of `obj.name = value`: data objects only, so analysis code
    can't patch library modules or classes shared by later executions."""
    check_attribute_name(name)
    if name in OUTPUT_METHODS or isinstance(obj, (ModuleType, type)):
        raise _unsafe(f"assigning {name!r} on {type(obj).__name__} is not allowed")
    setattr(obj, name, value)


def _checked_format(obj: object, method: object) -> object:
    # str.format resolves "{0.attr}" / "{0[key]}" itself, past the attribute
    # guard, so only plain fields like "{}", "{0}", "{name:.2f}" are allowed.
    if not isinstance(obj, str):
        raise _unsafe("format is only allowed on strings")
    check_format_string(obj)
    return method


def check_format_string(text: str) -> None:
    fields = [field for _, field, _, _ in string.Formatter().parse(text) if field]
    if any("." in field or "[" in field for field in fields):
        raise _unsafe(f"format fields with attribute or index lookups in {text!r}")


def _guard_style(value: object) -> object:
    import matplotlib
    import matplotlib.style

    if value is matplotlib.use:
        # Backend names can be "module://<any installed module>".
        raise _unsafe("matplotlib.use is not allowed")
    if value not in (matplotlib.style.use, matplotlib.style.context):
        return value
    style_function = cast(Callable[..., object], value)

    def checked_style(style: object, *args: object, **kwargs: object) -> object:
        # Style names and rc dicts only: matplotlib also takes file paths,
        # URLs and importable package names here.
        styles = style if isinstance(style, list) else [style]
        for item in styles:
            if not isinstance(item, dict) and item not in matplotlib.style.available:
                raise _unsafe(f"style {item!r} is not a built-in matplotlib style")
        return style_function(style, *args, **kwargs)

    return checked_style


def _checked_dispatch(method: Callable[..., object]) -> Callable[..., object]:
    def checked(*args: object, **kwargs: object) -> object:
        pending: list[object] = [*args, *kwargs.values()]
        while pending:
            item = pending.pop()
            if isinstance(item, str):
                check_attribute_name(item)
            elif isinstance(item, (list, tuple)):
                pending.extend(item)
            elif isinstance(item, dict):
                pending.extend(item.values())
        return method(*args, **kwargs)

    return checked


class AnalysisValidator(ast.NodeVisitor):
    ALLOWED_NODES = (
        ast.Module,
        ast.Import,
        ast.ImportFrom,
        ast.alias,
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.Delete,
        ast.Expr,
        ast.If,
        ast.IfExp,
        ast.For,
        ast.Break,
        ast.Continue,
        ast.Pass,
        ast.With,
        ast.withitem,
        ast.Assert,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.Lambda,
        ast.Raise,
        ast.Try,
        ast.ExceptHandler,
        ast.Name,
        ast.NamedExpr,
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

    def __init__(self, tree: ast.Module) -> None:
        self.call_targets = {
            id(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        # `obj.name = value` as the single target of a plain assignment is
        # rewritten into guarded_setattr; other attribute stores are rejected.
        self.attribute_stores = {
            id(node.targets[0])
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
        }
        # Names bound to the os module, whose attributes are checked statically.
        self.os_names = {
            alias.asname if alias.asname else "os"
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name == "os" or (not alias.asname and alias.name == "os.path")
        }

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, self.ALLOWED_NODES):
            raise _unsafe(f"{type(node).__name__} is not allowed")
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
                raise _unsafe(f"name {value!r} is not allowed")
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            check_import(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or node.module is None:
            raise _unsafe("relative imports are not allowed")
        check_import(node.module, tuple(alias.name for alias in node.names))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in OUTPUT_METHODS:
            if id(node) not in self.call_targets:
                raise _unsafe(f"attribute {node.attr!r} is not allowed")
        else:
            check_attribute_name(node.attr)
        if (
            node.attr in ("format", "format_map")
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            check_format_string(node.value.value)
        if isinstance(node.ctx, ast.Del) or (
            isinstance(node.ctx, ast.Store) and id(node) not in self.attribute_stores
        ):
            raise _unsafe(f"assigning or deleting {node.attr!r} is not allowed here")
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self.os_names
            and node.attr not in OS_ATTRIBUTES
        ):
            raise _unsafe(f"attribute {node.attr!r} of 'os' is not allowed")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.decorator_list:
            raise _unsafe("decorators are not allowed")
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


class AttributeGuard(ast.NodeTransformer):
    """Rewrites `obj.name` into `_getattr(obj, "name")` and `obj.name = value`
    into `_setattr(obj, "name", value)`, so every access is checked at runtime."""

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id="_getattr", ctx=ast.Load()),
                args=[node.value, ast.Constant(node.attr)],
                keywords=[],
            ),
            node,
        )

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        target = node.targets[0]
        if not isinstance(target, ast.Attribute):
            self.generic_visit(node)
            return node
        call = ast.Call(
            func=ast.Name(id="_setattr", ctx=ast.Load()),
            args=[
                self.visit(target.value),
                ast.Constant(target.attr),
                self.visit(node.value),
            ],
            keywords=[],
        )
        return ast.copy_location(ast.Expr(call), node)


def compile_analysis(code: str) -> CodeType:
    tree = ast.parse(code, filename="<analysis>", mode="exec")
    AnalysisValidator(tree).visit(tree)
    guarded_tree = AttributeGuard().visit(OutputGuard().visit(tree))
    return compile(ast.fix_missing_locations(guarded_tree), "<analysis>", "exec")


def analysis_globals(output_dir: Path) -> dict[str, object]:
    """The guarded globals compiled analysis code must be executed with."""
    return {
        "__builtins__": {**SAFE_BUILTINS, "__import__": restricted_import},
        "_save_output": partial(save_output, output_dir),
        "_getattr": guarded_getattr,
        "_setattr": guarded_setattr,
    }


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
