# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import dataclasses
from dataclasses import dataclass, asdict
from typing import Type, Callable

from .adapters import Driver, MatrixAdapter, MatrixSpyAdapter
from .html_formatter import HTMLTableFormatter, NotebookHTMLFormatter
from .latex_formatter import LatexFormatter, python_scientific_to_latex_times10


@dataclass
class MatSpyParams:
    max_rows: int = 11
    """Maximum number of rows in HTML and Latex output."""

    max_cols: int = 11
    """Maximum number of columns in HTML and Latex output."""

    num_after_dots: int = 1
    """
    If a matrix has more rows or columns than allowed then an ellipsis (three dots) is emitted to cover the excess.
    This parameter controls how many rows/columns are drawn at the end of the matrix.
    
    For example, a value of 1 means the final row and final column are emitted in addition to the top-left corner.
    A value of 2 means the final two rows and columns are emitted, with a correspondingly smaller top-left corner.
    
    Note: ignored for matrix formats without fast row/column indexing, such as COO.
    """

    cell_align: str = "center"
    """tabular: Horizontal text alignment for non-zero values."""

    indices: bool = False
    """Whether to show row/column indices."""

    indices_spy: bool = None
    """If not None then overrides 'indices' for spy plots only."""

    title: bool | str = True
    """Title of table or spy plot. If `True` then generate matrix description such as dimensions, nnz, datatype."""

    title_latex: bool = False
    """If not None then overrides `title` for Latex output only."""

    latex_matrix_env: str = "bmatrix"
    """Latex environment to use for matrices. For Jupyter this should be one supported by MathJax."""

    shading: str = "relative"
    """
    How to shade buckets:
     - `'binary'`: A nonempty bucket is considered full.
     - `'relative'`: A bucket is shaded relative to the fullness of the fullest bucket in the matrix.
     - `'absolute'`: A bucket is shaded according to how full it is.
    """

    shading_absolute_min: float = 0.2
    """
    if `shading == 'absolute'`: buckets with values less than this will be clipped to this value
    so they may be visible in the plot.
    """

    shading_relative_min: float = 0.4
    """
    if `shading == 'relative'`: the lightest non-zeros have this value.
    """

    shading_relative_max_percentile: float = 0.99
    """
    if `shading == 'relative'`: define what a 'full' bucket is as a percentile of the nonzero buckets.
    A simple max would allow one or two outliers to skew the entire range making the plot appear too light.
    """

    figsize: float = 3.5
    """Figure size for spy plots, of longest side, in default matplotlib units (inches)."""

    sparkline_size: float = 1
    """Figure size for sparklines, of longest side, in default matplotlib units (inches)."""

    dpi: float = None
    """spy: Default spy image DPI. If None the matplotlib default Figure dpi is used."""

    buckets: int = None
    """spy: Pixel count of longest side of spy image. If None then computed from size and DPI."""

    spy_aa_tweaks_enabled: bool = None
    """
    Whether to_sparkline() may tweak parameters like bucket count to prevent visible aliasing artifacts.
    If None then defaults to True if dpi and buckets are also None.
    """

    color_empty: tuple[float, float, float, float] | str = (1.0, 1.0, 1.0, 1.0)  # RGBA: empty space is white
    """Spy (and sparkline) color for empty space. Can be anything matplotlib accepts, like RGB or RGBA tuples."""

    color_full: tuple[float, float, float, float] | str = (0.0, 0.0, 1.0, 1.0)  # RGBA: non-zeros are blue
    """Spy (and sparkline) color for a full bucket. Can be anything matplotlib accepts, like RGB or RGBA tuples."""

    float_formatter: Callable[[float], str] = lambda f: format(f, ".4g")
    """
    A callable for converting floating point numbers to string.
    For convenience may also be a format string `fmt_str` and this will be done for you:
    `float_formatter = lambda f: format(f, fmt_str)`
    """

    float_formatter_latex: Callable[[float], str] = None
    """
    Make floating-point numbers look better in Latex. Notably enable conversion of scientific notation styles.
    Also accepts the original float to enable more complex formatting. Called as:
    `float_formatter_latex_postprocess(float_formatter(f), f)`
    """

    def set_precision(self, precision, g=True):
        """
        Precision to use for floating-point to string conversion.
        """
        fmt_str = f".{precision}{'g' if g else ''}"
        self.float_formatter = lambda f: format(f, fmt_str)

    def _assert_one_of(self, var, choices):
        if getattr(self, var) not in choices:
            raise ValueError(f"{var} must be one of: " + ", ".join(choices))

    def get(self, **kwargs):
        ret = dataclasses.replace(self)

        # Allow some explicit overwrites for convenience
        if "title" in kwargs and "title_latex" not in kwargs:
            kwargs["title_latex"] = kwargs["title"]

        if "figsize" in kwargs and "sparkline_size" not in kwargs:
            kwargs["sparkline_size"] = kwargs["figsize"]

        # Update all parameters with the ones in kwargs
        for key, value in kwargs.items():
            if hasattr(ret, key):
                setattr(ret, key, value)

        # Handy type conversions
        if isinstance(ret.float_formatter, str):
            fmt_str = ret.float_formatter
            ret.float_formatter = lambda f: format(f, fmt_str)

        # validate
        self._assert_one_of("shading", ['relative', 'absolute', 'binary'])
        self._assert_one_of("cell_align", ['center', 'left', 'right'])

        # Apply some default rules
        if ret.title_latex is None:
            ret.title_latex = ret.title

        if ret.indices_spy is None:
            ret.indices_spy = ret.indices

        if ret.spy_aa_tweaks_enabled is None:
            ret.spy_aa_tweaks_enabled = ret.buckets is None and ret.dpi is None

        if ret.float_formatter_latex is None:
            ret.float_formatter_latex = lambda f: python_scientific_to_latex_times10(ret.float_formatter(f))

        return ret

    def to_kwargs(self):
        return asdict(self)


params = MatSpyParams()
_drivers: list[Type[Driver]] = []
_driver_map: dict[str, Type[Driver]] = {}
_driver_registration_notify: list[Callable[[Type[Driver]], None]] = []


def register_driver(driver: Type[Driver]):
    _drivers.append(driver)

    for type_module, type_name, _ in driver.get_supported_types():
        _driver_map[".".join((type_module, type_name))] = driver

    for func in _driver_registration_notify:
        func(driver)


def _register_bundled():
    """
    Register the built-in drivers.
    """
    from .adapters.scipy_driver import SciPyDriver
    register_driver(SciPyDriver)

    from .adapters.list_like import ListDriver
    register_driver(ListDriver)


_register_bundled()


def _get_driver(mat):
    if isinstance(mat, list):
        type_str = "builtins.list"
    elif isinstance(mat, tuple):
        type_str = "builtins.tuple"
    else:
        type_str = ".".join((mat.__module__, mat.__class__.__name__))
    driver = _driver_map.get(type_str, None)
    if not driver:
        raise AttributeError("Unsupported type: " + type_str)
    return driver


def _get_adapter(mat) -> MatrixAdapter:
    adapter = _get_driver(mat).adapt(mat)
    if not adapter:
        raise AttributeError("Unsupported matrix")

    return adapter


def _get_spy_adapter(mat) -> MatrixSpyAdapter:
    if isinstance(mat, MatrixSpyAdapter):
        return mat

    adapter = _get_driver(mat).adapt_spy(mat)
    if not adapter:
        raise AttributeError("Unsupported matrix")

    return adapter


def to_html(mat, notebook=False, precision=None, **kwargs) -> str:
    options = params.get(**kwargs)
    if precision is not None:
        options.set_precision(precision)
    adapter = _get_adapter(mat)

    if notebook:
        formatter = NotebookHTMLFormatter(**options.to_kwargs())
    else:
        formatter = HTMLTableFormatter(**options.to_kwargs())

    return str(formatter.format(adapter))


def to_latex(mat, precision=None, **kwargs):
    options = params.get(**kwargs)
    if precision is not None:
        options.set_precision(precision)
    adapter = _get_adapter(mat)

    formatter = LatexFormatter(**options.to_kwargs())

    return str(formatter.format(adapter))


def to_spy_heatmap(mat, buckets=500, **kwargs):
    options = params.get(**kwargs)
    options.buckets = buckets
    adapter = _get_spy_adapter(mat)

    from .spy_renderer import get_spy_heatmap
    heatmap = get_spy_heatmap(adapter, **options.to_kwargs())
    return heatmap


def spy_to_mpl(mat, **kwargs):
    from . import spy_renderer
    return spy_renderer.spy_to_mpl(mat, **kwargs)


def spy(mat, **kwargs):
    from . import spy_renderer
    spy_renderer.spy(mat, **kwargs)


def to_sparkline(mat, retscale=False, scale=None, html_border="1px solid black", **kwargs):
    from . import spy_renderer
    return spy_renderer.to_sparkline(mat=mat, retscale=retscale, scale=scale, html_border=html_border, **kwargs)


def mdisplay(mat, method="html", **kwargs):
    from IPython.display import display, HTML, Latex

    if method == "html":
        display(HTML(to_html(mat, notebook=True, **kwargs)))
    elif method == "latex":
        display(Latex('$' + to_latex(mat, **kwargs) + '$'))
    elif method == "sparkline":
        display(HTML(to_sparkline(mat, **kwargs)))
    elif method == "spy":
        spy(mat, **kwargs)
    else:
        raise ValueError("Unknown method: " + method)


def _register_jupyter_formatter(mime_type: str, repr_method: Callable):
    """
    See https://ipython.readthedocs.io/en/stable/config/integrating.html
    """
    # This import is unnecessary but makes static type checking work.
    # noinspection PyProtectedMember
    from IPython import get_ipython

    try:
        formatter = get_ipython().display_formatter.formatters[mime_type]
    except AttributeError:
        # not running in a notebook
        return

    for driver in _drivers:
        for type_module, type_name, register_with_jupyter in driver.get_supported_types():
            if register_with_jupyter:
                formatter.for_type_by_name(type_module, type_name, repr_method)


__all__ = ["to_html", "to_latex", "to_sparkline", "to_spy_heatmap", "spy_to_mpl", "spy", "mdisplay"]
