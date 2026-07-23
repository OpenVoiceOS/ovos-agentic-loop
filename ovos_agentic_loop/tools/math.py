# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MathToolBox — safe arithmetic and mathematical operations as agent tools.

Provides four tools:
- ``evaluate_expression`` — evaluate a numeric expression via Python's ``ast``
  module (no ``eval``; only arithmetic nodes are permitted).
- ``solve_equation`` — solve a linear or simple polynomial equation of the form
  ``f(x) = 0`` expressed as a string, using ``sympy`` when available or a
  numeric fallback otherwise.
- ``unit_convert`` — convert a value between common measurement units.
- ``statistics_summary`` — compute mean, median, std dev, min, max for a list
  of numbers.
"""
import ast
import math
import operator
import statistics
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agent_tools import AgentTool, ToolArguments, ToolBox, ToolOutput
from pydantic import Field


# ---------------------------------------------------------------------------
# Safe expression evaluator
# ---------------------------------------------------------------------------

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_FUNCS: Dict[str, Any] = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "lcm": getattr(math, "lcm", None),  # Python 3.9+
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

# Remove None entries (lcm on Python < 3.9)
_ALLOWED_FUNCS = {k: v for k, v in _ALLOWED_FUNCS.items() if v is not None}


def _safe_eval(node: ast.expr) -> float:
    """
    Recursively evaluate an AST expression using only allowed operations.

    Args:
        node: AST expression node.

    Returns:
        Numeric result.

    Raises:
        ValueError: If the expression contains disallowed constructs.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_FUNCS:
            val = _ALLOWED_FUNCS[node.id]
            if callable(val):
                raise ValueError(f"'{node.id}' is a function, not a constant")
            return float(val)
        raise ValueError(f"Unknown name: '{node.id}'")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Attribute access not permitted")
        fname = node.func.id
        if fname not in _ALLOWED_FUNCS or not callable(_ALLOWED_FUNCS[fname]):
            raise ValueError(f"Function not permitted: '{fname}'")
        evaluated_args = [_safe_eval(a) for a in node.args]
        # Functions that require integer arguments
        _INT_ARG_FUNCS = {"factorial", "gcd", "lcm"}
        if fname in _INT_ARG_FUNCS:
            evaluated_args = [int(a) for a in evaluated_args]
        return float(_ALLOWED_FUNCS[fname](*evaluated_args))

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"Operator not permitted: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return float(_ALLOWED_OPS[op_type](left, right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"Unary operator not permitted: {op_type.__name__}")
        operand = _safe_eval(node.operand)
        return float(_ALLOWED_OPS[op_type](operand))

    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Unit conversion table
# ---------------------------------------------------------------------------
# Format: {category: {unit_name: factor_to_si_base}}
# SI base units: metre, kilogram, second, celsius (as special case), litre
_UNIT_TABLE: Dict[str, Dict[str, float]] = {
    "length": {
        "mm": 0.001,
        "cm": 0.01,
        "m": 1.0,
        "km": 1000.0,
        "in": 0.0254,
        "inch": 0.0254,
        "ft": 0.3048,
        "foot": 0.3048,
        "feet": 0.3048,
        "yd": 0.9144,
        "yard": 0.9144,
        "mi": 1609.344,
        "mile": 1609.344,
        "miles": 1609.344,
        "nmi": 1852.0,
        "ly": 9.4607e15,
    },
    "mass": {
        "mg": 1e-6,
        "g": 0.001,
        "kg": 1.0,
        "t": 1000.0,
        "tonne": 1000.0,
        "oz": 0.028349523125,
        "lb": 0.45359237,
        "lbs": 0.45359237,
        "pound": 0.45359237,
        "pounds": 0.45359237,
        "stone": 6.35029318,
        "ton": 907.18474,   # short ton (US)
    },
    "volume": {
        "ml": 0.001,
        "l": 1.0,
        "litre": 1.0,
        "liter": 1.0,
        "cl": 0.01,
        "dl": 0.1,
        "m3": 1000.0,
        "cm3": 0.001,
        "mm3": 1e-6,
        "tsp": 0.00492892,
        "tbsp": 0.01478676,
        "floz": 0.0295735,
        "cup": 0.236588,
        "pt": 0.473176,
        "pint": 0.473176,
        "qt": 0.946353,
        "quart": 0.946353,
        "gal": 3.785411784,
        "gallon": 3.785411784,
    },
    "time": {
        "ns": 1e-9,
        "us": 1e-6,
        "ms": 0.001,
        "s": 1.0,
        "sec": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "min": 60.0,
        "minute": 60.0,
        "minutes": 60.0,
        "h": 3600.0,
        "hr": 3600.0,
        "hour": 3600.0,
        "hours": 3600.0,
        "d": 86400.0,
        "day": 86400.0,
        "days": 86400.0,
        "week": 604800.0,
        "weeks": 604800.0,
        "year": 31557600.0,
        "years": 31557600.0,
    },
    "speed": {
        "m/s": 1.0,
        "mps": 1.0,
        "km/h": 1.0 / 3.6,
        "kph": 1.0 / 3.6,
        "mph": 0.44704,
        "knot": 0.514444,
        "knots": 0.514444,
        "ft/s": 0.3048,
    },
    "area": {
        "mm2": 1e-6,
        "cm2": 1e-4,
        "m2": 1.0,
        "km2": 1e6,
        "ha": 1e4,
        "hectare": 1e4,
        "acre": 4046.8564224,
        "ft2": 0.09290304,
        "sqft": 0.09290304,
        "yd2": 0.83612736,
        "mi2": 2589988.110336,
    },
    "data": {
        "bit": 1.0,
        "byte": 8.0,
        "b": 8.0,
        "kb": 8e3,
        "kib": 8 * 1024,
        "mb": 8e6,
        "mib": 8 * 1024 ** 2,
        "gb": 8e9,
        "gib": 8 * 1024 ** 3,
        "tb": 8e12,
        "tib": 8 * 1024 ** 4,
        "pb": 8e15,
    },
}

# Build a flat lookup: unit_name_lower → (category, factor)
_FLAT_UNITS: Dict[str, tuple] = {}
for _cat, _units in _UNIT_TABLE.items():
    for _uname, _factor in _units.items():
        _FLAT_UNITS[_uname.lower()] = (_cat, _factor)


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between Celsius, Fahrenheit, and Kelvin."""
    # Normalise to Celsius first
    fu = from_unit.lower().rstrip(".")
    tu = to_unit.lower().rstrip(".")

    celsius_aliases = {"c", "celsius", "°c"}
    fahrenheit_aliases = {"f", "fahrenheit", "°f"}
    kelvin_aliases = {"k", "kelvin"}

    if fu in celsius_aliases:
        celsius = value
    elif fu in fahrenheit_aliases:
        celsius = (value - 32) * 5 / 9
    elif fu in kelvin_aliases:
        celsius = value - 273.15
    else:
        raise ValueError(f"Unknown temperature unit: '{from_unit}'")

    if tu in celsius_aliases:
        return celsius
    elif tu in fahrenheit_aliases:
        return celsius * 9 / 5 + 32
    elif tu in kelvin_aliases:
        return celsius + 273.15
    else:
        raise ValueError(f"Unknown temperature unit: '{to_unit}'")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class EvaluateExpressionArgs(ToolArguments):
    """Arguments for ``evaluate_expression``."""

    expression: str = Field(
        ...,
        description=(
            "A mathematical expression to evaluate. "
            "Supports +, -, *, /, //, %, **, and functions: "
            "abs, round, sqrt, ceil, floor, log, log10, log2, exp, "
            "sin, cos, tan, asin, acos, atan, atan2, degrees, radians, "
            "factorial, gcd, lcm. Constants: pi, e, tau, inf."
        ),
    )


class EvaluateExpressionOutput(ToolOutput):
    """Output of ``evaluate_expression``."""

    result: float = Field(..., description="Numeric result of the expression.")
    expression: str = Field(..., description="The original expression string.")
    error: Optional[str] = Field(None, description="Error message, if evaluation failed.")


class UnitConvertArgs(ToolArguments):
    """Arguments for ``unit_convert``."""

    value: float = Field(..., description="Numeric value to convert.")
    from_unit: str = Field(..., description="Source unit (e.g. 'km', 'lb', 'celsius').")
    to_unit: str = Field(..., description="Target unit (e.g. 'mi', 'kg', 'fahrenheit').")


class UnitConvertOutput(ToolOutput):
    """Output of ``unit_convert``."""

    result: float = Field(..., description="Converted value.")
    from_unit: str = Field(..., description="Source unit.")
    to_unit: str = Field(..., description="Target unit.")
    category: str = Field(..., description="Measurement category (length, mass, …).")
    error: Optional[str] = Field(None, description="Error message, if conversion failed.")


class StatisticsSummaryArgs(ToolArguments):
    """Arguments for ``statistics_summary``."""

    numbers: List[float] = Field(
        ..., description="List of numeric values to summarise."
    )


class StatisticsSummaryOutput(ToolOutput):
    """Output of ``statistics_summary``."""

    count: int = Field(..., description="Number of values.")
    mean: float = Field(..., description="Arithmetic mean.")
    median: float = Field(..., description="Median value.")
    stdev: Optional[float] = Field(None, description="Sample standard deviation (None when count < 2).")
    minimum: float = Field(..., description="Minimum value.")
    maximum: float = Field(..., description="Maximum value.")
    total: float = Field(..., description="Sum of all values.")
    error: Optional[str] = Field(None, description="Error message, if computation failed.")


class SolveEquationArgs(ToolArguments):
    """Arguments for ``solve_equation``."""

    equation: str = Field(
        ...,
        description=(
            "An equation to solve for 'x', written as a Python expression "
            "equal to zero (e.g. 'x**2 - 4' solves x²=4, '2*x + 6' solves 2x=-6). "
            "Requires sympy for exact symbolic solutions; falls back to numeric "
            "root-finding otherwise."
        ),
    )


class SolveEquationOutput(ToolOutput):
    """Output of ``solve_equation``."""

    solutions: List[float] = Field(..., description="Real-valued solutions for x.")
    equation: str = Field(..., description="The original equation string.")
    method: str = Field(..., description="'symbolic' (sympy) or 'numeric' (fallback).")
    error: Optional[str] = Field(None, description="Error message, if solving failed.")


# ---------------------------------------------------------------------------
# ToolBox
# ---------------------------------------------------------------------------

class MathToolBox(ToolBox):
    """
    A ``ToolBox`` plugin that exposes safe mathematical operations.

    **Tools provided**:

    - ``evaluate_expression`` — evaluate a numeric expression without ``eval``;
      supports common math functions and constants.
    - ``solve_equation`` — solve a polynomial or transcendental equation for *x*;
      uses ``sympy`` when installed, numeric fallback otherwise.
    - ``unit_convert`` — convert a value between length, mass, volume, time,
      speed, area, data, and temperature units.
    - ``statistics_summary`` — compute descriptive statistics for a list of
      numbers (mean, median, stdev, min, max, sum).

    No external dependencies are required for ``evaluate_expression``,
    ``unit_convert``, and ``statistics_summary``. ``solve_equation`` benefits
    from ``sympy`` but degrades gracefully to SciPy / Newton's method.

    Entry point group: ``opm.agents.toolbox``
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 bus: Optional[Any] = None) -> None:
        """
        Initialise the toolbox.

        Args:
            config: Plugin configuration dict (currently unused).
        """
        super().__init__(toolbox_id="ovos-math-tools", config=config, bus=bus)

    # ------------------------------------------------------------------
    # evaluate_expression
    # ------------------------------------------------------------------

    def _evaluate_expression(self, args: EvaluateExpressionArgs) -> EvaluateExpressionOutput:
        """
        Safely evaluate a numeric expression.

        Uses Python's ``ast`` module — no ``eval`` is called. Only arithmetic
        operators and whitelisted math functions are permitted.

        Args:
            args: Validated ``EvaluateExpressionArgs``.

        Returns:
            ``EvaluateExpressionOutput`` with the numeric result or an error.
        """
        try:
            tree = ast.parse(args.expression.strip(), mode="eval")
            result = _safe_eval(tree.body)
            return EvaluateExpressionOutput(result=result, expression=args.expression)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
            return EvaluateExpressionOutput(
                result=float("nan"),
                expression=args.expression,
                error=str(exc),
            )
        except SyntaxError as exc:
            return EvaluateExpressionOutput(
                result=float("nan"),
                expression=args.expression,
                error=f"Syntax error: {exc.msg}",
            )

    # ------------------------------------------------------------------
    # unit_convert
    # ------------------------------------------------------------------

    def _unit_convert(self, args: UnitConvertArgs) -> UnitConvertOutput:
        """
        Convert *value* from *from_unit* to *to_unit*.

        Supports length, mass, volume, time, speed, area, data, and temperature.

        Args:
            args: Validated ``UnitConvertArgs``.

        Returns:
            ``UnitConvertOutput`` with the converted value or an error.
        """
        fu = args.from_unit.strip().lower()
        tu = args.to_unit.strip().lower()

        # Temperature is a special case (affine, not linear)
        temp_aliases = {"c", "celsius", "°c", "f", "fahrenheit", "°f", "k", "kelvin"}
        if fu in temp_aliases or tu in temp_aliases:
            try:
                result = _convert_temperature(args.value, fu, tu)
                return UnitConvertOutput(
                    result=result,
                    from_unit=args.from_unit,
                    to_unit=args.to_unit,
                    category="temperature",
                )
            except ValueError as exc:
                return UnitConvertOutput(
                    result=float("nan"),
                    from_unit=args.from_unit,
                    to_unit=args.to_unit,
                    category="temperature",
                    error=str(exc),
                )

        if fu not in _FLAT_UNITS:
            return UnitConvertOutput(
                result=float("nan"),
                from_unit=args.from_unit,
                to_unit=args.to_unit,
                category="unknown",
                error=f"Unknown unit: '{args.from_unit}'",
            )
        if tu not in _FLAT_UNITS:
            return UnitConvertOutput(
                result=float("nan"),
                from_unit=args.from_unit,
                to_unit=args.to_unit,
                category="unknown",
                error=f"Unknown unit: '{args.to_unit}'",
            )

        from_cat, from_factor = _FLAT_UNITS[fu]
        to_cat, to_factor = _FLAT_UNITS[tu]

        if from_cat != to_cat:
            return UnitConvertOutput(
                result=float("nan"),
                from_unit=args.from_unit,
                to_unit=args.to_unit,
                category=from_cat,
                error=f"Incompatible categories: '{from_cat}' vs '{to_cat}'",
            )

        result = args.value * from_factor / to_factor
        return UnitConvertOutput(
            result=result,
            from_unit=args.from_unit,
            to_unit=args.to_unit,
            category=from_cat,
        )

    # ------------------------------------------------------------------
    # statistics_summary
    # ------------------------------------------------------------------

    def _statistics_summary(self, args: StatisticsSummaryArgs) -> StatisticsSummaryOutput:
        """
        Compute descriptive statistics for a list of numbers.

        Args:
            args: Validated ``StatisticsSummaryArgs``.

        Returns:
            ``StatisticsSummaryOutput`` with mean, median, stdev, min, max, sum.
        """
        nums = args.numbers
        if not nums:
            return StatisticsSummaryOutput(
                count=0, mean=0.0, median=0.0, stdev=None,
                minimum=0.0, maximum=0.0, total=0.0,
                error="Empty list provided.",
            )
        try:
            stdev = statistics.stdev(nums) if len(nums) >= 2 else None
            return StatisticsSummaryOutput(
                count=len(nums),
                mean=statistics.mean(nums),
                median=statistics.median(nums),
                stdev=stdev,
                minimum=min(nums),
                maximum=max(nums),
                total=sum(nums),
            )
        except Exception as exc:  # noqa: BLE001
            return StatisticsSummaryOutput(
                count=len(nums), mean=0.0, median=0.0, stdev=None,
                minimum=0.0, maximum=0.0, total=0.0,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # solve_equation
    # ------------------------------------------------------------------

    def _solve_equation(self, args: SolveEquationArgs) -> SolveEquationOutput:
        """
        Solve an equation ``f(x) = 0`` for *x*.

        Tries ``sympy.solve`` first; falls back to ``scipy.optimize.brentq``
        on a grid of intervals; final fallback is Newton's method via the
        safe expression evaluator.

        Args:
            args: Validated ``SolveEquationArgs``.

        Returns:
            ``SolveEquationOutput`` with real solutions or an error.
        """
        expr_str = args.equation.strip()

        # --- Attempt 1: sympy ---
        try:
            import sympy  # type: ignore[import-untyped]
            x = sympy.Symbol("x")
            expr = sympy.sympify(expr_str)
            raw = sympy.solve(expr, x)
            solutions = sorted(
                float(s) for s in raw if s.is_real
            )
            return SolveEquationOutput(
                solutions=solutions,
                equation=args.equation,
                method="symbolic",
            )
        except ImportError:
            pass
        except Exception:  # noqa: BLE001
            pass

        # --- Attempt 2: numeric root finding using safe evaluator ---
        try:
            tree = ast.parse(expr_str, mode="eval")
        except SyntaxError as exc:
            return SolveEquationOutput(
                solutions=[],
                equation=args.equation,
                method="numeric",
                error=f"Syntax error: {exc.msg}",
            )

        def _f(x_val: float) -> float:
            """Substitute x = x_val into the expression tree."""
            # Temporarily register x as a constant
            _ALLOWED_FUNCS["x"] = x_val
            try:
                return _safe_eval(tree.body)
            finally:
                _ALLOWED_FUNCS.pop("x", None)

        solutions: List[float] = []
        # Scan range [-1000, 1000] for sign changes, then refine with bisection
        search_range = list(range(-1000, 1001))
        prev_val: Optional[float] = None
        prev_x: Optional[float] = None
        seen: set = set()

        for xi in search_range:
            try:
                fx = _f(float(xi))
            except Exception:  # noqa: BLE001
                prev_val = None
                continue

            if prev_val is not None and prev_val * fx < 0:
                # Sign change → bisect
                lo, hi = float(prev_x), float(xi)  # type: ignore[arg-type]
                for _ in range(60):
                    mid = (lo + hi) / 2
                    try:
                        fmid = _f(mid)
                    except Exception:  # noqa: BLE001
                        break
                    if abs(fmid) < 1e-12 or (hi - lo) < 1e-12:
                        break
                    if _f(lo) * fmid < 0:
                        hi = mid
                    else:
                        lo = mid
                root = round((lo + hi) / 2, 10)
                rounded = round(root, 6)
                if rounded not in seen:
                    seen.add(rounded)
                    solutions.append(root)

            if abs(fx) < 1e-10:
                rounded = round(float(xi), 6)
                if rounded not in seen:
                    seen.add(rounded)
                    solutions.append(float(xi))

            prev_val = fx
            prev_x = float(xi)

        solutions.sort()
        return SolveEquationOutput(
            solutions=solutions,
            equation=args.equation,
            method="numeric",
        )

    # ------------------------------------------------------------------
    # discover_tools
    # ------------------------------------------------------------------

    def discover_tools(self) -> List[AgentTool]:
        """
        Return the math ``AgentTool`` instances.

        Returns:
            List of four tools: evaluate_expression, unit_convert,
            statistics_summary, solve_equation.
        """
        return [
            AgentTool(
                name="evaluate_expression",
                description=(
                    "Evaluate a mathematical expression and return its numeric result. "
                    "Supports arithmetic operators (+, -, *, /, //, %, **) and functions: "
                    "abs, round, sqrt, ceil, floor, log, log10, log2, exp, sin, cos, tan, "
                    "asin, acos, atan, atan2, degrees, radians, factorial, gcd, lcm. "
                    "Constants: pi, e, tau, inf. Safe — no eval()."
                ),
                argument_schema=EvaluateExpressionArgs,
                output_schema=EvaluateExpressionOutput,
                tool_call=self._evaluate_expression,
            ),
            AgentTool(
                name="unit_convert",
                description=(
                    "Convert a numeric value between measurement units. "
                    "Supported categories: length (mm/cm/m/km/in/ft/yd/mi), "
                    "mass (mg/g/kg/t/oz/lb), volume (ml/l/tsp/tbsp/cup/pt/qt/gal), "
                    "time (ns/ms/s/min/h/d/week/year), speed (m/s/km/h/mph/knot), "
                    "area (m2/km2/ha/acre/sqft), data (bit/byte/KB/MB/GB/TB), "
                    "temperature (celsius/fahrenheit/kelvin)."
                ),
                argument_schema=UnitConvertArgs,
                output_schema=UnitConvertOutput,
                tool_call=self._unit_convert,
            ),
            AgentTool(
                name="statistics_summary",
                description=(
                    "Compute descriptive statistics for a list of numbers: "
                    "count, mean, median, standard deviation, min, max, and sum."
                ),
                argument_schema=StatisticsSummaryArgs,
                output_schema=StatisticsSummaryOutput,
                tool_call=self._statistics_summary,
            ),
            AgentTool(
                name="solve_equation",
                description=(
                    "Solve an equation f(x) = 0 for x. Express the equation as a "
                    "Python expression equal to zero (e.g. 'x**2 - 4' for x²=4, "
                    "'2*x + 6' for 2x=-6). Uses sympy for exact symbolic solutions "
                    "when available; falls back to numeric root-finding otherwise."
                ),
                argument_schema=SolveEquationArgs,
                output_schema=SolveEquationOutput,
                tool_call=self._solve_equation,
            ),
        ]
