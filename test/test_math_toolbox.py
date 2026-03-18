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
"""Unit tests for MathToolBox."""
import math
import pytest

from ovos_agentic_loop.tools.math import (
    EvaluateExpressionArgs,
    MathToolBox,
    SolveEquationArgs,
    StatisticsSummaryArgs,
    UnitConvertArgs,
)


@pytest.fixture
def tb() -> MathToolBox:
    return MathToolBox()


# ---------------------------------------------------------------------------
# discover_tools
# ---------------------------------------------------------------------------

class TestDiscoverTools:
    def test_returns_four_tools(self, tb):
        tools = tb.discover_tools()
        assert len(tools) == 4

    def test_tool_names(self, tb):
        names = {t.name for t in tb.discover_tools()}
        assert names == {
            "evaluate_expression",
            "unit_convert",
            "statistics_summary",
            "solve_equation",
        }

    def test_toolbox_id(self, tb):
        assert tb.toolbox_id == "ovos-math-tools"


# ---------------------------------------------------------------------------
# evaluate_expression
# ---------------------------------------------------------------------------

class TestEvaluateExpression:
    def test_addition(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="2 + 3"))
        assert out.result == pytest.approx(5.0)
        assert out.error is None

    def test_subtraction(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="10 - 4"))
        assert out.result == pytest.approx(6.0)

    def test_multiplication(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="3 * 7"))
        assert out.result == pytest.approx(21.0)

    def test_division(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="10 / 4"))
        assert out.result == pytest.approx(2.5)

    def test_floor_division(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="10 // 3"))
        assert out.result == pytest.approx(3.0)

    def test_modulo(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="10 % 3"))
        assert out.result == pytest.approx(1.0)

    def test_power(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="2 ** 10"))
        assert out.result == pytest.approx(1024.0)

    def test_unary_neg(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="-5"))
        assert out.result == pytest.approx(-5.0)

    def test_nested(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="(2 + 3) * 4"))
        assert out.result == pytest.approx(20.0)

    def test_sqrt(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="sqrt(144)"))
        assert out.result == pytest.approx(12.0)

    def test_pi_constant(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="pi"))
        assert out.result == pytest.approx(math.pi)

    def test_e_constant(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="e"))
        assert out.result == pytest.approx(math.e)

    def test_trig(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="sin(pi / 2)"))
        assert out.result == pytest.approx(1.0, abs=1e-10)

    def test_log(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="log10(1000)"))
        assert out.result == pytest.approx(3.0)

    def test_factorial(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="factorial(5)"))
        assert out.result == pytest.approx(120.0)

    def test_division_by_zero(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="1 / 0"))
        assert out.error is not None
        assert math.isnan(out.result)

    def test_syntax_error(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="2 +* 3"))
        assert out.error is not None

    def test_disallowed_name(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="os"))
        assert out.error is not None

    def test_disallowed_call(self, tb):
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression="__import__('os')"))
        assert out.error is not None

    def test_expression_preserved(self, tb):
        expr = "2 + 2"
        out = tb._evaluate_expression(EvaluateExpressionArgs(expression=expr))
        assert out.expression == expr


# ---------------------------------------------------------------------------
# unit_convert
# ---------------------------------------------------------------------------

class TestUnitConvert:
    def test_km_to_mi(self, tb):
        out = tb._unit_convert(UnitConvertArgs(value=1.0, from_unit="km", to_unit="mi"))
        assert out.result == pytest.approx(0.621371, rel=1e-4)
        assert out.error is None
        assert out.category == "length"

    def test_mi_to_km(self, tb):
        out = tb._unit_convert(UnitConvertArgs(value=1.0, from_unit="mi", to_unit="km"))
        assert out.result == pytest.approx(1.609344, rel=1e-4)

    def test_m_to_cm(self, tb):
        out = tb._unit_convert(UnitConvertArgs(value=1.0, from_unit="m", to_unit="cm"))
        assert out.result == pytest.approx(100.0)

    def test_kg_to_lb(self, tb):
        out = tb._unit_convert(UnitConvertArgs(value=1.0, from_unit="kg", to_unit="lb"))
        assert out.result == pytest.approx(2.20462, rel=1e-4)

    def test_celsius_to_fahrenheit(self, tb):
        out = tb._unit_convert(UnitConvertArgs(value=0.0, from_unit="celsius", to_unit="fahrenheit"))
        assert out.result == pytest.approx(32.0)
        assert out.category == "temperature"

    def test_fahrenheit_to_celsius(self, tb):
        out = tb._unit_convert(UnitConvertArgs(value=212.0, from_unit="fahrenheit", to_unit="celsius"))
        assert out.result == pytest.approx(100.0)

    def test_celsius_to_kelvin(self, tb):
        out = tb._unit_convert(UnitConvertArgs(value=0.0, from_unit="celsius", to_unit="kelvin"))
        assert out.result == pytest.approx(273.15)

    def test_litre_to_gallon(self, tb):
        out = tb._unit_convert(UnitConvertArgs(value=1.0, from_unit="l", to_unit="gal"))
        assert out.result == pytest.approx(0.264172, rel=1e-4)

    def test_hours_to_seconds(self, tb):
        out = tb._unit_convert(UnitConvertArgs(value=1.0, from_unit="h", to_unit="s"))
        assert out.result == pytest.approx(3600.0)

    def test_incompatible_categories(self, tb):
        out = tb._unit_convert(UnitConvertArgs(value=1.0, from_unit="km", to_unit="kg"))
        assert out.error is not None

    def test_unknown_from_unit(self, tb):
        out = tb._unit_convert(UnitConvertArgs(value=1.0, from_unit="xyz", to_unit="m"))
        assert out.error is not None

    def test_unknown_to_unit(self, tb):
        out = tb._unit_convert(UnitConvertArgs(value=1.0, from_unit="m", to_unit="xyz"))
        assert out.error is not None

    def test_mb_to_gb(self, tb):
        out = tb._unit_convert(UnitConvertArgs(value=1024.0, from_unit="mb", to_unit="gb"))
        assert out.result == pytest.approx(1.024, rel=1e-4)


# ---------------------------------------------------------------------------
# statistics_summary
# ---------------------------------------------------------------------------

class TestStatisticsSummary:
    def test_basic(self, tb):
        out = tb._statistics_summary(StatisticsSummaryArgs(numbers=[1.0, 2.0, 3.0, 4.0, 5.0]))
        assert out.count == 5
        assert out.mean == pytest.approx(3.0)
        assert out.median == pytest.approx(3.0)
        assert out.minimum == pytest.approx(1.0)
        assert out.maximum == pytest.approx(5.0)
        assert out.total == pytest.approx(15.0)
        assert out.stdev is not None
        assert out.error is None

    def test_single_element(self, tb):
        out = tb._statistics_summary(StatisticsSummaryArgs(numbers=[42.0]))
        assert out.count == 1
        assert out.mean == pytest.approx(42.0)
        assert out.stdev is None  # undefined for n=1

    def test_empty_list(self, tb):
        out = tb._statistics_summary(StatisticsSummaryArgs(numbers=[]))
        assert out.error is not None

    def test_negative_numbers(self, tb):
        out = tb._statistics_summary(StatisticsSummaryArgs(numbers=[-3.0, -1.0, 1.0, 3.0]))
        assert out.mean == pytest.approx(0.0)
        assert out.minimum == pytest.approx(-3.0)
        assert out.maximum == pytest.approx(3.0)

    def test_stdev_two_elements(self, tb):
        out = tb._statistics_summary(StatisticsSummaryArgs(numbers=[0.0, 10.0]))
        assert out.stdev is not None
        assert out.stdev == pytest.approx(math.sqrt(50), rel=1e-6)


# ---------------------------------------------------------------------------
# solve_equation
# ---------------------------------------------------------------------------

class TestSolveEquation:
    def test_linear(self, tb):
        out = tb._solve_equation(SolveEquationArgs(equation="2*x + 6"))
        assert out.error is None
        assert any(abs(s - (-3.0)) < 1e-4 for s in out.solutions)

    def test_quadratic_two_roots(self, tb):
        out = tb._solve_equation(SolveEquationArgs(equation="x**2 - 4"))
        roots = sorted(out.solutions)
        assert any(abs(r - (-2.0)) < 1e-4 for r in roots)
        assert any(abs(r - 2.0) < 1e-4 for r in roots)

    def test_quadratic_no_real_root(self, tb):
        # x² + 1 = 0 has no real roots
        out = tb._solve_equation(SolveEquationArgs(equation="x**2 + 1"))
        assert out.solutions == [] or all(abs(s) > 1e4 for s in out.solutions)

    def test_cubic(self, tb):
        # x³ - x = x(x-1)(x+1) → roots at -1, 0, 1
        out = tb._solve_equation(SolveEquationArgs(equation="x**3 - x"))
        roots = sorted(out.solutions)
        for expected in [-1.0, 0.0, 1.0]:
            assert any(abs(r - expected) < 1e-3 for r in roots), f"Missing root {expected}"

    def test_equation_preserved(self, tb):
        eq = "x - 7"
        out = tb._solve_equation(SolveEquationArgs(equation=eq))
        assert out.equation == eq

    def test_syntax_error(self, tb):
        out = tb._solve_equation(SolveEquationArgs(equation="x +* 1"))
        assert out.error is not None
        assert out.solutions == []
