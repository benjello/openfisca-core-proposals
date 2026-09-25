import numpy
import pytest

from openfisca_core.entities import Entity
from openfisca_core.periods import DateUnit
from openfisca_core.variables import Variable


entity = Entity("person", "persons", "", "")


def test_initial_formula_requires_as_of():
    class InvalidVariable(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH

        def initial_formula(person, period):  # noqa: N805
            return numpy.zeros(1)

    with pytest.raises(ValueError, match="initial_formula without as_of"):
        InvalidVariable()


def test_initial_formula_is_mutually_exclusive_with_formula():
    class InvalidVariable(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True

        def formula(person, period):  # noqa: N805
            return 0

        def initial_formula(person, period):  # noqa: N805
            return 1

    with pytest.raises(ValueError, match="formula and initial_formula"):
        InvalidVariable()


def test_initial_formula_dispatches_in_date_order():
    class StatefulVariable(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True

        def initial_formula(person, period):  # noqa: N805
            return 1

        def initial_formula_2024_06(person, period):  # noqa: N805
            return 2

        def initial_formula_2025(person, period):  # noqa: N805
            return 3

    variable = StatefulVariable()

    assert variable.get_initial_formula("2024-05") is StatefulVariable.initial_formula
    assert (
        variable.get_initial_formula("2024-06")
        is StatefulVariable.initial_formula_2024_06
    )
    assert (
        variable.get_initial_formula("2025-02") is StatefulVariable.initial_formula_2025
    )


def test_initial_formula_rejects_invalid_date_name():
    class InvalidVariable(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True

        def initial_formula_2024_13(person, period):  # noqa: N805
            return 1

    with pytest.raises(ValueError, match="Unrecognized initial_formula"):
        InvalidVariable()


def test_initial_formula_respects_variable_end():
    class InvalidVariable(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True
        end = "2024-12-31"

        def initial_formula_2025(person, period):  # noqa: N805
            return 1

    with pytest.raises(ValueError, match="ends on"):
        InvalidVariable()


def test_initial_formula_inherits_earlier_baseline_formulas():
    class BaselineVariable(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True

        def initial_formula(person, period):  # noqa: N805
            return 1

        def initial_formula_2024(person, period):  # noqa: N805
            return 2

    class ReformVariable(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True

        def initial_formula_2025(person, period):  # noqa: N805
            return 3

    variable = ReformVariable(baseline_variable=BaselineVariable())

    assert variable.get_initial_formula("2024") is BaselineVariable.initial_formula_2024
    assert variable.get_initial_formula("2025") is ReformVariable.initial_formula_2025


def test_has_initial_formula():
    class StatefulVariable(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True

        def initial_formula(person, period):  # noqa: N805
            return 1

    assert StatefulVariable().has_initial_formula
