import numpy
import pytest

from openfisca_core.entities import Entity
from openfisca_core.periods import DateUnit
from openfisca_core.populations import Population
from openfisca_core.simulations import Simulation
from openfisca_core.taxbenefitsystems import TaxBenefitSystem
from openfisca_core.variables import Variable


entity = Entity("person", "persons", "", "")


def make_simulation(*variable_classes, count=3):
    tax_benefit_system = TaxBenefitSystem([entity])
    person_entity = tax_benefit_system.person_entity
    for variable_class in variable_classes:
        tax_benefit_system.add_variable(variable_class)
    population = Population(person_entity)
    population.count = count
    population.ids = [str(index) for index in range(count)]
    return Simulation(tax_benefit_system, {person_entity.key: population})


def test_transition_formula_requires_as_of():
    class InvalidVariable(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH

        def transition_formula(person, period):  # noqa: N805
            return [], []

    with pytest.raises(ValueError, match="transition_formula without as_of"):
        InvalidVariable()


def test_transition_formula_is_mutually_exclusive_with_formula():
    class InvalidVariable(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True

        def formula(person, period):  # noqa: N805
            return 0

        def transition_formula(person, period):  # noqa: N805
            return [], []

    with pytest.raises(ValueError, match="mutually exclusive"):
        InvalidVariable()


def test_transition_formula_dispatches_by_date():
    class State(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True

        def transition_formula(person, period):  # noqa: N805
            return [], []

        def transition_formula_2025(person, period):  # noqa: N805
            return [], []

    variable = State()
    assert variable.get_transition_formula("2024") is State.transition_formula
    assert variable.get_transition_formula("2025") is State.transition_formula_2025
    assert not variable.is_input_variable()


def test_initial_formula_runs_before_transitions():
    calls = []

    class State(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True

        def initial_formula(person, period):  # noqa: N805
            calls.append("initial")
            return numpy.array([10, 20, 30])

        def transition_formula(person, period):  # noqa: N805
            calls.append("transition")
            previous = person("State", period.last_month)
            return numpy.arange(3), previous + 1

    simulation = make_simulation(State)

    numpy.testing.assert_array_equal(
        simulation.calculate("State", "2024-01"),
        [10, 20, 30],
    )
    numpy.testing.assert_array_equal(
        simulation.calculate("State", "2024-02"),
        [11, 21, 31],
    )
    assert calls == ["initial", "transition"]


def test_transition_formula_accepts_boolean_selector_and_scalar_value():
    class State(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True

        def transition_formula(person, period):  # noqa: N805
            return numpy.array([True, False, True]), 9

    simulation = make_simulation(State)
    simulation.set_input("State", "2024-01", [1, 2, 3])

    numpy.testing.assert_array_equal(
        simulation.calculate("State", "2024-02"),
        [9, 2, 9],
    )


def test_explicit_input_prevents_transition_at_same_period():
    calls = []

    class State(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True

        def transition_formula(person, period):  # noqa: N805
            calls.append(period)
            return [0], [9]

    simulation = make_simulation(State)
    simulation.set_input("State", "2024-01", [1, 2, 3])

    numpy.testing.assert_array_equal(
        simulation.calculate("State", "2024-01"),
        [1, 2, 3],
    )
    assert calls == []


def test_transition_formula_runs_once_per_period():
    calls = []

    class State(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True

        def transition_formula(person, period):  # noqa: N805
            calls.append(period)
            return [0], [9]

    simulation = make_simulation(State)
    simulation.set_input("State", "2024-01", [1, 2, 3])
    simulation.calculate("State", "2024-02")
    simulation.calculate("State", "2024-02")

    assert len(calls) == 1


def test_transition_formula_requires_initial_state():
    class State(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True

        def transition_formula(person, period):  # noqa: N805
            return [0], [9]

    with pytest.raises(ValueError, match="no initial state"):
        make_simulation(State).calculate("State", "2024-01")


def test_transition_formula_validates_result_lengths():
    class State(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True

        def transition_formula(person, period):  # noqa: N805
            return [0, 1], [9]

    simulation = make_simulation(State)
    simulation.set_input("State", "2024-01", [1, 2, 3])

    with pytest.raises(ValueError, match="match the number"):
        simulation.calculate("State", "2024-02")


def test_temporal_recursion_is_allowed_but_exact_cycle_is_stopped():
    class State(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True

        def transition_formula(person, period):  # noqa: N805
            previous = person("State", period.last_month)
            return numpy.arange(3), previous + 1

    simulation = make_simulation(State)
    simulation.set_input("State", "2024-01", [0, 0, 0])
    numpy.testing.assert_array_equal(
        simulation.calculate("State", "2024-03"),
        [2, 2, 2],
    )

    class CyclicState(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True

        def transition_formula(person, period):  # noqa: N805
            person("CyclicState", period)
            return [], []

    cyclic_simulation = make_simulation(CyclicState)
    cyclic_simulation.set_input("CyclicState", "2024-01", [0, 0, 0])
    numpy.testing.assert_array_equal(
        cyclic_simulation.calculate("CyclicState", "2024-02"),
        [0, 0, 0],
    )
