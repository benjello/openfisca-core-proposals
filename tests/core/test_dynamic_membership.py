import numpy
import pytest

from openfisca_core import entities, periods, taxbenefitsystems, variables
from openfisca_core.simulations import SimulationBuilder


def build_simulation():
    person = entities.SingleEntity("person", "persons", "A person", "")
    household = entities.GroupEntity(
        "household",
        "households",
        "A household",
        "",
        roles=[{"key": "member"}],
    )
    tax_benefit_system = taxbenefitsystems.TaxBenefitSystem([person, household])

    class salary(variables.Variable):
        value_type = float
        entity = person
        definition_period = periods.DateUnit.MONTH

    class household_income(variables.Variable):
        value_type = float
        entity = household
        definition_period = periods.DateUnit.MONTH

        def formula(population, period):
            return population.sum(population.members("salary", period), period=period)

    tax_benefit_system.add_variables(salary, household_income)
    simulation = SimulationBuilder().build_default_simulation(
        tax_benefit_system,
        count=4,
    )
    simulation.household.ids = numpy.array(["household_0", "household_1"])
    simulation.household.set_members_entity_id([0, 0, 1, 1])
    return simulation


def test_membership_snapshots_carry_forward_chronologically() -> None:
    simulation = build_simulation()
    values = numpy.array([10, 20, 30, 40])

    simulation.household.set_members_for_period("2024-02", [0, 1, 0, 1])
    simulation.household.set_members_for_period("2024-04", [0, 1, 1, 0])

    numpy.testing.assert_array_equal(
        simulation.household.sum(values, period="2024-01"),
        [30, 70],
    )
    numpy.testing.assert_array_equal(
        simulation.household.sum(values, period="2024-03"),
        [40, 60],
    )
    numpy.testing.assert_array_equal(
        simulation.household.sum(values, period="2024-05"),
        [50, 50],
    )


def test_period_aware_aggregations_and_projections() -> None:
    simulation = build_simulation()
    values = numpy.array([10, 20, 30, 40])
    simulation.household.set_members_for_period("2024-02", [0, 1, 0, 1])

    numpy.testing.assert_array_equal(
        simulation.household.nb_persons(period="2024-03"),
        [2, 2],
    )
    numpy.testing.assert_array_equal(
        simulation.household.value_nth_person(1, values, period="2024-03"),
        [30, 40],
    )
    numpy.testing.assert_array_equal(
        simulation.household.project(numpy.array([100, 200]), period="2024-03"),
        [100, 200, 100, 200],
    )


def test_membership_change_invalidates_inherited_cached_periods() -> None:
    simulation = build_simulation()
    for month in ("2024-02", "2024-03"):
        simulation.set_input("salary", month, [10, 20, 30, 40])
        numpy.testing.assert_array_equal(
            simulation.calculate("household_income", month),
            [30, 70],
        )

    simulation.household.set_members_for_period("2024-02", [0, 1, 0, 1])

    numpy.testing.assert_array_equal(
        simulation.calculate("household_income", "2024-02"),
        [40, 60],
    )
    numpy.testing.assert_array_equal(
        simulation.calculate("household_income", "2024-03"),
        [40, 60],
    )


def test_dynamic_membership_is_preserved_and_isolated_by_clone() -> None:
    simulation = build_simulation()
    simulation.household.set_members_for_period("2024-02", [0, 1, 0, 1])

    clone = simulation.clone()
    clone.household.set_members_for_period("2024-02", [0, 1, 1, 0])

    assert clone.household.members is clone.persons
    numpy.testing.assert_array_equal(
        simulation.household._get_members_entity_id("2024-03"),
        [0, 1, 0, 1],
    )
    numpy.testing.assert_array_equal(
        clone.household._get_members_entity_id("2024-03"),
        [0, 1, 1, 0],
    )


@pytest.mark.parametrize(
    ("membership", "message"),
    [
        ([], "cannot be empty"),
        ([[0, 1], [1, 0]], "one-dimensional"),
        ([0, 1, 0], "one entry per member"),
        ([0.0, 1.0, 0.0, 1.0], "integer IDs"),
        ([0, -1, 0, 1], "non-negative IDs"),
        ([0, 2, 0, 1], "existing group"),
        ([0, 0, 0, 0], "number of groups"),
    ],
)
def test_set_members_for_period_rejects_invalid_membership(
    membership,
    message,
) -> None:
    simulation = build_simulation()

    with pytest.raises(ValueError, match=message):
        simulation.household.set_members_for_period("2024-02", membership)
