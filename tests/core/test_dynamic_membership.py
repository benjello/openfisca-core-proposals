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
            return population.sum(population.members("salary", period))

    class household_size(variables.Variable):
        value_type = int
        entity = person
        definition_period = periods.DateUnit.MONTH

        def formula(population, _period):
            return population.household.nb_persons()

    tax_benefit_system.add_variables(salary, household_income, household_size)
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


def test_membership_change_invalidates_other_population_results_not_inputs() -> None:
    simulation = build_simulation()
    simulation.set_input("salary", "2024-02", [10, 20, 30, 40])
    numpy.testing.assert_array_equal(
        simulation.calculate("household_size", "2024-02"),
        [2, 2, 2, 2],
    )

    simulation.household.set_members_for_period("2024-02", [0, 0, 0, 1])

    numpy.testing.assert_array_equal(
        simulation.calculate("household_size", "2024-02"),
        [3, 3, 3, 1],
    )
    numpy.testing.assert_array_equal(
        simulation.calculate("salary", "2024-02"),
        [10, 20, 30, 40],
    )


def test_membership_change_invalidates_annual_caches_across_populations() -> None:
    simulation = build_simulation()
    annual_period = periods.period("2024")
    household_holder = simulation.household.get_holder("household_income")
    person_holder = simulation.persons.get_holder("household_size")
    household_holder._memory_storage.put(numpy.array([30, 70]), annual_period)
    person_holder._memory_storage.put(numpy.array([2, 2, 2, 2]), annual_period)

    simulation.household.set_members_for_period("2024-02", [0, 1, 0, 1])

    assert household_holder.get_array(annual_period) is None
    assert person_holder.get_array(annual_period) is None


def test_top_level_projector_uses_explicit_membership_period() -> None:
    simulation = build_simulation()
    simulation.household.set_members_for_period("2024-02", [0, 1, 0, 1])
    simulation.set_input("household_income", "2024-02", [100, 200])

    numpy.testing.assert_array_equal(
        simulation.persons.household("household_income", "2024-02"),
        [100, 200, 100, 200],
    )


def test_dynamic_membership_is_preserved_and_isolated_by_clone() -> None:
    simulation = build_simulation()
    simulation.household.set_members_for_period("2024-02", [0, 1, 0, 1])
    simulation.household.members_position
    simulation.household.ordered_members_map
    simulation.household._get_members_position("2024-03")
    simulation.household._get_ordered_members_map("2024-03")

    clone = simulation.clone()
    assert clone.household.members is clone.persons
    assert not numpy.shares_memory(simulation.persons.ids, clone.persons.ids)
    assert not numpy.shares_memory(simulation.household.ids, clone.household.ids)
    for source, copied in (
        (simulation.household._members_entity_id, clone.household._members_entity_id),
        (simulation.household._members_role, clone.household._members_role),
        (simulation.household._members_position, clone.household._members_position),
        (
            simulation.household._ordered_members_map,
            clone.household._ordered_members_map,
        ),
    ):
        assert not numpy.shares_memory(source, copied)
    for attribute in (
        "_members_entity_id_by_period",
        "_members_role_by_period",
        "_members_position_by_period",
        "_ordered_members_map_by_period",
    ):
        source = getattr(simulation.household, attribute)[periods.period("2024-02")]
        copied = getattr(clone.household, attribute)[periods.period("2024-02")]
        assert not numpy.shares_memory(source, copied)

    clone.household.set_members_for_period("2024-02", [0, 1, 1, 0])

    numpy.testing.assert_array_equal(
        simulation.household._get_members_entity_id("2024-03"),
        [0, 1, 0, 1],
    )
    numpy.testing.assert_array_equal(
        clone.household._get_members_entity_id("2024-03"),
        [0, 1, 1, 0],
    )


def test_snapshot_periods_with_the_same_start_are_rejected() -> None:
    simulation = build_simulation()
    simulation.household.set_members_for_period("2024-01", [0, 1, 0, 1])

    with pytest.raises(ValueError, match="same start"):
        simulation.household.set_members_for_period(
            "year:2024-01:1",
            [0, 0, 1, 1],
        )


def test_membership_events_are_rejected_out_of_order() -> None:
    simulation = build_simulation()
    simulation.household.set_members_for_period("2024-03", [0, 1, 0, 1])

    with pytest.raises(ValueError, match="must be registered in order"):
        simulation.household.set_members_for_period("2024-02", [0, 0, 1, 1])


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
