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
        roles=[
            {"key": "parent", "max": 1},
            {"key": "child", "plural": "children"},
        ],
    )
    tax_benefit_system = taxbenefitsystems.TaxBenefitSystem([person, household])

    class group_value(variables.Variable):
        value_type = int
        entity = household
        definition_period = periods.DateUnit.MONTH

    class is_parent(variables.Variable):
        value_type = bool
        entity = person
        definition_period = periods.DateUnit.MONTH

        def formula(population, _period):
            return population.has_role(household.flattened_roles[0])

    tax_benefit_system.add_variables(group_value, is_parent)
    simulation = SimulationBuilder.build_default_simulation(
        tax_benefit_system,
        count=4,
    )
    simulation.household.ids = numpy.array(["household_0", "household_1"])
    simulation.household.set_members_entity_id([0, 0, 1, 1])
    parent, child = household.flattened_roles
    simulation.household.members_role = [parent, child, parent, child]
    return simulation, parent, child


def test_roles_carry_forward_and_are_read_for_requested_period() -> None:
    simulation, parent, child = build_simulation()
    simulation.household.set_roles_for_period(
        "2024-02",
        [child, parent, child, parent],
    )

    numpy.testing.assert_array_equal(
        simulation.persons.has_role(parent, "2024-01"),
        [True, False, True, False],
    )
    numpy.testing.assert_array_equal(
        simulation.persons.has_role(parent, "2024-03"),
        [False, True, False, True],
    )
    numpy.testing.assert_array_equal(
        simulation.household.sum(
            numpy.array([10, 20, 30, 40]),
            role=parent,
            period="2024-03",
        ),
        [20, 40],
    )


def test_membership_and_roles_can_change_atomically() -> None:
    simulation, parent, child = build_simulation()

    simulation.household.set_members_for_period(
        "2024-02",
        [0, 1, 0, 1],
        members_role=[parent, parent, child, child],
    )

    numpy.testing.assert_array_equal(
        simulation.household.nb_persons(role=parent, period="2024-03"),
        [1, 1],
    )
    numpy.testing.assert_array_equal(
        simulation.household.project(
            numpy.array([100, 200]),
            role=child,
            period="2024-03",
        ),
        [0, 0, 100, 200],
    )


def test_membership_change_preserves_roles_set_at_the_same_instant() -> None:
    simulation, parent, child = build_simulation()
    supplied_roles = [parent, child, child, parent]
    simulation.household.set_roles_for_period("2024-02", supplied_roles)

    simulation.household.set_members_for_period("2024-02", [0, 1, 0, 1])

    numpy.testing.assert_array_equal(
        simulation.household._get_members_role("2024-02"),
        supplied_roles,
    )


def test_role_change_invalidates_cached_group_values() -> None:
    simulation, parent, child = build_simulation()
    holder = simulation.household.get_holder("group_value")
    holder._memory_storage.put(numpy.array([1, 2]), "2024-02")

    simulation.household.set_roles_for_period(
        "2024-02",
        [child, parent, child, parent],
    )

    assert holder.get_array("2024-02") is None


def test_role_change_invalidates_person_formula_cache() -> None:
    simulation, parent, child = build_simulation()
    numpy.testing.assert_array_equal(
        simulation.calculate("is_parent", "2024-02"),
        [True, False, True, False],
    )

    simulation.household.set_roles_for_period(
        "2024-02",
        [child, parent, child, parent],
    )

    numpy.testing.assert_array_equal(
        simulation.calculate("is_parent", "2024-02"),
        [False, True, False, True],
    )


@pytest.mark.parametrize(
    ("roles", "message"),
    [
        ([None, None, None, None], "roles defined"),
        ([[None], [None], [None], [None]], "one-dimensional"),
        ([None], "one entry per member"),
    ],
)
def test_invalid_role_snapshots_are_rejected(roles, message) -> None:
    simulation, _, _ = build_simulation()

    with pytest.raises(ValueError, match=message):
        simulation.household.set_roles_for_period("2024-02", roles)


def test_role_capacity_is_validated_without_mutating_membership() -> None:
    simulation, parent, child = build_simulation()

    with pytest.raises(ValueError, match="at most 1"):
        simulation.household.set_members_for_period(
            "2024-02",
            [0, 0, 1, 1],
            members_role=[parent, parent, child, child],
        )

    assert simulation.household._get_snapshot_period("2024-02") is None
