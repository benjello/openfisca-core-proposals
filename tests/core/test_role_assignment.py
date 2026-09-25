import numpy

from openfisca_core import entities, taxbenefitsystems
from openfisca_core.simulations import SimulationBuilder


def build_simulation():
    person = entities.SingleEntity("person", "persons", "A person", "")
    household = entities.GroupEntity(
        "household",
        "households",
        "A household",
        "",
        roles=[
            {"key": "first_parent", "max": 1},
            {"key": "second_parent", "max": 1},
            {"key": "child", "plural": "children"},
        ],
    )
    tax_benefit_system = taxbenefitsystems.TaxBenefitSystem([person, household])
    simulation = SimulationBuilder.build_default_simulation(
        tax_benefit_system,
        count=6,
    )
    simulation.household.ids = numpy.array(["household_0", "household_1"])
    simulation.household.set_members_entity_id([0, 0, 0, 1, 1, 1])
    first_parent, second_parent, child = household.flattened_roles
    simulation.household.members_role = [
        first_parent,
        second_parent,
        child,
        first_parent,
        child,
        child,
    ]
    return simulation, first_parent, second_parent, child


def test_stayers_keep_roles_and_mover_keeps_role_when_available() -> None:
    simulation, first_parent, second_parent, child = build_simulation()

    roles = simulation.household.auto_assign_roles_for_period(
        "2024-02",
        [0, 1, 0, 1, 1, 1],
    )

    assert roles.tolist() == [
        first_parent,
        second_parent,
        child,
        first_parent,
        child,
        child,
    ]


def test_mover_climbs_when_same_role_is_full() -> None:
    simulation, first_parent, second_parent, child = build_simulation()
    simulation.household.members_role = [
        first_parent,
        second_parent,
        child,
        second_parent,
        child,
        child,
    ]

    roles = simulation.household.auto_assign_roles_for_period(
        "2024-02",
        [0, 1, 0, 1, 0, 1],
    )

    assert roles[1] == first_parent
    assert roles[4] == child


def test_mover_descends_when_same_role_is_full() -> None:
    simulation, first_parent, second_parent, _ = build_simulation()

    roles = simulation.household.auto_assign_roles_for_period(
        "2024-02",
        [1, 0, 0, 1, 1, 1],
    )

    assert roles[0] == second_parent
    assert roles[1] == first_parent


def test_first_parent_is_guaranteed_with_row_order_tie_break() -> None:
    simulation, first_parent, _, child = build_simulation()
    simulation.household.count = 3
    simulation.household.ids = numpy.array(
        ["household_0", "household_1", "household_2"],
    )

    roles = simulation.household.auto_assign_roles_for_period(
        "2024-02",
        [0, 0, 1, 1, 2, 2],
    )

    assert roles[4] == first_parent
    assert roles[5] == child


def test_assignment_can_be_committed_atomically() -> None:
    simulation, first_parent, second_parent, child = build_simulation()
    membership = [0, 1, 0, 1, 1, 1]
    roles = simulation.household.auto_assign_roles_for_period(
        "2024-02",
        membership,
    )

    simulation.household.set_members_for_period(
        "2024-02",
        membership,
        members_role=roles,
    )

    assert simulation.household._get_members_role("2024-03").tolist() == [
        first_parent,
        second_parent,
        child,
        first_parent,
        child,
        child,
    ]
