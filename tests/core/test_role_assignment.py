import numpy
import pytest

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


def test_assignment_uses_latest_role_snapshot() -> None:
    simulation, first_parent, second_parent, child = build_simulation()
    simulation.household.set_roles_for_period(
        "2024-02",
        [second_parent, first_parent, child, first_parent, child, child],
    )

    roles = simulation.household.auto_assign_roles_for_period(
        "2024-03",
        [0, 0, 0, 1, 1, 1],
    )

    assert roles.tolist() == [
        second_parent,
        first_parent,
        child,
        first_parent,
        child,
        child,
    ]


def test_roles_are_remapped_by_id_when_person_count_is_unchanged() -> None:
    simulation, first_parent, second_parent, child = build_simulation()
    simulation.persons.activate_dynamic_mode("2024-01", [10, 20, 30, 40, 50, 60])
    simulation.persons._set_period_identity_snapshot(
        "2024-02",
        numpy.array([20, 10, 30, 40, 50, 60]),
    )

    roles = simulation.household.auto_assign_roles_for_period(
        "2024-02",
        [0, 0, 0, 1, 1, 1],
    )

    assert roles.tolist() == [
        second_parent,
        first_parent,
        child,
        first_parent,
        child,
        child,
    ]


def test_carry_forward_roles_are_validated_against_new_membership() -> None:
    simulation, _, _, _ = build_simulation()

    with pytest.raises(ValueError, match="at most 1"):
        simulation.household.set_members_for_period(
            "2024-02",
            [0, 0, 0, 0, 1, 1],
        )


def reference_assignment(population, membership):
    old_membership = population.members_entity_id
    old_roles = population.members_role
    roles = population.entity.flattened_roles
    role_rank = {role: index for index, role in enumerate(roles)}
    result = numpy.empty(len(membership), dtype=object)
    assigned = numpy.zeros(len(membership), dtype=bool)
    used = [dict.fromkeys(roles, 0) for _ in range(population.count)]

    for row, group_id in enumerate(membership):
        if old_membership[row] == group_id:
            result[row] = old_roles[row]
            assigned[row] = True
            used[group_id][old_roles[row]] += 1

    for row, group_id in enumerate(membership):
        if assigned[row]:
            continue
        old_rank = role_rank.get(old_roles[row], 0)
        search_order = [old_rank, *range(old_rank - 1, -1, -1)]
        search_order.extend(range(old_rank + 1, len(roles)))
        for rank in search_order:
            role = roles[rank]
            if role.max is None or used[group_id][role] < role.max:
                result[row] = role
                assigned[row] = True
                used[group_id][role] += 1
                break

    primary_role = roles[0]
    for group_id in numpy.unique(membership):
        if used[group_id][primary_role]:
            continue
        members = numpy.flatnonzero(membership == group_id)
        promoted = min(members, key=lambda row: (role_rank[result[row]], row))
        result[promoted] = primary_role

    return result


@pytest.mark.parametrize("seed", range(20))
def test_vectorized_assignment_matches_sequential_strategy(seed) -> None:
    simulation, first_parent, second_parent, child = build_simulation()
    rng = numpy.random.default_rng(seed)
    person_count = 120
    group_count = 12
    old_membership = numpy.arange(person_count) % group_count
    rng.shuffle(old_membership)
    new_membership = numpy.arange(person_count) % group_count
    rng.shuffle(new_membership)
    old_roles = numpy.full(person_count, child, dtype=object)
    for group_id in range(group_count):
        rows = numpy.flatnonzero(old_membership == group_id)
        old_roles[rows[0]] = first_parent
        old_roles[rows[1]] = second_parent
    simulation.persons.count = person_count
    simulation.household.count = group_count
    simulation.household.members_entity_id = old_membership
    simulation.household.members_role = old_roles

    expected = reference_assignment(simulation.household, new_membership)
    actual = simulation.household.auto_assign_roles_for_period(
        "2024-02",
        new_membership,
    )

    assert actual.tolist() == expected.tolist()
    for group_id in range(group_count):
        group_roles = actual[new_membership == group_id]
        assert numpy.count_nonzero(group_roles == first_parent) <= 1
        assert numpy.count_nonzero(group_roles == second_parent) <= 1
