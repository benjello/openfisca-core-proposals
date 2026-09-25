import numpy
import pytest

from openfisca_core.simulations import SimulationBuilder


def test_set_members_entity_id_updates_structure_and_invalidates_cached_values(
    tax_benefit_system,
) -> None:
    population = SimulationBuilder.build_default_simulation(
        tax_benefit_system,
        count=4,
    ).household
    old_roles = population.members_role
    population.members_position
    population.ordered_members_map

    population.set_members_entity_id([0, 2, 0, 2])

    numpy.testing.assert_array_equal(population.members_entity_id, [0, 2, 0, 2])
    assert population.count == 3
    numpy.testing.assert_array_equal(population.members_position, [0, 0, 1, 1])
    numpy.testing.assert_array_equal(population.ordered_members_map, [0, 2, 1, 3])
    assert population.members_role is not old_roles


@pytest.mark.parametrize(
    ("members_entity_id", "message"),
    [
        ([], "cannot be empty"),
        ([[0, 1], [1, 0]], "one-dimensional"),
        ([0, 1, 2], "one entry per member"),
        ([0.0, 0.0, 1.0, 1.0], "integer IDs"),
        ([0, 0, -1, 1], "non-negative IDs"),
    ],
)
def test_set_members_entity_id_rejects_invalid_values(
    tax_benefit_system,
    members_entity_id,
    message,
) -> None:
    population = SimulationBuilder.build_default_simulation(
        tax_benefit_system,
        count=4,
    ).household

    with pytest.raises(ValueError, match=message):
        population.set_members_entity_id(members_entity_id)


def test_members_entity_id_property_rejects_sparse_values(
    tax_benefit_system,
) -> None:
    population = SimulationBuilder.build_default_simulation(
        tax_benefit_system,
        count=4,
    ).household

    with pytest.raises(ValueError, match="too sparse"):
        population.members_entity_id = numpy.array([0, 0, 1, 1_000_000_000])
