import numpy
import pytest

from openfisca_core import entities, periods, taxbenefitsystems, variables
from openfisca_core.simulations import SimulationBuilder
from openfisca_core.tools import simulation_dumper


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

    class rent(variables.Variable):
        value_type = float
        entity = household
        definition_period = periods.DateUnit.MONTH

    tax_benefit_system.add_variables(salary, rent)
    simulation = SimulationBuilder.build_default_simulation(
        tax_benefit_system,
        count=3,
    )
    simulation.household.ids = numpy.array(["household_0", "household_1"])
    simulation.household.set_members_entity_id([0, 0, 1])
    simulation.persons.activate_dynamic_mode("2024-01", [10, 20, 30])
    simulation.household.activate_dynamic_mode("2024-01")
    return simulation, tax_benefit_system


def test_membership_creates_groups_and_keeps_non_contiguous_groups_empty() -> None:
    simulation, _ = build_simulation()
    simulation.household.set_members_for_period("2024-02", [0, 2, 2])

    assert simulation.household.get_count_for_period("2024-03") == 3
    numpy.testing.assert_array_equal(
        simulation.household.nb_persons(period="2024-03"),
        [1, 0, 2],
    )
    numpy.testing.assert_array_equal(
        simulation.household.sum(
            numpy.array([10, 20, 30]),
            period="2024-03",
        ),
        [10, 0, 50],
    )


def test_membership_dissolves_groups_and_count_carries_forward() -> None:
    simulation, _ = build_simulation()
    simulation.household.set_members_for_period("2024-02", [0, 2, 2])
    simulation.household.set_members_for_period("2024-04", [0, 0, 0])

    assert simulation.household.get_count_for_period("2024-03") == 3
    assert simulation.household.get_count_for_period("2024-05") == 1
    numpy.testing.assert_array_equal(
        simulation.household.nb_persons(period="2024-05"),
        [3],
    )


def test_group_inputs_and_projection_use_period_count() -> None:
    simulation, _ = build_simulation()
    simulation.household.set_members_for_period("2024-02", [0, 2, 2])
    simulation.set_input("rent", "2024-02", [100, 0, 300])

    numpy.testing.assert_array_equal(
        simulation.household.project(
            simulation.calculate("rent", "2024-02"),
            period="2024-02",
        ),
        [100, 300, 300],
    )


def test_all_persons_and_groups_can_disappear() -> None:
    simulation, _ = build_simulation()
    simulation.persons.register_deaths("2024-02", [10, 20, 30])
    simulation.household.set_members_for_period("2024-02", [])
    simulation.set_input("salary", "2024-02", [])
    simulation.set_input("rent", "2024-02", [])

    assert simulation.persons.get_count_for_period("2024-02") == 0
    assert simulation.household.get_count_for_period("2024-02") == 0
    assert simulation.household.nb_persons(period="2024-02").size == 0
    assert (
        simulation.household.sum(
            simulation.calculate("salary", "2024-02"),
            period="2024-02",
        ).size
        == 0
    )


def test_dump_restore_preserves_dynamic_group_counts(tmp_path) -> None:
    simulation, tax_benefit_system = build_simulation()
    simulation.household.set_members_for_period("2024-02", [0, 2, 2])
    directory = tmp_path / "simulation"

    simulation_dumper.dump_simulation(simulation, str(directory))
    restored = simulation_dumper.restore_simulation(
        str(directory),
        tax_benefit_system,
    )

    assert restored.household.get_count_for_period("2024-03") == 3
    numpy.testing.assert_array_equal(
        restored.household.nb_persons(period="2024-03"),
        [1, 0, 2],
    )


@pytest.mark.parametrize(
    ("membership", "message"),
    [
        ([-1, 0, 0], "non-negative"),
        ([0.0, 1.0, 1.0], "integer IDs"),
    ],
)
def test_dynamic_membership_still_rejects_invalid_group_ids(
    membership,
    message,
) -> None:
    simulation, _ = build_simulation()

    with pytest.raises(ValueError, match=message):
        simulation.household.set_members_for_period("2024-02", membership)
