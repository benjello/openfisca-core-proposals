import numpy
import pytest

from openfisca_core import entities, taxbenefitsystems
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
    simulation = SimulationBuilder.build_default_simulation(
        tax_benefit_system,
        count=2,
    )
    return simulation, tax_benefit_system


def test_identity_snapshots_carry_forward_chronologically() -> None:
    simulation, _ = build_simulation()
    simulation.persons.activate_dynamic_mode("2024-01", [10, 20])

    assert simulation.persons.get_count_for_period("2024-03") == 2
    mapping = simulation.persons.get_period_id_to_rownum("2024-03")
    assert mapping[10] == 0
    assert mapping[20] == 1


def test_remap_uses_ids_when_count_is_unchanged() -> None:
    simulation, _ = build_simulation()
    simulation.persons.activate_dynamic_mode("2024-01", [10, 20])
    simulation.persons._set_period_identity_snapshot("2024-02", numpy.array([20, 10]))

    result = simulation.persons.remap_array(
        numpy.array([100, 200]),
        "2024-01",
        "2024-03",
        default=-1,
    )

    numpy.testing.assert_array_equal(result, [200, 100])


def test_clone_keeps_independent_identity_indexes() -> None:
    simulation, _ = build_simulation()
    simulation.persons.activate_dynamic_mode("2024-01", [10, 20])

    clone = simulation.clone()
    clone.persons._set_period_identity_snapshot("2024-02", numpy.array([20, 10]))

    assert simulation.persons.get_period_id_to_rownum("2024-02")[10] == 0
    assert clone.persons.get_period_id_to_rownum("2024-02")[10] == 1
    assert not numpy.shares_memory(
        simulation.persons._permanent_ids,
        clone.persons._permanent_ids,
    )
    assert not numpy.shares_memory(
        simulation.persons._id_to_rownum,
        clone.persons._id_to_rownum,
    )
    assert not numpy.shares_memory(
        simulation.persons.get_period_id_to_rownum("2024-01"),
        clone.persons.get_period_id_to_rownum("2024-01"),
    )


def test_identity_snapshot_periods_with_the_same_start_are_rejected() -> None:
    simulation, _ = build_simulation()
    simulation.persons.activate_dynamic_mode("2024-01", [10, 20])

    with pytest.raises(ValueError, match="same start"):
        simulation.persons.snapshot_period("year:2024-01:1")


def test_dump_restore_preserves_identity_snapshots(tmp_path) -> None:
    simulation, tax_benefit_system = build_simulation()
    simulation.persons.activate_dynamic_mode("2024-01", [10, 20])
    simulation.persons._set_period_identity_snapshot("2024-02", numpy.array([20, 10]))
    simulation.household.set_members_for_period("2024-02", [1, 0])
    directory = tmp_path / "simulation"

    simulation_dumper.dump_simulation(simulation, str(directory))
    restored = simulation_dumper.restore_simulation(
        str(directory),
        tax_benefit_system,
    )

    assert restored.persons._dynamic
    assert restored.persons.get_period_id_to_rownum("2024-03")[10] == 1
    numpy.testing.assert_array_equal(
        restored.household._get_members_entity_id("2024-03"),
        [1, 0],
    )


@pytest.mark.parametrize(
    ("permanent_ids", "message"),
    [
        ([[0, 1]], "one-dimensional"),
        ([0], "one entry"),
        ([0.0, 1.0], "integer IDs"),
        ([-1, 1], "non-negative"),
        ([1, 1], "duplicates"),
    ],
)
def test_activate_dynamic_mode_rejects_invalid_ids(permanent_ids, message) -> None:
    simulation, _ = build_simulation()

    with pytest.raises(ValueError, match=message):
        simulation.persons.activate_dynamic_mode("2024-01", permanent_ids)


def test_sparse_permanent_ids_are_rejected_before_dense_allocation() -> None:
    simulation, _ = build_simulation()

    with pytest.raises(ValueError, match="too sparse"):
        simulation.persons.activate_dynamic_mode("2024-01", [0, 1_000_000_000])
