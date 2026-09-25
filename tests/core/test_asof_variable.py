from __future__ import annotations

import numpy
import pytest

from openfisca_core.entities import Entity
from openfisca_core.holders import Holder, set_input_divide_by_period
from openfisca_core.indexed_enums import Enum
from openfisca_core.periods import DateUnit, period
from openfisca_core.populations import Population
from openfisca_core.variables import Variable


entity = Entity("person", "persons", "", "")


class AsOfVariable(Variable):
    value_type = int
    entity = entity
    definition_period = DateUnit.MONTH
    as_of = "start"


class RegularVariable(Variable):
    value_type = int
    entity = entity
    definition_period = DateUnit.MONTH


def make_holder(variable_class=AsOfVariable, count=3):
    population = Population(entity)
    population.count = count
    population.simulation = None
    return Holder(variable_class(), population)


def test_as_of_persists_latest_state_forward():
    holder = make_holder()
    holder.set_input("2024-01", [1, 2, 3])
    holder.set_input("2024-03", [1, 9, 3])

    numpy.testing.assert_array_equal(holder.get_array("2024-02"), [1, 2, 3])
    numpy.testing.assert_array_equal(holder.get_array("2024-04"), [1, 9, 3])


def test_as_of_returns_none_before_first_state():
    holder = make_holder()
    holder.set_input("2024-02", [1, 2, 3])

    assert holder.get_array("2024-01") is None


def test_as_of_dense_inputs_replace_all_values():
    holder = make_holder()
    holder.set_input("2024-01", [1, 2, 3])
    holder.set_input("2024-02", [1, 8, 3])
    holder.set_input("2024-03", [1, 8, 3])

    assert len(holder._as_of_patches) == 2
    _, indices, values = holder._as_of_patches[0]
    numpy.testing.assert_array_equal(indices, [0, 1, 2])
    numpy.testing.assert_array_equal(values, [1, 8, 3])


def test_as_of_reconstructs_retroactive_state():
    holder = make_holder()
    holder.set_input("2024-01", [10, 20, 30])
    holder.set_input("2024-03", [10, 21, 30])
    holder.set_input("2024-02", [11, 20, 30])

    numpy.testing.assert_array_equal(holder.get_array("2024-01"), [10, 20, 30])
    numpy.testing.assert_array_equal(holder.get_array("2024-02"), [11, 20, 30])
    numpy.testing.assert_array_equal(holder.get_array("2024-03"), [10, 21, 30])


def test_as_of_accepts_state_before_initial_base():
    holder = make_holder()
    holder.set_input("2024-03", [3, 3, 3])
    holder.set_input("2024-01", [1, 1, 1])

    numpy.testing.assert_array_equal(holder.get_array("2024-02"), [1, 1, 1])
    numpy.testing.assert_array_equal(holder.get_array("2024-03"), [3, 3, 3])


def test_as_of_arrays_are_immutable_defensive_copies():
    holder = make_holder()
    source = numpy.array([1, 2, 3])
    holder.set_input("2024-01", source)
    source[0] = 99

    result = holder.get_array("2024-01")
    numpy.testing.assert_array_equal(result, [1, 2, 3])
    assert not result.flags.writeable


def test_as_of_clone_has_independent_patch_index():
    holder = make_holder()
    holder.set_input("2024-01", [1, 2, 3])
    clone = holder.clone(holder.population)
    clone.set_input("2024-02", [4, 2, 3])

    numpy.testing.assert_array_equal(holder.get_array("2024-02"), [1, 2, 3])
    numpy.testing.assert_array_equal(clone.get_array("2024-02"), [4, 2, 3])


def test_as_of_snapshots_are_bounded_fifo():
    class TwoSnapshotVariable(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = "start"
        snapshot_count = 2

    holder = make_holder(TwoSnapshotVariable)
    holder.set_input("2024-01", [1, 1, 1])
    holder.set_input("2024-02", [2, 2, 2])

    # Reading the oldest entry must not promote it as an LRU cache would.
    holder.get_array("2024-01")
    holder.set_input("2024-03", [3, 3, 3])

    assert list(holder._as_of_snapshots) == [
        period("2024-02").start,
        period("2024-03").start,
    ]


def test_as_of_retroactive_write_invalidates_later_snapshots():
    holder = make_holder()
    holder.set_input("2024-01", [1, 1, 1])
    holder.set_input("2024-03", [1, 3, 1])
    holder.get_array("2024-04")

    holder.set_input("2024-02", [2, 1, 1])

    assert period("2024-03").start not in holder._as_of_snapshots
    assert period("2024-04").start not in holder._as_of_snapshots
    numpy.testing.assert_array_equal(holder.get_array("2024-04"), [1, 3, 1])


def test_as_of_snapshots_support_forward_and_backward_access():
    holder = make_holder()
    holder.set_input("2024-01", [1, 1, 1])
    holder.set_input("2024-03", [3, 1, 1])
    holder.set_input("2024-05", [5, 1, 1])

    numpy.testing.assert_array_equal(holder.get_array("2024-06"), [5, 1, 1])
    numpy.testing.assert_array_equal(holder.get_array("2024-02"), [1, 1, 1])
    numpy.testing.assert_array_equal(holder.get_array("2024-04"), [3, 1, 1])


def test_snapshot_count_must_be_positive():
    class InvalidVariable(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = "start"
        snapshot_count = 0

    with pytest.raises(ValueError, match="snapshot_count.*positive"):
        InvalidVariable()


def test_set_input_sparse_updates_selected_values():
    holder = make_holder()
    holder.set_input("2024-01", [1, 2, 3])
    holder.set_input_sparse("2024-02", [0, 2], [10, 30])

    numpy.testing.assert_array_equal(holder.get_array("2024-02"), [10, 2, 30])


def test_set_input_sparse_broadcasts_scalar_and_ignores_unchanged_values():
    holder = make_holder()
    holder.set_input("2024-01", [1, 2, 1])
    holder.set_input_sparse("2024-02", [0, 2], 1)

    assert holder._as_of_patches == []


def test_set_input_sparse_broadcasts_scalar_enum():
    class Status(Enum):
        inactive = "Inactive"
        active = "Active"

    class EnumState(Variable):
        value_type = Enum
        possible_values = Status
        default_value = Status.inactive
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True

    holder = make_holder(EnumState)
    holder.set_input("2024-01", ["inactive", "inactive", "inactive"])
    holder.set_input_sparse("2024-02", [0, 2], Status.active)

    assert holder.get_array("2024-02").decode_to_str().tolist() == [
        "active",
        "inactive",
        "active",
    ]


def test_set_input_sparse_requires_as_of_and_base():
    with pytest.raises(ValueError, match="only available for as_of"):
        make_holder(RegularVariable).set_input_sparse("2024-01", [0], [1])

    with pytest.raises(ValueError, match="Call set_input first"):
        make_holder().set_input_sparse("2024-01", [0], [1])


@pytest.mark.parametrize(
    ("indices", "values", "message"),
    [
        ([0.5], [1], "integer array"),
        ([[0]], [1], "one-dimensional"),
        ([0, 0], [1, 2], "duplicates"),
        ([0, 1], [1], "match the number"),
    ],
)
def test_set_input_sparse_validates_shape(indices, values, message):
    holder = make_holder()
    holder.set_input("2024-01", [1, 2, 3])

    with pytest.raises(ValueError, match=message):
        holder.set_input_sparse("2024-02", indices, values)


def test_set_input_sparse_validates_bounds():
    holder = make_holder()
    holder.set_input("2024-01", [1, 2, 3])

    with pytest.raises(IndexError, match="between 0 and 2"):
        holder.set_input_sparse("2024-02", [3], [1])


def test_set_input_sparse_before_base_has_clear_error():
    holder = make_holder()
    holder.set_input("2024-02", [1, 2, 3])

    with pytest.raises(ValueError, match="Use set_input to move the base"):
        holder.set_input_sparse("2024-01", [0], [5])


def test_as_of_end_uses_period_end_for_lookup():
    class AsOfEndVariable(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = "end"

    holder = make_holder(AsOfEndVariable)
    holder.set_input("2024-02", [4, 5, 6])

    assert holder.get_array(period("day:2024-01-31:2")) is not None


def test_regular_variable_is_unchanged():
    holder = make_holder(RegularVariable)
    holder.set_input("2024-01", [1, 2, 3])

    assert holder.get_array("2024-02") is None


@pytest.mark.parametrize(
    ("declared", "expected"),
    [(True, "start"), ("start", "start"), ("end", "end"), (False, False)],
)
def test_as_of_normalization(declared, expected):
    class TestVariable(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = declared

    assert TestVariable().as_of == expected


def test_as_of_defaults_to_false():
    assert RegularVariable().as_of is False


def test_as_of_rejects_invalid_value():
    class InvalidVariable(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = "monthly"

    with pytest.raises(ValueError, match="as_of"):
        InvalidVariable()


def test_as_of_rejects_set_input_helper():
    class InvalidVariable(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True
        set_input = set_input_divide_by_period

    with pytest.raises(ValueError, match="incompatible"):
        InvalidVariable()


def test_as_of_rejects_eternity_definition_period():
    class InvalidVariable(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.ETERNITY
        as_of = True

    with pytest.raises(ValueError, match="ETERNITY"):
        InvalidVariable()


def test_as_of_delete_arrays_removes_period_events():
    holder = make_holder()
    holder.set_input("2024-01", [1, 1, 1])
    holder.set_input_sparse("2024-02", [0], [2])
    holder.set_input_sparse("2024-03", [1], [3])

    holder.delete_arrays("2024-02")

    assert holder.get_known_periods() == [period("2024-01"), period("2024-03")]
    numpy.testing.assert_array_equal(holder.get_array("2024-03"), [1, 3, 1])


def test_as_of_delete_arrays_clears_dependent_history_with_base():
    holder = make_holder()
    holder.set_input("2024-01", [1, 1, 1])
    holder.set_input_sparse("2024-02", [0], [2])

    holder.delete_arrays("2024-01")

    assert holder.get_known_periods() == []
    assert holder.get_array("2024-02") is None


def test_as_of_memory_usage_includes_base_patches_and_snapshots():
    holder = make_holder(count=3)
    holder.set_input("2024-01", [1, 1, 1])
    holder.set_input_sparse("2024-02", [0], [2])
    holder.get_array("2024-03")

    arrays = [holder._as_of_base]
    arrays.extend(
        array
        for _, indices, values in holder._as_of_patches
        for array in (indices, values)
    )
    arrays.extend(array for array, _ in holder._as_of_snapshots.values())
    unique_arrays = {id(array): array for array in arrays}

    usage = holder.get_memory_usage()
    assert usage["nb_arrays"] == len(unique_arrays)
    array_bytes = sum(array.nbytes for array in unique_arrays.values())
    assert usage["storage_overhead_bytes"] > 0
    assert usage["total_nb_bytes"] == array_bytes + usage["storage_overhead_bytes"]


def test_as_of_does_not_persist_past_variable_end():
    class EndedState(Variable):
        value_type = int
        entity = entity
        definition_period = DateUnit.MONTH
        as_of = True
        end = "2024-02-29"

    holder = make_holder(EndedState)
    holder.set_input("2024-02", [1, 2, 3])

    assert holder.get_array("2024-03") is None
