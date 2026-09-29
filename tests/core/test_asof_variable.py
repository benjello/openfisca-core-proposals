from __future__ import annotations

import numpy
import pytest

from openfisca_core.entities import Entity
from openfisca_core.holders import Holder, set_input_divide_by_period
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
