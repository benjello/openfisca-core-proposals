from __future__ import annotations

import numpy

from openfisca_core.entities import Entity
from openfisca_core.holders import Holder
from openfisca_core.periods import DateUnit, period
from openfisca_core.populations import Population
from openfisca_core.variables import Variable


entity = Entity("person", "persons", "", "")


class AsOfVariable(Variable):
    value_type = int
    entity = entity
    definition_period = DateUnit.MONTH
    as_of = True


def make_holder(count=100_000):
    population = Population(entity)
    population.count = count
    population.simulation = None
    return Holder(AsOfVariable(), population)


def month(index):
    return f"{2020 + index // 12}-{index % 12 + 1:02d}"


def changes(count, periods_count=24, change_rate=0.01):
    random = numpy.random.default_rng(42)
    changed_count = int(count * change_rate)
    return [
        (
            random.choice(count, size=changed_count, replace=False),
            random.integers(0, 100, size=changed_count, dtype=numpy.int32),
        )
        for _ in range(periods_count)
    ]


def populated_holder(count=100_000, periods_count=24):
    holder = make_holder(count)
    holder.set_input(month(0), numpy.zeros(count, dtype=numpy.int32))
    for index, (indices, values) in enumerate(changes(count, periods_count), 1):
        holder.set_input_sparse(month(index), indices, values)
    return holder


def test_sparse_storage_size():
    count = 100_000
    periods_count = 24
    holder = populated_holder(count, periods_count)

    sparse_bytes = holder._as_of_base.nbytes + sum(
        indices.nbytes + values.nbytes for _, indices, values in holder._as_of_patches
    )
    dense_bytes = count * numpy.dtype(numpy.int32).itemsize * (periods_count + 1)

    assert sparse_bytes < dense_bytes / 5


def test_reconstruct_forward(benchmark):
    holder = populated_holder()
    requested_periods = [period(month(index)) for index in range(25)]

    def reconstruct():
        holder._as_of_snapshots.clear()
        for requested_period in requested_periods:
            holder.get_array(requested_period)

    benchmark(reconstruct)


def test_reconstruct_backward(benchmark):
    holder = populated_holder()
    requested_periods = [period(month(index)) for index in reversed(range(25))]

    def reconstruct():
        holder._as_of_snapshots.clear()
        for requested_period in requested_periods:
            holder.get_array(requested_period)

    benchmark(reconstruct)


def test_set_input_sparse(benchmark):
    count = 100_000
    updates = changes(count)
    base = numpy.zeros(count, dtype=numpy.int32)

    def write_sparse():
        holder = make_holder(count)
        holder.set_input(month(0), base)
        for index, (indices, values) in enumerate(updates, 1):
            holder.set_input_sparse(month(index), indices, values)

    benchmark(write_sparse)
