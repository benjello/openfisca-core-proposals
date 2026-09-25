from __future__ import annotations

import bisect
import os
import sys
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy
import psutil

from openfisca_core import (
    commons,
    errors,
    periods,
    types,
)
from openfisca_core import (
    data_storage as storage,
)
from openfisca_core import (
    indexed_enums as enums,
)

from . import types as t


class Holder:
    """A holder keeps tracks of a variable values after they have been calculated, or set as an input."""

    def __init__(self, variable, population) -> None:
        self.population = population
        self.variable = variable
        self.simulation = population.simulation
        self._eternal = self.variable.definition_period == periods.DateUnit.ETERNITY
        self._as_of = self.variable.as_of
        self._memory_storage = storage.InMemoryStorage(is_eternal=self._eternal)
        if self._as_of:
            self._as_of_base = None
            self._as_of_base_instant = None
            self._as_of_base_source = None
            self._as_of_patches = []
            self._as_of_patch_instants = []
            self._as_of_patch_sources = []
            self._as_of_snapshots = OrderedDict()
            self._as_of_max_snapshots = self.variable.snapshot_count
            self._as_of_transition_computed = set()
            self._as_of_known_periods = set()
            self._as_of_explicit_periods = set()
            self._as_of_calculated_periods = set()

        # By default, do not activate on-disk storage, or variable dropping
        self._disk_storage = None
        self._on_disk_storable = False
        self._do_not_store = False
        if self.simulation and self.simulation.memory_config:
            if (
                self.variable.name
                not in self.simulation.memory_config.priority_variables
            ):
                self._disk_storage = self.create_disk_storage()
                self._on_disk_storable = True
            if self.variable.name in self.simulation.memory_config.variables_to_drop:
                self._do_not_store = True

    def clone(self, population: t.CorePopulation) -> t.Holder:
        """Copy the holder just enough to be able to run a new simulation without modifying the original simulation."""
        new = commons.empty_clone(self)
        new_dict = new.__dict__

        for key, value in self.__dict__.items():
            if key not in ("population", "formula", "simulation"):
                new_dict[key] = value

        if self._as_of:
            new_dict["_as_of_patches"] = list(self._as_of_patches)
            new_dict["_as_of_patch_instants"] = list(self._as_of_patch_instants)
            new_dict["_as_of_patch_sources"] = list(self._as_of_patch_sources)
            new_dict["_as_of_snapshots"] = OrderedDict()
            new_dict["_as_of_transition_computed"] = set(
                self._as_of_transition_computed,
            )
            new_dict["_as_of_known_periods"] = set(self._as_of_known_periods)
            new_dict["_as_of_explicit_periods"] = set(self._as_of_explicit_periods)
            new_dict["_as_of_calculated_periods"] = set(
                self._as_of_calculated_periods,
            )

        new_dict["population"] = population
        new_dict["simulation"] = population.simulation

        return new

    def create_disk_storage(self, directory=None, preserve=False):
        if directory is None:
            directory = self.simulation.data_storage_dir
        storage_dir = os.path.join(directory, self.variable.name)
        if not os.path.isdir(storage_dir):
            os.mkdir(storage_dir)
        return storage.OnDiskStorage(
            storage_dir,
            self._eternal,
            preserve_storage_dir=preserve,
        )

    def delete_arrays(self, period=None) -> None:
        """If ``period`` is ``None``, remove all known values of the variable.

        If ``period`` is not ``None``, only remove all values for any period included in period (e.g. if period is "2017", values for "2017-01", "2017-07", etc. would be removed)
        """
        if self._as_of:
            self._delete_as_of(period)
        self._memory_storage.delete(period)
        if self._disk_storage:
            self._disk_storage.delete(period)

    def _delete_as_of(self, period=None) -> None:
        if period is None:
            self._as_of_base = None
            self._as_of_base_instant = None
            self._as_of_base_source = None
            self._as_of_patches = []
            self._as_of_patch_instants = []
            self._as_of_patch_sources = []
            self._as_of_snapshots.clear()
            self._as_of_transition_computed.clear()
            self._as_of_known_periods.clear()
            self._as_of_explicit_periods.clear()
            self._as_of_calculated_periods.clear()
            return

        period = periods.period(period)
        if (
            self._as_of_base_instant is not None
            and period.start <= self._as_of_base_instant <= period.stop
        ):
            replacement_position = next(
                (
                    position
                    for position, (patch, source) in enumerate(
                        zip(
                            self._as_of_patches,
                            self._as_of_patch_sources,
                            strict=True,
                        ),
                    )
                    if patch[0] > period.stop and source == "explicit_dense"
                ),
                None,
            )
            if replacement_position is None:
                self._delete_as_of()
                return

            replacement = self._as_of_patches[replacement_position]
            self._as_of_base = self._immutable_array(replacement[2])
            self._as_of_base_instant = replacement[0]
            self._as_of_base_source = "explicit_dense"
            retained_patches = [
                (patch, source)
                for patch, source in zip(
                    self._as_of_patches[replacement_position + 1 :],
                    self._as_of_patch_sources[replacement_position + 1 :],
                    strict=True,
                )
                if source != "calculated"
            ]
            self._as_of_patches = [patch for patch, _ in retained_patches]
            self._as_of_patch_sources = [source for _, source in retained_patches]
            self._as_of_patch_instants = [patch[0] for patch in self._as_of_patches]
            self._as_of_snapshots.clear()
            self._cache_as_of_snapshot(
                self._as_of_base_instant,
                self._as_of_base,
                0,
            )
            self._as_of_explicit_periods = {
                known_period
                for known_period in self._as_of_explicit_periods
                if self._as_of_reference_instant(known_period)
                >= self._as_of_base_instant
                and not period.contains(known_period)
            }
            self._as_of_known_periods = set(self._as_of_explicit_periods)
            self._as_of_calculated_periods.clear()
            self._as_of_transition_computed = {
                self._as_of_reference_instant(explicit_period)
                for explicit_period in self._as_of_explicit_periods
            }
            return

        retained_patches = [
            (patch, source)
            for patch, source in zip(
                self._as_of_patches,
                self._as_of_patch_sources,
                strict=True,
            )
            if not period.start <= patch[0] <= period.stop
        ]
        self._as_of_patches = [patch for patch, _ in retained_patches]
        self._as_of_patch_sources = [source for _, source in retained_patches]
        self._as_of_patch_instants = [patch[0] for patch in self._as_of_patches]
        self._as_of_snapshots.clear()
        self._as_of_transition_computed = {
            instant
            for instant in self._as_of_transition_computed
            if not period.start <= instant <= period.stop
        }
        self._as_of_known_periods = {
            known_period
            for known_period in self._as_of_known_periods
            if not period.contains(known_period)
        }
        self._as_of_explicit_periods = {
            known_period
            for known_period in self._as_of_explicit_periods
            if not period.contains(known_period)
        }
        self._as_of_calculated_periods = {
            known_period
            for known_period in self._as_of_calculated_periods
            if not period.contains(known_period)
        }
        self._invalidate_as_of_calculations_from(period)

    def get_array(self, period):
        """Get the value of the variable for the given period.

        If the value is not known, return ``None``.
        """
        if self.variable.is_neutralized:
            return self.default_array()
        if self._as_of:
            period = periods.period(period)
            if self.variable.end and period.start.date > self.variable.end:
                return None
            return self._get_as_of(period)
        value = self._memory_storage.get(period)
        if value is not None:
            return value
        if self._disk_storage:
            return self._disk_storage.get(period)
        return None

    def _get_as_of(self, period):
        return self._reconstruct_as_of(self._as_of_reference_instant(period))

    def _as_of_reference_instant(self, period):
        return period.start if self._as_of == "start" else period.stop

    def _reconstruct_as_of(self, target):
        if self._as_of_base is None or target < self._as_of_base_instant:
            return None

        patch_count = bisect.bisect_right(self._as_of_patch_instants, target)
        cached = self._as_of_snapshots.get(target)
        if cached is not None:
            return cached[0]

        best_instant = None
        best_array = self._as_of_base
        first_patch = 0
        for snapshot_instant, (
            snapshot,
            snapshot_patch_count,
        ) in self._as_of_snapshots.items():
            if snapshot_instant < target and (
                best_instant is None or snapshot_instant > best_instant
            ):
                best_instant = snapshot_instant
                best_array = snapshot
                first_patch = snapshot_patch_count

        result = best_array
        for _, indices, values in self._as_of_patches[first_patch:patch_count]:
            if result is best_array:
                result = result.copy()
            result[indices] = values
        if result is not best_array:
            result.flags.writeable = False
        self._cache_as_of_snapshot(target, result, patch_count)
        return result

    def _cache_as_of_snapshot(self, instant, array, patch_count) -> None:
        self._as_of_snapshots[instant] = (array, patch_count)
        if len(self._as_of_snapshots) > self._as_of_max_snapshots:
            self._as_of_snapshots.popitem(last=False)

    def _invalidate_as_of_snapshots_from(self, instant) -> None:
        for cached_instant in [
            cached for cached in self._as_of_snapshots if cached >= instant
        ]:
            del self._as_of_snapshots[cached_instant]

    @staticmethod
    def _immutable_array(value):
        result = value.copy()
        result.flags.writeable = False
        return result

    def _insert_as_of_patch(self, instant, indices, values, source) -> None:
        indices = self._immutable_array(indices.astype(numpy.int32, copy=False))
        values = self._immutable_array(values)
        position = bisect.bisect_right(self._as_of_patch_instants, instant)
        self._invalidate_as_of_snapshots_from(instant)
        self._as_of_patch_instants.insert(position, instant)
        self._as_of_patches.insert(position, (instant, indices, values))
        self._as_of_patch_sources.insert(position, source)

    def _record_as_of_period(self, period, source) -> None:
        self._as_of_known_periods.add(period)
        if source == "calculated":
            self._as_of_calculated_periods.add(period)
        else:
            self._as_of_explicit_periods.add(period)

    def _invalidate_as_of_calculations_from(
        self,
        period,
        *,
        preserve_calculated_base=False,
    ) -> None:
        cutoff = period.start
        reference = self._as_of_reference_instant(period)
        if (
            self._as_of_base_source == "calculated"
            and self._as_of_base_instant >= cutoff
            and not preserve_calculated_base
        ):
            retained_patches = [
                (patch, source)
                for patch, source in zip(
                    self._as_of_patches,
                    self._as_of_patch_sources,
                    strict=True,
                )
                if source != "calculated"
            ]
            self._as_of_base = None
            self._as_of_base_instant = None
            self._as_of_base_source = None
            self._as_of_patches = [patch for patch, _ in retained_patches]
            self._as_of_patch_sources = [source for _, source in retained_patches]
            self._as_of_patch_instants = [patch[0] for patch in self._as_of_patches]
            self._as_of_snapshots.clear()
            self._as_of_calculated_periods.clear()
            self._as_of_known_periods = set(self._as_of_explicit_periods)
            self._as_of_transition_computed = {
                self._as_of_reference_instant(explicit_period)
                for explicit_period in self._as_of_explicit_periods
            }
            return

        retained_patches = [
            (patch, source)
            for patch, source in zip(
                self._as_of_patches,
                self._as_of_patch_sources,
                strict=True,
            )
            if source != "calculated" or patch[0] < cutoff
        ]
        self._as_of_patches = [patch for patch, _ in retained_patches]
        self._as_of_patch_sources = [source for _, source in retained_patches]
        self._as_of_patch_instants = [patch[0] for patch in self._as_of_patches]
        invalidated_periods = {
            known_period
            for known_period in self._as_of_calculated_periods
            if self._as_of_reference_instant(known_period) >= reference
            and not (
                preserve_calculated_base
                and known_period.start == self._as_of_base_instant
            )
        }
        self._as_of_calculated_periods -= invalidated_periods
        self._as_of_known_periods -= invalidated_periods - self._as_of_explicit_periods
        self._as_of_transition_computed = {
            instant
            for instant in self._as_of_transition_computed
            if instant < reference
            or any(
                self._as_of_reference_instant(explicit_period) == instant
                for explicit_period in self._as_of_explicit_periods
            )
        }
        self._invalidate_as_of_snapshots_from(cutoff)

    def _set_as_of(self, period, value, source) -> None:
        instant = period.start
        if source != "calculated":
            self._invalidate_as_of_calculations_from(period)
        self._record_as_of_period(period, source)
        self._as_of_transition_computed.add(self._as_of_reference_instant(period))
        if self._as_of_base is None:
            self._as_of_base = self._immutable_array(value)
            self._as_of_base_instant = instant
            self._as_of_base_source = source
            self._cache_as_of_snapshot(instant, self._as_of_base, 0)
            return

        if instant < self._as_of_base_instant:
            previous_base = self._as_of_base
            previous_base_instant = self._as_of_base_instant
            previous_base_source = self._as_of_base_source
            self._as_of_base = self._immutable_array(value)
            self._as_of_base_instant = instant
            self._as_of_base_source = source
            if previous_base_source == "explicit_dense":
                indices = numpy.arange(len(previous_base))
                self._insert_as_of_patch(
                    previous_base_instant,
                    indices,
                    previous_base,
                    previous_base_source,
                )
            else:
                changed = previous_base != self._as_of_base
                if changed.any():
                    indices = numpy.flatnonzero(changed)
                    self._insert_as_of_patch(
                        previous_base_instant,
                        indices,
                        previous_base[indices],
                        previous_base_source,
                    )
            self._cache_as_of_snapshot(instant, self._as_of_base, 0)
            return

        previous = self._reconstruct_as_of(instant)
        changed = value != previous
        if source == "explicit_dense" or changed.any():
            indices = (
                numpy.arange(len(value))
                if source == "explicit_dense"
                else numpy.flatnonzero(changed)
            )
            self._insert_as_of_patch(instant, indices, value[indices], source)
            patch_count = bisect.bisect_right(self._as_of_patch_instants, instant)
            self._cache_as_of_snapshot(
                instant,
                self._immutable_array(value),
                patch_count,
            )

    def set_input_sparse(
        self,
        period,
        indices,
        values,
        *,
        as_of_source="explicit_sparse",
    ) -> None:
        """Set selected values of an ``as_of`` variable as a sparse patch."""
        if not self._as_of:
            msg = (
                f"set_input_sparse is only available for as_of variables; "
                f'"{self.variable.name}" does not declare as_of.'
            )
            raise ValueError(msg)
        if self._as_of_base is None:
            msg = (
                "set_input_sparse requires an initial state. Call set_input first "
                "to establish the base."
            )
            raise ValueError(msg)

        period = periods.period(period)
        self._check_set_period(period)
        if period.start < self._as_of_base_instant:
            msg = (
                f"Cannot set a sparse value at {period.start}, before the as_of "
                f"base at {self._as_of_base_instant}. Use set_input to move the base."
            )
            raise ValueError(msg)

        raw_indices = numpy.asarray(indices)
        if raw_indices.ndim != 1 or raw_indices.dtype.kind not in "iu":
            msg = "set_input_sparse indices must be a one-dimensional integer array."
            raise ValueError(msg)
        indices = raw_indices.astype(numpy.intp, copy=False)
        if len(numpy.unique(indices)) != len(indices):
            msg = "set_input_sparse indices must not contain duplicates."
            raise ValueError(msg)
        if ((indices < 0) | (indices >= self.population.count)).any():
            msg = (
                "set_input_sparse indices must be between 0 and "
                f"{self.population.count - 1}."
            )
            raise IndexError(msg)

        values = numpy.asarray(values)
        scalar_value = values.ndim == 0
        if scalar_value:
            values = values.reshape(1)
        if self.variable.value_type == enums.Enum:
            values = self.variable.possible_values.encode(values)
        if scalar_value:
            values = numpy.full(len(indices), values[0], dtype=self.variable.dtype)
        if values.ndim != 1 or len(values) != len(indices):
            msg = (
                "set_input_sparse values must be one-dimensional and match the "
                f"number of indices ({len(indices)})."
            )
            raise ValueError(msg)
        try:
            values = values.astype(self.variable.dtype, copy=False)
        except (TypeError, ValueError) as error:
            msg = (
                f'set_input_sparse values for "{self.variable.name}" cannot be '
                f"converted to {self.variable.dtype}."
            )
            raise ValueError(msg) from error

        if len(indices) == 0:
            if as_of_source == "calculated":
                self._record_as_of_period(period, as_of_source)
            return
        if as_of_source != "calculated":
            self._invalidate_as_of_calculations_from(
                period,
                preserve_calculated_base=(
                    self._as_of_base_source == "calculated"
                    and self._as_of_base_instant == period.start
                ),
            )
        self._record_as_of_period(period, as_of_source)
        previous = self._reconstruct_as_of(period.start)
        changed = values != previous[indices]
        if changed.any():
            self._insert_as_of_patch(
                period.start,
                indices[changed],
                values[changed],
                as_of_source,
            )
        self._as_of_transition_computed.add(self._as_of_reference_instant(period))

    def get_memory_usage(self) -> t.MemoryUsage:
        """Get data about the virtual memory usage of the Holder.

        Returns:
            Memory usage data.

        Examples:
            >>> from pprint import pprint

            >>> from openfisca_core import (
            ...     entities,
            ...     populations,
            ...     simulations,
            ...     taxbenefitsystems,
            ...     variables,
            ... )

            >>> entity = entities.Entity("", "", "", "")

            >>> class MyVariable(variables.Variable):
            ...     definition_period = periods.DateUnit.YEAR
            ...     entity = entity
            ...     value_type = int

            >>> population = populations.Population(entity)
            >>> variable = MyVariable()
            >>> holder = Holder(variable, population)

            >>> tbs = taxbenefitsystems.TaxBenefitSystem([entity])
            >>> entities = {entity.key: population}
            >>> simulation = simulations.Simulation(tbs, entities)
            >>> holder.simulation = simulation

            >>> pprint(holder.get_memory_usage(), indent=3)
            {  'cell_size': nan,
               'dtype': <class 'numpy.int32'>,
               'nb_arrays': 0,
               'nb_cells_by_array': 0,
               'total_nb_bytes': 0...

        """
        usage = t.MemoryUsage(
            nb_cells_by_array=self.population.count,
            dtype=self.variable.dtype,
        )

        if self._as_of:
            arrays = [self._as_of_base]
            arrays.extend(
                array
                for _, indices, values in self._as_of_patches
                for array in (indices, values)
            )
            arrays.extend(snapshot for snapshot, _ in self._as_of_snapshots.values())
            unique_arrays = {id(array): array for array in arrays if array is not None}
            array_bytes = sum(array.nbytes for array in unique_arrays.values())
            overhead_bytes = sum(
                max(sys.getsizeof(array) - array.nbytes, 0)
                for array in unique_arrays.values()
            )
            overhead_bytes += sum(
                sys.getsizeof(container)
                for container in (
                    self._as_of_patches,
                    self._as_of_patch_instants,
                    self._as_of_patch_sources,
                    self._as_of_snapshots,
                    self._as_of_transition_computed,
                    self._as_of_known_periods,
                    self._as_of_explicit_periods,
                    self._as_of_calculated_periods,
                )
            )
            overhead_bytes += sum(map(sys.getsizeof, self._as_of_patches))
            overhead_bytes += sum(
                sys.getsizeof(snapshot) for snapshot in self._as_of_snapshots.values()
            )
            usage.update(
                {
                    "nb_arrays": len(unique_arrays),
                    "total_nb_bytes": array_bytes + overhead_bytes,
                    "cell_size": numpy.dtype(self.variable.dtype).itemsize,
                    "storage_overhead_bytes": overhead_bytes,
                },
            )
        else:
            usage.update(self._memory_storage.get_memory_usage())

        if self.simulation and self.simulation.trace:
            nb_requests = self.simulation.tracer.get_nb_requests(self.variable.name)
            usage.update(
                {
                    "nb_requests": nb_requests,
                    "nb_requests_by_array": (
                        nb_requests / float(usage["nb_arrays"])
                        if usage["nb_arrays"] > 0
                        else numpy.nan
                    ),
                },
            )

        return usage

    def get_known_periods(self):
        """Get the list of periods the variable value is known for."""
        if self._as_of:
            return sorted(self._as_of_known_periods)
        return list(self._memory_storage.get_known_periods()) + list(
            self._disk_storage.get_known_periods() if self._disk_storage else [],
        )

    def set_input(
        self,
        period: types.Period,
        array: numpy.ndarray | Sequence[Any],
    ) -> numpy.ndarray | None:
        """Set a Variable's array of values of a given Period.

        Args:
            period: The period at which the value is set.
            array: The input value for the variable.

        Returns:
            The set input array.

        Note:
            If a ``set_input`` property has been set for the variable, this
            method may accept inputs for periods not matching the
            ``definition_period`` of the Variable. To read
            more about this, check the `documentation`_.

        Examples:
            >>> from openfisca_core import entities, populations, variables

            >>> entity = entities.Entity("", "", "", "")

            >>> class MyVariable(variables.Variable):
            ...     definition_period = periods.DateUnit.YEAR
            ...     entity = entity
            ...     value_type = float

            >>> variable = MyVariable()

            >>> population = populations.Population(entity)
            >>> population.count = 2

            >>> holder = Holder(variable, population)
            >>> holder.set_input("2018", numpy.array([12.5, 14]))
            >>> holder.get_array("2018")
            array([12.5, 14. ], dtype=float32)

            >>> holder.set_input("2018", [12.5, 14])
            >>> holder.get_array("2018")
            array([12.5, 14. ], dtype=float32)

        .. _documentation:
            https://openfisca.org/doc/coding-the-legislation/35_periods.html#set-input-automatically-process-variable-inputs-defined-for-periods-not-matching-the-definition-period

        """
        period = periods.period(period)

        if period.unit == periods.DateUnit.ETERNITY and not self._eternal:
            error_message = os.linesep.join(
                [
                    "Unable to set a value for variable {1} for {0}.",
                    "{1} is only defined for {2}s. Please adapt your input.",
                ],
            ).format(
                periods.DateUnit.ETERNITY.upper(),
                self.variable.name,
                self.variable.definition_period,
            )
            raise errors.PeriodMismatchError(
                self.variable.name,
                period,
                self.variable.definition_period,
                error_message,
            )
        if self.variable.is_neutralized:
            warning_message = f"You cannot set a value for the variable {self.variable.name}, as it has been neutralized. The value you provided ({array}) will be ignored."
            return warnings.warn(warning_message, Warning, stacklevel=2)
        if self.variable.value_type in (float, int) and isinstance(array, str):
            array = commons.eval_expression(array)
        if self.variable.set_input:
            return self.variable.set_input(self, period, array)
        return self._set(period, array)

    def _to_array(self, value):
        if not isinstance(value, numpy.ndarray):
            value = numpy.asarray(value)
        if value.ndim == 0:
            # 0-dim arrays are casted to scalar when they interact with float. We don't want that.
            value = value.reshape(1)
        if len(value) != self.population.count:
            msg = f'Unable to set value "{value}" for variable "{self.variable.name}", as its length is {len(value)} while there are {self.population.count} {self.population.entity.plural} in the simulation.'
            raise ValueError(
                msg,
            )
        if self.variable.value_type == enums.Enum:
            value = self.variable.possible_values.encode(value)
        if value.dtype != self.variable.dtype:
            try:
                value = value.astype(self.variable.dtype)
            except ValueError:
                msg = f'Unable to set value "{value}" for variable "{self.variable.name}", as the variable dtype "{self.variable.dtype}" does not match the value dtype "{value.dtype}".'
                raise ValueError(
                    msg,
                )
        return value

    def _set(self, period, value, *, as_of_source="explicit_dense") -> None:
        value = self._to_array(value)
        self._check_set_period(period)

        if self._as_of:
            self._set_as_of(period, value, as_of_source)
            return

        should_store_on_disk = (
            self._on_disk_storable
            and self._memory_storage.get(period) is None
            and psutil.virtual_memory().percent  # If there is already a value in memory, replace it and don't put a new value in the disk storage
            >= self.simulation.memory_config.max_memory_occupation_pc
        )

        if should_store_on_disk:
            self._disk_storage.put(value, period)
        else:
            self._memory_storage.put(value, period)

    def _check_set_period(self, period) -> None:
        if not self._eternal:
            if period is None:
                msg = (
                    f"A period must be specified to set values, except for variables with "
                    f"{periods.DateUnit.ETERNITY.upper()} as as period_definition."
                )
                raise ValueError(
                    msg,
                )
            if self.variable.definition_period != period.unit or period.size > 1:
                name = self.variable.name
                period_size_adj = (
                    f"{period.unit}"
                    if (period.size == 1)
                    else f"{period.size}-{period.unit}s"
                )
                error_message = os.linesep.join(
                    [
                        f'Unable to set a value for variable "{name}" for {period_size_adj}-long period "{period}".',
                        f'"{name}" can only be set for one {self.variable.definition_period} at a time. Please adapt your input.',
                        f'If you are the maintainer of "{name}", you can consider adding it a set_input attribute to enable automatic period casting.',
                    ],
                )

                raise errors.PeriodMismatchError(
                    self.variable.name,
                    period,
                    self.variable.definition_period,
                    error_message,
                )

    def put_in_cache(self, value, period) -> None:
        if self._do_not_store:
            return

        if (
            self.simulation.opt_out_cache
            and self.simulation.tax_benefit_system.cache_blacklist
            and self.variable.name in self.simulation.tax_benefit_system.cache_blacklist
        ):
            return

        self._set(period, value)

    def default_array(self):
        """Return a new array of the appropriate length for the entity, filled with the variable default values."""
        return self.variable.default_array(self.population.count)
