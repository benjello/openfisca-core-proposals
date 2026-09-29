from __future__ import annotations

import traceback
from collections.abc import Sequence
from typing import TypeVar

import numpy

from openfisca_core import holders, indexed_enums, periods

from . import types as t
from ._errors import (
    IncompatibleOptionsError,
    InvalidArraySizeError,
    InvalidOptionError,
    PeriodValidityError,
)

#: Type variable for a covariant data type.
_DT_co = TypeVar("_DT_co", covariant=True, bound=t.VarDType)


class CorePopulation:
    """Base class to build populations from.

    Args:
        entity: The :class:`~entities.CoreEntity` of the population.
        *__args: Variable length argument list.
        **__kwds: Arbitrary keyword arguments.

    """

    #: The number :class:`~entities.CoreEntity` members in the population.
    count: int = 0

    #: The :class:`~entities.CoreEntity` of the population.
    entity: t.CoreEntity

    #: A pseudo index for the members of the population.
    ids: Sequence[str] = []

    #: The :class:`~simulations.Simulation` for which the population is calculated.
    simulation: None | t.Simulation = None

    def __init__(self, entity: t.CoreEntity, *__args: object, **__kwds: object) -> None:
        self.entity = entity
        self._holders: t.HolderByVariable = {}
        self._dynamic = False
        self._permanent_ids = None
        self._id_to_rownum = None
        self._period_index = {}

    def __call__(
        self,
        variable_name: t.VariableName,
        period: t.PeriodLike,
        options: None | Sequence[t.Option] = None,
    ) -> None | t.VarArray:
        """Calculate ``variable_name`` for ``period``, using the formula if it exists.

        Args:
            variable_name: The name of the variable to calculate.
            period: The period to calculate the variable for.
            options: The options to use for the calculation.

        Returns:
            None: If there is no :class:`~simulations.Simulation`.
            ndarray[generic]: The result of the calculation.

        Raises:
            IncompatibleOptionsError: If the options are incompatible.
            InvalidOptionError: If the option is invalid.

        Examples:
            >>> from openfisca_core import (
            ...     entities,
            ...     periods,
            ...     populations,
            ...     simulations,
            ...     taxbenefitsystems,
            ...     variables,
            ... )

            >>> class Person(entities.SingleEntity): ...

            >>> person = Person("person", "people", "", "")
            >>> period = periods.Period.eternity()
            >>> population = populations.CorePopulation(person)
            >>> population.count = 3
            >>> population("salary", period)

            >>> tbs = taxbenefitsystems.TaxBenefitSystem([person])
            >>> person.set_tax_benefit_system(tbs)
            >>> simulation = simulations.Simulation(tbs, {person.key: population})
            >>> population("salary", period)
            Traceback (most recent call last):
            VariableNotFoundError: You tried to calculate or to set a value ...

            >>> class Salary(variables.Variable):
            ...     definition_period = periods.ETERNITY
            ...     entity = person
            ...     value_type = int

            >>> tbs.add_variable(Salary)
            <openfisca_core.populations._core_population.Salary object at...

            >>> population(Salary().name, period)
            array([0, 0, 0], dtype=int32)

            >>> class Tax(Salary):
            ...     default_value = 100.0
            ...     definition_period = periods.ETERNITY
            ...     entity = person
            ...     value_type = float

            >>> tbs.add_variable(Tax)
            <openfisca_core.populations._core_population.Tax object at...

            >>> population(Tax().name, period)
            array([100., 100., 100.], dtype=float32)

            >>> population(Tax().name, period, [populations.ADD])
            Traceback (most recent call last):
            ValueError: Unable to ADD constant variable 'Tax' over the perio...

            >>> population(Tax().name, period, [populations.DIVIDE])
            Traceback (most recent call last):
            ValueError: Unable to DIVIDE constant variable 'Tax' over the pe...

            >>> population(Tax().name, period, [populations.ADD, populations.DIVIDE])
            Traceback (most recent call last):
            IncompatibleOptionsError: Options ADD and DIVIDE are incompatibl...

            >>> population(Tax().name, period, ["LAGRANGIAN"])
            Traceback (most recent call last):
            InvalidOptionError: Option LAGRANGIAN is not a valid option (try...

        """
        if self.simulation is None:
            return None

        calculate = t.Calculate(
            variable=variable_name,
            period=periods.period(period),
            option=options,
        )

        self.entity.check_variable_defined_for_entity(calculate.variable)
        self.check_period_validity(calculate.variable, calculate.period)

        result = None
        result_period = calculate.period
        if not isinstance(calculate.option, Sequence):
            result = self.simulation.calculate(
                calculate.variable,
                calculate.period,
            )
        elif t.Option.ADD in calculate.option and t.Option.DIVIDE in calculate.option:
            raise IncompatibleOptionsError(variable_name)
        elif t.Option.ADD in calculate.option:
            result = self.simulation.calculate_add(
                calculate.variable,
                calculate.period,
            )
            variable = self.entity.get_variable(calculate.variable)
            result_period = calculate.period.get_subperiods(
                variable.definition_period,
            )[-1]
        elif t.Option.DIVIDE in calculate.option:
            result = self.simulation.calculate_divide(
                calculate.variable,
                calculate.period,
            )
        else:
            raise InvalidOptionError(calculate.option[0], variable_name)

        return self._remap_to_outer_period(
            result,
            calculate.variable,
            result_period,
        )

    def empty_array(self) -> t.FloatArray:
        """Return an empty array.

        Returns:
            ndarray[float32]: An empty array.

        Examples:
            >>> import numpy

            >>> from openfisca_core import populations as p

            >>> class Population(p.CorePopulation): ...

            >>> population = Population(None)
            >>> population.empty_array()
            array([], dtype=float32)

            >>> population.count = 3
            >>> population.empty_array()
            array([0., 0., 0.], dtype=float32)

        """
        return numpy.zeros(self.count, dtype=t.FloatDType)

    def filled_array(
        self, value: _DT_co, dtype: None | t.DTypeLike = None
    ) -> t.Array[_DT_co]:
        """Return an array filled with a value.

        Args:
            value: The value to fill the array with.
            dtype: The data type of the array.

        Returns:
            ndarray[generic]: An array filled with the value.

        Examples:
            >>> import numpy

            >>> from openfisca_core import populations

            >>> class Population(populations.CorePopulation): ...

            >>> population = Population(None)
            >>> population.count = 3
            >>> population.filled_array(1)
            array([1, 1, 1])

            >>> population.filled_array(numpy.float32(1))
            array([1., 1., 1.], dtype=float32)

            >>> population.filled_array(1, dtype=str)
            array(['1', '1', '1'], dtype='<U1')

            >>> population.filled_array("hola", dtype=numpy.uint8)
            Traceback (most recent call last):
            ValueError: could not convert string to float: 'hola'

        """
        return numpy.full(self.count, value, dtype)

    def get_index(self, id: str) -> int:
        """Return the index of an `id``.

        Args:
            id: The id to get the index for.

        Returns:
            int: The index of the id.

        Examples:
            >>> from openfisca_core import entities, populations

            >>> class Person(entities.SingleEntity): ...

            >>> person = Person("person", "people", "", "")
            >>> population = populations.CorePopulation(person)
            >>> population.ids = ["Juan", "Megan", "Brahim"]

            >>> population.get_index("Megan")
            1

            >>> population.get_index("Ibrahim")
            Traceback (most recent call last):
            ValueError: 'Ibrahim' is not in list

        """
        return self.ids.index(id)

    # Calculations

    def check_array_compatible_with_entity(self, array: t.VarArray) -> None:
        """Check if an array is compatible with the population.

        Args:
            array: The array to check.

        Raises:
            InvalidArraySizeError: If the array is not compatible.

        Examples:
            >>> import numpy

            >>> from openfisca_core import entities, populations

            >>> class Person(entities.SingleEntity): ...

            >>> person = Person("person", "people", "", "")
            >>> population = populations.CorePopulation(person)
            >>> population.count = 3

            >>> array = numpy.array([1, 2, 3])
            >>> population.check_array_compatible_with_entity(array)

            >>> array = numpy.array([1, 2, 3, 4])
            >>> population.check_array_compatible_with_entity(array)
            Traceback (most recent call last):
            InvalidArraySizeError: Input [1 2 3 4] is not a valid value for t...

        """
        if self.count == array.size:
            return
        raise InvalidArraySizeError(array, self.entity.key, self.count)

    @staticmethod
    def check_period_validity(
        variable_name: t.VariableName,
        period: None | t.PeriodLike = None,
    ) -> None:
        """Check if a period is valid.

        Args:
            variable_name: The name of the variable.
            period: The period to check.

        Raises:
            PeriodValidityError: If the period is not valid.

        Examples:
            >>> from openfisca_core import entities, periods, populations

            >>> class Person(entities.SingleEntity): ...

            >>> person = Person("person", "people", "", "")
            >>> period = periods.Period("2017-04")
            >>> population = populations.CorePopulation(person)

            >>> population.check_period_validity("salary")
            Traceback (most recent call last):
            PeriodValidityError: You requested computation of variable "sala...

            >>> population.check_period_validity("salary", 2017)

            >>> population.check_period_validity("salary", "2017-04")

            >>> population.check_period_validity("salary", period)

        """
        if isinstance(period, (int, str, periods.Period)):
            return
        stack = traceback.extract_stack()
        filename, line_number, _, line_of_code = stack[-3]
        raise PeriodValidityError(variable_name, filename, line_number, line_of_code)

    # Helpers

    def get_holder(self, variable_name: t.VariableName) -> t.Holder:
        """Return the holder of a variable.

        Args:
            variable_name: The name of the variable.

        Returns:
            Holder: The holder of the variable.

        Examples:
            >>> from openfisca_core import (
            ...     entities,
            ...     holders,
            ...     periods,
            ...     populations,
            ...     simulations,
            ...     taxbenefitsystems,
            ...     simulations,
            ...     variables,
            ... )

            >>> class Person(entities.SingleEntity): ...

            >>> person = Person("person", "people", "", "")

            >>> class Salary(variables.Variable):
            ...     definition_period = periods.WEEK
            ...     entity = person
            ...     value_type = int

            >>> tbs = taxbenefitsystems.TaxBenefitSystem([person])
            >>> person.set_tax_benefit_system(tbs)
            >>> population = populations.SinglePopulation(person)
            >>> simulation = simulations.Simulation(tbs, {person.key: population})
            >>> population.get_holder("income_tax")
            Traceback (most recent call last):
            VariableNotFoundError: You tried to calculate or to set a value ...

            >>> tbs.add_variable(Salary)
            <openfisca_core.populations._core_population.Salary object at...

            >>> salary = Salary()
            >>> population.get_holder(salary.name)
            <openfisca_core.holders.holder.Holder object at ...

        """
        self.entity.check_variable_defined_for_entity(variable_name)
        holder = self._holders.get(variable_name)
        if holder:
            return holder
        variable = self.entity.get_variable(variable_name)
        self._holders[variable_name] = holder = holders.Holder(variable, self)
        return holder

    def get_memory_usage(
        self,
        variables: None | Sequence[t.VariableName] = None,
    ) -> t.MemoryUsageByVariable:
        """Return the memory usage of the population per variable.

        Args:
            variables: The variables to get the memory usage for.

        Returns:
            MemoryUsageByVariable: The memory usage of the population per variable.

        Examples:
            >>> from openfisca_core import (
            ...     entities,
            ...     holders,
            ...     periods,
            ...     populations,
            ...     simulations,
            ...     taxbenefitsystems,
            ...     simulations,
            ...     variables,
            ... )

            >>> class Person(entities.SingleEntity): ...

            >>> person = Person("person", "people", "", "")

            >>> class Salary(variables.Variable):
            ...     definition_period = periods.WEEK
            ...     entity = person
            ...     value_type = int

            >>> tbs = taxbenefitsystems.TaxBenefitSystem([person])
            >>> population = populations.SinglePopulation(person)
            >>> simulation = simulations.Simulation(tbs, {person.key: population})
            >>> salary = Salary()
            >>> holder = holders.Holder(salary, population)
            >>> population._holders[salary.name] = holder

            >>> population.get_memory_usage()
            {'total_nb_bytes': 0, 'by_variable': {'Salary': {'nb_cells_by...}}}

            >>> population.get_memory_usage([salary.name])
            {'total_nb_bytes': 0, 'by_variable': {'Salary': {'nb_cells_by...}}}

        """
        holders_memory_usage = {
            variable_name: holder.get_memory_usage()
            for variable_name, holder in self._holders.items()
            if variables is None or variable_name in variables
        }

        total_memory_usage = sum(
            holder_memory_usage["total_nb_bytes"]
            for holder_memory_usage in holders_memory_usage.values()
        )

        return t.MemoryUsageByVariable(
            total_nb_bytes=total_memory_usage,
            by_variable=holders_memory_usage,
        )

    def activate_dynamic_mode(
        self,
        period: t.PeriodLike,
        permanent_ids=None,
    ) -> None:
        """Enable period identity snapshots from an initial population state."""
        if permanent_ids is None:
            permanent_ids = numpy.arange(self.count, dtype=numpy.intp)
        permanent_ids = self._validate_permanent_ids(permanent_ids, self.count)
        self._dynamic = True
        self._permanent_ids = permanent_ids.copy()
        self._set_period_identity_snapshot(period, permanent_ids)

    def register_births(self, period: t.PeriodLike, permanent_ids) -> None:
        """Register explicit new permanent IDs as alive from ``period``.

        This does not assign the new persons to groups. Call each group
        population's :meth:`set_members_for_period` explicitly afterwards.
        """
        self._check_person_lifecycle()
        new_ids = self._validate_lifecycle_ids(permanent_ids, "birth")
        known_ids = (
            numpy.array([], dtype=numpy.intp)
            if self._permanent_ids is None
            else self._permanent_ids
        )
        already_known = numpy.intersect1d(new_ids, known_ids)
        if already_known.size:
            raise ValueError(
                f"Cannot register already known permanent IDs: {already_known.tolist()}",
            )

        snapshot_period = periods.period(period)
        alive_ids = self._get_alive_ids_for_period(snapshot_period)
        self._permanent_ids = numpy.sort(numpy.concatenate((known_ids, new_ids)))
        self._set_period_identity_snapshot(
            snapshot_period,
            numpy.concatenate((alive_ids, new_ids)),
        )
        self._invalidate_lifecycle_caches(snapshot_period)

    def register_deaths(self, period: t.PeriodLike, permanent_ids) -> None:
        """Register explicit permanent IDs as absent from ``period`` onward."""
        self._check_person_lifecycle()
        dead_ids = self._validate_lifecycle_ids(permanent_ids, "death")
        known_ids = (
            numpy.array([], dtype=numpy.intp)
            if self._permanent_ids is None
            else self._permanent_ids
        )
        unknown_ids = numpy.setdiff1d(dead_ids, known_ids)
        if unknown_ids.size:
            raise ValueError(f"Unknown permanent IDs: {unknown_ids.tolist()}")

        snapshot_period = periods.period(period)
        alive_ids = self._get_alive_ids_for_period(snapshot_period)
        not_alive = numpy.setdiff1d(dead_ids, alive_ids)
        if not_alive.size:
            raise ValueError(
                f"Permanent IDs are not alive at {snapshot_period}: "
                f"{not_alive.tolist()}",
            )
        self._set_period_identity_snapshot(
            snapshot_period,
            alive_ids[~numpy.isin(alive_ids, dead_ids)],
        )
        self._invalidate_lifecycle_caches(snapshot_period)

    def _check_person_lifecycle(self) -> None:
        if not self._dynamic:
            raise ValueError("activate_dynamic_mode must be called first")
        if not self.entity.is_person:
            raise ValueError("Births and deaths are only supported for persons")

    @classmethod
    def _validate_lifecycle_ids(cls, permanent_ids, event):
        if numpy.asarray(permanent_ids).size == 0:
            raise ValueError(f"{event} permanent_ids cannot be empty")
        ids = cls._validate_permanent_ids(permanent_ids)
        return ids

    @staticmethod
    def _validate_permanent_ids(permanent_ids, expected_count=None):
        ids = numpy.asarray(permanent_ids)
        if ids.ndim != 1:
            raise ValueError("permanent_ids must be a one-dimensional array")
        if expected_count is not None and len(ids) != expected_count:
            raise ValueError(
                "permanent_ids must contain one entry per population member "
                f"(expected {expected_count}, got {len(ids)})",
            )
        if not numpy.issubdtype(ids.dtype, numpy.integer):
            raise ValueError("permanent_ids must contain integer IDs")
        if numpy.any(ids < 0):
            raise ValueError("permanent_ids must contain non-negative IDs")
        if len(numpy.unique(ids)) != len(ids):
            raise ValueError("permanent_ids must not contain duplicates")
        return ids.astype(numpy.intp, copy=False)

    @staticmethod
    def _build_id_to_rownum(permanent_ids):
        if len(permanent_ids) == 0:
            return numpy.array([], dtype=numpy.intp)
        result = numpy.full(int(numpy.max(permanent_ids)) + 1, -1, dtype=numpy.intp)
        result[permanent_ids] = numpy.arange(len(permanent_ids), dtype=numpy.intp)
        result.flags.writeable = False
        return result

    def _set_period_identity_snapshot(self, period, permanent_ids) -> None:
        snapshot_period = periods.period(period)
        id_to_rownum = self._build_id_to_rownum(permanent_ids)
        self._period_index[snapshot_period] = {
            "count": len(permanent_ids),
            "id_to_rownum": id_to_rownum,
        }
        latest_period = max(self._period_index, key=lambda item: item.start)
        latest = self._period_index[latest_period]
        self._id_to_rownum = latest["id_to_rownum"]
        self.count = latest["count"]

    def snapshot_period(self, period: t.PeriodLike) -> None:
        """Snapshot the current row ordering for ``period``."""
        if self._id_to_rownum is None:
            permanent_ids = numpy.arange(self.count, dtype=numpy.intp)
        else:
            permanent_ids = self._ids_by_row(self._id_to_rownum)
        self._set_period_identity_snapshot(period, permanent_ids)

    @staticmethod
    def _ids_by_row(id_to_rownum):
        alive_ids = numpy.flatnonzero(id_to_rownum >= 0)
        return alive_ids[numpy.argsort(id_to_rownum[alive_ids])].astype(numpy.intp)

    def _get_alive_ids_for_period(self, period):
        snapshot = self._get_period_identity_snapshot(period)
        if snapshot is None:
            return numpy.arange(self.count, dtype=numpy.intp)
        return self._ids_by_row(snapshot["id_to_rownum"])

    def _get_period_identity_snapshot(self, period):
        if not self._period_index:
            return None
        requested = periods.period(period)
        candidates = (
            (snapshot_period, snapshot)
            for snapshot_period, snapshot in self._period_index.items()
            if snapshot_period.start <= requested.start
        )
        latest = max(candidates, key=lambda item: item[0].start, default=None)
        return None if latest is None else latest[1]

    def get_period_id_to_rownum(self, period):
        """Return the latest identity mapping applicable to ``period``."""
        snapshot = self._get_period_identity_snapshot(period)
        return None if snapshot is None else snapshot["id_to_rownum"]

    def get_count_for_period(self, period) -> int:
        """Return the latest population count applicable to ``period``."""
        snapshot = self._get_period_identity_snapshot(period)
        return self.count if snapshot is None else snapshot["count"]

    def remap_array(self, array, from_period, to_period, default=0):
        """Remap an array between period row spaces using permanent IDs."""
        from_snapshot = self._get_period_identity_snapshot(from_period)
        to_snapshot = self._get_period_identity_snapshot(to_period)
        if from_snapshot is None or to_snapshot is None:
            return array
        if len(array) != from_snapshot["count"]:
            raise ValueError(
                f"Array has {len(array)} rows, expected {from_snapshot['count']} "
                f"for period {periods.period(from_period)}",
            )

        result = numpy.full(to_snapshot["count"], default, dtype=array.dtype)
        from_mapping = from_snapshot["id_to_rownum"]
        to_mapping = to_snapshot["id_to_rownum"]
        shared_size = min(len(from_mapping), len(to_mapping))
        if shared_size:
            shared = (from_mapping[:shared_size] >= 0) & (to_mapping[:shared_size] >= 0)
            permanent_ids = numpy.flatnonzero(shared)
            result[to_mapping[permanent_ids]] = array[from_mapping[permanent_ids]]
        if isinstance(array, indexed_enums.EnumArray):
            return indexed_enums.EnumArray(result, array.possible_values)
        return result

    def _invalidate_period_caches(self, snapshot_period, next_start=None) -> None:
        if next_start is None:
            next_starts = [
                item.start
                for item in self._period_index
                if item.start > snapshot_period.start
            ]
            next_start = min(next_starts, default=None)
        for holder in self._holders.values():
            for known_period in list(holder.get_known_periods()):
                if known_period.start < snapshot_period.start:
                    continue
                if next_start is None or known_period.start < next_start:
                    holder.delete_arrays(known_period)

    def _invalidate_lifecycle_caches(self, snapshot_period) -> None:
        populations = (
            self.simulation.populations.values()
            if self.simulation is not None
            else (self,)
        )
        for population in populations:
            population._invalidate_period_caches(snapshot_period)

    def _remap_to_outer_period(self, result, variable_name, result_period):
        if not self._dynamic or result is None or self.simulation is None:
            return result
        calculation_stack = self.simulation._calculation_stack
        if not calculation_stack:
            return result
        outer_period = calculation_stack[-1]
        variable = self.entity.get_variable(variable_name)
        return self.remap_array(
            result,
            result_period,
            outer_period,
            variable.default_value,
        )


__all__ = ["CorePopulation"]
