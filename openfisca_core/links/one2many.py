"""One-to-many entity links."""

from __future__ import annotations

import numpy

from openfisca_core import errors

from .link import Link, LinkResolutionError, _role_matches


class One2ManyLink(Link):
    """Aggregate target members that point to each source entity."""

    def _target_values(self, variable_name: str, period) -> numpy.ndarray:
        try:
            return self._target_population(variable_name, period)
        except (errors.CycleError, errors.SpiralError):
            raise
        except Exception as error:
            message = (
                f"Link '{self.name}' could not calculate target variable "
                f"'{variable_name}': {error}"
            )
            raise LinkResolutionError(message) from error

    def _source_rows(self, period) -> numpy.ndarray:
        try:
            source_ids = self._target_population(self.link_field, period)
        except (errors.CycleError, errors.SpiralError):
            raise
        except Exception as error:
            message = (
                f"Link '{self.name}' could not read link field "
                f"'{self.link_field}': {error}"
            )
            raise LinkResolutionError(message) from error
        return self._resolve_ids(source_ids, self._source_population)

    def _valid_values(self, variable_name: str, period, role=None, condition=None):
        values = self._target_values(variable_name, period)
        source_rows = self._source_rows(period)
        valid = self._filter(source_rows, role, condition, period)
        return source_rows[valid], values[valid]

    def _filter(self, source_rows, role, condition, period) -> numpy.ndarray:
        valid = source_rows >= 0
        if role is not None:
            if self.role_field is None:
                message = f"Link '{self.name}' has no role_field"
                raise ValueError(message)
            try:
                roles = self._target_population(self.role_field, period)
            except (errors.CycleError, errors.SpiralError):
                raise
            except Exception as error:
                message = (
                    f"Link '{self.name}' could not read role field "
                    f"'{self.role_field}': {error}"
                )
                raise LinkResolutionError(message) from error
            valid &= _role_matches(roles, role)
        if condition is not None:
            condition = numpy.asarray(condition, dtype=bool)
            if condition.shape != source_rows.shape:
                message = (
                    f"Link '{self.name}' condition has shape {condition.shape}; "
                    f"expected {source_rows.shape}"
                )
                raise ValueError(message)
            valid &= condition
        return valid

    def sum(
        self, variable_name: str, period, *, role=None, condition=None
    ) -> numpy.ndarray:
        """Sum target values for each source entity."""
        source_rows, values = self._valid_values(variable_name, period, role, condition)
        result = numpy.zeros(self._source_population.count, dtype=values.dtype)
        numpy.add.at(result, source_rows, values)
        return result

    def count(self, period=None, *, role=None, condition=None) -> numpy.ndarray:
        """Count target members for each source entity."""
        source_rows = self._source_rows(period)
        valid = self._filter(source_rows, role, condition, period)
        return numpy.bincount(
            source_rows[valid],
            minlength=self._source_population.count,
        )

    def any(
        self, variable_name: str, period, *, role=None, condition=None
    ) -> numpy.ndarray:
        """Return whether any target value is truthy for each source."""
        source_rows, values = self._valid_values(variable_name, period, role, condition)
        result = numpy.zeros(self._source_population.count, dtype=bool)
        numpy.logical_or.at(result, source_rows, values)
        return result

    def all(
        self, variable_name: str, period, *, role=None, condition=None
    ) -> numpy.ndarray:
        """Return whether all target values are truthy for each source."""
        source_rows, values = self._valid_values(variable_name, period, role, condition)
        result = numpy.ones(self._source_population.count, dtype=bool)
        numpy.logical_and.at(result, source_rows, values)
        return result

    def min(
        self, variable_name: str, period, *, role=None, condition=None
    ) -> numpy.ndarray:
        """Return the minimum target value, or zero for an empty source."""
        return self._extreme(
            variable_name, period, numpy.minimum, numpy.inf, role, condition
        )

    def max(
        self, variable_name: str, period, *, role=None, condition=None
    ) -> numpy.ndarray:
        """Return the maximum target value, or zero for an empty source."""
        return self._extreme(
            variable_name, period, numpy.maximum, -numpy.inf, role, condition
        )

    def _extreme(self, variable_name, period, operation, initial, role, condition):
        source_rows, values = self._valid_values(variable_name, period, role, condition)
        dtype = numpy.result_type(values.dtype, type(initial))
        result = numpy.full(self._source_population.count, initial, dtype=dtype)
        operation.at(result, source_rows, values)
        result[result == initial] = 0
        return result

    def avg(
        self, variable_name: str, period, *, role=None, condition=None
    ) -> numpy.ndarray:
        """Return the average target value, or zero for an empty source."""
        total = self.sum(variable_name, period, role=role, condition=condition)
        count = self.count(period, role=role, condition=condition)
        return numpy.divide(
            total,
            count,
            out=numpy.zeros(self._source_population.count, dtype=float),
            where=count != 0,
        )

    def nth(
        self,
        n: int,
        variable_name: str,
        period,
        *,
        role=None,
        condition=None,
    ) -> numpy.ndarray:
        """Return the n-th matching target value for each source."""
        if n < 0:
            raise ValueError("n must be non-negative")
        source_rows, values = self._valid_values(variable_name, period, role, condition)
        result = numpy.zeros(self._source_population.count, dtype=values.dtype)
        positions = numpy.zeros(self._source_population.count, dtype=numpy.intp)
        for source_row, value in zip(source_rows, values):
            if positions[source_row] == n:
                result[source_row] = value
            positions[source_row] += 1
        return result

    def get_by_role(
        self,
        variable_name: str,
        period,
        role_value,
        *,
        condition=None,
    ) -> numpy.ndarray:
        """Return the last matching target value for each source."""
        source_rows, values = self._valid_values(
            variable_name,
            period,
            role_value,
            condition,
        )
        result = numpy.zeros(self._source_population.count, dtype=values.dtype)
        result[source_rows] = values
        return result


__all__ = ["One2ManyLink"]
