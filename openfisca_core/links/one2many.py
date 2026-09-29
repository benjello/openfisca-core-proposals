"""One-to-many entity links."""

from __future__ import annotations

import numpy

from openfisca_core import errors

from .link import Link, LinkResolutionError


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

    def _valid_values(self, variable_name: str, period):
        values = self._target_values(variable_name, period)
        source_rows = self._source_rows(period)
        valid = source_rows >= 0
        return source_rows[valid], values[valid]

    def sum(self, variable_name: str, period) -> numpy.ndarray:
        """Sum target values for each source entity."""
        source_rows, values = self._valid_values(variable_name, period)
        result = numpy.zeros(self._source_population.count, dtype=values.dtype)
        numpy.add.at(result, source_rows, values)
        return result

    def count(self, period=None) -> numpy.ndarray:
        """Count target members for each source entity."""
        source_rows = self._source_rows(period)
        valid = source_rows >= 0
        return numpy.bincount(
            source_rows[valid],
            minlength=self._source_population.count,
        )

    def any(self, variable_name: str, period) -> numpy.ndarray:
        """Return whether any target value is truthy for each source."""
        source_rows, values = self._valid_values(variable_name, period)
        result = numpy.zeros(self._source_population.count, dtype=bool)
        numpy.logical_or.at(result, source_rows, values)
        return result

    def all(self, variable_name: str, period) -> numpy.ndarray:
        """Return whether all target values are truthy for each source."""
        source_rows, values = self._valid_values(variable_name, period)
        result = numpy.ones(self._source_population.count, dtype=bool)
        numpy.logical_and.at(result, source_rows, values)
        return result

    def min(self, variable_name: str, period) -> numpy.ndarray:
        """Return the minimum target value, or zero for an empty source."""
        return self._extreme(variable_name, period, numpy.minimum, numpy.inf)

    def max(self, variable_name: str, period) -> numpy.ndarray:
        """Return the maximum target value, or zero for an empty source."""
        return self._extreme(variable_name, period, numpy.maximum, -numpy.inf)

    def _extreme(self, variable_name, period, operation, initial):
        source_rows, values = self._valid_values(variable_name, period)
        dtype = numpy.result_type(values.dtype, type(initial))
        result = numpy.full(self._source_population.count, initial, dtype=dtype)
        operation.at(result, source_rows, values)
        result[result == initial] = 0
        return result

    def avg(self, variable_name: str, period) -> numpy.ndarray:
        """Return the average target value, or zero for an empty source."""
        total = self.sum(variable_name, period)
        count = self.count(period)
        return numpy.divide(
            total,
            count,
            out=numpy.zeros(self._source_population.count, dtype=float),
            where=count != 0,
        )


__all__ = ["One2ManyLink"]
