"""Many-to-one entity links."""

from __future__ import annotations

import numpy

from openfisca_core import errors, indexed_enums

from .link import Link, LinkResolutionError


class Many2OneLink(Link):
    """Navigate from each source member to at most one target entity."""

    def get(self, variable_name: str, period) -> numpy.ndarray:
        """Return a target variable projected to source rows."""
        if not self.is_resolved:
            message = f"Link '{self.name}' is not bound to a simulation"
            raise LinkResolutionError(message)

        simulation = self._source_population.simulation
        try:
            target_ids = self._source_population(self.link_field, period)
        except (errors.CycleError, errors.SpiralError):
            raise
        except Exception as error:
            message = (
                f"Link '{self.name}' could not read link field "
                f"'{self.link_field}': {error}"
            )
            raise LinkResolutionError(message) from error

        try:
            target_values = self._target_population(variable_name, period)
        except (errors.CycleError, errors.SpiralError):
            raise
        except Exception as error:
            message = (
                f"Link '{self.name}' could not calculate target variable "
                f"'{variable_name}': {error}"
            )
            raise LinkResolutionError(message) from error

        target_rows = self._resolve_ids(target_ids, self._target_population)
        variable = simulation.tax_benefit_system.get_variable(variable_name)
        default = variable.default_value if variable is not None else 0
        if isinstance(default, indexed_enums.Enum):
            default = default.index

        result = numpy.full(
            self._source_population.count,
            default,
            dtype=target_values.dtype,
        )
        valid = target_rows >= 0
        result[valid] = target_values[target_rows[valid]]
        if isinstance(target_values, indexed_enums.EnumArray):
            return indexed_enums.EnumArray(result, target_values.possible_values)
        return result

    def __call__(self, variable_name: str, period) -> numpy.ndarray:
        return self.get(variable_name, period)


__all__ = ["Many2OneLink"]
