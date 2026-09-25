"""Many-to-one entity links."""

from __future__ import annotations

import numpy

from openfisca_core import indexed_enums, periods

from .link import Link, LinkResolutionError, _role_matches


class Many2OneLink(Link):
    """Navigate from each source member to at most one target entity."""

    def get(self, variable_name: str, period) -> numpy.ndarray:
        """Return a target variable projected to source rows."""
        if not self.is_resolved:
            message = f"Link '{self.name}' is not bound to a simulation"
            raise LinkResolutionError(message)

        simulation = self._source_population.simulation
        try:
            target_ids = self._get_target_ids(period)
        except Exception as error:
            message = (
                f"Link '{self.name}' could not read link field "
                f"'{self.link_field}': {error}"
            )
            raise LinkResolutionError(message) from error

        try:
            target_values = simulation.calculate(variable_name, period)
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

    def _get_target_ids(self, period) -> numpy.ndarray:
        """Read target IDs from the source population's link field."""
        return self._source_population.simulation.calculate(self.link_field, period)

    def __call__(self, variable_name: str, period) -> numpy.ndarray:
        return self.get(variable_name, period)

    @property
    def role(self) -> numpy.ndarray | None:
        """Return source roles for this link, when configured."""
        if self.role_field is None:
            return None
        return self._source_population.simulation.calculate(
            self.role_field,
            periods.ETERNITY,
        )

    def has_role(self, role_value) -> numpy.ndarray:
        """Return which source rows have the requested role."""
        roles = self.role
        if roles is None:
            message = f"Link '{self.name}' has no role_field"
            raise ValueError(message)
        return _role_matches(roles, role_value)

    def get_by_role(self, variable_name: str, period, *, role_value) -> numpy.ndarray:
        """Project values only for source rows with the requested role."""
        result = self.get(variable_name, period)
        variable = self._source_population.simulation.tax_benefit_system.get_variable(
            variable_name
        )
        default = variable.default_value if variable is not None else 0
        if isinstance(default, indexed_enums.Enum):
            default = default.index
        filtered = result.copy()
        filtered[~self.has_role(role_value)] = default
        return filtered

    def rank(self, variable_name: str, period, *, condition=True) -> numpy.ndarray:
        """Rank source members within the linked group population."""
        target = self._target_population
        if (
            not hasattr(target, "members_position")
            or target.members is not self._source_population
        ):
            message = (
                f"Link '{self.name}' rank requires its target to group "
                "its source population"
            )
            raise ValueError(message)
        criteria = self._source_population.simulation.calculate(variable_name, period)
        return self._source_population.get_rank(target, criteria, condition=condition)


__all__ = ["Many2OneLink"]
