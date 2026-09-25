"""Many-to-one entity links."""

from __future__ import annotations

from collections.abc import Sequence

import numpy

from openfisca_core import errors, indexed_enums, periods
from openfisca_core.populations import types as population_types
from openfisca_core.populations._errors import (
    IncompatibleOptionsError,
    InvalidOptionError,
)

from .link import Link, LinkResolutionError, _role_matches


class Many2OneLink(Link):
    """Navigate from each source member to at most one target entity."""

    def get(self, variable_name: str, period, options=None) -> numpy.ndarray:
        """Return a target variable projected to source rows."""
        if not self.is_resolved:
            message = f"Link '{self.name}' is not bound to a simulation"
            raise LinkResolutionError(message)

        try:
            target_ids = self._get_target_ids(period)
        except (errors.CycleError, errors.SpiralError):
            raise
        except Exception as error:
            message = (
                f"Link '{self.name}' could not read link field "
                f"'{self.link_field}': {error}"
            )
            raise LinkResolutionError(message) from error

        if isinstance(options, Sequence):
            if (
                population_types.Option.ADD in options
                and population_types.Option.DIVIDE in options
            ):
                raise IncompatibleOptionsError(variable_name)
            if options and not any(
                option in options for option in population_types.Option
            ):
                raise InvalidOptionError(options[0], variable_name)

        try:
            target_values = self._target_population(
                variable_name,
                period,
                options=options,
            )
        except (errors.CycleError, errors.SpiralError):
            raise
        except Exception as error:
            message = (
                f"Link '{self.name}' could not calculate target variable "
                f"'{variable_name}': {error}"
            )
            raise LinkResolutionError(message) from error

        return self._project_values(
            target_values,
            target_ids,
            self._default_value(variable_name),
        )

    def _project_values(self, target_values, target_ids, default) -> numpy.ndarray:
        """Project target values through this link's ID resolution."""
        target_rows = self._resolve_ids(target_ids, self._target_population)
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

    def _default_value(self, variable_name: str):
        variable = self._source_population.simulation.tax_benefit_system.get_variable(
            variable_name
        )
        default = variable.default_value if variable is not None else 0
        if isinstance(default, indexed_enums.Enum):
            return default.index
        return default

    def _get_target_ids(self, period) -> numpy.ndarray:
        """Read target IDs from the source population's link field."""
        return self._source_population(self.link_field, period)

    def __call__(self, variable_name: str, period, *, options=None) -> numpy.ndarray:
        return self.get(variable_name, period, options=options)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if self._target_population is None:
            raise AttributeError(f"Link '{self.name}' is not bound to a simulation")
        target_link = self._target_population.links.get(name)
        if isinstance(target_link, Many2OneLink):
            return _ChainedLink((self, target_link))
        target_key = self._target_population.entity.key
        raise AttributeError(f"Entity '{target_key}' has no many-to-one link '{name}'")

    @property
    def role(self) -> numpy.ndarray | None:
        """Return source roles for this link, when configured."""
        if self.role_field is None:
            return None
        return self._get_roles(periods.ETERNITY)

    def _get_roles(self, period) -> numpy.ndarray:
        try:
            return self._source_population(self.role_field, period)
        except (errors.CycleError, errors.SpiralError):
            raise
        except Exception as error:
            message = (
                f"Link '{self.name}' could not read role field "
                f"'{self.role_field}': {error}"
            )
            raise LinkResolutionError(message) from error

    def has_role(self, role_value, period=periods.ETERNITY) -> numpy.ndarray:
        """Return which source rows have the requested role."""
        if self.role_field is None:
            message = f"Link '{self.name}' has no role_field"
            raise ValueError(message)
        return _role_matches(self._get_roles(period), role_value)

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
        filtered[~self.has_role(role_value, period)] = default
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
        criteria = self._source_population(variable_name, period)
        return self._source_population.get_rank(target, criteria, condition=condition)


class _ChainedLink:
    """A sequence of many-to-one links resolved from left to right."""

    def __init__(self, links: tuple[Many2OneLink, ...]) -> None:
        self._links = links

    def get(self, variable_name: str, period, options=None) -> numpy.ndarray:
        """Resolve the final variable and project it through every prior link."""
        result = self._links[-1].get(variable_name, period, options=options)
        default = self._links[-1]._default_value(variable_name)
        for link in reversed(self._links[:-1]):
            target_ids = link._get_target_ids(period)
            result = link._project_values(result, target_ids, default)
        return result

    def __call__(self, variable_name: str, period, *, options=None) -> numpy.ndarray:
        return self.get(variable_name, period, options=options)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        target_population = self._links[-1]._target_population
        target_link = target_population.links.get(name)
        if isinstance(target_link, Many2OneLink):
            return _ChainedLink((*self._links, target_link))
        target_key = target_population.entity.key
        raise AttributeError(f"Entity '{target_key}' has no many-to-one link '{name}'")


__all__ = ["Many2OneLink"]
