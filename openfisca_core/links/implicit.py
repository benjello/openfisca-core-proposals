"""Links backed by OpenFisca group membership arrays."""

from __future__ import annotations

import numpy

from openfisca_core import projectors

from .link import _role_matches
from .many2one import Many2OneLink, _ChainedLink
from .one2many import One2ManyLink


class _ProjectedCallable:
    """Callable proxy that preserves attributes and projects group results."""

    def __init__(self, function, link) -> None:
        self._function = function
        self._link = link

    def __call__(self, *args, **kwargs):
        result = self._function(*args, **kwargs)
        return self._link._project_entity_result(result)

    def __getattr__(self, name: str):
        attribute = getattr(self._function, name)
        if callable(attribute):
            return _ProjectedCallable(attribute, self._link)
        return attribute


class ImplicitMany2OneLink(Many2OneLink):
    """A person-to-group link backed by ``members_entity_id``."""

    def __init__(self, group_entity_key: str) -> None:
        super().__init__(group_entity_key, "", group_entity_key)

    def _get_target_ids(self, period) -> numpy.ndarray:
        return self._target_population.members_entity_id

    @staticmethod
    def _resolve_ids(ids, population) -> numpy.ndarray:
        """Membership values are row numbers, regardless of public group IDs."""
        rows = numpy.asarray(ids, dtype=numpy.intp)
        valid = (rows >= 0) & (rows < population.count)
        return numpy.where(valid, rows, -1)

    @property
    def role(self) -> numpy.ndarray:
        return self._target_population.members_role

    def _project_entity_result(self, result):
        """Project a result known to have one value per target entity."""
        result = numpy.asarray(result)
        if result.size != self._target_population.count:
            message = (
                f"Link '{self.name}' expected {self._target_population.count} "
                f"entity values, got {result.size}"
            )
            raise ValueError(message)
        return self._target_population.project(result)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        target_link = self._target_population.links.get(name)
        if isinstance(target_link, Many2OneLink):
            return _ChainedLink((self, target_link))
        target_attribute = getattr(self._target_population, name)
        function = getattr(target_attribute, "__func__", target_attribute)
        if isinstance(target_attribute, projectors.Projector) or getattr(
            function,
            "projectable",
            False,
        ):
            return _ProjectedCallable(target_attribute, self)
        return target_attribute


class ImplicitOne2ManyLink(One2ManyLink):
    """A group-to-person link backed by ``members_entity_id``."""

    def __init__(
        self,
        name: str,
        group_entity_key: str,
        person_entity_key: str,
    ) -> None:
        super().__init__(name, "", person_entity_key)
        self.group_entity_key = group_entity_key

    def _source_rows(self, period) -> numpy.ndarray:
        return self._source_population.members_entity_id

    def _filter(self, source_rows, role, condition, period) -> numpy.ndarray:
        valid = source_rows >= 0
        if role is not None:
            valid &= _role_matches(self._source_population.members_role, role)
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


__all__ = ["ImplicitMany2OneLink", "ImplicitOne2ManyLink"]
