"""Base class for links between entity populations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy


def _role_matches(role_array, role_value) -> numpy.ndarray:
    """Compare role objects and raw role keys uniformly."""
    expected = getattr(role_value, "key", role_value)
    return numpy.fromiter(
        (getattr(role, "key", role) == expected for role in role_array),
        dtype=bool,
        count=len(role_array),
    )


class LinkResolutionError(RuntimeError):
    """Raised when a link cannot be resolved in a simulation."""


class Link:
    """A named relationship between a source and a target entity."""

    def __init__(
        self,
        name: str,
        link_field: str,
        target_entity_key: str,
        *,
        role_field: str | None = None,
        position_field: str | None = None,
    ) -> None:
        self.name = name
        self.link_field = link_field
        self.target_entity_key = target_entity_key
        self.role_field = role_field
        self.position_field = position_field
        self._source_population = None
        self._target_population = None

    def attach(self, source_population) -> None:
        """Bind the link to its source population."""
        self._source_population = source_population

    def resolve(self, populations: Mapping[str, object]) -> None:
        """Resolve the target population from a simulation's populations."""
        try:
            self._target_population = populations[self.target_entity_key]
        except KeyError as error:
            available = ", ".join(sorted(populations))
            message = (
                f"Link '{self.name}' targets unknown entity "
                f"'{self.target_entity_key}' (available: {available})"
            )
            raise LinkResolutionError(message) from error

    @property
    def is_resolved(self) -> bool:
        """Whether both ends of this link are bound."""
        return (
            self._source_population is not None and self._target_population is not None
        )

    @staticmethod
    def _resolve_ids(ids, population) -> numpy.ndarray:
        """Map entity identifiers to valid row numbers, using -1 for missing IDs."""
        ids = numpy.asarray(ids)
        rows = numpy.full(ids.shape, -1, dtype=numpy.intp)
        id_to_rownum = getattr(population, "_id_to_rownum", None)

        if isinstance(id_to_rownum, Mapping):
            for index, identifier in numpy.ndenumerate(ids):
                row = id_to_rownum.get(identifier.item(), -1)
                if 0 <= row < population.count:
                    rows[index] = row
            return rows

        if id_to_rownum is not None:
            mapping = numpy.asarray(id_to_rownum)
            if numpy.issubdtype(ids.dtype, numpy.integer):
                valid_ids = (ids >= 0) & (ids < mapping.size)
                mapped = mapping[ids[valid_ids]]
                valid_rows = (mapped >= 0) & (mapped < population.count)
                rows[valid_ids] = numpy.where(valid_rows, mapped, -1)
            return rows

        population_ids = numpy.asarray(population.ids)
        if population_ids.size == population.count:
            row_by_id = {
                identifier.item(): row for row, identifier in enumerate(population_ids)
            }
            for index, identifier in numpy.ndenumerate(ids):
                rows[index] = row_by_id.get(identifier.item(), -1)
            if numpy.issubdtype(ids.dtype, numpy.integer):
                direct = (rows < 0) & (ids >= 0) & (ids < population.count)
                rows[direct] = ids[direct]
            return rows

        if numpy.issubdtype(ids.dtype, numpy.integer):
            valid = (ids >= 0) & (ids < population.count)
            rows[valid] = ids[valid]
        return rows

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"link_field={self.link_field!r}, target={self.target_entity_key!r})"
        )


__all__ = ["Link", "LinkResolutionError", "_role_matches"]
