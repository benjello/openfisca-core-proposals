from __future__ import annotations

import typing

import numpy

from openfisca_core import entities, indexed_enums, periods, projectors

from . import types as t
from .population import Population


class GroupPopulation(Population):
    def __init__(self, entity: t.GroupEntity, members: t.Members) -> None:
        super().__init__(entity)
        self.members = members
        self._members_entity_id = None
        self._members_role = None
        self._members_position = None
        self._ordered_members_map = None
        self._members_entity_id_by_period = {}
        self._members_role_by_period = {}
        self._members_position_by_period = {}
        self._ordered_members_map_by_period = {}

    def clone(self, simulation):
        result = GroupPopulation(self.entity, simulation.persons)
        result.simulation = simulation
        result._holders = {
            variable: holder.clone(result)
            for (variable, holder) in self._holders.items()
        }
        result.count = self.count
        result.ids = self.ids
        result._members_entity_id = self._members_entity_id
        result._members_role = self._members_role
        result._members_position = self._members_position
        result._ordered_members_map = self._ordered_members_map
        result._members_entity_id_by_period = dict(
            self._members_entity_id_by_period,
        )
        result._members_role_by_period = dict(self._members_role_by_period)
        return result

    @staticmethod
    def _compute_members_position(members_entity_id):
        nb_entities = numpy.max(members_entity_id) + 1
        nb_persons = len(members_entity_id)
        order = numpy.argsort(members_entity_id, kind="stable")
        sorted_ids = members_entity_id[order]
        group_sizes = numpy.bincount(sorted_ids, minlength=nb_entities)
        group_starts = numpy.empty(nb_entities, dtype=numpy.intp)
        group_starts[0] = 0
        numpy.cumsum(group_sizes[:-1], out=group_starts[1:])
        positions_sorted = numpy.arange(nb_persons) - group_starts[sorted_ids]
        result = numpy.empty(nb_persons, dtype=numpy.int32)
        result[order] = positions_sorted
        return result

    def _get_snapshot_period(self, period):
        if period is None or not self._members_entity_id_by_period:
            return None

        requested = periods.period(period)
        candidates = (
            snapshot_period
            for snapshot_period in self._members_entity_id_by_period
            if snapshot_period.start <= requested.start
        )
        return max(candidates, key=lambda item: item.start, default=None)

    def _get_members_entity_id(self, period=None):
        snapshot_period = self._get_snapshot_period(period)
        if snapshot_period is None:
            return self._members_entity_id
        return self._members_entity_id_by_period[snapshot_period]

    def _get_members_role(self, period=None):
        if period is None or not self._members_role_by_period:
            return self.members_role

        requested = periods.period(period)
        candidates = (
            snapshot_period
            for snapshot_period in self._members_role_by_period
            if snapshot_period.start <= requested.start
        )
        snapshot_period = max(
            candidates,
            key=lambda item: item.start,
            default=None,
        )
        if snapshot_period is None:
            return self.members_role
        return self._members_role_by_period[snapshot_period]

    def _get_members_position(self, period=None):
        snapshot_period = self._get_snapshot_period(period)
        if snapshot_period is None:
            return self.members_position
        if snapshot_period not in self._members_position_by_period:
            self._members_position_by_period[snapshot_period] = (
                self._compute_members_position(
                    self._members_entity_id_by_period[snapshot_period],
                )
            )
        return self._members_position_by_period[snapshot_period]

    def _get_ordered_members_map(self, period=None):
        snapshot_period = self._get_snapshot_period(period)
        if snapshot_period is None:
            return self.ordered_members_map
        if snapshot_period not in self._ordered_members_map_by_period:
            self._ordered_members_map_by_period[snapshot_period] = numpy.argsort(
                self._members_entity_id_by_period[snapshot_period],
                kind="stable",
            )
        return self._ordered_members_map_by_period[snapshot_period]

    def _validate_members_role(self, members_role, members_entity_id):
        role_array = numpy.asarray(list(members_role), dtype=object)
        if role_array.ndim != 1:
            raise ValueError("members_role must be a one-dimensional array")
        if len(role_array) != len(members_entity_id):
            raise ValueError(
                "members_role must contain one entry per member "
                f"(expected {len(members_entity_id)}, got {len(role_array)})",
            )

        valid_roles = self.entity.flattened_roles
        invalid_roles = [role for role in role_array if role not in valid_roles]
        if invalid_roles:
            raise ValueError(
                "members_role must contain roles defined by the group entity",
            )

        for role in valid_roles:
            if role.max is None:
                continue
            counts = numpy.bincount(
                members_entity_id[role_array == role],
                minlength=self.count,
            )
            if numpy.any(counts > role.max):
                raise ValueError(
                    f"Role {role.key!r} allows at most {role.max} member(s) per group",
                )
        return role_array

    def _invalidate_period_caches(self, snapshot_period, next_start=None) -> None:
        for holder in self._holders.values():
            for known_period in list(holder.get_known_periods()):
                if known_period.start < snapshot_period.start:
                    continue
                if next_start is None or known_period.start < next_start:
                    holder.delete_arrays(known_period)

    def set_members_for_period(
        self,
        period,
        members_entity_id,
        members_role=None,
    ) -> None:
        """Set membership from ``period`` until the next explicit snapshot.

        This first dynamic-membership API keeps both person and group counts
        constant. Group creation and dissolution are handled by a later API.
        """
        snapshot_period = periods.period(period)
        array = numpy.asarray(members_entity_id)
        if array.ndim != 1:
            raise ValueError("members_entity_id must be a one-dimensional array")
        if array.size == 0:
            raise ValueError("members_entity_id cannot be empty")
        if len(array) != self.members.count:
            raise ValueError(
                "members_entity_id must contain one entry per member "
                f"(expected {self.members.count}, got {len(array)})",
            )
        if not numpy.issubdtype(array.dtype, numpy.integer):
            raise ValueError("members_entity_id must contain integer IDs")
        if numpy.any(array < 0):
            raise ValueError("members_entity_id must contain non-negative IDs")
        if numpy.any(array >= self.count):
            raise ValueError(
                "members_entity_id must reference an existing group "
                f"(valid IDs are 0 to {self.count - 1})",
            )
        if int(numpy.max(array)) + 1 != self.count:
            raise ValueError(
                "set_members_for_period cannot change the number of groups",
            )

        role_snapshot = (
            None
            if members_role is None
            else self._validate_members_role(members_role, array)
        )
        snapshot = array.astype(numpy.intp, copy=True)
        snapshot.flags.writeable = False
        self._members_entity_id_by_period[snapshot_period] = snapshot
        if role_snapshot is not None:
            role_snapshot.flags.writeable = False
            self._members_role_by_period[snapshot_period] = role_snapshot
        self._members_position_by_period.pop(snapshot_period, None)
        self._ordered_members_map_by_period.pop(snapshot_period, None)

        next_starts = [
            item.start
            for item in self._members_entity_id_by_period
            if item.start > snapshot_period.start
        ]
        next_start = min(next_starts, default=None)
        self._invalidate_period_caches(snapshot_period, next_start)

    def set_roles_for_period(self, period, members_role) -> None:
        """Set member roles from ``period`` until the next role snapshot."""
        snapshot_period = periods.period(period)
        membership = self._get_members_entity_id(snapshot_period)
        role_snapshot = self._validate_members_role(members_role, membership)
        role_snapshot.flags.writeable = False
        self._members_role_by_period[snapshot_period] = role_snapshot

        next_starts = [
            item.start
            for item in self._members_role_by_period
            if item.start > snapshot_period.start
        ]
        self._invalidate_period_caches(
            snapshot_period,
            min(next_starts, default=None),
        )

    def _members_have_role(self, role, period=None):
        members_role = self._get_members_role(period)
        if role.subroles:
            return numpy.logical_or.reduce(
                [members_role == subrole for subrole in role.subroles],
            )
        return members_role == role

    @property
    def members_position(self):
        if self._members_position is None and self.members_entity_id is not None:
            self._members_position = self._compute_members_position(
                self.members_entity_id,
            )

        return self._members_position

    @members_position.setter
    def members_position(self, members_position) -> None:
        self._members_position = members_position

    @property
    def members_entity_id(self):
        return self._members_entity_id

    @members_entity_id.setter
    def members_entity_id(self, members_entity_id) -> None:
        self._members_entity_id = members_entity_id

    def set_members_entity_id(self, members_entity_id) -> None:
        """Set and validate the group entity ID of each member."""
        array = numpy.asarray(members_entity_id)
        if array.ndim != 1:
            raise ValueError("members_entity_id must be a one-dimensional array")
        if array.size == 0:
            raise ValueError("members_entity_id cannot be empty")
        if len(array) != self.members.count:
            raise ValueError(
                "members_entity_id must contain one entry per member "
                f"(expected {self.members.count}, got {len(array)})",
            )
        if not numpy.issubdtype(array.dtype, numpy.integer):
            raise ValueError("members_entity_id must contain integer IDs")
        if numpy.any(array < 0):
            raise ValueError("members_entity_id must contain non-negative IDs")

        self._members_entity_id = array.astype(numpy.intp, copy=False)
        self.count = int(numpy.max(array)) + 1
        self._members_position = None
        self._ordered_members_map = None
        self._members_role = None

    @property
    def members_role(self):
        if self._members_role is None:
            default_role = self.entity.flattened_roles[0]
            self._members_role = numpy.repeat(default_role, len(self.members_entity_id))
        return self._members_role

    @members_role.setter
    def members_role(self, members_role: typing.Iterable[entities.Role]) -> None:
        if members_role is not None:
            self._members_role = numpy.array(list(members_role))

    @property
    def ordered_members_map(self):
        """Mask to group the persons by entity
        This function only caches the map value, to see what the map is used for, see value_nth_person method.
        """
        if self._ordered_members_map is None:
            self._ordered_members_map = numpy.argsort(self.members_entity_id)
        return self._ordered_members_map

    # Helpers

    def get_role(self, role_name):
        return next(
            (role for role in self.entity.flattened_roles if role.key == role_name),
            None,
        )

    #  Aggregation persons -> entity

    @projectors.projectable
    def sum(self, array, role=None, period=None):
        """Return the sum of ``array`` for the members of the entity.

        ``array`` must have the dimension of the number of persons in the simulation

        If ``role`` is provided, only the entity member with the given role are taken into account.

        Example:
        >>> salaries = household.members(
        ...     "salary", "2018-01"
        ... )  # e.g. [2000, 1500, 0, 0, 0]
        >>> household.sum(salaries)
        >>> array([3500])

        """
        self.entity.check_role_validity(role)
        self.members.check_array_compatible_with_entity(array)
        members_entity_id = self._get_members_entity_id(period)
        if role is not None:
            role_filter = self._members_have_role(role, period)
            return numpy.bincount(
                members_entity_id[role_filter],
                weights=array[role_filter],
                minlength=self.count,
            )
        return numpy.bincount(
            members_entity_id,
            weights=array,
            minlength=self.count,
        )

    @projectors.projectable
    def any(self, array, role=None, period=None):
        """Return ``True`` if ``array`` is ``True`` for any members of the entity.

        ``array`` must have the dimension of the number of persons in the simulation

        If ``role`` is provided, only the entity member with the given role are taken into account.

        Example:
        >>> salaries = household.members(
        ...     "salary", "2018-01"
        ... )  # e.g. [2000, 1500, 0, 0, 0]
        >>> household.any(salaries >= 1800)
        >>> array([True])

        """
        sum_in_entity = self.sum(array, role=role, period=period)
        return sum_in_entity > 0

    @projectors.projectable
    def reduce(self, array, reducer, neutral_element, role=None, period=None):
        self.members.check_array_compatible_with_entity(array)
        self.entity.check_role_validity(role)
        position_in_entity = self._get_members_position(period)
        role_filter = (
            self._members_have_role(role, period) if role is not None else True
        )
        filtered_array = numpy.where(role_filter, array, neutral_element)

        result = self.filled_array(
            neutral_element,
        )  # Neutral value that will be returned if no one with the given role exists.

        # We loop over the positions in the entity
        # Looping over the entities is tempting, but potentially slow if there are a lot of entities
        biggest_entity_size = numpy.max(position_in_entity) + 1

        for p in range(biggest_entity_size):
            values = self.value_nth_person(
                p,
                filtered_array,
                default=neutral_element,
                period=period,
            )
            result = reducer(result, values)

        return result

    @projectors.projectable
    def all(self, array, role=None, period=None):
        """Return ``True`` if ``array`` is ``True`` for all members of the entity.

        ``array`` must have the dimension of the number of persons in the simulation

        If ``role`` is provided, only the entity member with the given role are taken into account.

        Example:
        >>> salaries = household.members(
        ...     "salary", "2018-01"
        ... )  # e.g. [2000, 1500, 0, 0, 0]
        >>> household.all(salaries >= 1800)
        >>> array([False])

        """
        return self.reduce(
            array,
            reducer=numpy.logical_and,
            neutral_element=True,
            role=role,
            period=period,
        )

    @projectors.projectable
    def max(self, array, role=None, period=None):
        """Return the maximum value of ``array`` for the entity members.

        ``array`` must have the dimension of the number of persons in the simulation

        If ``role`` is provided, only the entity member with the given role are taken into account.

        Example:
        >>> salaries = household.members(
        ...     "salary", "2018-01"
        ... )  # e.g. [2000, 1500, 0, 0, 0]
        >>> household.max(salaries)
        >>> array([2000])

        """
        return self.reduce(
            array,
            reducer=numpy.maximum,
            neutral_element=-numpy.inf,
            role=role,
            period=period,
        )

    @projectors.projectable
    def min(self, array, role=None, period=None):
        """Return the minimum value of ``array`` for the entity members.

        ``array`` must have the dimension of the number of persons in the simulation

        If ``role`` is provided, only the entity member with the given role are taken into account.

        Example:
        >>> salaries = household.members(
        ...     "salary", "2018-01"
        ... )  # e.g. [2000, 1500, 0, 0, 0]
        >>> household.min(salaries)
        >>> array([0])
        >>> household.min(
        ...     salaries, role=Household.PARENT
        ... )  # Assuming the 1st two persons are parents
        >>> array([1500])

        """
        return self.reduce(
            array,
            reducer=numpy.minimum,
            neutral_element=numpy.inf,
            role=role,
            period=period,
        )

    @projectors.projectable
    def nb_persons(self, role=None, period=None):
        """Returns the number of persons contained in the entity.

        If ``role`` is provided, only the entity member with the given role are taken into account.
        """
        if role:
            role_condition = self._members_have_role(role, period)
            return self.sum(role_condition, period=period)
        return numpy.bincount(
            self._get_members_entity_id(period),
            minlength=self.count,
        )

    # Projection person -> entity

    @projectors.projectable
    def value_from_person(self, array, role, default=0, period=None):
        """Get the value of ``array`` for the person with the unique role ``role``.

        ``array`` must have the dimension of the number of persons in the simulation

        If such a person does not exist, return ``default`` instead

        The result is a vector which dimension is the number of entities
        """
        self.entity.check_role_validity(role)
        if role.max != 1:
            msg = f"You can only use value_from_person with a role that is unique in {self.key}. Role {role.key} is not unique."
            raise Exception(
                msg,
            )
        self.members.check_array_compatible_with_entity(array)
        members_map = self._get_ordered_members_map(period)
        result = self.filled_array(default, dtype=array.dtype)
        if isinstance(array, indexed_enums.EnumArray):
            result = indexed_enums.EnumArray(result, array.possible_values)
        role_filter = self._members_have_role(role, period)
        entity_filter = self.any(role_filter, period=period)

        result[entity_filter] = array[members_map][role_filter[members_map]]

        return result

    @projectors.projectable
    def value_nth_person(self, n, array, default=0, period=None):
        """Get the value of array for the person whose position in the entity is n.

        Note that this position is arbitrary, and that members are not sorted.

        If the nth person does not exist, return  ``default`` instead.

        The result is a vector which dimension is the number of entities.
        """
        self.members.check_array_compatible_with_entity(array)
        positions = self._get_members_position(period)
        nb_persons_per_entity = self.nb_persons(period=period)
        members_map = self._get_ordered_members_map(period)
        result = self.filled_array(default, dtype=array.dtype)
        # For households that have at least n persons, set the result as the value of criteria for the person for which the position is n.
        # The map is needed b/c the order of the nth persons of each household in the persons vector is not necessarily the same than the household order.
        result[nb_persons_per_entity > n] = array[members_map][
            positions[members_map] == n
        ]

        if isinstance(array, indexed_enums.EnumArray):
            result = indexed_enums.EnumArray(result, array.possible_values)

        return result

    @projectors.projectable
    def value_from_first_person(self, array, period=None):
        return self.value_nth_person(0, array, period=period)

    # Projection entity -> person(s)

    def project(self, array, role=None, period=None):
        self.check_array_compatible_with_entity(array)
        self.entity.check_role_validity(role)
        members_entity_id = self._get_members_entity_id(period)
        if role is None:
            return array[members_entity_id]
        role_condition = self._members_have_role(role, period)
        return numpy.where(role_condition, array[members_entity_id], 0)
