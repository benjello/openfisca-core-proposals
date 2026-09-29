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
        result.ids = self.ids[:]
        result._members_entity_id = self._members_entity_id
        result._members_role = self._members_role
        result._members_position = self._members_position
        result._ordered_members_map = self._ordered_members_map
        result._members_entity_id_by_period = dict(
            self._members_entity_id_by_period,
        )
        result._members_role_by_period = dict(self._members_role_by_period)
        result._dynamic = self._dynamic
        result._permanent_ids = (
            None if self._permanent_ids is None else self._permanent_ids.copy()
        )
        result._id_to_rownum = self._id_to_rownum
        result._period_index = {
            period: dict(snapshot) for period, snapshot in self._period_index.items()
        }
        return result

    @staticmethod
    def _compute_members_position(members_entity_id):
        if len(members_entity_id) == 0:
            return numpy.array([], dtype=numpy.int32)
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

    def _resolve_period(self, period):
        if period is not None or self.simulation is None:
            return period
        calculation_stack = self.simulation._calculation_stack
        return calculation_stack[-1] if calculation_stack else None

    def _get_members_entity_id(self, period=None):
        period = self._resolve_period(period)
        snapshot_period = self._get_snapshot_period(period)
        if snapshot_period is None:
            return self._members_entity_id
        return self._members_entity_id_by_period[snapshot_period]

    def _get_members_role(self, period=None):
        period = self._resolve_period(period)
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
        period = self._resolve_period(period)
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
        period = self._resolve_period(period)
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

    def _check_members_array(self, array, period=None) -> None:
        expected_count = (
            self.members.get_count_for_period(period)
            if period is not None and self.members._dynamic
            else self.members.count
        )
        if array.size != expected_count:
            from ._errors import InvalidArraySizeError

            raise InvalidArraySizeError(
                array,
                self.members.entity.key,
                expected_count,
            )

    def _check_group_array(self, array, period=None) -> None:
        expected_count = (
            self.get_count_for_period(period)
            if period is not None and self._dynamic
            else self.count
        )
        if array.size != expected_count:
            from ._errors import InvalidArraySizeError

            raise InvalidArraySizeError(array, self.entity.key, expected_count)

    def activate_dynamic_mode(self, period, permanent_ids=None) -> None:
        """Enable dense integer group IDs and period group-count snapshots."""
        if permanent_ids is not None and not numpy.array_equal(
            permanent_ids,
            numpy.arange(self.count),
        ):
            raise ValueError(
                "Dynamic group permanent_ids must be the dense range "
                f"0 to {self.count - 1}",
            )
        super().activate_dynamic_mode(period, permanent_ids)
        self.ids = self._get_alive_ids_for_period(period)

    def set_members_for_period(
        self,
        period,
        members_entity_id,
        members_role=None,
    ) -> None:
        """Set membership from ``period`` until the next explicit snapshot.

        In static mode, person and group counts remain fixed. In dynamic mode,
        the membership length follows the period's person count. Group IDs are
        dense integers from zero through the largest referenced ID. Interior
        unreferenced IDs remain empty groups; unreferenced terminal IDs are
        dissolved. An empty membership therefore dissolves every group.
        """
        snapshot_period = self._validate_structure_period(period)
        array = numpy.asarray(members_entity_id)
        if array.ndim != 1:
            raise ValueError("members_entity_id must be a one-dimensional array")
        if array.size == 0 and not self._dynamic:
            raise ValueError("members_entity_id cannot be empty")
        expected_count = (
            self.members.get_count_for_period(snapshot_period)
            if self.members._dynamic
            else self.members.count
        )
        if len(array) != expected_count:
            raise ValueError(
                "members_entity_id must contain one entry per member "
                f"(expected {expected_count}, got {len(array)})",
            )
        if array.size and not numpy.issubdtype(array.dtype, numpy.integer):
            raise ValueError("members_entity_id must contain integer IDs")
        if array.size and numpy.any(array < 0):
            raise ValueError("members_entity_id must contain non-negative IDs")
        self._validate_id_density(array, "members_entity_id")
        if not self._dynamic and numpy.any(array >= self.count):
            raise ValueError(
                "members_entity_id must reference an existing group "
                f"(valid IDs are 0 to {self.count - 1})",
            )
        if not self._dynamic and int(numpy.max(array)) + 1 != self.count:
            raise ValueError(
                "set_members_for_period cannot change the number of groups",
            )

        if members_role is None:
            current_roles = self._carry_forward_roles(snapshot_period)
            if any(role is None for role in current_roles):
                raise ValueError(
                    "members_role is required for new members; call "
                    "auto_assign_roles_for_period for explicit assignment",
                )
            if len(current_roles) == len(array):
                role_snapshot = self._validate_members_role(current_roles, array)
            elif len(self.entity.flattened_roles) == 1 or array.size == 0:
                role_snapshot = numpy.repeat(
                    self.entity.flattened_roles[0],
                    len(array),
                )
            else:
                raise ValueError(
                    "members_role is required when the person count changes "
                    "for an entity with multiple roles",
                )
        else:
            role_snapshot = self._validate_members_role(members_role, array)
        snapshot = array.astype(numpy.intp, copy=True)
        snapshot.flags.writeable = False
        self._members_entity_id_by_period[snapshot_period] = snapshot
        if role_snapshot is not None:
            role_snapshot.flags.writeable = False
            self._members_role_by_period[snapshot_period] = role_snapshot
        self._members_position_by_period.pop(snapshot_period, None)
        self._ordered_members_map_by_period.pop(snapshot_period, None)

        if self._dynamic:
            group_count = int(numpy.max(snapshot)) + 1 if snapshot.size else 0
            known_group_count = (
                0 if self._permanent_ids is None else len(self._permanent_ids)
            )
            self._permanent_ids = numpy.arange(
                max(group_count, known_group_count),
                dtype=numpy.intp,
            )
            self._set_period_identity_snapshot(
                snapshot_period,
                numpy.arange(group_count, dtype=numpy.intp),
            )
            self.ids = numpy.arange(group_count, dtype=numpy.intp)

        next_starts = [
            item.start
            for item in self._members_entity_id_by_period
            if item.start > snapshot_period.start
        ]
        next_start = min(next_starts, default=None)
        self._invalidate_structure_caches(snapshot_period, next_start)

    def set_roles_for_period(self, period, members_role) -> None:
        """Set member roles from ``period`` until the next role snapshot."""
        snapshot_period = self._validate_structure_period(period)
        membership = self._get_members_entity_id(snapshot_period)
        role_snapshot = self._validate_members_role(members_role, membership)
        role_snapshot.flags.writeable = False
        self._members_role_by_period[snapshot_period] = role_snapshot

        next_starts = [
            item.start
            for item in self._members_role_by_period
            if item.start > snapshot_period.start
        ]
        self._invalidate_structure_caches(
            snapshot_period,
            min(next_starts, default=None),
        )

    def _invalidate_structure_caches(self, snapshot_period, next_start) -> None:
        populations = (
            self.simulation.populations.values()
            if self.simulation is not None
            else (self,)
        )
        for population in populations:
            population._invalidate_period_caches(snapshot_period, next_start)

    def _validate_structure_period(self, period):
        snapshots = {
            **self._members_entity_id_by_period,
            **self._members_role_by_period,
        }
        if self._dynamic:
            snapshots.update(self._period_index)
        return self._validate_snapshot_period(period, snapshots)

    def _get_previous_structure_period(self, period):
        target_period = periods.period(period)
        candidates = [
            snapshot_period
            for snapshot_period in (
                *self._members_entity_id_by_period,
                *self._members_role_by_period,
                *self.members._period_index,
            )
            if snapshot_period.start < target_period.start
        ]
        return max(candidates, key=lambda item: item.start, default=None)

    def _carry_forward_roles(self, period):
        source_period = self._get_previous_structure_period(period)
        if source_period is None:
            return self.members_role
        old_roles = numpy.asarray(self._get_members_role(source_period), dtype=object)
        if self.members._dynamic:
            old_roles = self.members.remap_array(
                old_roles,
                source_period,
                period,
                default=None,
            )
        return old_roles

    def _members_have_role(self, role, period=None):
        period = self._resolve_period(period)
        members_role = self._get_members_role(period)
        if role.subroles:
            return numpy.logical_or.reduce(
                [members_role == subrole for subrole in role.subroles],
            )
        return members_role == role

    def auto_assign_roles_for_period(
        self,
        period,
        members_entity_id,
        previous_period=None,
    ):
        """Assign roles deterministically after a membership change.

        Stayers retain their role. Movers are processed in person-row order and
        try their previous role first, then successively higher roles, then
        lower roles. Role capacities are respected. Finally, every non-empty
        group is guaranteed to contain the first declared role (normally
        ``first_parent``): the group's best-ranked member, with row order as a
        tie-breaker, is promoted when necessary.

        This method only computes roles. Pass its result to
        :meth:`set_members_for_period` to commit membership and roles together.
        """
        new_membership = numpy.asarray(members_entity_id)
        if new_membership.ndim != 1:
            raise ValueError("members_entity_id must be a one-dimensional array")
        expected_count = (
            self.members.get_count_for_period(period)
            if self.members._dynamic
            else self.members.count
        )
        if len(new_membership) != expected_count:
            raise ValueError(
                "members_entity_id must contain one entry per member "
                f"(expected {expected_count}, got {len(new_membership)})",
            )
        if new_membership.size and not numpy.issubdtype(
            new_membership.dtype,
            numpy.integer,
        ):
            raise ValueError("members_entity_id must contain integer IDs")
        if numpy.any(new_membership < 0) or (
            not self._dynamic and numpy.any(new_membership >= self.count)
        ):
            raise ValueError("members_entity_id contains an unknown group ID")
        self._validate_id_density(new_membership, "members_entity_id")

        new_membership = new_membership.astype(numpy.intp, copy=False)
        if self._dynamic:
            group_count = (
                int(numpy.max(new_membership)) + 1 if new_membership.size else 0
            )
        else:
            group_count = self.count

        if previous_period is None:
            source_period = self._get_previous_structure_period(period)
            if source_period is None:
                source_period = periods.period(period)
        else:
            source_period = periods.period(previous_period)
        old_membership = self._get_members_entity_id(source_period)
        old_roles = self._get_members_role(source_period)
        roles = self.entity.flattened_roles
        if self.members._dynamic:
            old_membership = self.members.remap_array(
                numpy.asarray(old_membership),
                source_period,
                period,
                default=-1,
            )
            old_roles = self.members.remap_array(
                numpy.asarray(old_roles, dtype=object),
                source_period,
                period,
                default=roles[0],
            )
        role_rank = {role: index for index, role in enumerate(roles)}
        role_count = len(roles)
        old_role_indices = numpy.fromiter(
            (role_rank.get(role, 0) for role in old_roles),
            dtype=numpy.intp,
            count=len(new_membership),
        )
        is_stayer = numpy.asarray(old_membership) == new_membership
        result_indices = numpy.full(len(new_membership), -1, dtype=numpy.intp)
        result_indices[is_stayer] = old_role_indices[is_stayer]
        used = numpy.zeros((group_count, role_count), dtype=numpy.intp)
        numpy.add.at(
            used,
            (new_membership[is_stayer], old_role_indices[is_stayer]),
            1,
        )

        search_orders = []
        for old_rank in range(role_count):
            order = [old_rank, *range(old_rank - 1, -1, -1)]
            order.extend(range(old_rank + 1, role_count))
            search_orders.append(order)

        for row, group_id in enumerate(new_membership):
            if is_stayer[row]:
                continue
            for rank in search_orders[old_role_indices[row]]:
                role = roles[rank]
                if role.max is None or used[int(group_id), rank] < role.max:
                    result_indices[row] = rank
                    used[int(group_id), rank] += 1
                    break
            if result_indices[row] < 0:
                raise ValueError(f"No role is available in group {group_id}")

        active_groups = numpy.unique(new_membership)
        missing_primary = active_groups[used[active_groups, 0] == 0]
        if missing_primary.size:
            rows = numpy.flatnonzero(numpy.isin(new_membership, missing_primary))
            order = numpy.lexsort(
                (rows, result_indices[rows], new_membership[rows]),
            )
            ordered_rows = rows[order]
            ordered_groups = new_membership[ordered_rows]
            first_in_group = numpy.concatenate(
                ([True], ordered_groups[1:] != ordered_groups[:-1]),
            )
            result_indices[ordered_rows[first_in_group]] = 0

        role_array = numpy.asarray(roles, dtype=object)
        result = role_array[result_indices]
        return self._validate_members_role(result, new_membership)

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
        self._validate_id_density(array, "members_entity_id")

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
        period = self._resolve_period(period)
        self.entity.check_role_validity(role)
        self._check_members_array(array, period)
        members_entity_id = self._get_members_entity_id(period)
        group_count = (
            self.get_count_for_period(period)
            if period is not None and self._dynamic
            else self.count
        )
        if role is not None:
            role_filter = self._members_have_role(role, period)
            return numpy.bincount(
                members_entity_id[role_filter],
                weights=array[role_filter],
                minlength=group_count,
            )
        return numpy.bincount(
            members_entity_id,
            weights=array,
            minlength=group_count,
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
        period = self._resolve_period(period)
        self._check_members_array(array, period)
        self.entity.check_role_validity(role)
        position_in_entity = self._get_members_position(period)
        role_filter = (
            self._members_have_role(role, period) if role is not None else True
        )
        filtered_array = numpy.where(role_filter, array, neutral_element)

        group_count = (
            self.get_count_for_period(period)
            if period is not None and self._dynamic
            else self.count
        )
        result = numpy.full(group_count, neutral_element)
        if position_in_entity.size == 0:
            return result

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
        period = self._resolve_period(period)
        if role:
            role_condition = self._members_have_role(role, period)
            return self.sum(role_condition, period=period)
        group_count = (
            self.get_count_for_period(period)
            if period is not None and self._dynamic
            else self.count
        )
        return numpy.bincount(
            self._get_members_entity_id(period),
            minlength=group_count,
        )

    # Projection person -> entity

    @projectors.projectable
    def value_from_person(self, array, role, default=0, period=None):
        """Get the value of ``array`` for the person with the unique role ``role``.

        ``array`` must have the dimension of the number of persons in the simulation

        If such a person does not exist, return ``default`` instead

        The result is a vector which dimension is the number of entities
        """
        period = self._resolve_period(period)
        self.entity.check_role_validity(role)
        if role.max != 1:
            msg = f"You can only use value_from_person with a role that is unique in {self.key}. Role {role.key} is not unique."
            raise Exception(
                msg,
            )
        self._check_members_array(array, period)
        members_map = self._get_ordered_members_map(period)
        group_count = (
            self.get_count_for_period(period)
            if period is not None and self._dynamic
            else self.count
        )
        result = numpy.full(group_count, default, dtype=array.dtype)
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
        period = self._resolve_period(period)
        self._check_members_array(array, period)
        positions = self._get_members_position(period)
        nb_persons_per_entity = self.nb_persons(period=period)
        members_map = self._get_ordered_members_map(period)
        group_count = (
            self.get_count_for_period(period)
            if period is not None and self._dynamic
            else self.count
        )
        result = numpy.full(group_count, default, dtype=array.dtype)
        if positions.size == 0:
            return result
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
        period = self._resolve_period(period)
        self._check_group_array(array, period)
        self.entity.check_role_validity(role)
        members_entity_id = self._get_members_entity_id(period)
        if role is None:
            return array[members_entity_id]
        role_condition = self._members_have_role(role, period)
        return numpy.where(role_condition, array[members_entity_id], 0)
