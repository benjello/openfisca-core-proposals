import os

import numpy

from openfisca_core.data_storage import OnDiskStorage
from openfisca_core.indexed_enums import Enum
from openfisca_core import periods
from openfisca_core.periods import DateUnit
from openfisca_core.simulations import Simulation


def dump_simulation(simulation, directory) -> None:
    """Write simulation data to directory, so that it can be restored later."""
    parent_directory = os.path.abspath(os.path.join(directory, os.pardir))
    if not os.path.isdir(parent_directory):  # To deal with reforms
        os.mkdir(parent_directory)
    if not os.path.isdir(directory):
        os.mkdir(directory)

    if os.listdir(directory):
        msg = f"Directory '{directory}' is not empty"
        raise ValueError(msg)

    entities_dump_dir = os.path.join(directory, "__entities__")
    os.mkdir(entities_dump_dir)

    for entity in simulation.populations.values():
        # Dump entity structure
        _dump_entity(entity, entities_dump_dir)

        # Dump variable values
        for holder in entity._holders.values():
            _dump_holder(holder, directory)


def restore_simulation(directory, tax_benefit_system, **kwargs):
    """Restore simulation from directory."""
    simulation = Simulation(
        tax_benefit_system,
        tax_benefit_system.instantiate_entities(),
    )

    entities_dump_dir = os.path.join(directory, "__entities__")
    for population in simulation.populations.values():
        if population.entity.is_person:
            continue
        person_count = _restore_entity(population, entities_dump_dir)

    for population in simulation.populations.values():
        if not population.entity.is_person:
            continue
        _restore_entity(population, entities_dump_dir)
        if population._dynamic:
            latest = max(population._period_index, key=lambda period: period.start)
            population.count = population._period_index[latest]["count"]
        else:
            population.count = person_count

    variables_to_restore = (
        variable for variable in os.listdir(directory) if variable != "__entities__"
    )
    for variable in variables_to_restore:
        _restore_holder(simulation, variable, directory)

    return simulation


def _dump_holder(holder, directory) -> None:
    disk_storage = holder.create_disk_storage(directory, preserve=True)
    for period in holder.get_known_periods():
        value = holder.get_array(period)
        disk_storage.put(value, period)


def _dump_entity(population, directory) -> None:
    path = os.path.join(directory, population.entity.key)
    os.mkdir(path)
    numpy.save(os.path.join(path, "id.npy"), population.ids)

    if population._period_index:
        snapshot_periods = sorted(
            population._period_index,
            key=lambda period: period.start,
        )
        numpy.save(
            os.path.join(path, "identity_periods.npy"),
            numpy.asarray([str(period) for period in snapshot_periods]),
        )
        numpy.save(
            os.path.join(path, "identity_counts.npy"),
            numpy.asarray(
                [
                    population._period_index[period]["count"]
                    for period in snapshot_periods
                ],
                dtype=numpy.intp,
            ),
        )
        for index, period in enumerate(snapshot_periods):
            numpy.save(
                os.path.join(path, f"id_to_rownum_{index}.npy"),
                population._period_index[period]["id_to_rownum"],
            )
        numpy.save(os.path.join(path, "permanent_ids.npy"), population._permanent_ids)

    if population.entity.is_person:
        return

    numpy.save(os.path.join(path, "members_position.npy"), population.members_position)
    numpy.save(
        os.path.join(path, "members_entity_id.npy"), population.members_entity_id
    )

    flattened_roles = population.entity.flattened_roles
    if len(flattened_roles) == 0:
        encoded_roles = numpy.int16(0)
    else:
        encoded_roles = numpy.select(
            [population.members_role == role for role in flattened_roles],
            [role.key for role in flattened_roles],
            default="",
        )
    numpy.save(os.path.join(path, "members_role.npy"), encoded_roles)

    membership_periods = sorted(
        population._members_entity_id_by_period,
        key=lambda period: period.start,
    )
    if membership_periods:
        numpy.save(
            os.path.join(path, "membership_periods.npy"),
            numpy.asarray([str(period) for period in membership_periods]),
        )
        for index, snapshot_period in enumerate(membership_periods):
            numpy.save(
                os.path.join(path, f"members_entity_id_{index}.npy"),
                population._members_entity_id_by_period[snapshot_period],
            )

    role_periods = sorted(
        population._members_role_by_period,
        key=lambda period: period.start,
    )
    if role_periods:
        numpy.save(
            os.path.join(path, "role_periods.npy"),
            numpy.asarray([str(period) for period in role_periods]),
        )
        for index, snapshot_period in enumerate(role_periods):
            role_snapshot = population._members_role_by_period[snapshot_period]
            encoded_snapshot = numpy.select(
                [role_snapshot == role for role in flattened_roles],
                [role.key for role in flattened_roles],
                default="",
            )
            numpy.save(
                os.path.join(path, f"members_role_{index}.npy"),
                encoded_snapshot,
            )


def _restore_entity(population, directory):
    path = os.path.join(directory, population.entity.key)

    population.ids = numpy.load(os.path.join(path, "id.npy"))

    identity_periods_path = os.path.join(path, "identity_periods.npy")
    if os.path.exists(identity_periods_path):
        snapshot_periods = numpy.load(identity_periods_path)
        snapshot_counts = numpy.load(os.path.join(path, "identity_counts.npy"))
        population._dynamic = True
        population._permanent_ids = numpy.load(
            os.path.join(path, "permanent_ids.npy"),
        )
        for index, (snapshot_period, count) in enumerate(
            zip(snapshot_periods, snapshot_counts),
        ):
            mapping = numpy.load(os.path.join(path, f"id_to_rownum_{index}.npy"))
            mapping.flags.writeable = False
            population._period_index[periods.period(str(snapshot_period))] = {
                "count": int(count),
                "id_to_rownum": mapping,
            }
        latest = max(population._period_index, key=lambda period: period.start)
        population._id_to_rownum = population._period_index[latest]["id_to_rownum"]

    if population.entity.is_person:
        return None

    population.members_position = numpy.load(os.path.join(path, "members_position.npy"))
    population.members_entity_id = numpy.load(
        os.path.join(path, "members_entity_id.npy")
    )
    encoded_roles = numpy.load(os.path.join(path, "members_role.npy"))

    flattened_roles = population.entity.flattened_roles
    if len(flattened_roles) == 0:
        population.members_role = numpy.int16(0)
    else:
        population.members_role = numpy.select(
            [encoded_roles == role.key for role in flattened_roles],
            list(flattened_roles),
            default=None,
        )

    membership_periods_path = os.path.join(path, "membership_periods.npy")
    if os.path.exists(membership_periods_path):
        for index, snapshot_period in enumerate(
            numpy.load(membership_periods_path),
        ):
            membership = numpy.load(
                os.path.join(path, f"members_entity_id_{index}.npy"),
            )
            membership.flags.writeable = False
            population._members_entity_id_by_period[
                periods.period(str(snapshot_period))
            ] = membership

    role_periods_path = os.path.join(path, "role_periods.npy")
    if os.path.exists(role_periods_path):
        for index, snapshot_period in enumerate(numpy.load(role_periods_path)):
            encoded_snapshot = numpy.load(
                os.path.join(path, f"members_role_{index}.npy"),
            )
            role_snapshot = numpy.select(
                [encoded_snapshot == role.key for role in flattened_roles],
                list(flattened_roles),
                default=None,
            )
            role_snapshot.flags.writeable = False
            population._members_role_by_period[periods.period(str(snapshot_period))] = (
                role_snapshot
            )
    person_count = len(population.members_entity_id)
    if not population._dynamic:
        population.count = max(population.members_entity_id) + 1
    return person_count


def _restore_holder(simulation, variable_name, directory) -> None:
    storage_dir = os.path.join(directory, variable_name)

    holder = simulation.get_holder(variable_name)

    is_variable_eternal = holder.variable.definition_period == DateUnit.ETERNITY

    disk_storage = OnDiskStorage(
        storage_dir,
        is_eternal=is_variable_eternal,
        preserve_storage_dir=True,
        enums=(
            {storage_dir: holder.variable.possible_values}
            if holder.variable.value_type == Enum
            and holder.variable.possible_values is not None
            else {}
        ),
    )
    disk_storage.restore()

    for period in disk_storage.get_known_periods():
        value = disk_storage.get(period)
        holder.put_in_cache(value, period)
