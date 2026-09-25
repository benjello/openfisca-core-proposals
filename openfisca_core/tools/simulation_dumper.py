import os

import numpy

from openfisca_core.data_storage import OnDiskStorage
from openfisca_core.indexed_enums import Enum, EnumArray
from openfisca_core.periods import DateUnit, instant, period as parse_period
from openfisca_core.simulations import Simulation


AS_OF_STATE_FILE = "__as_of_state.npz"


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
        population.count = person_count

    variables_to_restore = (
        variable for variable in os.listdir(directory) if variable != "__entities__"
    )
    for variable in variables_to_restore:
        _restore_holder(simulation, variable, directory)

    return simulation


def _dump_holder(holder, directory) -> None:
    disk_storage = holder.create_disk_storage(directory, preserve=True)
    for known_period in holder.get_known_periods():
        value = holder.get_array(known_period)
        disk_storage.put(value, known_period)
    if holder.variable.as_of and holder._as_of_base is not None:
        state = {
            "base": holder._as_of_base,
            "base_instant": str(holder._as_of_base_instant),
            "base_source": holder._as_of_base_source,
            "patch_instants": [str(item) for item in holder._as_of_patch_instants],
            "patch_sources": holder._as_of_patch_sources,
            "known_periods": [
                str(item) for item in sorted(holder._as_of_known_periods)
            ],
            "explicit_periods": [
                str(item) for item in sorted(holder._as_of_explicit_periods)
            ],
            "calculated_periods": [
                str(item) for item in sorted(holder._as_of_calculated_periods)
            ],
            "transition_computed": [
                str(item) for item in sorted(holder._as_of_transition_computed)
            ],
        }
        for index, (_, indices, values) in enumerate(holder._as_of_patches):
            state[f"patch_{index}_indices"] = indices
            state[f"patch_{index}_values"] = values
        numpy.savez(os.path.join(disk_storage.storage_dir, AS_OF_STATE_FILE), **state)


def _dump_entity(population, directory) -> None:
    path = os.path.join(directory, population.entity.key)
    os.mkdir(path)
    numpy.save(os.path.join(path, "id.npy"), population.ids)

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


def _restore_entity(population, directory):
    path = os.path.join(directory, population.entity.key)

    population.ids = numpy.load(os.path.join(path, "id.npy"))

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
    person_count = len(population.members_entity_id)
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

    as_of_state_path = os.path.join(storage_dir, AS_OF_STATE_FILE)
    if holder.variable.as_of and os.path.isfile(as_of_state_path):
        _restore_as_of_holder(holder, as_of_state_path)
        return

    for known_period in disk_storage.get_known_periods():
        value = disk_storage.get(known_period)
        holder.put_in_cache(value, known_period)


def _restore_as_of_holder(holder, state_path) -> None:
    with numpy.load(state_path, allow_pickle=False) as state:

        def restore_array(value, *, enum=False):
            array = state[value]
            if enum and holder.variable.value_type == Enum:
                array = EnumArray(array, holder.variable.possible_values)
            return holder._immutable_array(array)

        holder._as_of_base = restore_array("base", enum=True)
        holder._as_of_base_instant = instant(str(state["base_instant"]))
        holder._as_of_base_source = str(state["base_source"])
        holder._as_of_patch_instants = [
            instant(str(value)) for value in state["patch_instants"]
        ]
        holder._as_of_patch_sources = state["patch_sources"].tolist()
        holder._as_of_patches = [
            (
                patch_instant,
                restore_array(f"patch_{index}_indices"),
                restore_array(f"patch_{index}_values", enum=True),
            )
            for index, patch_instant in enumerate(holder._as_of_patch_instants)
        ]
        holder._as_of_known_periods = {
            parse_period(str(value)) for value in state["known_periods"]
        }
        holder._as_of_explicit_periods = {
            parse_period(str(value)) for value in state["explicit_periods"]
        }
        holder._as_of_calculated_periods = {
            parse_period(str(value)) for value in state["calculated_periods"]
        }
        holder._as_of_transition_computed = {
            instant(str(value)) for value in state["transition_computed"]
        }
