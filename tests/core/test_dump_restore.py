import shutil
import tempfile

from numpy import testing
from openfisca_country_template import situation_examples

from openfisca_core.periods import DateUnit, period
from openfisca_core.simulations import SimulationBuilder
from openfisca_core.tools import simulation_dumper
from openfisca_core.variables import Variable


def test_dump(tax_benefit_system) -> None:
    directory = tempfile.mkdtemp(prefix="openfisca_")
    simulation = SimulationBuilder().build_from_entities(
        tax_benefit_system,
        situation_examples.couple,
    )
    calculated_value = simulation.calculate("disposable_income", "2018-01")
    simulation_dumper.dump_simulation(simulation, directory)

    simulation_2 = simulation_dumper.restore_simulation(directory, tax_benefit_system)

    # Check entities structure have been restored

    testing.assert_array_equal(simulation.person.ids, simulation_2.person.ids)
    testing.assert_array_equal(simulation.person.count, simulation_2.person.count)
    testing.assert_array_equal(simulation.household.ids, simulation_2.household.ids)
    testing.assert_array_equal(simulation.household.count, simulation_2.household.count)
    testing.assert_array_equal(
        simulation.household.members_position,
        simulation_2.household.members_position,
    )
    testing.assert_array_equal(
        simulation.household.members_entity_id,
        simulation_2.household.members_entity_id,
    )
    testing.assert_array_equal(
        simulation.household.members_role,
        simulation_2.household.members_role,
    )

    # Check calculated values are in cache

    disposable_income_holder = simulation_2.household.get_holder("disposable_income")
    cached_value = disposable_income_holder.get_array("2018-01")
    assert cached_value is not None
    testing.assert_array_equal(cached_value, calculated_value)

    shutil.rmtree(directory)


def test_dump_and_restore_as_of_history(tax_benefit_system, tmp_path) -> None:
    class PersistentState(Variable):
        value_type = int
        entity = tax_benefit_system.person_entity
        definition_period = DateUnit.MONTH
        as_of = True

    tax_benefit_system.add_variable(PersistentState)
    simulation = SimulationBuilder().build_from_entities(
        tax_benefit_system,
        situation_examples.couple,
    )
    simulation.set_input("PersistentState", "2018-01", [1, 2])
    simulation.get_holder("PersistentState").set_input_sparse(
        "2018-03",
        [1],
        [7],
    )
    directory = tmp_path / "simulation"

    simulation_dumper.dump_simulation(simulation, str(directory))
    restored = simulation_dumper.restore_simulation(
        str(directory),
        tax_benefit_system,
    )

    holder = restored.get_holder("PersistentState")
    assert holder.get_known_periods() == [period("2018-01"), period("2018-03")]
    testing.assert_array_equal(holder.get_array("2018-02"), [1, 2])
    testing.assert_array_equal(holder.get_array("2018-04"), [1, 7])
