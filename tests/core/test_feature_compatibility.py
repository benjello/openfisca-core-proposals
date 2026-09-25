import numpy
import pytest

from openfisca_core import entities, periods, populations, taxbenefitsystems, variables
from openfisca_core.links import Many2OneLink
from openfisca_core.simulations import SimulationBuilder
from openfisca_core.tools import simulation_dumper


def build_dynamic_simulation(*, state_variable=None):
    person = entities.SingleEntity("person", "persons", "", "")
    household = entities.GroupEntity(
        "household",
        "households",
        "",
        "",
        roles=[{"key": "member"}],
    )
    tax_benefit_system = taxbenefitsystems.TaxBenefitSystem([person, household])

    class salary(variables.Variable):
        value_type = float
        entity = person
        definition_period = periods.MONTH

    class rent(variables.Variable):
        value_type = float
        entity = household
        definition_period = periods.MONTH

    tax_benefit_system.add_variables(salary, rent)
    if state_variable is not None:
        state_variable.entity = person
        tax_benefit_system.add_variable(state_variable)

    simulation = SimulationBuilder.build_default_simulation(
        tax_benefit_system,
        count=2,
    )
    simulation.household.set_members_entity_id([0, 0])
    simulation.persons.activate_dynamic_mode("2024-01", [10, 20])
    simulation.household.activate_dynamic_mode("2024-01")
    return simulation, tax_benefit_system


def add_person_and_household(simulation) -> None:
    simulation.persons.register_births("2024-02", [30])
    member = simulation.household.entity.flattened_roles[0]
    simulation.household.set_members_for_period(
        "2024-02",
        [0, 1, 1],
        members_role=[member, member, member],
    )


def test_implicit_links_use_historical_population_sizes() -> None:
    simulation, _ = build_dynamic_simulation()
    simulation.set_input("salary", "2024-01", [10, 20])
    simulation.set_input("rent", "2024-01", [100])
    add_person_and_household(simulation)
    simulation.set_input("salary", "2024-02", [10, 20, 30])
    simulation.set_input("rent", "2024-02", [100, 200])
    member = simulation.household.entity.flattened_roles[0]

    numpy.testing.assert_array_equal(
        simulation.persons.household("rent", "2024-01"),
        [100, 100],
    )
    numpy.testing.assert_array_equal(
        simulation.household.persons.count("2024-01"),
        [2],
    )
    numpy.testing.assert_array_equal(
        simulation.persons.household.sum(
            numpy.array([10, 20]),
            period="2024-01",
        ),
        [30, 30],
    )
    numpy.testing.assert_array_equal(
        simulation.persons.household.first_person("salary", "2024-01"),
        [10, 10],
    )
    numpy.testing.assert_array_equal(
        simulation.persons.household.has_role(member, "2024-01"),
        [True, True],
    )

    numpy.testing.assert_array_equal(
        simulation.persons.household("rent", "2024-02"),
        [100, 200, 200],
    )
    numpy.testing.assert_array_equal(
        simulation.household.persons.count("2024-02"),
        [1, 2],
    )


def test_explicit_links_resolve_historical_permanent_ids() -> None:
    person = entities.SingleEntity("person", "persons", "", "")
    person.add_link(Many2OneLink("mother", "mother_id", "person"))
    tax_benefit_system = taxbenefitsystems.TaxBenefitSystem([person])

    class age(variables.Variable):
        value_type = int
        entity = person
        definition_period = periods.MONTH

    class mother_id(variables.Variable):
        value_type = int
        entity = person
        definition_period = periods.MONTH
        default_value = -1

    tax_benefit_system.add_variables(age, mother_id)
    simulation = SimulationBuilder.build_default_simulation(
        tax_benefit_system,
        count=2,
    )
    simulation.persons.activate_dynamic_mode("2024-01", [10, 20])
    simulation.set_input("age", "2024-01", [40, 20])
    simulation.set_input("mother_id", "2024-01", [-1, 10])
    simulation.persons.register_deaths("2024-02", [10])
    simulation.set_input("age", "2024-02", [20])
    simulation.set_input("mother_id", "2024-02", [-1])

    numpy.testing.assert_array_equal(
        simulation.persons.mother("age", "2024-01"),
        [0, 40],
    )
    numpy.testing.assert_array_equal(
        simulation.persons.mother("age", "2024-02"),
        [0],
    )


def test_as_of_state_follows_permanent_ids_through_lifecycle(tmp_path) -> None:
    class state(variables.Variable):
        value_type = int
        definition_period = periods.MONTH
        as_of = True

    simulation, tax_benefit_system = build_dynamic_simulation(state_variable=state)
    simulation.set_input("state", "2024-01", [1, 2])
    add_person_and_household(simulation)

    numpy.testing.assert_array_equal(
        simulation.calculate("state", "2024-02"),
        [1, 2, 0],
    )
    simulation.get_holder("state").set_input_sparse("2024-02", [2], [3])
    simulation.persons.register_deaths("2024-03", [10])
    simulation.set_input("state", "2024-03", [20, 30])

    numpy.testing.assert_array_equal(
        simulation.calculate("state", "2024-01"),
        [1, 2],
    )
    numpy.testing.assert_array_equal(
        simulation.calculate("state", "2024-02"),
        [1, 2, 3],
    )
    numpy.testing.assert_array_equal(
        simulation.calculate("state", "2024-03"),
        [20, 30],
    )

    directory = tmp_path / "simulation"
    simulation_dumper.dump_simulation(simulation, str(directory))
    restored = simulation_dumper.restore_simulation(
        str(directory),
        tax_benefit_system,
    )
    numpy.testing.assert_array_equal(
        restored.calculate("state", "2024-02"),
        [1, 2, 3],
    )
    numpy.testing.assert_array_equal(
        restored.calculate("state", "2024-03"),
        [20, 30],
    )


def test_as_of_transition_uses_the_period_population() -> None:
    class state(variables.Variable):
        value_type = int
        definition_period = periods.MONTH
        as_of = True

        def initial_formula(population, _period):
            return numpy.arange(population.count) + 1

        def transition_formula(population, period):
            previous = population("state", period.last_month)
            return numpy.arange(population.count), previous + 1

    simulation, _ = build_dynamic_simulation(state_variable=state)
    add_person_and_household(simulation)

    numpy.testing.assert_array_equal(
        simulation.calculate("state", "2024-01"),
        [1, 2],
    )
    numpy.testing.assert_array_equal(
        simulation.calculate("state", "2024-02"),
        [2, 3, 1],
    )


@pytest.mark.parametrize("permanent_ids", ([10, 20], [1, 0]))
def test_dynamic_activation_migrates_existing_as_of_state(
    permanent_ids,
    tmp_path,
) -> None:
    class state(variables.Variable):
        value_type = int
        definition_period = periods.MONTH
        as_of = True

    simulation, tax_benefit_system = build_dynamic_simulation(
        state_variable=state,
    )
    simulation = SimulationBuilder.build_default_simulation(
        tax_benefit_system,
        count=2,
    )
    simulation.set_input("state", "2024-01", [1, 2])
    simulation.get_holder("state").set_input_sparse("2024-02", [0], [3])

    simulation.persons.activate_dynamic_mode("2024-01", permanent_ids)

    numpy.testing.assert_array_equal(
        simulation.calculate("state", "2024-02"),
        [3, 2],
    )
    numpy.testing.assert_array_equal(
        simulation.clone().calculate("state", "2024-02"),
        [3, 2],
    )

    directory = tmp_path / "simulation"
    simulation_dumper.dump_simulation(simulation, str(directory))
    restored = simulation_dumper.restore_simulation(
        str(directory),
        tax_benefit_system,
    )
    numpy.testing.assert_array_equal(
        restored.calculate("state", "2024-02"),
        [3, 2],
    )


def test_retroactive_birth_preserves_future_sparse_input() -> None:
    class state(variables.Variable):
        value_type = int
        definition_period = periods.MONTH
        as_of = True

        def initial_formula(population, _period):
            return numpy.arange(population.count) + 1

    simulation, _ = build_dynamic_simulation(state_variable=state)
    simulation.calculate("state", "2024-03")
    holder = simulation.get_holder("state")
    holder.set_input_sparse("2024-04", [1], [9])

    simulation.persons.register_births("2024-02", [30])

    assert periods.period("2024-04") in holder._input_periods
    numpy.testing.assert_array_equal(
        simulation.calculate("state", "2024-04"),
        [1, 9, 3],
    )


def test_link_add_uses_last_subperiod_identity_space() -> None:
    person = entities.SingleEntity("person", "persons", "", "")
    person.add_link(Many2OneLink("parent", "parent_id", "person"))
    tax_benefit_system = taxbenefitsystems.TaxBenefitSystem([person])

    class score(variables.Variable):
        value_type = int
        entity = person
        definition_period = periods.MONTH

    class parent_id(variables.Variable):
        value_type = int
        entity = person
        definition_period = periods.MONTH
        default_value = -1
        as_of = True

    tax_benefit_system.add_variables(score, parent_id)
    simulation = SimulationBuilder.build_default_simulation(
        tax_benefit_system,
        count=2,
    )
    simulation.persons.activate_dynamic_mode("2024-01", [10, 20])
    simulation.set_input("score", "2024-01", [10, 2])
    simulation.set_input("parent_id", "2024-01", [-1, 10])
    simulation.persons.register_deaths("2024-02", [10])
    simulation.persons.register_births("2024-03", [30])
    simulation.set_input("score", "2024-12", [2, 3])
    simulation.set_input("parent_id", "2024-12", [30, -1])

    numpy.testing.assert_array_equal(
        simulation.persons.parent("score", "2024", options=[populations.ADD]),
        [3, 0],
    )


def test_link_add_reads_annual_link_before_projecting_last_subperiod() -> None:
    person = entities.SingleEntity("person", "persons", "", "")
    person.add_link(Many2OneLink("parent", "parent_id", "person"))
    tax_benefit_system = taxbenefitsystems.TaxBenefitSystem([person])

    class score(variables.Variable):
        value_type = int
        entity = person
        definition_period = periods.MONTH

        def formula(population, period):
            return period.start.month

    class parent_id(variables.Variable):
        value_type = int
        entity = person
        definition_period = periods.YEAR
        default_value = -1

    tax_benefit_system.add_variables(score, parent_id)
    simulation = SimulationBuilder.build_default_simulation(
        tax_benefit_system,
        count=3,
    )
    simulation.persons.activate_dynamic_mode("2024-01", [10, 20, 30])
    simulation.set_input("parent_id", "2024", [-1, 10, -1])
    simulation.persons.register_deaths("2024-02", [30])
    simulation.persons.register_births("2024-03", [40])
    simulation.persons.register_deaths("2024-04", [40])

    numpy.testing.assert_array_equal(
        simulation.persons.parent("score", "2024", options=[populations.ADD]),
        [0, 78],
    )
