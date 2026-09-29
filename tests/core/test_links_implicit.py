import numpy
import pytest
from types import SimpleNamespace

from openfisca_core import entities, periods, taxbenefitsystems, variables
from openfisca_core.links import Many2OneLink
from openfisca_core.links.implicit import ImplicitMany2OneLink, ImplicitOne2ManyLink
from openfisca_core.simulations import SimulationBuilder
from openfisca_core.simulations.simulation import Simulation


@pytest.fixture
def simulation():
    person = entities.SingleEntity("person", "persons", "", "")
    household = entities.GroupEntity(
        "household",
        "households",
        "",
        "",
        roles=[{"key": "adult", "max": 1}, {"key": "child"}],
    )
    tax_benefit_system = taxbenefitsystems.TaxBenefitSystem([person, household])

    class salary(variables.Variable):
        value_type = int
        entity = person
        definition_period = periods.YEAR

    class rent(variables.Variable):
        value_type = int
        entity = household
        definition_period = periods.YEAR

    tax_benefit_system.add_variable(salary)
    tax_benefit_system.add_variable(rent)
    return SimulationBuilder().build_from_dict(
        tax_benefit_system,
        {
            "persons": {
                "a": {"salary": {"2024": 10}},
                "b": {"salary": {"2024": 20}},
                "c": {"salary": {"2024": 5}},
            },
            "households": {
                "first": {
                    "adult": ["a"],
                    "child": ["b"],
                    "rent": {"2024": 100},
                },
                "second": {"adult": ["c"], "rent": {"2024": 80}},
            },
        },
    )


def test_group_membership_creates_implicit_links(simulation):
    assert isinstance(simulation.persons.household, ImplicitMany2OneLink)
    assert isinstance(simulation.household.persons, ImplicitOne2ManyLink)

    numpy.testing.assert_array_equal(
        simulation.persons.household("rent", "2024"),
        [100, 100, 80],
    )
    numpy.testing.assert_array_equal(
        simulation.household.persons.sum("salary", "2024"),
        [30, 5],
    )


def test_implicit_link_keeps_projector_shortcuts_working(simulation):
    salaries = simulation.persons("salary", "2024")

    numpy.testing.assert_array_equal(
        simulation.persons.household.sum(salaries),
        [30, 30, 5],
    )
    numpy.testing.assert_array_equal(
        simulation.persons.household.first_person("salary", "2024"),
        [10, 10, 5],
    )


def test_implicit_link_remains_compatible_with_get_rank(simulation):
    salaries = simulation.persons("salary", "2024")

    numpy.testing.assert_array_equal(
        simulation.persons.get_rank(simulation.persons.household, salaries),
        [0, 1, 0],
    )


def test_implicit_membership_rows_are_not_public_group_ids(simulation):
    simulation.household.ids = numpy.array([1, 0])

    numpy.testing.assert_array_equal(
        simulation.persons.household("rent", "2024"), [100, 100, 80]
    )


def test_clone_rebinds_explicit_and_implicit_links(simulation):
    simulation.persons.entity.add_link(
        Many2OneLink("explicit_household", "unused", "household")
    )
    simulation._resolve_links()

    clone = simulation.clone()

    for population_key, population in simulation.populations.items():
        cloned_population = clone.populations[population_key]
        for name, link in population.links.items():
            cloned_link = cloned_population.links[name]
            assert cloned_link is not link
            assert cloned_link._source_population is cloned_population
            assert (
                cloned_link._target_population
                is clone.populations[link.target_entity_key]
            )

    assert clone.household.members is clone.persons
    assert clone.household.members is not simulation.persons
    numpy.testing.assert_array_equal(
        clone.persons.household("rent", "2024"), [100, 100, 80]
    )


def test_implicit_links_use_the_tax_benefit_system_person_key():
    individu = entities.SingleEntity("individu", "individus", "", "")
    menage = entities.GroupEntity(
        "menage",
        "menages",
        "",
        "",
        roles=[{"key": "membre"}],
    )
    tax_benefit_system = taxbenefitsystems.TaxBenefitSystem([individu, menage])
    simulation = SimulationBuilder().build_default_simulation(
        tax_benefit_system,
        count=2,
    )

    link = simulation.menage.individus
    assert link.target_entity_key == "individu"
    assert link._target_population is simulation.persons


def test_link_resolution_tolerates_entities_without_links():
    population = SimpleNamespace(entity=SimpleNamespace(key="legacy"))
    simulation = object.__new__(Simulation)
    simulation.persons = population
    simulation.populations = {"legacy": population}

    simulation._resolve_links()

    assert population.links == {}
