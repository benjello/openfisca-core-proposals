import numpy
import pytest

from openfisca_core import entities, periods, taxbenefitsystems, variables
from openfisca_core.links import Many2OneLink, One2ManyLink
from openfisca_core.simulations import SimulationBuilder


@pytest.fixture(scope="module")
def link_simulation():
    person_count = 10_000
    household_count = 1_000
    person = entities.SingleEntity("person", "persons", "", "")
    household = entities.GroupEntity(
        "household",
        "households",
        "",
        "",
        roles=[{"key": "member"}],
    )
    person.add_link(Many2OneLink("household", "household_id", "household"))
    household.add_link(One2ManyLink("residents", "household_id", "person"))
    tax_benefit_system = taxbenefitsystems.TaxBenefitSystem([person, household])

    class household_id(variables.Variable):
        value_type = int
        entity = person
        definition_period = periods.ETERNITY

    class salary(variables.Variable):
        value_type = float
        entity = person
        definition_period = periods.YEAR

    class rent(variables.Variable):
        value_type = float
        entity = household
        definition_period = periods.YEAR

    for variable in (household_id, salary, rent):
        tax_benefit_system.add_variable(variable)

    simulation = SimulationBuilder().build_default_simulation(
        tax_benefit_system,
        count=person_count,
    )
    simulation.household.count = household_count
    simulation.household.ids = numpy.arange(household_count)
    simulation.set_input(
        "household_id",
        periods.ETERNITY,
        numpy.arange(person_count) % household_count,
    )
    simulation.set_input("salary", "2024", numpy.arange(person_count, dtype=float))
    simulation.set_input("rent", "2024", numpy.arange(household_count, dtype=float))
    return simulation


def test_bench_many2one_get(benchmark, link_simulation):
    benchmark(link_simulation.persons.household.get, "rent", "2024")


def test_bench_one2many_sum(benchmark, link_simulation):
    benchmark(link_simulation.household.residents.sum, "salary", "2024")
