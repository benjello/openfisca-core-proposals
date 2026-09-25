import numpy
import pytest

from openfisca_core import entities, periods, taxbenefitsystems, variables
from openfisca_core.links.implicit import ImplicitMany2OneLink, ImplicitOne2ManyLink
from openfisca_core.simulations import SimulationBuilder


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
