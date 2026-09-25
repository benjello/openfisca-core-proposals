import numpy
import pytest

from openfisca_core import entities, periods, taxbenefitsystems, variables
from openfisca_core.links import Many2OneLink, One2ManyLink
from openfisca_core.links.link import LinkResolutionError
from openfisca_core.simulations import SimulationBuilder


@pytest.fixture
def simulation():
    person = entities.SingleEntity("person", "persons", "", "")
    household = entities.GroupEntity(
        "household",
        "households",
        "",
        "",
        roles=[{"key": "member"}],
    )
    person.add_link(
        Many2OneLink(
            "household",
            "household_id",
            "household",
            role_field="resident_role",
        )
    )
    household.add_link(
        One2ManyLink(
            "residents",
            "household_id",
            "person",
            role_field="resident_role",
        )
    )
    tax_benefit_system = taxbenefitsystems.TaxBenefitSystem([person, household])

    class salary(variables.Variable):
        value_type = int
        entity = person
        definition_period = periods.YEAR

    class household_id(variables.Variable):
        value_type = int
        entity = person
        definition_period = periods.ETERNITY
        default_value = -1

    class rent(variables.Variable):
        value_type = int
        entity = household
        definition_period = periods.YEAR

    class resident_role(variables.Variable):
        value_type = str
        entity = person
        definition_period = periods.YEAR

    for variable in (salary, household_id, resident_role, rent):
        tax_benefit_system.add_variable(variable)

    result = SimulationBuilder().build_from_dict(
        tax_benefit_system,
        {
            "persons": {
                "a": {
                    "salary": {"2024": 10},
                    "household_id": {"ETERNITY": 0},
                    "resident_role": {"2024": "adult", "2025": "child"},
                },
                "b": {
                    "salary": {"2024": 20},
                    "household_id": {"ETERNITY": 0},
                    "resident_role": {"2024": "child", "2025": "adult"},
                },
                "c": {
                    "salary": {"2024": 5},
                    "household_id": {"ETERNITY": 1},
                    "resident_role": {"2024": "adult", "2025": "child"},
                },
                "outside": {"salary": {"2024": 99}},
            },
            "households": {
                "first": {"member": ["a", "b"]},
                "second": {"member": ["c", "outside"]},
                "empty": {"member": []},
            },
        },
    )
    return result


def test_one2many_numeric_aggregations(simulation):
    link = simulation.household.residents

    numpy.testing.assert_array_equal(link.sum("salary", "2024"), [30, 5, 0])
    numpy.testing.assert_array_equal(link.count("2024"), [2, 1, 0])
    numpy.testing.assert_array_equal(link.min("salary", "2024"), [10, 5, 0])
    numpy.testing.assert_array_equal(link.max("salary", "2024"), [20, 5, 0])
    numpy.testing.assert_array_equal(link.avg("salary", "2024"), [15, 5, 0])


def test_one2many_boolean_aggregations(simulation):
    link = simulation.household.residents

    numpy.testing.assert_array_equal(link.any("salary", "2024"), [True, True, False])
    numpy.testing.assert_array_equal(link.all("salary", "2024"), [True, True, True])


def test_one2many_rejects_variables_from_another_entity(simulation):
    with pytest.raises(LinkResolutionError, match="defined for 'households'"):
        simulation.household.residents.sum("rent", "2024")


def test_one2many_role_condition_and_nth(simulation):
    link = simulation.household.residents
    condition = numpy.array([True, False, False, True])

    numpy.testing.assert_array_equal(
        link.sum("salary", "2024", role="adult"),
        [10, 5, 0],
    )
    numpy.testing.assert_array_equal(
        link.sum("salary", "2024", condition=condition),
        [10, 0, 0],
    )
    numpy.testing.assert_array_equal(link.nth(1, "salary", "2024"), [20, 0, 0])

    with pytest.raises(ValueError, match="condition has shape"):
        link.count("2024", condition=[True])


def test_many2one_role_and_rank(simulation):
    link = simulation.persons.household

    numpy.testing.assert_array_equal(
        link.has_role("adult", "2024"),
        [True, False, True, False],
    )
    numpy.testing.assert_array_equal(link.rank("salary", "2024"), [0, 1, 0, 1])


def test_role_filters_use_the_operation_period(simulation):
    numpy.testing.assert_array_equal(
        simulation.persons.household.has_role("adult", "2025"),
        [False, True, False, False],
    )
    numpy.testing.assert_array_equal(
        simulation.household.residents.count("2025", role="adult"),
        [1, 0, 0],
    )


def test_links_validate_role_field_entity(simulation):
    simulation.persons.household.role_field = "rent"
    simulation.household.residents.role_field = "rent"

    with pytest.raises(LinkResolutionError, match="defined for 'households'"):
        simulation.persons.household.has_role("adult", "2024")
    with pytest.raises(LinkResolutionError, match="defined for 'households'"):
        simulation.household.residents.count("2024", role="adult")
