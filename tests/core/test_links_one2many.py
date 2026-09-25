import numpy
import pytest

from openfisca_core import entities, periods, taxbenefitsystems, variables
from openfisca_core.links import One2ManyLink
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
    household.add_link(One2ManyLink("residents", "household_id", "person"))
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

    for variable in (salary, household_id):
        tax_benefit_system.add_variable(variable)

    result = SimulationBuilder().build_from_dict(
        tax_benefit_system,
        {
            "persons": {
                "a": {"salary": {"2024": 10}, "household_id": {"ETERNITY": 0}},
                "b": {"salary": {"2024": 20}, "household_id": {"ETERNITY": 0}},
                "c": {"salary": {"2024": 5}, "household_id": {"ETERNITY": 1}},
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
