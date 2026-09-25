import numpy
import pytest

from openfisca_core import (
    entities,
    indexed_enums,
    periods,
    taxbenefitsystems,
    variables,
)
from openfisca_core.links import Many2OneLink
from openfisca_core.links.link import LinkResolutionError
from openfisca_core.populations import ADD, DIVIDE
from openfisca_core.populations._errors import (
    IncompatibleOptionsError,
    InvalidOptionError,
)
from openfisca_core.simulations import SimulationBuilder


class HousingStatus(indexed_enums.Enum):
    tenant = "Tenant"
    owner = "Owner"


@pytest.fixture
def simulation():
    person = entities.SingleEntity("person", "persons", "", "")
    person.add_link(Many2OneLink("mother", "mother_id", "person"))
    tax_benefit_system = taxbenefitsystems.TaxBenefitSystem([person])

    class age(variables.Variable):
        value_type = int
        entity = person
        definition_period = periods.YEAR

    class mother_id(variables.Variable):
        value_type = int
        entity = person
        definition_period = periods.ETERNITY
        default_value = -1

    class housing_status(variables.Variable):
        value_type = indexed_enums.Enum
        possible_values = HousingStatus
        default_value = HousingStatus.tenant
        entity = person
        definition_period = periods.YEAR

    for variable in (age, mother_id, housing_status):
        tax_benefit_system.add_variable(variable)

    result = SimulationBuilder().build_default_simulation(tax_benefit_system, count=3)
    result.set_input("mother_id", "2024", [-1, 0, 1])
    result.set_input("age", "2024", [45, 25, 5])
    result.set_input("housing_status", "2024", ["owner", "tenant", "tenant"])
    return result


def test_many2one_is_bound_and_maps_missing_ids(simulation):
    link = simulation.persons.mother

    assert link.is_resolved
    numpy.testing.assert_array_equal(link("age", "2024"), [0, 45, 25])


def test_many2one_uses_non_identity_id_mapping(simulation):
    simulation.persons._id_to_rownum = numpy.array([2, 0, 1], dtype=numpy.intp)

    numpy.testing.assert_array_equal(
        simulation.persons.mother("age", "2024"), [0, 5, 45]
    )


def test_many2one_preserves_enum_array(simulation):
    result = simulation.persons.mother("housing_status", "2024")

    assert isinstance(result, indexed_enums.EnumArray)
    numpy.testing.assert_array_equal(
        result.decode_to_str(), ["tenant", "owner", "tenant"]
    )


def test_many2one_contextualizes_link_field_errors(simulation):
    simulation.persons.mother.link_field = "unknown_id"

    with pytest.raises(LinkResolutionError, match="Link 'mother'.*'unknown_id'"):
        simulation.persons.mother("age", "2024")


def test_many2one_rejects_variables_from_another_entity():
    person = entities.SingleEntity("person", "persons", "", "")
    household = entities.GroupEntity(
        "household", "households", "", "", roles=[{"key": "member"}]
    )
    person.add_link(Many2OneLink("mother", "mother_id", "person"))
    system = taxbenefitsystems.TaxBenefitSystem([person, household])

    class mother_id(variables.Variable):
        value_type = int
        entity = person
        definition_period = periods.ETERNITY
        default_value = -1

    class rent(variables.Variable):
        value_type = int
        entity = household
        definition_period = periods.YEAR

    system.add_variables(mother_id, rent)
    simulation = SimulationBuilder().build_default_simulation(system, count=2)

    with pytest.raises(LinkResolutionError, match="defined for 'households'"):
        simulation.persons.mother("rent", "2024")


def test_many2one_rank_rejects_non_group_target(simulation):
    with pytest.raises(ValueError, match="rank requires its target to group"):
        simulation.persons.mother.rank("age", "2024")


def test_many2one_supports_add_and_divide(simulation):
    link = simulation.persons.mother

    numpy.testing.assert_array_equal(
        link("age", "2024", options=[ADD]),
        [0, 45, 25],
    )
    numpy.testing.assert_allclose(
        link("age", "2024-01", options=[DIVIDE]),
        [0, 45 / 12, 25 / 12],
    )


def test_many2one_rejects_invalid_options(simulation):
    link = simulation.persons.mother

    with pytest.raises(IncompatibleOptionsError):
        link("age", "2024", options=[ADD, DIVIDE])
    with pytest.raises(InvalidOptionError):
        link("age", "2024", options=["INVALID"])
