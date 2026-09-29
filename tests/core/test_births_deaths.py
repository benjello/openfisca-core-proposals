import numpy
import pytest

from openfisca_core import entities, periods, populations, taxbenefitsystems, variables
from openfisca_core.simulations import SimulationBuilder


def build_simulation():
    person = entities.SingleEntity("person", "persons", "A person", "")
    household = entities.GroupEntity(
        "household",
        "households",
        "A household",
        "",
        roles=[
            {"key": "first_parent", "max": 1},
            {"key": "child", "plural": "children"},
        ],
    )
    tax_benefit_system = taxbenefitsystems.TaxBenefitSystem([person, household])

    class salary(variables.Variable):
        value_type = float
        entity = person
        definition_period = periods.DateUnit.MONTH

    class previous_salary(variables.Variable):
        value_type = float
        entity = person
        definition_period = periods.DateUnit.MONTH

        def formula(population, period):
            return population("salary", period.last_month)

    class two_month_salary(variables.Variable):
        value_type = float
        entity = person
        definition_period = periods.DateUnit.MONTH

        def formula(population, _period):
            return population(
                "salary",
                "month:2024-01:2",
                options=[populations.ADD],
            )

    class annual_income(variables.Variable):
        value_type = float
        entity = person
        definition_period = periods.DateUnit.YEAR

    class household_income(variables.Variable):
        value_type = float
        entity = household
        definition_period = periods.DateUnit.MONTH

        def formula(population, period):
            salary = population.members("salary", period)
            return population.sum(salary, period=period)

    class monthly_annual_income(variables.Variable):
        value_type = float
        entity = person
        definition_period = periods.DateUnit.MONTH

        def formula(population, period):
            return population("annual_income", period, options=[populations.DIVIDE])

    tax_benefit_system.add_variables(
        salary,
        previous_salary,
        two_month_salary,
        annual_income,
        monthly_annual_income,
        household_income,
    )
    simulation = SimulationBuilder.build_default_simulation(
        tax_benefit_system,
        count=2,
    )
    simulation.household.ids = numpy.array(["household"])
    simulation.household.set_members_entity_id([0, 0])
    first_parent, child = household.flattened_roles
    simulation.household.members_role = [first_parent, child]
    simulation.persons.activate_dynamic_mode("2024-01", [10, 20])
    return simulation, first_parent, child


def apply_february_lifecycle(simulation) -> None:
    simulation.persons.register_deaths("2024-02", [10])
    simulation.persons.register_births("2024-02", [30])


def test_births_and_deaths_are_explicit_period_operations() -> None:
    simulation, _, _ = build_simulation()
    apply_february_lifecycle(simulation)

    assert simulation.persons.get_count_for_period("2024-01") == 2
    assert simulation.persons.get_count_for_period("2024-03") == 2
    numpy.testing.assert_array_equal(
        simulation.persons._get_alive_ids_for_period("2024-03"),
        [20, 30],
    )


def test_cross_period_formula_remaps_by_permanent_id_and_invalidates_cache() -> None:
    simulation, _, _ = build_simulation()
    simulation.set_input("salary", "2024-01", [100, 200])
    numpy.testing.assert_array_equal(
        simulation.calculate("previous_salary", "2024-02"),
        [100, 200],
    )

    apply_february_lifecycle(simulation)

    numpy.testing.assert_array_equal(
        simulation.calculate("previous_salary", "2024-02"),
        [200, 0],
    )


def test_add_and_divide_remap_across_periods() -> None:
    simulation, _, _ = build_simulation()
    simulation.set_input("salary", "2024-01", [100, 200])
    simulation.set_input("annual_income", "2024", [1200, 2400])
    apply_february_lifecycle(simulation)
    simulation.set_input("salary", "2024-02", [250, 300])

    numpy.testing.assert_array_equal(
        simulation.calculate("two_month_salary", "2024-02"),
        [450, 300],
    )
    numpy.testing.assert_array_equal(
        simulation.calculate("monthly_annual_income", "2024-02"),
        [200, 0],
    )


def test_role_assignment_remaps_previous_roles_by_permanent_id() -> None:
    simulation, first_parent, child = build_simulation()
    apply_february_lifecycle(simulation)

    roles = simulation.household.auto_assign_roles_for_period(
        "2024-02",
        [0, 0],
    )

    assert roles.tolist() == [child, first_parent]


def test_lifecycle_invalidates_group_caches_before_explicit_membership_update() -> None:
    simulation, first_parent, child = build_simulation()
    simulation.set_input("salary", "2024-02", [100, 200])
    numpy.testing.assert_array_equal(
        simulation.calculate("household_income", "2024-02"),
        [300],
    )

    apply_february_lifecycle(simulation)
    roles = simulation.household.auto_assign_roles_for_period("2024-02", [0, 0])
    simulation.household.set_members_for_period(
        "2024-02",
        [0, 0],
        members_role=roles,
    )
    simulation.set_input("salary", "2024-02", [250, 300])

    assert roles.tolist() == [child, first_parent]
    numpy.testing.assert_array_equal(
        simulation.calculate("household_income", "2024-02"),
        [550],
    )


@pytest.mark.parametrize(
    ("method", "ids", "message"),
    [
        ("register_births", [], "cannot be empty"),
        ("register_births", [-1], "non-negative"),
        ("register_births", [30, 30], "duplicates"),
        ("register_births", [10], "already known"),
        ("register_deaths", [], "cannot be empty"),
        ("register_deaths", [-1], "non-negative"),
        ("register_deaths", [10, 10], "duplicates"),
        ("register_deaths", [999], "Unknown"),
    ],
)
def test_lifecycle_rejects_invalid_ids(method, ids, message) -> None:
    simulation, _, _ = build_simulation()

    with pytest.raises(ValueError, match=message):
        getattr(simulation.persons, method)("2024-02", ids)


def test_death_of_absent_id_is_rejected() -> None:
    simulation, _, _ = build_simulation()
    simulation.persons.register_deaths("2024-02", [10])

    with pytest.raises(ValueError, match="not alive"):
        simulation.persons.register_deaths("2024-03", [10])


def test_lifecycle_events_before_activation_are_rejected_without_mutation() -> None:
    simulation, _, _ = build_simulation()

    with pytest.raises(ValueError, match="must be registered in order"):
        simulation.persons.register_births("2023-12", [30])

    numpy.testing.assert_array_equal(simulation.persons._permanent_ids, [10, 20])
    numpy.testing.assert_array_equal(
        simulation.persons._get_alive_ids_for_period("2024-01"),
        [10, 20],
    )


def test_out_of_order_lifecycle_events_are_rejected() -> None:
    simulation, _, _ = build_simulation()
    simulation.persons.register_births("2024-03", [30])

    with pytest.raises(ValueError, match="must be registered in order"):
        simulation.persons.register_deaths("2024-02", [10])


def test_snapshot_period_captures_the_final_ordered_state() -> None:
    simulation, _, _ = build_simulation()
    simulation.persons.register_deaths("2024-02", [10])

    simulation.persons.snapshot_period("2024-03")

    assert simulation.persons.get_count_for_period("2024-03") == 1
    numpy.testing.assert_array_equal(
        simulation.persons._get_alive_ids_for_period("2024-03"),
        [20],
    )


def test_lifecycle_requires_explicit_activation() -> None:
    simulation, _, _ = build_simulation()
    other = SimulationBuilder.build_default_simulation(
        simulation.tax_benefit_system,
        count=2,
    )

    with pytest.raises(ValueError, match="activate_dynamic_mode"):
        other.persons.register_births("2024-02", [2])
