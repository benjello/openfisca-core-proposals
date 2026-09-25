"""This module contains the _BuildDefaultSimulation class."""

from collections.abc import Mapping

import numpy
from numpy.typing import NDArray as Array
from typing_extensions import Self

from .simulation import Simulation
from .typing import Entity, Population, TaxBenefitSystem


class _BuildDefaultSimulation:
    """Build a default simulation.

    Args:
        tax_benefit_system(TaxBenefitSystem): The tax-benefit system.
        count(int): The number of persons.
        group_members: Group entity IDs, indexed by group entity key and member.

    Examples:
        >>> from openfisca_core import entities, taxbenefitsystems

        >>> role = {"key": "stray", "plural": "stray", "label": "", "doc": ""}
        >>> single_entity = entities.Entity("dog", "dogs", "", "")
        >>> group_entity = entities.GroupEntity("pack", "packs", "", "", [role])
        >>> test_entities = [single_entity, group_entity]
        >>> tax_benefit_system = taxbenefitsystems.TaxBenefitSystem(test_entities)
        >>> count = 1
        >>> builder = (
        ...     _BuildDefaultSimulation(tax_benefit_system, count)
        ...     .add_count()
        ...     .add_ids()
        ...     .add_members_entity_id()
        ...     .add_id_to_rownum()
        ... )

        >>> builder.count
        1

        >>> sorted(builder.populations.keys())
        ['dog', 'pack']

        >>> sorted(builder.simulation.populations.keys())
        ['dog', 'pack']

    """

    #: The number of Population.
    count: int

    #: Optional member assignments indexed by group entity key.
    group_members: Mapping[str, Array] | None

    #: The built populations.
    populations: dict[str, Population[Entity]]

    #: The built simulation.
    simulation: Simulation

    def __init__(
        self,
        tax_benefit_system: TaxBenefitSystem,
        count: int,
        group_members: Mapping[str, Array] | None = None,
    ) -> None:
        self.count = count
        self.populations = tax_benefit_system.instantiate_entities()
        self.simulation = Simulation(tax_benefit_system, self.populations)
        self.group_members = group_members

        if group_members is not None:
            group_keys = {
                population.entity.key
                for population in self.populations.values()
                if hasattr(population, "members_entity_id")
            }
            unknown_keys = set(group_members) - group_keys
            if unknown_keys:
                keys = ", ".join(repr(key) for key in sorted(unknown_keys, key=str))
                raise ValueError(f"Unknown group entity keys: {keys}")

    def add_count(self) -> Self:
        """Add the number of Population to the simulation.

        Returns:
            _BuildDefaultSimulation: The builder.

        Examples:
            >>> from openfisca_core import entities, taxbenefitsystems

            >>> role = {"key": "stray", "plural": "stray", "label": "", "doc": ""}
            >>> single_entity = entities.Entity("dog", "dogs", "", "")
            >>> group_entity = entities.GroupEntity("pack", "packs", "", "", [role])
            >>> test_entities = [single_entity, group_entity]
            >>> tax_benefit_system = taxbenefitsystems.TaxBenefitSystem(test_entities)
            >>> count = 2
            >>> builder = _BuildDefaultSimulation(tax_benefit_system, count)

            >>> builder.add_count()
            <..._BuildDefaultSimulation object at ...>

            >>> builder.populations["dog"].count
            2

            >>> builder.populations["pack"].count
            2

        """
        for population in self.populations.values():
            population.count = self.count

        return self

    def add_ids(self) -> Self:
        """Add the populations ids to the simulation.

        Returns:
            _BuildDefaultSimulation: The builder.

        Examples:
            >>> from openfisca_core import entities, taxbenefitsystems

            >>> role = {"key": "stray", "plural": "stray", "label": "", "doc": ""}
            >>> single_entity = entities.Entity("dog", "dogs", "", "")
            >>> group_entity = entities.GroupEntity("pack", "packs", "", "", [role])
            >>> test_entities = [single_entity, group_entity]
            >>> tax_benefit_system = taxbenefitsystems.TaxBenefitSystem(test_entities)
            >>> count = 2
            >>> builder = _BuildDefaultSimulation(tax_benefit_system, count)

            >>> builder.add_ids()
            <..._BuildDefaultSimulation object at ...>

            >>> builder.populations["dog"].ids
            array([0, 1])

            >>> builder.populations["pack"].ids
            array([0, 1])

        """
        for population in self.populations.values():
            population.ids = numpy.array(range(self.count))

        return self

    def add_id_to_rownum(self) -> Self:
        """Add an identity ID-to-row index to each population."""
        for population in self.populations.values():
            population._id_to_rownum = numpy.arange(
                population.count,
                dtype=numpy.intp,
            )

        return self

    def add_members_entity_id(self) -> Self:
        """Add the group entity ID of each member.

        By default, each member has its own group entity. ``group_members`` can
        specify another structure for individual group entities.

        Returns:
            _BuildDefaultSimulation: The builder.

        Examples:
            >>> from openfisca_core import entities, taxbenefitsystems

            >>> role = {"key": "stray", "plural": "stray", "label": "", "doc": ""}
            >>> single_entity = entities.Entity("dog", "dogs", "", "")
            >>> group_entity = entities.GroupEntity("pack", "packs", "", "", [role])
            >>> test_entities = [single_entity, group_entity]
            >>> tax_benefit_system = taxbenefitsystems.TaxBenefitSystem(test_entities)
            >>> count = 2
            >>> builder = _BuildDefaultSimulation(tax_benefit_system, count)

            >>> builder.add_members_entity_id()
            <..._BuildDefaultSimulation object at ...>

            >>> population = builder.populations["pack"]

            >>> hasattr(population, "members_entity_id")
            True

            >>> population.members_entity_id
            array([0, 1])

        """
        for population in self.populations.values():
            if hasattr(population, "members_entity_id"):
                key = population.entity.key
                if self.group_members is not None and key in self.group_members:
                    population.set_members_entity_id(self.group_members[key])
                    population.ids = numpy.arange(population.count)
                else:
                    population.members_entity_id = numpy.array(range(self.count))

        return self
