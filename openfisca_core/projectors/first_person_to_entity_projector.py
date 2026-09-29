from .projector import Projector


class FirstPersonToEntityProjector(Projector):
    """For instance famille.first_person."""

    def __init__(self, entity, parent=None) -> None:
        self.target_entity = entity
        self.reference_entity = entity.members
        self.parent = parent

    def _transform(self, result, period=None):
        return self.target_entity.value_from_first_person(result, period=period)

    def transform(self, result):
        return self._transform(result)
