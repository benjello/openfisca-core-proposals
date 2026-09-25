from .projector import Projector


class EntityToPersonProjector(Projector):
    """For instance person.family."""

    def __init__(self, entity, parent=None) -> None:
        self.reference_entity = entity
        self.parent = parent

    def transform(self, result, period=None):
        return self.reference_entity.project(result, period=period)
