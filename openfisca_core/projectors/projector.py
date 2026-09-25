import inspect

from openfisca_core.projectors import helpers


class Projector:
    reference_entity = None
    parent = None

    def __getattr__(self, attribute):
        projector = helpers.get_projector_from_shortcut(
            self.reference_entity,
            attribute,
            parent=self,
        )
        if projector:
            return projector

        reference_attr = getattr(self.reference_entity, attribute)
        if not hasattr(reference_attr, "projectable"):
            return reference_attr

        def projector_function(*args, **kwargs):
            result = reference_attr(*args, **kwargs)
            period = (
                inspect.signature(reference_attr)
                .bind(*args, **kwargs)
                .arguments.get("period")
            )
            return self.transform_and_bubble_up(result, period)

        return projector_function

    def __call__(self, *args, **kwargs):
        result = self.reference_entity(*args, **kwargs)
        period = kwargs.get("period", args[1] if len(args) > 1 else None)
        return self.transform_and_bubble_up(result, period)

    def transform_and_bubble_up(self, result, period=None):
        transformed_result = self.transform(result, period)
        if self.parent is None:
            return transformed_result
        return self.parent.transform_and_bubble_up(transformed_result, period)

    def transform(self, result, period=None):
        return NotImplementedError()
