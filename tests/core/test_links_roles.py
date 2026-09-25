import numpy

from openfisca_core import entities
from openfisca_core.links.link import _role_matches


def test_role_matching_accepts_role_objects_and_raw_keys():
    group = entities.GroupEntity(
        "household",
        "households",
        "",
        "",
        roles=[{"key": "adult"}, {"key": "child"}],
    )
    roles = numpy.array([group.ADULT, "child"], dtype=object)

    numpy.testing.assert_array_equal(_role_matches(roles, "adult"), [True, False])
    numpy.testing.assert_array_equal(_role_matches(roles, group.CHILD), [False, True])
