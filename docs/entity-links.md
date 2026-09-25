# Entity links

Entity links provide named relationships between populations without replacing
OpenFisca's existing group projectors.

## Explicit links

Declare links on entities before constructing the tax-benefit system:

```python
from openfisca_core.links import Many2OneLink, One2ManyLink

person.add_link(Many2OneLink("mother", "mother_id", "person"))
employer.add_link(One2ManyLink("employees", "employer_id", "person"))
```

The link field stores target IDs for `Many2OneLink`, and source IDs on each
target member for `One2ManyLink`. Invalid or missing IDs produce the target
variable's default value. Population IDs and dynamic `_id_to_rownum` mappings
are supported.

```python
mother_age = simulation.persons.mother("age", period)
payroll = simulation.employer.employees.sum("salary", period)
```

One-to-many links provide `sum`, `count`, `avg`, `min`, `max`, `any`, `all`,
`nth`, and `get_by_role`. Aggregations accept optional `role` and `condition`
filters. Many-to-one links provide `has_role`, `get_by_role`, and group `rank`.

`Many2OneLink.get` and its callable shorthand accept `options=[ADD]` or
`options=[DIVIDE]`. Many-to-one links can be chained, for example
`simulation.persons.mother.mother("age", period)`.

## Group links

Every group population receives implicit links in both directions. Existing
projector syntax remains valid:

```python
rent_per_person = person.household("rent", period)
income_per_person = person.household.sum(person("salary", period))
income_per_household = household.persons.sum("salary", period)
```

Projection is based on the operation's semantics, not array length. Calls on a
group projector or projectable group method return entity-sized data and are
projected to members. Calls through `.members` already return member-sized data
and are not projected, including when entity and member counts happen to match.
