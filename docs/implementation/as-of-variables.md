# Stateful variables with `as_of`

An `as_of` variable stores a vector state that remains valid until a later
state or sparse patch changes it. This is useful for statuses, counters, and
other stocks whose values do not reset every period.

## Declaration

```python
class employment_status(Variable):
    value_type = int
    entity = Person
    definition_period = MONTH
    as_of = "start"
    snapshot_count = 3
```

`as_of="start"` reads the state at the requested period's first instant.
`as_of="end"` reads it at the last instant. `as_of=True` is an alias for
`"start"`. A variable cannot combine `as_of` with a `set_input` period helper.

The holder stores one immutable dense base and sorted sparse patches containing
only changed indices and values. Reconstructed arrays are immutable. A bounded
FIFO cache keeps dense snapshots for repeated and forward reads;
`snapshot_count` controls its size and defaults to 3.

## Inputs

Use `set_input` to establish the base or replace a complete state:

```python
simulation.set_input("employment_status", "2025-01", [1, 1, 2])
```

When the changed indices are already known, avoid constructing a complete
input vector:

```python
holder = simulation.get_holder("employment_status")
holder.set_input_sparse("2025-02", [0, 2], [2, 1])
```

`set_input_sparse` requires an existing base. Indices must be a unique,
one-dimensional integer array within population bounds. A scalar value is
broadcast to all selected indices.

## Stateful formulas

An `initial_formula` can establish the first complete state. A
`transition_formula` returns `(selector, values)` and is applied at most once
per period. The selector can be integer indices or a boolean population mask.

```python
class employment_status(Variable):
    value_type = int
    entity = Person
    definition_period = MONTH
    as_of = "start"

    def initial_formula(person, period):
        return person.filled_array(1)

    def transition_formula(person, period):
        previous = person("employment_status", period.last_month)
        changes = person("starts_job", period)
        return changes, numpy.where(changes, 2, previous)[changes]
```

The initial formula runs before any transition when no state exists. Explicit
inputs take precedence at their period. Reading the same variable at an earlier
period from a transition is supported; an exact variable-period cycle is
stopped and leaves the previous state unchanged. Regular `formula` and
`transition_formula` declarations are mutually exclusive.

Both formula families support dated variants such as
`initial_formula_2025_01` and `transition_formula_2025_01`.

With full tracing enabled, stateful nodes expose `formula_type` as `"initial"`
or `"transition"`. Pass `show_formula_type=True` to computation-log methods to
include the tag in text output.

## Operational notes

- Sparse state currently lives in memory and does not use `OnDiskStorage`.
- Retroactive writes invalidate snapshots at and after the changed instant.
- Backward reads remain correct but may reconstruct from the base.
- Returned arrays are read-only; copy one before modifying it.

Run focused tests with:

```console
python -m pytest tests/core/test_asof_variable.py \
  tests/core/test_initial_formula.py tests/core/test_transition_formula.py
```

Run the source benchmarks explicitly with:

```console
python -m pytest benchmarks/test_bench_asof.py --benchmark-sort=name
```
