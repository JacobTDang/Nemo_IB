#!/bin/sh
# Create the SQLite schema before the server starts.
#
# A few tools on the modelling server read book state -- theses,
# thesis_evolution -- which does not exist on a data-source host. Without the
# tables they fail with "no such table"; with them they return an empty result,
# which is the truthful answer for a box that holds no positions.
#
# No error suppression: if the schema cannot be created, the container must
# fail rather than serve tools that will break one call later.
set -e
python -c "from state.schema import init_schema; init_schema()"
exec "$@"
