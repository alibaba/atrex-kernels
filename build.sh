#!/usr/bin/env bash
set -euo pipefail

ATREX_REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ATREX_REPOSITORY"

PYTHON_EXECUTABLE="${PYTHON:-python3}"
DIST_DIRECTORY="$ATREX_REPOSITORY/dist"

git submodule update --init --recursive -- third_party

"$PYTHON_EXECUTABLE" -m build --wheel --no-isolation

wheel_path=""
for candidate in "$DIST_DIRECTORY"/atrex-*.whl; do
    [[ -f "$candidate" ]] || continue
    if [[ -z "$wheel_path" || "$candidate" -nt "$wheel_path" ]]; then
        wheel_path="$candidate"
    fi
done

if [[ -z "$wheel_path" ]]; then
    echo "ATREX wheel was not produced under $DIST_DIRECTORY" >&2
    exit 1
fi

# Resolve and install the wheel's declared runtime dependencies without forcing
# an expensive reinstall of an already-satisfied GPU dependency stack.
"$PYTHON_EXECUTABLE" -m pip install "$wheel_path"
# The project version is intentionally stable during local development, so the
# first command may consider ATREX itself already satisfied. Replace only the
# freshly built project wheel after dependency resolution has completed.
"$PYTHON_EXECUTABLE" -m pip install "$wheel_path" --force-reinstall --no-deps

echo "Built and installed $wheel_path"
