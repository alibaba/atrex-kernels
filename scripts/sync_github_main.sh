#!/usr/bin/env bash
set -euo pipefail

GITHUB_URL="${GITHUB_URL:-https://github.com/alibaba/atrex-kernels.git}"
TARGET_BRANCH="${TARGET_BRANCH:-dev}"
NO_PUSH="${NO_PUSH:-false}"

git fetch --no-tags origin "refs/heads/${TARGET_BRANCH}:refs/remotes/origin/${TARGET_BRANCH}"
git fetch --no-tags "$GITHUB_URL" "refs/heads/main:refs/remotes/github/main"

INTERNAL_SHA="$(git rev-parse "refs/remotes/origin/${TARGET_BRANCH}")"
EXTERNAL_SHA="$(git rev-parse refs/remotes/github/main)"

echo "internal ${TARGET_BRANCH}: ${INTERNAL_SHA}"
echo "external main:      ${EXTERNAL_SHA}"

if [[ "$INTERNAL_SHA" == "$EXTERNAL_SHA" ]]; then
    echo "Already synchronized"
    exit 0
fi

if ! git merge-base --is-ancestor "$INTERNAL_SHA" "$EXTERNAL_SHA"; then
    echo "Refusing to overwrite divergent internal history" >&2
    echo "Move internal-only changes to GitHub main before retrying" >&2
    exit 1
fi

if [[ "$NO_PUSH" == "true" ]]; then
    echo "NO_PUSH=true; fast-forward is valid but was not pushed"
    exit 0
fi

git push origin "${EXTERNAL_SHA}:refs/heads/${TARGET_BRANCH}"
echo "Synchronized internal ${TARGET_BRANCH} to GitHub main @ ${EXTERNAL_SHA}"
