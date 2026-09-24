#!/usr/bin/env bash
set -euo pipefail

: "${WORKER_USER:?请在仓库 Variables 配置 WORKER_USER}"
: "${WORKER_KEY:?请在仓库 Secrets 配置 WORKER_KEY}"

url_encode() {
    V="$1" python3 -c 'import os, urllib.parse; print(urllib.parse.quote(os.environ["V"], safe=""))'
}

USER_ENCODED="$(url_encode "$WORKER_USER")"
KEY_ENCODED="$(url_encode "$WORKER_KEY")"

git config --global credential.helper store
umask 077
{
    printf 'https://%s:%s@code.alibaba-inc.com\n' "$USER_ENCODED" "$KEY_ENCODED"
    printf 'https://%s:%s@gitlab.alibaba-inc.com\n' "$USER_ENCODED" "$KEY_ENCODED"
} > "$HOME/.git-credentials"
chmod 600 "$HOME/.git-credentials"

git config --global --unset-all url."https://code.alibaba-inc.com/".insteadOf 2>/dev/null || true
for pattern in \
        "git@code.alibaba-inc.com:" \
        "git@gitlab.alibaba-inc.com:" \
        "https://gitlab.alibaba-inc.com/"; do
    git config --global --add url."https://code.alibaba-inc.com/".insteadOf "$pattern"
done
