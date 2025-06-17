#!/usr/bin/env bash
set -euo pipefail
# Always run a *login* shell so ~/.aws + profiles are loaded
bash -l -c "$*"