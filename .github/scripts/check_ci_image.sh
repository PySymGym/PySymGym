#!/usr/bin/env bash
# Check whether the CI image for a Dockerfile hash is published on GHCR.
# Called by the ensure-image gate jobs of the self-hosted build workflows
# (see docs/developer.rst, "Self-hosted end-to-end workflows").
#
# Usage: check_ci_image.sh <dockerfile-hash>
# Requires: GITHUB_TOKEN with packages:read on the repository. The package
# is private, so the token is exchanged for a scoped GHCR pull token first.
#
# Exit codes:
#   0  the image is published
#   1  the image is not published yet (retryable)
#   2  usage, authentication, or network error (not retryable)
set -euo pipefail

PACKAGE="pysymgym/pysymgym"

if [ $# -ne 1 ] || [ -z "${1:-}" ]; then
    echo "::error::usage: check_ci_image.sh <dockerfile-hash>" >&2
    exit 2
fi
tag="$1"

if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "::error::GITHUB_TOKEN is not set; cannot authenticate to GHCR" >&2
    exit 2
fi

token_response=$(curl -sf -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    "https://ghcr.io/token?scope=repository:${PACKAGE}:pull") || {
    echo "::error::GHCR token exchange failed (network error, or GITHUB_TOKEN lacks packages:read on the repository)" >&2
    exit 2
}
pull_token=$(sed -n 's/.*"token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' <<<"$token_response")
if [ -z "$pull_token" ]; then
    echo "::error::GHCR token exchange returned no token: ${token_response}" >&2
    exit 2
fi

status=$(curl -s -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer ${pull_token}" \
    -H "Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.v2+json" \
    "https://ghcr.io/v2/${PACKAGE}/manifests/${tag}") || {
    echo "::error::GHCR manifest lookup failed (network error)" >&2
    exit 2
}

case "$status" in
    200)
        echo "image ghcr.io/${PACKAGE}:${tag} is published"
        exit 0
        ;;
    404)
        echo "image ghcr.io/${PACKAGE}:${tag} is not published yet"
        exit 1
        ;;
    *)
        echo "::error::GHCR manifest lookup returned HTTP ${status} for ${PACKAGE}:${tag}" >&2
        exit 2
        ;;
esac
