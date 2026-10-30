#!/usr/bin/env bash
# Check whether the CI image for a Dockerfile hash is published on GHCR.
# Called by the ensure-image gate jobs of the self-hosted build workflows
# (see docs/developer.rst, "Self-hosted end-to-end workflows").
#
# Usage: check_ci_image.sh <dockerfile-hash>
# Requires: GITHUB_TOKEN with packages:read on the repository.
#
# The check goes through the GitHub REST API (api.github.com) rather than a
# raw GHCR token exchange: the API authenticates with GITHUB_TOKEN directly
# and is the supported path for listing a private container package's tags.
# (The raw `ghcr.io/token` exchange has proven unreliable for the workflow
# token, so it is avoided.)
#
# Exit codes:
#   0  the image is published
#   1  the image is not published yet (retryable)
#   2  usage, authentication, or network error (not retryable)
set -euo pipefail

ORG="PySymGym"
PACKAGE="pysymgym"
API="https://api.github.com/orgs/${ORG}/packages/container/${PACKAGE}/versions"

if [ $# -ne 1 ] || [ -z "${1:-}" ]; then
    echo "::error::usage: check_ci_image.sh <dockerfile-hash>" >&2
    exit 2
fi
tag="$1"

if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "::error::GITHUB_TOKEN is not set; cannot authenticate to the GitHub API" >&2
    exit 2
fi

# Page through the package's versions (one per distinct image) and look for
# the tag. The tag is a 40-hex sha1, which cannot collide with a digest
# (`sha256:<64-hex>`), so a quoted substring match is unambiguous. An empty
# page marks the end of the list.
page=1
while :; do
    response=$(curl -s -w $'\n%{http_code}' \
        -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github+json" \
        "${API}?per_page=100&page=${page}") || {
        echo "::error::GitHub API request failed (network error) for ${API}" >&2
        exit 2
    }
    code=${response##*$'\n'}
    body=${response%$'\n'*}

    case "$code" in
        200)
            if grep -q "\"${tag}\"" <<<"$body"; then
                echo "image ghcr.io/${ORG}/${PACKAGE}:${tag} is published"
                exit 0
            fi
            if ! grep -q '"version":' <<<"$body"; then
                # Empty page: the whole version list was scanned.
                echo "image ghcr.io/${ORG}/${PACKAGE}:${tag} is not published yet"
                exit 1
            fi
            page=$(( page + 1 ))
            ;;
        404)
            # The package does not exist yet, so the image is not published.
            echo "image ghcr.io/${ORG}/${PACKAGE}:${tag} is not published yet (package not found)"
            exit 1
            ;;
        401|403)
            echo "::error::GitHub API auth failed (HTTP ${code}); GITHUB_TOKEN lacks packages:read on ${ORG}/${PACKAGE}" >&2
            echo "  response: $(head -c 300 <<<"$body")" >&2
            exit 2
            ;;
        *)
            echo "::error::GitHub API returned HTTP ${code} for ${API}" >&2
            echo "  response: $(head -c 300 <<<"$body")" >&2
            exit 2
            ;;
    esac
done
