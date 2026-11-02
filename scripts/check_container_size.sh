#!/usr/bin/env bash
set -euo pipefail
IMAGE_NAME="${1:-telco-churn-api:1.0}"; LIMIT_MB="${2:-600}"
SIZE_BYTES="$(docker image inspect "$IMAGE_NAME" --format='{{.Size}}')"; SIZE_MB="$((SIZE_BYTES / 1024 / 1024))"
printf 'Image: %s\nSize: %s MB\nLimit: %s MB\n' "$IMAGE_NAME" "$SIZE_MB" "$LIMIT_MB"
if (( SIZE_MB >= LIMIT_MB )); then echo "FAIL: image is not below the configured size limit." >&2; exit 1; fi
echo "PASS: image is below the configured size limit."
