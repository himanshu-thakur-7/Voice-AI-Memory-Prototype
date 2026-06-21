#!/usr/bin/env bash
# Vercel build step — substitutes the BACKEND_URL env var into the static HTML so the
# Vercel-hosted page can call the Render-hosted backend. Single source of truth lives
# at web/static/index.html; this script copies + rewrites into frontend/dist/.
set -euo pipefail

DEST_DIR="frontend/dist"
SRC="web/static/index.html"

mkdir -p "$DEST_DIR"

if [[ -z "${BACKEND_URL:-}" ]]; then
  echo "⚠️  BACKEND_URL env var is unset — frontend will use relative paths"
  echo "    and only work when the backend is co-served. Set BACKEND_URL on the"
  echo "    Vercel project (e.g. https://your-backend.onrender.com) to fix."
fi

# Strip a trailing slash on BACKEND_URL so concatenating /api/... doesn't double-slash.
SANITIZED="${BACKEND_URL%/}"

# Substitute the placeholder. Use a delimiter unlikely to appear in URLs (|) and escape
# any | in the URL itself (paranoid but cheap).
ESCAPED="${SANITIZED//|/\\|}"
sed "s|__API_BASE_URL__|${ESCAPED}|g" "$SRC" > "$DEST_DIR/index.html"

echo "✅ Built $DEST_DIR/index.html with BACKEND_URL=${SANITIZED:-(empty)}"
