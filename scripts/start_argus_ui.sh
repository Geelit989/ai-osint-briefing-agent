#!/bin/bash
# Launch only the two local workspace processes; Ollama is managed separately.
set -euo pipefail

ARGUS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ARGUS_ROOT"
export PYTHONPATH="$ARGUS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [ -d "$ARGUS_ROOT/.local/node/bin" ]; then
  export PATH="$ARGUS_ROOT/.local/node/bin:$PATH"
fi
ARGUS_PYTHON="${ARGUS_PYTHON:-$ARGUS_ROOT/.venv/bin/python}"
export ARGUS_API_PORT="${ARGUS_API_PORT:-8000}"
export ARGUS_UI_PORT="${ARGUS_UI_PORT:-3000}"

"$ARGUS_PYTHON" - <<'PY'
import importlib.util
import os
import socket
import sys

if sys.version_info[:2] != (3, 14):
    raise SystemExit('Use the existing Python 3.14 environment (set ARGUS_PYTHON if needed).')
for module in ('fastapi', 'uvicorn'):
    if importlib.util.find_spec(module) is None:
        raise SystemExit('Install adapter dependencies: .venv/bin/python -m pip install -r requirements-ui.txt')
for name in ('ARGUS_API_PORT', 'ARGUS_UI_PORT'):
    port = int(os.environ[name])
    if not 1 <= port <= 65535:
        raise SystemExit(f'{name} must be a valid local port.')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            raise SystemExit(f'Port {port} is already in use. Stop the existing service or choose another local port.')
PY

if ! command -v node >/dev/null || ! command -v npm >/dev/null; then
  echo "Install Node.js 24 LTS, then run: npm --prefix web ci" >&2
  exit 1
fi
if [ ! -d web/node_modules ]; then
  echo "Install frontend dependencies first: npm --prefix web ci" >&2
  exit 1
fi
ARGUS_NEXT_MODE="start"
if [ "${ARGUS_UI_DEV:-0}" = "1" ]; then
  ARGUS_NEXT_MODE="dev"
elif [ ! -f web/.next/BUILD_ID ]; then
  npm --prefix web run build
fi

ARGUS_API_PID=""
ARGUS_WEB_PID=""
argus_cleanup() {
  trap - EXIT INT TERM
  [ -z "$ARGUS_API_PID" ] || kill "$ARGUS_API_PID" 2>/dev/null || true
  [ -z "$ARGUS_WEB_PID" ] || kill "$ARGUS_WEB_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap argus_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$ARGUS_PYTHON" -m uvicorn osint_agent.workspace.api:app --host 127.0.0.1 --port "$ARGUS_API_PORT" &
ARGUS_API_PID=$!
node web/node_modules/next/dist/bin/next "$ARGUS_NEXT_MODE" web --hostname 127.0.0.1 --port "$ARGUS_UI_PORT" &
ARGUS_WEB_PID=$!
echo "ARGUS workspace: http://127.0.0.1:$ARGUS_UI_PORT"
echo "API documentation: http://127.0.0.1:$ARGUS_API_PORT/docs"
echo "Press Ctrl-C to stop both workspace services."
while kill -0 "$ARGUS_API_PID" 2>/dev/null && kill -0 "$ARGUS_WEB_PID" 2>/dev/null; do
  sleep 1
done
echo "A workspace service exited; stopping its companion." >&2
exit 1
