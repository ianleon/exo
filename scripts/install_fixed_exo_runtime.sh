#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_PATH="${EXO_APP_PATH:-/Applications/EXO.app}"
RESOURCES_DIR="$APP_PATH/Contents/Resources"
RUNTIME_DIR="$RESOURCES_DIR/exo"
SOURCE_RUNTIME_MARKER="$RUNTIME_DIR/.source-runtime"
UV_BIN="${UV_BIN:-}"
BACKUP_ROOT="$REPO_ROOT/.install-backups"

log() {
  printf '[install-fixed-exo] %s\n' "$*" >&2
}

die() {
  printf '[install-fixed-exo] ERROR: %s\n' "$*" >&2
  exit 1
}

find_uv() {
  if [[ -n "$UV_BIN" && -x "$UV_BIN" ]]; then
    printf '%s\n' "$UV_BIN"
    return
  fi
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return
  fi
  if [[ -x /opt/homebrew/bin/uv ]]; then
    printf '%s\n' /opt/homebrew/bin/uv
    return
  fi
  if command -v brew >/dev/null 2>&1; then
    log "Installing uv with Homebrew"
    brew install uv
    command -v uv
    return
  fi
  die "uv is not installed and Homebrew is unavailable"
}

stop_exo() {
  log "Stopping existing EXO processes"
  osascript -e 'tell application "EXO" to quit' >/dev/null 2>&1 || true
  sleep 2
  pkill -f "$APP_PATH/Contents/MacOS/EXO" >/dev/null 2>&1 || true
  pkill -x EXO >/dev/null 2>&1 || true
  pkill -f "$APP_PATH/Contents/Resources/exo/exo" >/dev/null 2>&1 || true
  pkill -f "$APP_PATH/Contents/Resources/exo/_internal/macmon" >/dev/null 2>&1 || true
  sleep 2
}

prepare_environment() {
  local uv="$1"

  log "Preparing Python environment"
  "$uv" sync --group dev

  log "Installing runtime MLX dependencies from wheels/source artifacts"
  "$uv" pip install 'mlx==0.31.2'
  "$uv" pip install --no-deps \
    'mlx-lm @ git+https://github.com/rltakashige/mlx-lm@6a3df6cd6b00a347ee40f12d97a182aaf86ea599' \
    'mflux @ git+https://github.com/evanev7/mflux@0fdd4cca9468dd92d8c2511c88031e937655eb01' \
    'mlx-vlm>=0.3.11' \
    'torch==2.10.0' \
    'torchvision==0.25.0' \
    'torchaudio==2.10.0'
  "$uv" pip install \
    sympy \
    pillow \
    sentencepiece \
    networkx \
    matplotlib \
    pandas \
    pyarrow \
    hf-transfer \
    'protobuf>=3.20.3' \
    'safetensors>=0.4.3' \
    'accelerate>=0.26.0'

  log "Building dashboard assets"
  (cd "$REPO_ROOT/dashboard" && npm install && npm run build)
}

verify_source() {
  local uv="$1"

  log "Running focused checks for the macmon fix"
  "$uv" run ruff check \
    "$REPO_ROOT/src/exo/utils/info_gatherer/info_gatherer.py" \
    "$REPO_ROOT/src/exo/utils/info_gatherer/tests/test_macmon_monitor.py"
  "$uv" run basedpyright --project "$REPO_ROOT/pyproject.toml" \
    "$REPO_ROOT/src/exo/utils/info_gatherer/info_gatherer.py" \
    "$REPO_ROOT/src/exo/utils/info_gatherer/tests/test_macmon_monitor.py"
  "$uv" run pytest "$REPO_ROOT/src/exo/utils/info_gatherer/tests" -q

  log "Verifying runtime imports"
  "$uv" run python - <<'PY'
import mlx.core
import mlx_lm
import mlx_vlm
import torch
import exo.worker.engines.mlx.utils_mlx
print("runtime imports ok")
PY
}

backup_runtime() {
  mkdir -p "$BACKUP_ROOT"

  if [[ -e "$SOURCE_RUNTIME_MARKER" ]]; then
    local previous_backup
    previous_backup="$(cat "$SOURCE_RUNTIME_MARKER")"
    if [[ -d "$previous_backup" ]]; then
      log "Existing source runtime detected; original backup remains at $previous_backup"
      rm -rf "$RUNTIME_DIR"
      printf '%s\n' "$previous_backup"
      return
    fi
  fi

  if [[ ! -d "$RUNTIME_DIR" ]]; then
    local latest_backup
    latest_backup="$(find "$BACKUP_ROOT" -maxdepth 1 -type d -name 'exo-runtime-*' | sort | tail -1)"
    [[ -n "$latest_backup" ]] || die "Runtime directory not found and no backup exists: $RUNTIME_DIR"
    log "Runtime directory is absent; reusing latest backup at $latest_backup"
    printf '%s\n' "$latest_backup"
    return
  fi

  local timestamp backup
  timestamp="$(date +%Y%m%d-%H%M%S)"
  backup="$BACKUP_ROOT/exo-runtime-$timestamp"
  log "Backing up current runtime to $backup"
  cp -a "$RUNTIME_DIR" "$backup"
  rm -rf "$RUNTIME_DIR"
  printf '%s\n' "$backup"
}

install_runtime() {
  local uv="$1"
  local backup="$2"
  local source_macmon="$backup/_internal/macmon"

  [[ -x "$source_macmon" ]] || die "macmon not found in backup runtime: $source_macmon"

  log "Installing source-backed runtime into $RUNTIME_DIR"
  mkdir -p "$RUNTIME_DIR/_internal"
  cp "$source_macmon" "$RUNTIME_DIR/_internal/macmon"
  cp "$backup/_internal/libiconv.2.dylib" "$RUNTIME_DIR/_internal/libiconv.2.dylib"
  cp "$backup/_internal/libcharset.1.dylib" "$RUNTIME_DIR/_internal/libcharset.1.dylib"
  chmod +x "$RUNTIME_DIR/_internal/macmon"

  cat >"$RUNTIME_DIR/exo" <<EOF
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "\${HOME:-$HOME}/.exo/exo_log"
exec >>"\${HOME:-$HOME}/.exo/exo_log/source-runtime-launch.log" 2>&1
echo "[\$(date -u '+%Y-%m-%dT%H:%M:%SZ')] launching source runtime from $REPO_ROOT"
export PATH="$RUNTIME_DIR:\$PATH"
export PATH="$RUNTIME_DIR/_internal:\$PATH"
export EXO_SOURCE_RUNTIME=1
unset EXO_LIBP2P_NAMESPACE
cd "$REPO_ROOT"
exec "$REPO_ROOT/.venv/bin/exo" "\$@"
EOF
  chmod +x "$RUNTIME_DIR/exo"
  printf '%s\n' "$backup" >"$SOURCE_RUNTIME_MARKER"
}

rollback() {
  [[ -e "$SOURCE_RUNTIME_MARKER" ]] || die "No source runtime marker found at $SOURCE_RUNTIME_MARKER"
  local backup
  backup="$(cat "$SOURCE_RUNTIME_MARKER")"
  [[ -d "$backup" ]] || die "Backup runtime not found: $backup"

  stop_exo
  log "Restoring original runtime from $backup"
  rm -rf "$RUNTIME_DIR"
  cp -a "$backup" "$RUNTIME_DIR"
  log "Rollback complete"
}

launch_and_verify() {
  local start_line
  start_line=0
  if [[ -f "$HOME/.exo/exo_log/exo.log" ]]; then
    start_line="$(wc -l <"$HOME/.exo/exo_log/exo.log" | tr -d ' ')"
  fi

  log "Launching EXO"
  open "$APP_PATH"

  log "Waiting for EXO to start"
  local deadline
  deadline=$((SECONDS + 90))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 2 http://127.0.0.1:52415/state >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  curl -fsS --max-time 2 http://127.0.0.1:52415/state >/dev/null 2>&1 || die "EXO API did not become reachable"

  sleep 20

  if [[ -f "$HOME/.exo/exo_log/exo.log" ]]; then
    local new_log
    new_log="$(tail -n +"$((start_line + 1))" "$HOME/.exo/exo_log/exo.log")"
    if grep -q 'anyio.IncompleteRead: The stream was closed before the read operation could be completed' <<<"$new_log"; then
      die "IncompleteRead traceback still appeared after install"
    fi
    if grep -q 'MacMon stream closed before a complete metrics line' <<<"$new_log"; then
      log "Observed fixed MacMon EOF handling in the new log"
    else
      log "No MacMon EOF occurred during the verification window"
    fi
  fi

  log "Installed source-backed EXO runtime is running"
}

main() {
  cd "$REPO_ROOT"

  if [[ "${1:-}" == "--rollback" ]]; then
    rollback
    exit 0
  fi

  [[ -d "$APP_PATH" ]] || die "EXO app not found: $APP_PATH"

  local uv backup
  uv="$(find_uv)"
  prepare_environment "$uv"
  verify_source "$uv"
  stop_exo
  backup="$(backup_runtime)"
  install_runtime "$uv" "$backup"
  launch_and_verify
}

main "$@"
