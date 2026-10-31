#!/usr/bin/env bash
#
# oxr-mac.sh —— 管理本机（macOS / Apple Silicon）的 OXR 推理服务
#
# 对应 Linux 侧的 scripts/oxr-vllm.sh：那边用 systemd user unit，这里用
# launchd LaunchAgent（launchctl bootstrap gui/$UID）。
#
#   ./scripts/oxr-mac.sh install     安装 LaunchAgent（不加载、不启动）
#   ./scripts/oxr-mac.sh start       启动并等待就绪
#   ./scripts/oxr-mac.sh stop        停止（释放统一内存）
#   ./scripts/oxr-mac.sh restart     重启
#   ./scripts/oxr-mac.sh status      运行状态 + /health
#   ./scripts/oxr-mac.sh logs        看日志（可传行数，默认 60）
#   ./scripts/oxr-mac.sh check       环境自检（venv / 模型 / 端口 / 设备）
#   ./scripts/oxr-mac.sh run         前台直跑（调试用，不经过 launchd）
#   ./scripts/oxr-mac.sh uninstall   停止并移除 LaunchAgent
#
# 环境变量：
#   OXR_VENV        虚拟环境目录（默认 <repo>/.venv）
#   OXR_PORT        端口（默认 8001）
#   OXR_INSTANCES   模型实例数（默认 1；3 个约 1.38x 吞吐，各占 3.5-6 GB）
#   OXR_DEVICE      auto / mps / cpu
#   OXR_DTYPE       auto / fp16 / bf16 / fp32
#   HF_HOME         模型缓存（默认 ~/cot/cache/huggingface）
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${OXR_LABEL:-com.oxr.server}"
VENV="${OXR_VENV:-$REPO/.venv}"
PORT="${OXR_PORT:-8001}"
INSTANCES="${OXR_INSTANCES:-1}"
DEVICE="${OXR_DEVICE:-auto}"
DTYPE="${OXR_DTYPE:-auto}"
HF_HOME_DIR="${HF_HOME:-$HOME/cot/cache/huggingface}"
MODEL_ID="${OXR_MODEL:-SoMarkAI/OXR-1.0}"
AGENT_DIR="$HOME/Library/LaunchAgents"
PLIST="$AGENT_DIR/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs"
LOG="$LOG_DIR/oxr-server.log"
ERR="$LOG_DIR/oxr-server.err.log"
UID_N="$(id -u)"
DOMAIN="gui/$UID_N"
BASE="http://127.0.0.1:$PORT"

die() { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }
log() { printf '\033[1;34m[·]\033[0m %s\n' "$*"; }
ok()  { printf '\033[1;32m[✓]\033[0m %s\n' "$*"; }

find_model() {
  local m
  m=$(ls -d "$HF_HOME_DIR"/hub/models--SoMarkAI--OXR-1.0/snapshots/*/ 2>/dev/null | head -1)
  [[ -n "$m" ]] || return 1
  echo "${m%/}"
}

check() {
  log "仓库:    $REPO"
  [[ -x "$VENV/bin/python" ]] || die "缺少 $VENV/bin/python（先 uv sync）"
  log "python:  $("$VENV/bin/python" --version 2>&1)"
  local model
  if model=$(find_model); then
    log "模型:    $model"
    [[ -f "$model/model.safetensors" ]] || die "模型快照不完整（缺 model.safetensors）"
  else
    die "找不到 OXR 模型快照（HF_HOME=$HF_HOME_DIR）"
  fi
  "$VENV/bin/python" - <<'PY' || die "MPS 不可用"
import torch
assert torch.backends.mps.is_available(), "torch.backends.mps 不可用"
print(f"torch:   {torch.__version__}（mps 可用）")
PY
  PORT_OWNER=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | awk 'NR==2{print $1" pid="$2}')
  log "端口:    $PORT ${PORT_OWNER:+（已被 $PORT_OWNER 占用）}${PORT_OWNER:-（空闲）}"
  [[ -f "$PLIST" ]] && log "单元:    $PLIST（已安装）" || log "单元:    未安装"
}

gen_plist() {
  mkdir -p "$AGENT_DIR" "$LOG_DIR"
  cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$VENV/bin/python</string>
    <string>-m</string>
    <string>oxr_starter.server</string>
    <string>--host</string><string>127.0.0.1</string>
    <string>--port</string><string>$PORT</string>
    <string>--device</string><string>$DEVICE</string>
    <string>--dtype</string><string>$DTYPE</string>
    <string>--instances</string><string>$INSTANCES</string>
  </array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HF_HOME</key><string>$HF_HOME_DIR</string>
    <key>HF_HUB_OFFLINE</key><string>1</string>
    <key>PATH</key><string>$VENV/bin:/opt/homebrew/bin:/usr/bin:/bin</string>
    <key>PYTHONUNBUFFERED</key><string>1</string>
  </dict>
  <key>RunAtLoad</key><false/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$ERR</string>
  <key>ProcessType</key><string>Interactive</string>
  <key>ThrottleInterval</key><integer>10</integer>
</dict>
</plist>
PLIST_EOF
  log "已生成 $PLIST"
}

wait_ready() {
  local i
  log "等待就绪（最多 120s）…"
  for i in $(seq 1 60); do
    sleep 2
    if curl -sf -o /dev/null "$BASE/health"; then
      ok "服务就绪（约 $((i*2))s）"
      curl -s "$BASE/health"; echo
      return 0
    fi
  done
  die "启动超时，看日志：./scripts/oxr-mac.sh logs"
}

case "${1:-}" in
  install)
    check; gen_plist
    log "单元已安装（未加载、未启动）"
    log "启动：  ./scripts/oxr-mac.sh start"
    ;;
  start)
    # 每次都按当前环境变量重新生成 plist：OXR_PORT / OXR_INSTANCES / OXR_DEVICE
    # 改了就能直接 restart 生效，不需要先 install。
    gen_plist
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null && sleep 1
    launchctl bootstrap "$DOMAIN" "$PLIST" || die "启动失败，看：./scripts/oxr-mac.sh logs"
    wait_ready
    ;;
  stop)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null \
      && ok "已停止（内存已释放）" || log "本来就没在跑"
    ;;
  restart)
    gen_plist
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
    sleep 1
    launchctl bootstrap "$DOMAIN" "$PLIST" || die "重启失败"
    wait_ready
    ;;
  status)
    if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
      log "LaunchAgent 已加载"
      launchctl print "$DOMAIN/$LABEL" | grep -E "state|pid|last exit" | sed 's/^/  /'
      curl -sf "$BASE/health" && echo || log "健康检查失败"
    else
      log "未加载（未运行）"
    fi
    ;;
  logs)
    tail -n "${2:-60}" "$LOG" 2>/dev/null
    [[ -s "$ERR" ]] && { echo "---- stderr ----"; tail -n "${2:-20}" "$ERR"; }
    ;;
  check)
    check ;;
  uninstall)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
    rm -f "$PLIST"
    ok "已卸载"
    ;;
  run)
    check
    exec "$VENV/bin/python" -m oxr_starter.server \
      --host 127.0.0.1 --port "$PORT" \
      --device "$DEVICE" --dtype "$DTYPE" --instances "$INSTANCES"
    ;;
  *)
    sed -n '3,25p' "$0" | sed 's/^# \{0,1\}//'
    exit 1 ;;
esac
