#!/usr/bin/env bash
#
# oxr-vllm.sh —— 管理 OXR 的 vLLM 服务（systemd 用户单元）
#
#   ./oxr-vllm.sh install     安装并启动（开机自启）
#   ./oxr-vllm.sh uninstall   停止并移除
#   ./oxr-vllm.sh start|stop|restart|status|logs
#   ./oxr-vllm.sh run         前台直接跑（调试用，不经过 systemd）
#   ./oxr-vllm.sh check       检查环境（venv / 模型 / 插件 / 端口）
#
# 依赖：~/space/python/oxr-vllm（vllm 0.25.1 + ninja + oxr 插件）
#       HF 缓存里的 OXR-1.0 模型快照
#
set -uo pipefail

VENV="${OXR_VLLM_VENV:-$HOME/space/python/oxr-vllm}"
HF_HOME_DIR="${HF_HOME:-$HOME/cot/cache/huggingface}"
UNIT="oxr-vllm"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT_FILE="$UNIT_DIR/$UNIT.service"
PORT="${OXR_VLLM_PORT:-8001}"
PY_HEADERS="${OXR_PY_HEADERS:-/home/q/cot/lib/python_3.12.16_linux_amd64_gnu/include/python3.12}"
GPU_UTIL="${OXR_GPU_UTIL:-0.55}"
MAX_LEN="${OXR_MAX_LEN:-8192}"

die() { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }
log() { printf '\033[1;34m[·]\033[0m %s\n' "$*"; }

find_model() {
  local m
  m=$(ls -d "$HF_HOME_DIR"/hub/models--SoMarkAI--OXR-1.0/snapshots/*/ 2>/dev/null | head -1)
  [[ -n "$m" ]] || die "找不到 OXR 模型快照（HF_HOME=$HF_HOME_DIR）"
  echo "${m%/}"
}

check() {
  log "venv:    $VENV"
  [[ -x "$VENV/bin/vllm" ]] || die "缺少 $VENV/bin/vllm"
  [[ -x "$VENV/bin/ninja" ]] || die "缺少 ninja：uv pip install ninja"
  log "ninja:   $("$VENV/bin/ninja" --version 2>/dev/null || echo '?')"
  local model; model=$(find_model)
  log "模型:    $model"
  [[ -f "$model/model.safetensors" ]] || die "模型快照不完整"
  [[ -d "$model/plugins/vllm" ]] || die "模型里没有 vLLM 插件目录"
  log "头文件:  $PY_HEADERS/Python.h"
  [[ -f "$PY_HEADERS/Python.h" ]] || die "缺少 Python.h，设置 OXR_PY_HEADERS"
  log "端口:    $PORT $(_port_busy && echo '（已被占用）' || echo '（空闲）')"
}

_port_busy() { ss -ltn 2>/dev/null | grep -q ":$PORT "; }

gen_unit() {
  local model; model=$(find_model)
  cat > "$UNIT_FILE" <<UNIT
[Unit]
Description=OXR vLLM server (port $PORT)
After=graphical-session.target

[Service]
Type=simple
Environment=VLLM_PLUGINS=oxr
Environment=CPATH=$PY_HEADERS
Environment=PATH=$VENV/bin:/usr/local/bin:/usr/bin:/bin
Environment=HF_HOME=$HF_HOME_DIR
Environment=HF_HUB_OFFLINE=1
ExecStart=$VENV/bin/vllm serve $model \\
  --trust-remote-code --served-model-name oxr \\
  --chat-template $model/chat_template.jinja \\
  --dtype bfloat16 --gpu-memory-utilization $GPU_UTIL --max-model-len $MAX_LEN \\
  --host 127.0.0.1 --port $PORT
Restart=on-failure
RestartSec=10
TimeoutStartSec=300

[Install]
WantedBy=default.target
UNIT
  log "已生成 $UNIT_FILE"
}

case "${1:-}" in
  install)
    mkdir -p "$UNIT_DIR"; gen_unit
    systemctl --user daemon-reload
    systemctl --user enable --now "$UNIT"
    log "已安装并启动；等待就绪（最多 120s）…"
    for i in $(seq 1 60); do
      sleep 2
      curl -sf -o /dev/null "http://127.0.0.1:$PORT/v1/models" && { log "服务就绪（约 $((i*2))s）"; exit 0; }
    done
    systemctl --user --no-pager status "$UNIT" | tail -15
    die "启动超时，看日志：./oxr-vllm.sh logs"
    ;;
  uninstall)
    systemctl --user disable --now "$UNIT" 2>/dev/null
    rm -f "$UNIT_FILE"; systemctl --user daemon-reload
    log "已卸载"
    ;;
  start|stop|restart|status)
    systemctl --user "$1" "$UNIT" ;;
  logs)
    journalctl --user -u "$UNIT" -n "${2:-60}" --no-pager ;;
  run)
    check
    model=$(find_model)
    VLLM_PLUGINS=oxr CPATH="$PY_HEADERS" PATH="$VENV/bin:$PATH" \
    HF_HOME="$HF_HOME_DIR" HF_HUB_OFFLINE=1 \
      "$VENV/bin/vllm" serve "$model" \
        --trust-remote-code --served-model-name oxr \
        --chat-template "$model/chat_template.jinja" \
        --dtype bfloat16 --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_LEN" \
        --host 127.0.0.1 --port "$PORT" ;;
  check)
    check ;;
  *)
    sed -n '3,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 1 ;;
esac
