#!/usr/bin/env bash
set -Eeuo pipefail

trap 'code=$?; echo -e "\e[1;31m[ERROR]\e[0m ${BASH_SOURCE[0]}:${LINENO} exit $code"; exit $code' ERR

# ---------------------------
# Runpod Bootstrapper for Wan2.2 workflows
# - Installs ComfyUI, required nodes, and Wan2.2 models
# ---------------------------

: "${COMFYUI_REF:=v0.3.50}"
: "${AUTO_UPDATE:=true}"
: "${WORKSPACE:=/workspace}"
: "${COMFY_DIR:=${WORKSPACE%/}/ComfyUI}"
: "${HF_TOKEN:=}"

APT_PACKAGES=(
  git git-lfs curl ca-certificates build-essential pkg-config
  python3-dev python3-pip python3-venv
  libgl1 libglib2.0-0 ffmpeg libsm6 libxext6
)

BASE_PIP_PACKAGES=(
  "pip" "setuptools" "wheel"
  "huggingface_hub==0.25.2"
  "tqdm" "pyyaml" "psutil" "colorama" "imageio" "imageio-ffmpeg" "matplotlib"
  "av" "einops" "scipy" "kornia>=0.7.1"
  "safetensors>=0.4.2" "transformers>=4.28.1" "tokenizers>=0.13.3" "sentencepiece"
  "timm" "albumentations" "shapely" "soundfile" "pydub"
)

# === Custom Nodes required by workflows ===
NODES=(
  https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
  https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
  https://github.com/kijai/ComfyUI-KJNodes
  https://github.com/kijai/ComfyUI-WanVideoWrapper
  https://github.com/city96/ComfyUI-GGUF
  https://github.com/city96/ComfyUI-Qwen
  https://github.com/Fannovel16/comfyui_controlnet_aux
  https://github.com/sipherxyz/comfyui-art-venture
  https://github.com/rgthree/rgthree-comfy
  https://github.com/kijai/ComfyUI-GIMM-VFI
  https://github.com/yolain/ComfyUI-Easy-Use
  https://github.com/ai-shizuka/ComfyUI-tbox
  https://github.com/justUmen/Bjornulf_custom_nodes
)

# === Models required ===
DIFFUSION_MODELS=(
  https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors
  https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors
  https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors
  https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/unet/qwen_image_edit-q8_0.gguf
  https://huggingface.co/Comfy-Org/Wan22-GGUF/resolve/main/wan22_image_q8_0.gguf
)

VAE_MODELS=(
  https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors
  https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors
)

TEXT_ENCODERS=(
  https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5-xxl-enc-bf16.safetensors
  https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
  https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors
)

LORA_MODELS=(
  https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1_low_noise_model.safetensors
  https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors
  https://huggingface.co/Comfy-Org/Wan22-LoRAs/resolve/main/wan22_b0n1_toolkit2_000001250.safetensors
  https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors
)

FRAME_INTERP_MODELS=(
  https://huggingface.co/hzwer/Practical-RIFE/resolve/main/rife49.pth
)

UPSCALE_MODELS=(
  https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth
)

log()  { printf "\e[1;32m[SETUP]\e[0m %s\n" "$*"; }
warn() { printf "\e[1;33m[WARN ]\e[0m %s\n" "$*"; }

sudo_if() { if command -v sudo >/dev/null 2>&1; then sudo "$@"; else "$@"; fi; }

pipx() {
  if [[ -n "${COMFYUI_VENV_PIP:-}" && -x "${COMFYUI_VENV_PIP}" ]]; then
    "${COMFYUI_VENV_PIP}" "$@"
  else
    pip "$@"
  fi
}
pyx() {
  if [[ -n "${COMFYUI_VENV_PYTHON:-}" && -x "${COMFYUI_VENV_PYTHON}" ]]; then
    "${COMFYUI_VENV_PYTHON}" "$@"
  else
    python3 "$@"
  fi
}

fetch() {
  local url="$1" out="$2"
  shift 2 || true
  local auth=()
  [[ "$url" =~ ^https://huggingface\.co ]] && [[ -n "${HF_TOKEN}" ]] && auth=(-H "Authorization: Bearer ${HF_TOKEN}")
  mkdir -p "$(dirname "$out")"
  for i in 1 2 3; do
    curl -fL --retry 5 --retry-delay 2 "${auth[@]}" -o "$out.partial" "$url" && mv -f "$out.partial" "$out" || true
    if [[ -s "$out" && $(stat -c%s "$out") -ge 262144 ]]; then
      echo OK; return 0
    fi
    warn "download retry $i: $url"
    sleep 2
  done
  return 1
}

prepare_env() {
  log "Preparing environment..."
  [[ -f /opt/ai-dock/etc/environment.sh ]] && source /opt/ai-dock/etc/environment.sh
  [[ -f /opt/ai-dock/bin/venv-set.sh    ]] && source /opt/ai-dock/bin/venv-set.sh comfyui
  umask 002
  mkdir -p "${COMFY_DIR%/*}"
}

install_apt() {
  log "Installing apt packages..."
  sudo_if apt-get update -y
  sudo_if apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"
}

clone_comfyui() {
  log "Syncing ComfyUI..."
  if [[ -d "${COMFY_DIR}/.git" ]]; then
    ( cd "$COMFY_DIR"
      git fetch --tags --prune
      [[ "${COMFYUI_REF}" == "latest" ]] && COMFYUI_REF="$(git describe --tags "$(git rev-list --tags --max-count=1)")"
      git checkout -f "${COMFYUI_REF}"
      git pull --ff-only || true
    )
  else
    git clone https://github.com/comfyanonymous/ComfyUI "$COMFY_DIR"
    ( cd "$COMFY_DIR"
      git fetch --tags
      [[ "${COMFYUI_REF}" == "latest" ]] && COMFYUI_REF="$(git describe --tags "$(git rev-list --tags --max-count=1)")"
      git checkout -f "${COMFYUI_REF}"
    )
  fi
}

install_python_base() {
  pipx install --upgrade pip setuptools wheel
  pipx install --no-cache-dir "${BASE_PIP_PACKAGES[@]}"
  pipx install --no-cache-dir "opencv-contrib-python-headless==4.10.0.84"
  if [[ -f "${COMFY_DIR}/requirements.txt" ]]; then
    pipx install --no-cache-dir -r "${COMFY_DIR}/requirements.txt"
  fi
}

install_pytorch() {
  log "Installing PyTorch based on GPU..."
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 || echo "Unknown")
  if [[ "$GPU_NAME" == *"5090"* || "$GPU_NAME" == *"B200"* || "$GPU_NAME" == *"H200"* ]]; then
    echo "[INFO] Detected next-gen GPU ($GPU_NAME), installing PyTorch nightly (CUDA 12.5+)..."
    pipx install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu125
  else
    echo "[INFO] Installing stable PyTorch 2.4.1 (CUDA 12.1) for $GPU_NAME..."
    pipx install torch==2.4.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  fi
  pipx install xformers==0.0.28.post1
}

install_nodes() {
  log "Installing nodes..."
  local base="${COMFY_DIR}/custom_nodes"
  mkdir -p "$base"
  for repo in "${NODES[@]}"; do
    local dir="${repo##*/}"
    local path="$base/$dir"
    if [[ -d "$path/.git" ]]; then
      [[ ${AUTO_UPDATE,,} != "false" ]] && (cd "$path" && git pull --rebase || true)
    else
      git clone --recursive "$repo" "$path" || warn "clone failed: $repo"
    fi
    [[ -s "$path/requirements.txt" ]] && pipx install -r "$path/requirements.txt" || true
  done
}

make_model_dirs() {
  mkdir -p \
    "${COMFY_DIR}/models/diffusion_models" \
    "${COMFY_DIR}/models/vae" \
    "${COMFY_DIR}/models/text_encoders" \
    "${COMFY_DIR}/models/loras" \
    "${COMFY_DIR}/models/rife" \
    "${COMFY_DIR}/models/upscale_models"
}

fetch_models() {
  log "Fetching models..."
  for d in "${DIFFUSION_MODELS[@]}"; do fetch "$d" "${COMFY_DIR}/models/diffusion_models/$(basename "$d")"; done
  for d in "${VAE_MODELS[@]}"; do fetch "$d" "${COMFY_DIR}/models/vae/$(basename "$d")"; done
  for d in "${TEXT_ENCODERS[@]}"; do fetch "$d" "${COMFY_DIR}/models/text_encoders/$(basename "$d")"; done
  for d in "${LORA_MODELS[@]}"; do fetch "$d" "${COMFY_DIR}/models/loras/$(basename "$d")"; done
  for d in "${FRAME_INTERP_MODELS[@]}"; do fetch "$d" "${COMFY_DIR}/models/rife/$(basename "$d")"; done
}

main() {
  prepare_env
  install_apt
  clone_comfyui
  install_python_base
  install_pytorch
  install_nodes
  make_model_dirs
  fetch_models
  log "Provisioning complete."
}

main "$@"