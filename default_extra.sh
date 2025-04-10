# This file will be sourced in default.sh

# https://github.com/kingaigfcash/aigfcash-runpod-template

# Packages are installed after nodes

DEFAULT_WORKFLOW="https://raw.githubusercontent.com/kingaigfcash/aigfcash-runpod-template/refs/heads/main/workflows/default_workflow.json"

APT_PACKAGES=(
    #"package-1"
    #"package-2"
)

PIP_PACKAGES=(
    #"package-1"
    #"package-2"
)

NODES=(
	"https://github.com/ltdrdata/ComfyUI-Manager"
	"https://github.com/cubiq/ComfyUI_essentials"
	"https://github.com/Gourieff/ComfyUI-ReActor"
	"https://github.com/ltdrdata/ComfyUI-Impact-Pack"
	"https://github.com/ltdrdata/ComfyUI-Impact-Subpack"
	"https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
	"https://github.com/giriss/comfy-image-saver"
	"https://github.com/pythongosssss/ComfyUI-WD14-Tagger"
	"https://github.com/hylarucoder/comfyui-copilot"
	"https://github.com/kijai/ComfyUI-KJNodes"
	"https://github.com/KoreTeknology/ComfyUI-Universal-Styler"
	"https://github.com/LarryJane491/Lora-Training-in-Comfy"
	"https://github.com/LarryJane491/Image-Captioning-in-ComfyUI"
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"
)

WORKFLOWS=(
	"https://github.com/kingaigfcash/aigfcash-runpod-template.git"
)

# Initialize empty arrays for models
CHECKPOINT_MODELS=(
    "https://huggingface.co/RunDiffusion/Juggernaut-XI-v11/resolve/main/Juggernaut-XI-byRunDiffusion.safetensors"
    "https://huggingface.co/John6666/epicrealism-xl-v8kiss-sdxl/resolve/main/epicrealismXL_vx1Finalkiss.safetensors"
    "https://huggingface.co/TheImposterImposters/URPM-v2.3Final/resolve/main/uberRealisticPornMerge_v23Final.safetensors"
)
UNET_MODELS=()
VAE_MODELS=()
CLIP_MODELS=()
LORA_MODELS=()
CONTROLNET_MODELS=()
ESRGAN_MODELS=()
INSIGHTFACE_MODELS=(
    "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx"
)

# Ultralytics models (YOLOv8)
ULTRALYTICS_BBOX_MODELS=(
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt"
)

ULTRALYTICS_SEGM_MODELS=(
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt"
)

SAM_MODELS=(
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)

### DO NOT EDIT BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###

function provisioning_start() {
    if [[ ! -d /opt/environments/python ]]; then 
        export MAMBA_BASE=true
    fi
    source /opt/ai-dock/etc/environment.sh
    source /opt/ai-dock/bin/venv-set.sh comfyui

    # Initialize CLIP models (these are required)
    CLIP_MODELS=(
        "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/clip_l.safetensors"
        "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/t5xxl_fp16.safetensors"
    )

    # Add HuggingFace models if token is valid
    if provisioning_has_valid_hf_token; then
        CHECKPOINT_MODELS+=("https://huggingface.co/RunDiffusion/Juggernaut-XI-v11/resolve/main/Juggernaut-XI-byRunDiffusion.safetensors")
        UNET_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors")
        VAE_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors")
    else
        CHECKPOINT_MODELS+=("https://huggingface.co/RunDiffusion/Juggernaut-XI-v11/resolve/main/Juggernaut-XI-byRunDiffusion.safetensors")
        UNET_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors")
        VAE_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors")
        sed -i 's/flux1-dev\.safetensors/flux1-schnell.safetensors/g' /opt/ComfyUI/web/scripts/defaultGraph.js
    fi

    # Add Civitai models if token is valid
    if provisioning_has_valid_civitai_token; then
        CHECKPOINT_MODELS+=(
            #"https://civitai.com/api/download/models/782002"
            #"https://civitai.com/api/download/models/919063"
            #"https://civitai.com/api/download/models/1346244"
        )
    fi

    provisioning_print_header
    provisioning_get_apt_packages
    provisioning_get_nodes
    provisioning_get_pip_packages

    # Create model directories
    mkdir -p "${WORKSPACE}/ComfyUI/models/checkpoints"
    mkdir -p "${WORKSPACE}/ComfyUI/models/ultralytics/bbox"
    mkdir -p "${WORKSPACE}/ComfyUI/models/ultralytics/segm"
    mkdir -p "${WORKSPACE}/ComfyUI/models/sams"
    mkdir -p "${WORKSPACE}/ComfyUI/models/insightface"

    # Download models to appropriate directories
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/checkpoints" \
        "${CHECKPOINT_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/unet" \
        "${UNET_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/clip" \
        "${CLIP_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/lora" \
        "${LORA_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/controlnet" \
        "${CONTROLNET_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/vae" \
        "${VAE_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/esrgan" \
        "${ESRGAN_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/ultralytics/bbox" \
        "${ULTRALYTICS_BBOX_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/ultralytics/segm" \
        "${ULTRALYTICS_SEGM_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/sams" \
        "${SAM_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/insightface" \
        "${INSIGHTFACE_MODELS[@]}"
    provisioning_get_workflows
    provisioning_print_end
}

function pip_install() {
    if [[ -z $MAMBA_BASE ]]; then
            "$COMFYUI_VENV_PIP" install --no-cache-dir "$@"
        else
            micromamba run -n comfyui pip install --no-cache-dir "$@"
        fi
}

function provisioning_get_apt_packages() {
    if [[ -n $APT_PACKAGES ]]; then
            sudo $APT_INSTALL ${APT_PACKAGES[@]}
    fi
}

function provisioning_get_pip_packages() {
    if [[ -n $PIP_PACKAGES ]]; then
            pip_install ${PIP_PACKAGES[@]}
    fi
}

function provisioning_get_nodes() {
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="${WORKSPACE}/ComfyUI/custom_nodes/${dir}"
        requirements="${path}/requirements.txt"
        if [[ -d $path ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" ]]; then
                printf "Updating node: %s...\n" "${repo}"
                ( cd "$path" && git pull )
                if [[ -e $requirements ]]; then
                   pip_install -r "$requirements"
                fi
            fi
        else
            printf "Downloading node: %s...\n" "${repo}"
            git clone "${repo}" "${path}" --recursive
            if [[ -e $requirements ]]; then
                pip_install -r "${requirements}"
            fi
        fi
    done
}

function provisioning_get_workflows() {
    for repo in "${WORKFLOWS[@]}"; do
        dir=$(basename "$repo" .git)
        temp_path="/tmp/${dir}"
        target_path="${WORKSPACE}/ComfyUI/user/default/workflows"
        
        # Clone or update the repository
        if [[ -d "$temp_path" ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" ]]; then
                printf "Updating workflows: %s...\n" "${repo}"
                ( cd "$temp_path" && git pull )
            fi
        else
            printf "Cloning workflows: %s...\n" "${repo}"
            git clone "$repo" "$temp_path"
        fi
        
        # Create workflows directory if it does not exist
        mkdir -p "$target_path"
        
        # Copy workflow files to the target directory
        if [[ -d "$temp_path/workflows" ]]; then
            cp -r "$temp_path/workflows"/* "$target_path/"
            printf "Copied workflows to: %s\n" "$target_path"
        fi
    done
}

function provisioning_get_default_workflow() {
    if [[ -n $DEFAULT_WORKFLOW ]]; then
        workflow_json=$(curl -s "$DEFAULT_WORKFLOW")
        if [[ -n $workflow_json ]]; then
            echo "export const defaultGraph = $workflow_json;" > "${WORKSPACE}/ComfyUI/web/scripts/defaultGraph.js"
        fi
    fi
}

function provisioning_get_models() {
    if [[ -z $2 ]]; then return 1; fi
    dir=$(normalize_path "$1")
    mkdir -p "$dir"
    shift
    arr=("$@")
    printf "Downloading %s model(s) to %s...\n" "${#arr[@]}" "$dir"
    for url in "${arr[@]}"; do
        printf "Downloading: %s\n" "${url}"
        provisioning_download "${url}" "${dir}"
        printf "\n"
    done
}

# Normalize path to remove any double slashes
normalize_path() {
    echo "${1//\/\///}"
}

function provisioning_print_header() {
    printf "\n##############################################\n#                                            #\n#          Provisioning container            #\n#                                            #\n#         This will take some time           #\n#                                            #\n# Your container will be ready on completion #\n#                                            #\n##############################################\n\n"
    if [[ $DISK_GB_ALLOCATED -lt $DISK_GB_REQUIRED ]]; then
        printf "WARNING: Your allocated disk size (%sGB) is below the recommended %sGB - Some models will not be downloaded\n" "$DISK_GB_ALLOCATED" "$DISK_GB_REQUIRED"
    fi
}

function provisioning_print_end() {
    printf "\nProvisioning complete:  Web UI will start now\n\n"
}

function provisioning_has_valid_hf_token() {
    [[ -n "$HF_TOKEN" ]] || return 1
    url="https://huggingface.co/api/whoami-v2"

    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $HF_TOKEN" \
        -H "Content-Type: application/json")

    # Check if the token is valid
    if [ "$response" -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

function provisioning_has_valid_civitai_token() {
    [[ -n "$CIVITAI_TOKEN" ]] || return 1
    url="https://civitai.com/api/v1/models?hidden=1&limit=1"

    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $CIVITAI_TOKEN" \
        -H "Content-Type: application/json")

    # Check if the token is valid
    if [ "$response" -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

# Download from $1 URL to $2 file path
function provisioning_download() {
    local url="$1"
    local target_dir="$2"
    local auth_token=""
    local response=""
    local filename=""

    echo "Attempting to download from: $url"
    echo "Target directory: $target_dir"

    # Set auth token based on URL
    if [[ -n $HF_TOKEN && $url =~ ^https://([a-zA-Z0-9_-]+\.)?huggingface\.co(/|$|\?) ]]; then
        auth_token="$HF_TOKEN"
        echo "Using HuggingFace token"
    elif [[ -n $CIVITAI_TOKEN && $url =~ ^https://([a-zA-Z0-9_-]+\.)?civitai\.com(/|$|\?) ]]; then
        auth_token="$CIVITAI_TOKEN"
        echo "Using CivitAI token"
        
        # For CivitAI, get the actual filename from headers
        if [[ $url =~ /api/download/models/([0-9]+) ]]; then
            local model_id="${BASH_REMATCH[1]}"
            echo "Detected CivitAI model ID: $model_id"
            
            # Get the filename from Content-Disposition header
            local headers=$(curl -sI -H "Authorization: Bearer $CIVITAI_TOKEN" "$url")
            if [[ $headers =~ Content-Disposition:.*filename=\"?([^\";\r\n]+) ]]; then
                filename="${BASH_REMATCH[1]}"
            else
                # Fallback: Try to get filename from redirect URL
                local redirect_url=$(curl -sI -H "Authorization: Bearer $CIVITAI_TOKEN" "$url" | grep -i "^location:" | cut -d' ' -f2 | tr -d '\r')
                if [[ -n "$redirect_url" ]]; then
                    filename=$(basename "$redirect_url")
                else
                    # Last resort fallback
                    filename="model_${model_id}.safetensors"
                fi
            fi
            echo "Will save as: $filename"
        fi
    fi

    # Create target directory if it doesn't exist
    mkdir -p "$target_dir"

    # Get filename from URL
    if [[ -z $filename ]]; then
        filename=$(basename "$url")
        if [[ -z $filename ]]; then
            echo "ERROR: Could not determine filename from URL"
            return 1
        fi
    fi

    # Full path to target file
    local target_file="$target_dir/$filename"

    # Download the file using curl with minimal output
    echo "Downloading to: $target_file"
    if [[ -n $auth_token ]]; then
        echo "Downloading with authentication..."
        curl -sS -L -H "Authorization: Bearer $auth_token" -o "$target_file" "$url"
    else
        echo "Downloading without authentication..."
        curl -sS -L -o "$target_file" "$url"
    fi

    # Verify download
    if [[ -f "$target_file" ]]; then
        local filesize=$(stat -f%z "$target_file" 2>/dev/null || stat -c%s "$target_file" 2>/dev/null)
        if [[ $filesize -gt 0 ]]; then
            echo "Successfully downloaded: $filename ($(numfmt --to=iec-i --suffix=B $filesize))"
            return 0
        else
            echo "ERROR: Downloaded file is empty"
            rm -f "$target_file"
            return 1
        fi
    else
        echo "ERROR: File not found after download"
        return 1
    fi
}

if provisioning_has_valid_hf_token; then
    echo "Downloading bbox models..."
    for url in "${ULTRALYTICS_BBOX_MODELS[@]}"; do
        filename="face_yolov8m.pt"  # Fixed filename for bbox
        target_dir="$WORKSPACE/ComfyUI/models/ultralytics/bbox"
        if ! provisioning_download "$url" "$target_dir"; then
            echo "ERROR: Failed to download bbox model"
            continue
        fi
    done

    echo "Downloading segm models..."
    for url in "${ULTRALYTICS_SEGM_MODELS[@]}"; do
        filename="yolov8m-seg.pt"  # Fixed filename for segm
        target_dir="$WORKSPACE/ComfyUI/models/ultralytics/segm"
        if ! provisioning_download "$url" "$target_dir"; then
            echo "ERROR: Failed to download segm model"
            continue
        fi
    done

    echo "Starting SAM models download..."
    echo "Downloading $(echo ${SAM_MODELS[@]} | wc -w) model(s) to $WORKSPACE/ComfyUI/models/sams..."
    for url in "${SAM_MODELS[@]}"; do
        provisioning_download "$url" "$WORKSPACE/ComfyUI/models/sams"
    done
    
    echo "Starting Insightface models download..."
    echo "Downloading $(echo ${INSIGHTFACE_MODELS[@]} | wc -w) model(s) to $WORKSPACE/ComfyUI/models/insightface..."
    for url in "${INSIGHTFACE_MODELS[@]}"; do
        provisioning_download "$url" "$WORKSPACE/ComfyUI/models/insightface"
    done
else
    echo "ERROR: Invalid Hugging Face token. Cannot download models."
fi

provisioning_start