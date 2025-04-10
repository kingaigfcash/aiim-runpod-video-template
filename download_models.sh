#!/bin/bash

# Set workspace path
WORKSPACE_PATH="/workspace/models"

# Set maximum number of concurrent downloads
MAX_CONCURRENT=5
current_downloads=0

# Function to download file
download_model() {
    local url="$1"
    local dest="$2"
    
    # Create directory if it doesn't exist
    local dir=$(dirname "$dest")
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
    
    if [ -f "$dest" ]; then
        echo "✅ Skipping $(basename "$dest") - already exists"
        return
    fi
    
    # Wait if we've reached max concurrent downloads
    while [ $current_downloads -ge $MAX_CONCURRENT ]; do
        sleep 1
        # Count current background processes
        current_downloads=$(jobs -p | wc -l)
    done
    
    # Increment counter and start download in background
    ((current_downloads++))
    (
        if wget -q -O "$dest" "$url"; then
            echo "✅ Completed downloading $(basename "$dest")"
        else
            echo "❌ Failed downloading $(basename "$dest")"
            # Remove failed/incomplete download
            rm -f "$dest"
        fi
        # Decrement counter when download finishes
        ((current_downloads--))
    ) &
}

# Download models one by one
echo "Starting model downloads..."

# epicrealismXL_vxiiiAb3ast
download_model \
    "https://civitai.com/api/download/models/1346244?type=Model&format=SafeTensor&size=pruned&fp=fp16" \
    "$WORKSPACE_PATH/checkpoints/epicrealismXL_vxiiiAb3ast.safetensors"

# uberRealisticPornMergePonyxl_ponyxlHybridV1
download_model \
    "https://civitai.com/api/download/models/923681?type=Model&format=SafeTensor&size=full&fp=fp16" \
    "$WORKSPACE_PATH/checkpoints/uberRealisticPornMergePonyxl_ponyxlHybridV1.safetensors"


# Photon model
download_model \
    "https://civitai.com/api/download/models/90072?type=Model&format=SafeTensor&size=pruned&fp=fp16" \
    "$WORKSPACE_PATH/checkpoints/photon_v1.safetensors"

# UltraSharp upscaler
download_model \
    "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth" \
    "$WORKSPACE_PATH/upscale_models/4x-UltraSharp.pth"
download_model \
    "https://huggingface.co/Kim2091/ClearRealityV1/resolve/main/4x-ClearRealityV1.safetensors?download=true" \
    "$WORKSPACE_PATH/upscale_models/4x-ClearRealityV1.safetensors"
download_model \
    "https://huggingface.co/Kim2091/ClearRealityV1/resolve/main/4x-ClearRealityV1_Soft.safetensors?download=true" \
    "$WORKSPACE_PATH/upscale_models/4x-ClearRealityV1_Soft.safetensors"

# ControlNet Union SDXL
download_model \
    "https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors" \
    "$WORKSPACE_PATH/controlnet/SDXL/controlnet-union-sdxl-1.0/diffusion_pytorch_model_promax.safetensors"

# OpenPoseXL2 ControlNet
download_model \
    "https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/OpenPoseXL2.safetensors" \
    "$WORKSPACE_PATH/controlnet/OpenPoseXL2.safetensors"

# IC-Light Model
download_model \
    "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors" \
    "$WORKSPACE_PATH/diffusion_models/IC-Light/iclight_sd15_fbc.safetensors"

# CLIP Vision Models for IP-Adapter
download_model \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" \
    "$WORKSPACE_PATH/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"

download_model \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors" \
    "$WORKSPACE_PATH/clip_vision/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors"

# IP-Adapter Models
download_model \
    "https://huggingface.co/TheMistoAI/MistoLine/resolve/main/mistoLine_rank256.safetensors?download=true" \
    "$WORKSPACE_PATH/controlnet/mistoLine_rank256.safetensors"

download_model \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors" \
    "$WORKSPACE_PATH/ipadapter/ip-adapter_sd15.safetensors"

download_model \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_light_v11.bin" \
    "$WORKSPACE_PATH/ipadapter/ip-adapter_sd15_light_v11.bin"

download_model \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors" \
    "$WORKSPACE_PATH/ipadapter/ip-adapter-plus_sd15.safetensors"

download_model \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors" \
    "$WORKSPACE_PATH/ipadapter/ip-adapter-plus-face_sd15.safetensors"

download_model \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors" \
    "$WORKSPACE_PATH/ipadapter/ip-adapter-full-face_sd15.safetensors"

download_model \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_vit-G.safetensors" \
    "$WORKSPACE_PATH/ipadapter/ip-adapter_sd15_vit-G.safetensors"

download_model \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors" \
    "$WORKSPACE_PATH/ipadapter/ip-adapter_sdxl.safetensors"

download_model \
    "https://huggingface.co/ostris/ip-composition-adapter/resolve/main/ip_plus_composition_sd15.safetensors" \
    "$WORKSPACE_PATH/ipadapter/ip_plus_composition_sd15.safetensors"

download_model \
    "https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-Plus/resolve/main/ip_adapter_plus_general.bin" \
    "$WORKSPACE_PATH/ipadapter/Kolors-IP-Adapter-Plus.bin"

download_model \
    "https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus/resolve/main/ipa-faceid-plus.bin" \
    "$WORKSPACE_PATH/ipadapter/Kolors-IP-Adapter-FaceID-Plus.bin"

# Face Detection Model
download_model \
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt" \
    "$WORKSPACE_PATH/ultralytics/bbox/face_yolov8m.pt"

# SDXL IP-Adapter ViT-H Models
download_model \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors" \
    "$WORKSPACE_PATH/ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors"

download_model \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors" \
    "$WORKSPACE_PATH/ipadapter/ip-adapter_sdxl_vit-h.safetensors"

download_model \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors" \
    "$WORKSPACE_PATH/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors"

# PuLID Models
download_model \
    "https://huggingface.co/guozinan/PuLID/resolve/main/pulid_v1.1.safetensors" \
    "$WORKSPACE_PATH/pulid/pulid_v1.1.safetensors"

download_model \
    "https://huggingface.co/huchenlei/ipadapter_pulid/resolve/main/ip-adapter_pulid_sdxl_fp16.safetensors?download=true" \
    "$WORKSPACE_PATH/pulid/ip-adapter_pulid_sdxl_fp_16.safetensors"

# Flux1 model
#download_model \
#    "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors" \
#    "$WORKSPACE_PATH/checkpoints/flux1-dev-fp8.safetensors"
#download_model \
#    "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union/resolve/main/diffusion_pytorch_model.safetensors" \
#    "$WORKSPACE_PATH/controlnet/FLUX.1/InstantX-FLUX1-Dev-Union/diffusion_pytorch_model.safetensors"
#download_model \
#    "https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors" \
#    "$WORKSPACE_PATH/pulid/pulid_flux_v0.9.1.safetensors"

# InstantID Antelopev2 models
download_model \
    "https://huggingface.co/MonsterMMORPG/tools/resolve/main/1k3d68.onnx" \
    "$WORKSPACE_PATH/insightface/models/antelopev2/1k3d68.onnx"

download_model \
    "https://huggingface.co/MonsterMMORPG/tools/resolve/main/2d106det.onnx" \
    "$WORKSPACE_PATH/insightface/models/antelopev2/2d106det.onnx"

download_model \
    "https://huggingface.co/MonsterMMORPG/tools/resolve/main/genderage.onnx" \
    "$WORKSPACE_PATH/insightface/models/antelopev2/genderage.onnx"

download_model \
    "https://huggingface.co/MonsterMMORPG/tools/resolve/main/glintr100.onnx" \
    "$WORKSPACE_PATH/insightface/models/antelopev2/glintr100.onnx"

download_model \
    "https://huggingface.co/MonsterMMORPG/tools/resolve/main/scrfd_10g_bnkps.onnx" \
    "$WORKSPACE_PATH/insightface/models/antelopev2/scrfd_10g_bnkps.onnx"

# Wait for all background downloads to complete
wait

echo "All downloads completed!"