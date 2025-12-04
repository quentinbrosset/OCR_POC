from huggingface_hub import snapshot_download

print("📥 Downloading CLIP model")

snapshot_download(
    repo_id="openai/clip-vit-base-patch32",
    local_dir="models/clip-vit-base-patch32",
    local_dir_use_symlinks=False
)

print("✅ CLIP model downloaded!")