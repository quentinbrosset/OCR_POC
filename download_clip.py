from huggingface_hub import snapshot_download

print("📥 Downloading CLIP model")

snapshot_download(
    repo_id="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    local_dir="models/laion_clip",
    local_dir_use_symlinks=False
)

print("✅ CLIP model downloaded!")