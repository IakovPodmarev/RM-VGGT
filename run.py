import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from pathlib import Path

def get_runtime_device():
    if not torch.cuda.is_available():
        return "cpu"

    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    supported_arches = set(getattr(torch.cuda, "get_arch_list", lambda: [])())
    if supported_arches and arch not in supported_arches:
        print(
            f"CUDA is available, but this PyTorch build does not support GPU arch {arch}. "
            "Falling back to CPU."
        )
        return "cpu"

    return "cuda"


device = get_runtime_device()
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
if device == "cuda":
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
else:
    dtype = torch.float32

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images.
folder_path = Path(__file__).parent / "examples" / "kitchen" / "images"
image_names = [str(f.resolve()) for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=device == "cuda", dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
