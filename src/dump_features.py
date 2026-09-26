"""Dump per-image features for several backbones.

Images found recursively under --image_dir are saved as one .pt file per image, mirroring
the input layout under a separate folder per model:

    images/cat1/image1.jpg -> features/<model_name>/cat1/image1.pt
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# All backbones are ViT-H/14 (or the closest available) to match I-JEPA, whose smallest release is ViT-H/14
DINOV3_REPO_DIR = "/home/mchasmai_umass_edu/bird_parts/inat/dinov3"
# DINOv3 has no ViT-H/14; ViT-H+/16 (1280-d, 32 blocks, SwiGLU) is its ViT-H-sized model
DINOV3_WEIGHTS = "/work/pi_smaji_umass_edu/mchasmai/model_weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
IJEPA_HF_ID = "facebook/ijepa_vith14_1k"
# OpenAI's largest CLIP is ViT-L/14, so use the OpenCLIP ViT-H/14 trained on LAION-2B
CLIP_HF_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
# torchvision's only ViT-H/14 is SWAG (Instagram) pretrained, so use DeiT III trained on ImageNet-1K alone
IMAGENET_VIT_TIMM_ID = "deit3_huge_patch14_224.fb_in1k"

IMG_EXTS = ('.png', '.jpg', '.jpeg')


class DINOv3Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = torch.hub.load(DINOV3_REPO_DIR, "dinov3_vith16plus", source='local', weights=DINOV3_WEIGHTS)

    def forward(self, x):
        # Normed CLS token (1280-d)
        return self.model(x)


class IJEPAEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        from transformers import IJepaModel
        self.model = IJepaModel.from_pretrained(IJEPA_HF_ID, cache_dir=args.hf_cache_dir)

    def forward(self, x):
        # I-JEPA has no CLS token, so average-pool the normed patch tokens (1280-d)
        return self.model(pixel_values=x).last_hidden_state.mean(dim=1)


class CLIPEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection
        self.model = CLIPVisionModelWithProjection.from_pretrained(CLIP_HF_ID, cache_dir=args.hf_cache_dir)

    def forward(self, x):
        # Image embedding in the joint image-text space (1024-d, not L2-normalized)
        return self.model(pixel_values=x).image_embeds


class ImageNetViTEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        import timm
        self.model = timm.create_model(IMAGENET_VIT_TIMM_ID, pretrained=True, num_classes=0, cache_dir=args.hf_cache_dir)

    def forward(self, x):
        # Normed CLS token, classification head removed (1280-d)
        return self.model(x)


# Each model's preprocessing follows its pretraining normalization and resize interpolation
MODELS = {
    "dinov3_vith16plus": dict(encoder=DINOv3Encoder, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                              interpolation=transforms.InterpolationMode.BICUBIC),
    "ijepa_vith14": dict(encoder=IJEPAEncoder, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                         interpolation=transforms.InterpolationMode.BICUBIC),
    "clip_vith14": dict(encoder=CLIPEncoder, mean=CLIP_MEAN, std=CLIP_STD,
                        interpolation=transforms.InterpolationMode.BICUBIC),
    "imagenet_vith14": dict(encoder=ImageNetViTEncoder, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                            interpolation=transforms.InterpolationMode.BICUBIC),
}


parser = argparse.ArgumentParser()
parser.add_argument("--num_splits", type=int, default=1, help="Number of splits to divide the dataset into for feature extraction")
parser.add_argument("--split_idx", type=int, default=0, help="Index of the split to process")

parser.add_argument("--image_dir", type=str, default="images", help="Directory containing the images")
parser.add_argument("--save_dir", type=str, default="features", help="Directory to save the extracted features (one subfolder per model)")
parser.add_argument("--models", type=str, nargs="+", default=list(MODELS), choices=list(MODELS), help="Models to extract features with")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--hf_cache_dir", type=str, default="/scratch3/workspace/mchasmai_umass_edu-re_id/model_weights/hub",
                    help="HuggingFace hub cache for the I-JEPA, CLIP and DeiT III weights")


def get_transform(mean, std, interpolation):
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=interpolation),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            print(f"Skipping unreadable image {p}", flush=True)
            return None
        return self.transform(img), p


def collate_skip_unreadable(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    images, paths = zip(*batch)
    return torch.stack(images), list(paths)


@torch.no_grad()
def extract_features(model, loader, image_root, save_dir, device):
    for batch in tqdm.tqdm(loader):
        if batch is None:
            continue
        images, paths = batch

        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            features = model(images.to(device, non_blocking=True))
        features = features.float().cpu()

        for feat, p in zip(features, paths):
            feat_path = Path(save_dir) / Path(p).relative_to(image_root).with_suffix(".pt")
            feat_path.parent.mkdir(parents=True, exist_ok=True)
            # clone so each file stores only its own row rather than the whole batch
            torch.save(feat.clone(), feat_path)


if __name__ == "__main__":
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # recursive search for images in subdirectories
    img_paths = []
    for root, dirs, files in os.walk(args.image_dir):
        for file in files:
            if file.lower().endswith(IMG_EXTS):
                img_paths.append(os.path.join(root, file))

    img_paths = sorted(img_paths)
    print("Total", len(img_paths))

    img_paths = img_paths[args.split_idx::args.num_splits]
    print("In split", len(img_paths))

    for model_name in args.models:
        print(f"Extracting {model_name} features")
        cfg = MODELS[model_name]
        model = cfg["encoder"](args).eval().to(device)

        dataset = ImageDataset(img_paths, get_transform(cfg["mean"], cfg["std"], cfg["interpolation"]))
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=collate_skip_unreadable, pin_memory=device.type == "cuda")

        extract_features(model, loader, args.image_dir, os.path.join(args.save_dir, model_name), device)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
