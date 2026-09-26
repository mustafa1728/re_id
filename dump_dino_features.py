import argparse
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os
import tqdm
from torchvision.transforms import functional as F
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--num_splits", type=int, default=1, help="Number of splits to divide the dataset into for feature extraction")
parser.add_argument("--split_idx", type=int, default=0, help="Index of the split to process")

parser.add_argument("--image_dir", type=str, default="images", help="Directory containing the images")
parser.add_argument("--save_dir", type=str, default="features", help="Directory to save the extracted features")
parser.add_argument("--model", type=str, default="dinov2_vitb14", help="Model to use for feature extraction")
parser.add_argument("--flatten", action="store_true", default=False, help="Flatten output directory structure (default: preserve subdirectory layout)")

args = parser.parse_args()


if "dinov2" in args.model:
    model = torch.hub.load('facebookresearch/dinov2', args.model)
elif "dinov3" in args.model:

    REPO_DIR = "/home/mchasmai_umass_edu/bird_parts/inat/dinov3"
    if "vitb16" in args.model:
        MODEL_WEIGHTS = "/work/pi_smaji_umass_edu/mchasmai/model_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    elif "vitl16" in args.model:
        MODEL_WEIGHTS = "/work/pi_smaji_umass_edu/mchasmai/model_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    elif "vits16" in args.model:
        MODEL_WEIGHTS = "/work/pi_smaji_umass_edu/mchasmai/model_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    elif "vith16plus" in args.model:
        MODEL_WEIGHTS = "/work/pi_smaji_umass_edu/mchasmai/model_weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
    elif "vit7b16" in args.model:
        MODEL_WEIGHTS = "/work/pi_smaji_umass_edu/mchasmai/model_weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
    else:
        raise ValueError(f"Unsupported DINOv3 model: {args.model}")
    

    # DINOv3 ViT models pretrained on web images
    model = torch.hub.load(REPO_DIR, args.model, source='local', weights=MODEL_WEIGHTS)
else:
    raise ValueError(f"Unsupported model: {args.model}")




# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.eval()
model.to(device)

def get_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        # ResizeKeepAspectPad(518),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return transform




def extract_dino_features(image_paths, save_dir, batch_size=8, flatten=False, image_root=None):
    all_features = []
    seen_folders = {}
    transform = get_transform()
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            valid_paths = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                except:
                    continue
                images.append(transform(img))
                valid_paths.append(p)
            if not images:
                continue

            images = torch.stack(images).to(device)


            # Forward pass - CLS token
            with torch.amp.autocast('cuda', dtype=torch.float16):
                features = model(images)
                # features_flipped = model(images_flipped)


            
            for i in range(len(valid_paths)):
                p = Path(valid_paths[i])
                if flatten or image_root is None:
                    feat_path = Path(save_dir) / (p.stem + ".pt")
                else:
                    rel = p.relative_to(image_root)
                    feat_path = Path(save_dir) / rel.with_suffix(".pt")
                    feat_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(features[i].cpu(), feat_path)

                # flipped_feat_path = os.path.join(save_subdir, img_name.split(".")[0] + "_flipped.pt")
                # torch.save(features_flipped[i].cpu(), flipped_feat_path)

    # return torch.cat(all_features, dim=0)



# ROOT_DIR = "/scratch3/workspace/mchasmai_umass_edu-re_id/imat"
# IMG_ROOT = os.path.join(ROOT_DIR, "images")
# FEAT_ROOT = os.path.join(ROOT_DIR, f"features/{args.model}")
# os.makedirs(FEAT_ROOT, exist_ok=True)

IMG_ROOT = args.image_dir
FEAT_ROOT = os.path.join(args.save_dir, args.model)
os.makedirs(FEAT_ROOT, exist_ok=True)

# img_paths = []
# for img_name in os.listdir(IMG_ROOT):
#     img_paths.append(os.path.join(IMG_ROOT, img_name))


# recursive search for images in subdirectories
img_paths = []
for root, dirs, files in os.walk(IMG_ROOT):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_paths.append(os.path.join(root, file))


img_paths = sorted(img_paths)

print("Total", len(img_paths))

img_paths = img_paths[args.split_idx::args.num_splits]

print("In split", len(img_paths))

extract_dino_features(img_paths, save_dir=FEAT_ROOT, batch_size=128, flatten=args.flatten, image_root=IMG_ROOT)