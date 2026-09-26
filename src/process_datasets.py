"""
Converts the datasets loaded by reid_datasets.py from their original formats into
one common on-disk format:

    <out_root>/<dataset>/
        images/<category>/<img_name>.png
        img_name2label.json        {"<category>/<img_name>": label}
        img_name2attributes.json   {"<category>/<img_name>": {attribute: value}}
        stats.json                 dataset, task, num_images, num_categories, attributes

<category> is a folder-safe version of the label (e.g. aircraft variant "F-16A/B" ->
"F-16A_B"); img_name2label keeps the original label. The keys of img_name2label are
image paths relative to images/ (without extension), so features dumped with the
same directory layout can be loaded as feat_root/<key>.pt.

sri_data only ships tracklet embeddings (no images), so it is not processed. uco3d is
split into one dataset per object category ("uco3d_<category>"), same as reid_datasets.
Large datasets are subsampled to ~50K images (see the constants below), and giraffe_zebra
images are cropped to one image per annotated animal.

Usage:
    python src/process_datasets.py                                   # everything
    python src/process_datasets.py --datasets cars aircraft uco3d    # a subset
    python src/process_datasets.py --datasets all --dry_run          # labels/stats only
"""
import os
import re
import json
import zlib
import shutil
import argparse
import multiprocessing as mp
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
import tqdm


from config import DATA_ROOT, PROCESSED_DATA_ROOT as OUT_ROOT

BASE_DATASETS = ["cars", "aircraft", "stanford_products", "imat_products", "vehicle_reid", "more", "wild_track",
                 "cub", "nabirds", "inat", "ox_flower", "food", "plant_village", "met", "veri",
                 "happy_whale", "sea_turtle", "lynx", "giraffe_zebra"]

TASKS = {
    "cars": "fine-grained car model recognition",
    "aircraft": "fine-grained aircraft variant recognition",
    "stanford_products": "product instance re-identification",
    "imat_products": "fine-grained product recognition",
    "vehicle_reid": "vehicle re-identification",
    "more": "motorcycle re-identification",
    "uco3d": "object instance re-identification",
    "wild_track": "person re-identification",
    "cub": "fine-grained bird species recognition",
    "nabirds": "fine-grained bird species recognition (by sex / age / morph)",
    "inat": "fine-grained species recognition",
    "ox_flower": "fine-grained flower species recognition",
    "food": "fine-grained food dish recognition",
    "plant_village": "plant disease recognition",
    "met": "artwork instance re-identification",
    "veri": "vehicle re-identification",
    "happy_whale": "whale and dolphin re-identification",
    "sea_turtle": "sea turtle re-identification",
    "lynx": "lynx re-identification",
    "giraffe_zebra": "giraffe and zebra re-identification",
}

# stanford_products: keep this many of the classes with the most images
SOP_NUM_CLASSES = 4526
# imat_products: keep 1/IMAT_IMG_FRAC of the images in every class
IMAT_IMG_FRAC = 20
# food: keep 1/FOOD_IMG_FRAC of the images in every class
FOOD_IMG_FRAC = 2
# inat: keep this many species (at random) of train_mini, which has 50 images per species
INAT_NUM_CLASSES = 1000
# met: keep this many of the exhibits with the most images (-> 50,003 images)
MET_NUM_CLASSES = 5655
# happy_whale: keep this many of the individuals with the most images
HAPPY_WHALE_NUM_CLASSES = 5000


def read_txt(path):
    with open(path, "r") as f:
        lines = f.read().splitlines()
    return lines


def read_id_file(path):
    # "<id> <value>" per line, as in the CUB / NABirds metadata files
    return dict(line.split(" ", 1) for line in read_txt(path) if line)


def read_label_xml(path):
    with open(path, 'r', encoding='gbk') as file:
        xml_string = file.read()
    # Strip out the <?xml ... encoding="gb2312"?> header so it doesn't crash the parser
    return ET.fromstring(re.sub(r'<\?xml.*?\?>', '', xml_string))


def top_labels(label2count, num_labels, rng):
    # The num_labels labels with the most images; ties at the cutoff are broken at random
    labels = sorted(label2count)
    tiebreak = rng.random(len(labels))
    order = sorted(range(len(labels)), key=lambda i: (-label2count[labels[i]], tiebreak[i]))
    return {labels[i] for i in order[:num_labels]}


def sanitize(name):
    # Folder-safe version of a label, e.g. aircraft variant "F-16A/B" -> "F-16A_B"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")


def list_uco3d_categories(data_root):
    # reid_datasets clusters features of the masked frames, so use those images
    img_root = os.path.join(data_root, "uco3d/data_download/masked")
    cat2super = {}
    for super_cat in sorted(os.listdir(img_root)):
        super_dir = os.path.join(img_root, super_cat)
        if not os.path.isdir(super_dir):
            continue
        for category in sorted(os.listdir(super_dir)):
            cat2super[category] = super_cat
    return img_root, cat2super


def is_readable(path):
    try:
        with Image.open(path):
            return True
    except Exception:
        return False


def get_records(dataset_name, data_root, rng, num_workers):
    """Returns one {"src", "name", "label", "attrs"} dict per image of the dataset, plus an
    optional "crop" box [x1, y1, x2, y2] to cut the image out of "src"."""
    records = []

    if dataset_name == "cars":
        ROOT_DIR = os.path.join(data_root, "cars")

        # Use the train and test images together
        for split in ["train", "test"]:
            IMG_ROOT = os.path.join(ROOT_DIR, f"cars_{split}/cars_{split}")
            if not os.path.isdir(IMG_ROOT):
                IMG_ROOT = os.path.join(ROOT_DIR, f"cars_{split}")

            anns_path = os.path.join(ROOT_DIR, f'car_devkit/devkit/cars_{split}_annos.mat')
            anns = loadmat(anns_path)
            assert "class" in anns['annotations'].dtype.names, f"{anns_path} has no class labels"
            frame = [[i.flat[0] for i in line] for line in anns['annotations'][0]]

            columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
            df = pd.DataFrame(frame, columns=columns)

            for _, row in df.iterrows():
                records.append({
                    "src": os.path.join(IMG_ROOT, row['fname']),
                    # train and test file names overlap (both start at 00001.jpg)
                    "name": f"{split}_{row['fname'].split('.')[0]}",
                    "label": row['class'],
                    "attrs": {
                        "split": split,
                        "bbox": [int(row[k]) for k in ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']],
                    },
                })

    elif dataset_name == "aircraft":
        ROOT_DIR = os.path.join(data_root, "aircraft/fgvc-aircraft-2013b/data")
        IMG_ROOT = os.path.join(ROOT_DIR, "images")

        def get_cls(path):
            lines = read_txt(path)
            mapping = {}
            for line in lines:
                if line == "": continue
                mapping[line.split()[0]] = " ".join(line.split()[1:])
            return mapping

        boxes = {}
        for line in read_txt(os.path.join(ROOT_DIR, "images_box.txt")):
            if line == "": continue
            boxes[line.split()[0]] = [int(x) for x in line.split()[1:]]

        for split in ["train", "val", "test"]:
            variant = get_cls(os.path.join(ROOT_DIR, f"images_variant_{split}.txt"))
            family = get_cls(os.path.join(ROOT_DIR, f"images_family_{split}.txt"))
            manufacturer = get_cls(os.path.join(ROOT_DIR, f"images_manufacturer_{split}.txt"))
            for img_id, label in variant.items():
                records.append({
                    "src": os.path.join(IMG_ROOT, f"{img_id}.jpg"),
                    "name": img_id,
                    "label": label,
                    "attrs": {
                        "family": family[img_id],
                        "manufacturer": manufacturer[img_id],
                        "split": split,
                        "bbox": boxes[img_id],
                    },
                })

    elif dataset_name == "stanford_products":
        ROOT_DIR = os.path.join(data_root, "stanford_products/Stanford_Online_Products")

        train_ids = {line.split()[0] for line in read_txt(os.path.join(ROOT_DIR, "Ebay_train.txt"))[1:] if line}
        for line in read_txt(os.path.join(ROOT_DIR, "Ebay_info.txt"))[1:]:
            if line == "": continue
            image_id, class_id, super_class_id, path = line.split()
            img_id = os.path.basename(path).split(".")[0]
            records.append({
                "src": os.path.join(ROOT_DIR, path),
                "name": img_id,
                "label": img_id.split("_")[0],
                "attrs": {
                    "super_class": path.split("/")[0].replace("_final", ""),
                    "split": "train" if image_id in train_ids else "test",
                },
            })

        # Keep the SOP_NUM_CLASSES classes with the most images; most classes have
        # only a handful of images, so ties at the cutoff are broken at random
        selected_labels = top_labels(Counter(r["label"] for r in records), SOP_NUM_CLASSES, rng)
        records = [r for r in records if r["label"] in selected_labels]

    elif dataset_name == "imat_products":
        ROOT_DIR = os.path.join(data_root, "imat_products")
        IMG_ROOT = os.path.join(ROOT_DIR, "images")

        with open(os.path.join(ROOT_DIR, "metadata/train.json"), "r") as f:
            train_anns = json.load(f)

        # product_tree.json: level1 node -> level2 node -> level3 node -> [class ids]
        with open(os.path.join(ROOT_DIR, "metadata/product_tree.json"), "r") as f:
            product_tree = json.load(f)
        class2levels = {}
        for l1, l2_nodes in product_tree.items():
            for l2, l3_nodes in l2_nodes.items():
                for l3, classes in l3_nodes.items():
                    for c in classes:
                        class2levels[int(c)] = {"level1": l1, "level2": f"{l1}/{l2}", "level3": f"{l1}/{l2}/{l3}"}

        # Not every train image could be downloaded, and ~1/4 of the files on disk aren't
        # valid images, so only keep the readable ones (same as the feature dumps)
        on_disk = set(os.listdir(IMG_ROOT))
        img_ids = [img["id"] for img in train_anns["images"] if img["id"] in on_disk]
        with mp.Pool(num_workers) as pool:
            paths = [os.path.join(IMG_ROOT, img_id) for img_id in img_ids]
            readable = list(tqdm.tqdm(pool.imap(is_readable, paths, chunksize=256), total=len(paths), desc="Checking images"))

        img_id2label = {img["id"]: img["class"] for img in train_anns["images"]}
        label2img_ids = defaultdict(list)
        for img_id, ok in zip(img_ids, readable):
            if ok:
                label2img_ids[img_id2label[img_id]].append(img_id)

        # Keep 1/IMAT_IMG_FRAC of the images in every class (at least one, so no class is dropped)
        for label in sorted(label2img_ids):
            img_ids = sorted(label2img_ids[label])
            selected_img_ids = rng.choice(img_ids, size=max(1, len(img_ids) // IMAT_IMG_FRAC), replace=False)
            for img_id in sorted(selected_img_ids):
                records.append({
                    "src": os.path.join(IMG_ROOT, img_id),
                    "name": img_id.split(".")[0],
                    "label": label,
                    "attrs": dict(class2levels[label]),
                })

    elif dataset_name == "vehicle_reid":
        ROOT_DIR = os.path.join(data_root, "vehicle_reid/AIC21_Track2_ReID")
        IMG_ROOT = os.path.join(ROOT_DIR, "image_train")

        root = read_label_xml(os.path.join(ROOT_DIR, "train_label.xml"))

        # Each line of train_track.txt lists the images of one tracklet
        track_lines = [line for line in read_txt(os.path.join(ROOT_DIR, "train_track.txt")) if line.strip()]
        img2track = {img_name: track_id for track_id, line in enumerate(track_lines) for img_name in line.split()}

        for item in root.findall('.//Item'):
            img_name = item.get('imageName')
            records.append({
                "src": os.path.join(IMG_ROOT, img_name),
                "name": img_name.split(".")[0],
                "label": item.get('vehicleID'),
                "attrs": {"camera_id": item.get('cameraID'), "track_id": img2track.get(img_name)},
            })

    elif dataset_name == "more":
        ROOT_DIR = os.path.join(data_root, "more/MoRe")

        # paths look like pair01/camA/camA_id_03578_num_001.png
        for split in ["train", "test"]:
            for path in read_txt(os.path.join(ROOT_DIR, f"{split}_files.txt")):
                if path == "": continue
                pair, camera, fname = path.split("/")
                name = fname.split(".")[0]
                records.append({
                    "src": os.path.join(ROOT_DIR, path),
                    "name": name,
                    "label": name.split("_")[2],
                    "attrs": {"pair": pair, "camera": camera, "split": split},
                })

    elif dataset_name.startswith("uco3d_"):
        category = dataset_name[len("uco3d_"):]
        img_root, cat2super = list_uco3d_categories(data_root)
        if category not in cat2super:
            raise ValueError(f"uco3d category '{category}' not found under {img_root}")
        CAT_ROOT = os.path.join(img_root, cat2super[category], category)

        # each 3D sequence is one identity; each extracted frame is one observation
        for seq_id in sorted(os.listdir(CAT_ROOT)):
            seq_dir = os.path.join(CAT_ROOT, seq_id)
            if not os.path.isdir(seq_dir):
                continue
            for fname in sorted(os.listdir(seq_dir)):
                if not fname.lower().endswith((".jpg", ".png")):
                    continue
                name = fname.split(".")[0]
                records.append({
                    "src": os.path.join(seq_dir, fname),
                    "name": name,
                    "label": seq_id,
                    "attrs": {"super_category": cat2super[category], "category": category, "frame": int(name.split("_")[-1])},
                })

    elif dataset_name == "wild_track":
        IMG_ROOT = os.path.join(data_root, "wild_track/objects/person")

        # crops are stored as <person_id>/<person_id>_<frame>_<camera>.png
        for person_id in sorted(os.listdir(IMG_ROOT)):
            person_dir = os.path.join(IMG_ROOT, person_id)
            if not os.path.isdir(person_dir):
                continue
            for fname in sorted(os.listdir(person_dir)):
                name = fname.split(".")[0]
                _, frame, camera = name.split("_")
                records.append({
                    "src": os.path.join(person_dir, fname),
                    "name": name,
                    "label": person_id,
                    "attrs": {"camera": camera, "frame": int(frame)},
                })

    elif dataset_name in ["cub", "nabirds"]:
        # Both use the same metadata format, keyed by image id; bboxes are x y w h
        ROOT_DIR = os.path.join(data_root, "cub/CUB_200_2011" if dataset_name == "cub" else "nabirds/nabirds")
        paths = read_id_file(os.path.join(ROOT_DIR, "images.txt"))
        img_id2class = read_id_file(os.path.join(ROOT_DIR, "image_class_labels.txt"))
        class_names = read_id_file(os.path.join(ROOT_DIR, "classes.txt"))
        is_train = read_id_file(os.path.join(ROOT_DIR, "train_test_split.txt"))
        boxes = read_id_file(os.path.join(ROOT_DIR, "bounding_boxes.txt"))

        for img_id, path in paths.items():
            x, y, w, h = [int(float(v)) for v in boxes[img_id].split()]
            attrs = {"split": "train" if is_train[img_id] == "1" else "test", "bbox": [x, y, x + w, y + h]}
            label = class_names[img_id2class[img_id]]
            if dataset_name == "cub":
                # "001.Black_footed_Albatross" -> "Black_footed_Albatross"
                label = label.split(".", 1)[1]
            else:
                # The 555 NABirds classes split species by sex / age / morph where these look
                # different, e.g. "Mallard (Breeding male)" and "Mallard (Female/juvenile)"
                attrs["species"] = re.sub(r"\s*\(.*\)", "", label).strip()
            records.append({
                "src": os.path.join(ROOT_DIR, "images", path),
                "name": os.path.basename(path).split(".")[0],
                "label": label,
                "attrs": attrs,
            })

    elif dataset_name == "inat":
        ROOT_DIR = os.path.join(data_root, "inat")

        with open(os.path.join(ROOT_DIR, "train_mini.json"), "r") as f:
            anns = json.load(f)
        categories = {c["id"]: c for c in anns["categories"]}
        img_id2cat = {ann["image_id"]: ann["category_id"] for ann in anns["annotations"]}

        # train_mini has 50 images for each of its 10K species; keep INAT_NUM_CLASSES of them at random
        selected_cats = set(rng.choice(sorted(categories), size=INAT_NUM_CLASSES, replace=False).tolist())
        for img in anns["images"]:
            cat = categories[img_id2cat[img["id"]]]
            if cat["id"] not in selected_cats:
                continue
            records.append({
                "src": os.path.join(ROOT_DIR, img["file_name"]),
                "name": os.path.basename(img["file_name"]).split(".")[0],
                "label": cat["name"],
                "attrs": {k: cat[k] for k in ["common_name", "supercategory", "kingdom", "phylum", "class", "order", "family", "genus"]},
            })

    elif dataset_name == "ox_flower":
        ROOT_DIR = os.path.join(data_root, "ox_flower")

        # labels[i] is the class (1-102) of jpg/image_<i+1>.jpg
        labels = loadmat(os.path.join(ROOT_DIR, "imagelabels.mat"))["labels"][0]
        for i, label in enumerate(labels):
            name = f"image_{i + 1:05d}"
            records.append({
                "src": os.path.join(ROOT_DIR, "jpg", f"{name}.jpg"),
                "name": name,
                "label": int(label),
                "attrs": {},
            })

    elif dataset_name == "food":
        ROOT_DIR = os.path.join(data_root, "food/food-101")

        # meta/<split>.txt lists the images of the split as <class>/<img_id>
        label2names, path2split = defaultdict(list), {}
        for split in ["train", "test"]:
            for path in read_txt(os.path.join(ROOT_DIR, "meta", f"{split}.txt")):
                if path == "": continue
                label, name = path.split("/")
                label2names[label].append(name)
                path2split[path] = split

        # Keep 1/FOOD_IMG_FRAC of the images in every class
        for label in sorted(label2names):
            names = sorted(label2names[label])
            for name in sorted(rng.choice(names, size=len(names) // FOOD_IMG_FRAC, replace=False)):
                records.append({
                    "src": os.path.join(ROOT_DIR, "images", label, f"{name}.jpg"),
                    "name": name,
                    "label": label,
                    "attrs": {"split": path2split[f"{label}/{name}"]},
                })

    elif dataset_name == "plant_village":
        IMG_ROOT = os.path.join(data_root, "plant_village/PlantVillage-Dataset/raw/color")

        # class folders are <crop>___<disease>, e.g. "Apple___Apple_scab" or "Apple___healthy"
        for label in sorted(os.listdir(IMG_ROOT)):
            crop, disease = label.split("___")
            for fname in sorted(os.listdir(os.path.join(IMG_ROOT, label))):
                records.append({
                    "src": os.path.join(IMG_ROOT, label, fname),
                    # file names look like "<uuid>___RS_Erly.B 7778.JPG"
                    "name": sanitize(os.path.splitext(fname)[0]),
                    "label": label,
                    "attrs": {"crop": crop, "disease": disease},
                })

    elif dataset_name == "met":
        ROOT_DIR = os.path.join(data_root, "met")

        # Each exhibit is one class. Only the training images are on disk (the test / val
        # queries in test_met/ are not), and most of the 224K exhibits have only one or two
        # images, so keep the MET_NUM_CLASSES exhibits with the most images
        with open(os.path.join(ROOT_DIR, "MET_database.json"), "r") as f:
            db = json.load(f)
        selected_labels = top_labels(Counter(img["id"] for img in db), MET_NUM_CLASSES, rng)
        for img in db:
            if img["id"] not in selected_labels:
                continue
            # paths look like MET/<exhibit_id>/<n>.jpg
            records.append({
                "src": os.path.join(ROOT_DIR, img["path"]),
                "name": os.path.basename(img["path"]).split(".")[0],
                "label": img["id"],
                "attrs": {},
            })

    elif dataset_name == "veri":
        ROOT_DIR = os.path.join(data_root, "veri/VeRi")

        id2color = dict(line.split() for line in read_txt(os.path.join(ROOT_DIR, "list_color.txt")) if line)
        id2type = dict(line.split() for line in read_txt(os.path.join(ROOT_DIR, "list_type.txt")) if line)

        # image_query is a copy of part of image_test, so only train and test are used;
        # the two have disjoint vehicle ids
        for split in ["train", "test"]:
            root = read_label_xml(os.path.join(ROOT_DIR, f"{split}_label.xml"))
            for item in root.findall('.//Item'):
                img_name = item.get('imageName')
                records.append({
                    "src": os.path.join(ROOT_DIR, f"image_{split}", img_name),
                    "name": img_name.split(".")[0],
                    "label": item.get('vehicleID'),
                    "attrs": {
                        "camera_id": item.get('cameraID'),
                        "color": id2color[item.get('colorID')],
                        "type": id2type[item.get('typeID')],
                        "split": split,
                    },
                })

    elif dataset_name == "happy_whale":
        ROOT_DIR = os.path.join(data_root, "happy_whale/happy_whale")

        # Only train_images are labeled. Keep the HAPPY_WHALE_NUM_CLASSES individuals with the
        # most images; most have only one or two, so ties at the cutoff are broken at random
        df = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"))
        selected_labels = top_labels(Counter(df["individual_id"]), HAPPY_WHALE_NUM_CLASSES, rng)
        for row in df.to_dict("records"):
            if row["individual_id"] not in selected_labels:
                continue
            records.append({
                "src": os.path.join(ROOT_DIR, "train_images", row["image"]),
                "name": row["image"].split(".")[0],
                "label": row["individual_id"],
                "attrs": {"species": row["species"]},
            })

    elif dataset_name == "sea_turtle":
        ROOT_DIR = os.path.join(data_root, "sea_turtle")

        with open(os.path.join(ROOT_DIR, "annotations.json"), "r") as f:
            anns = json.load(f)
        images = {img["id"]: img for img in anns["images"]}

        # The images are already crops of one turtle head each (one annotation per image)
        for ann in anns["annotations"]:
            img = images[ann["image_id"]]
            records.append({
                "src": os.path.join(ROOT_DIR, img["path"]),
                "name": os.path.basename(img["path"]).split(".")[0],
                "label": ann["identity"],
                "attrs": {"position": ann["position"], "date": img["date"]},
            })

    elif dataset_name == "lynx":
        ROOT_DIR = os.path.join(data_root, "lynx/lynx")

        # Only the real camera-trap images, not CzechLynx_Synthetic. The mask / pose columns
        # (RLE segmentation and keypoints) are too large to keep as attributes
        df = pd.read_csv(os.path.join(ROOT_DIR, "CzechLynxDataset-Metadata-Real.csv"))
        attr_cols = ["source", "date", "encounter", "relative_age", "coat_pattern", "location", "trap_id",
                     "latitude", "longitude", "split-geo_aware", "split-time_open", "split-time_closed", "split-pose"]
        for row in df.to_dict("records"):
            records.append({
                "src": os.path.join(ROOT_DIR, row["path"]),
                "name": os.path.basename(row["path"]).split(".")[0],
                "label": row["unique_name"],
                "attrs": {k: None if pd.isna(row[k]) else row[k] for k in attr_cols},
            })

    elif dataset_name == "giraffe_zebra":
        ROOT_DIR = os.path.join(data_root, "giraffe_zebra/gzgc.coco")

        # val2020 / test2020 are empty, every annotated image is in train2020
        with open(os.path.join(ROOT_DIR, "annotations/instances_train2020.json"), "r") as f:
            anns = json.load(f)
        images = {img["id"]: img for img in anns["images"]}
        species = {c["id"]: c["name"] for c in anns["categories"]}

        # An image can show several animals, so every annotation is cropped out as its own image
        for ann in anns["annotations"]:
            img = images[ann["image_id"]]
            # bbox is x y w h, rotated by theta about its center; crop the axis-aligned box
            # around the rotated one (the same for either rotation direction)
            x, y, w, h = ann["bbox"]
            cos, sin = abs(np.cos(ann["theta"])), abs(np.sin(ann["theta"]))
            cx, cy, half_w, half_h = x + w / 2, y + h / 2, (w * cos + h * sin) / 2, (w * sin + h * cos) / 2
            crop = [max(0, int(cx - half_w)), max(0, int(cy - half_h)),
                    min(img["width"], int(np.ceil(cx + half_w))), min(img["height"], int(np.ceil(cy + half_h)))]
            records.append({
                "src": os.path.join(ROOT_DIR, "images/train2020", img["file_name"]),
                "name": f"{img['file_name'].split('.')[0]}_{ann['id']}",
                "label": ann["name"],
                "attrs": {"species": species[ann["category_id"]], "viewpoint": ann["viewpoint"], "image": img["file_name"],
                          "bbox": crop, "theta": ann["theta"]},
                "crop": crop,
            })

    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    return records


def save_png(job):
    key, src, dst, crop, overwrite = job
    if os.path.exists(dst) and not overwrite:
        return key, None
    # Write to a temp file first so an interrupted run never leaves a truncated image behind
    tmp = dst + ".tmp"
    try:
        with Image.open(src) as img:
            if crop is None and img.format == "PNG" and img.mode == "RGB":
                shutil.copyfile(src, tmp)
            else:
                if crop is not None:
                    img = img.crop(crop)
                img.convert("RGB").save(tmp, format="PNG")
        os.replace(tmp, dst)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        return key, repr(e)
    return key, None


def process_dataset(dataset_name, data_root, out_root, num_workers, overwrite=False, dry_run=False):
    print(f"\n===== {dataset_name} =====")
    # Same per-dataset seeding as reid_datasets, so reruns draw the identical subsample
    rng = np.random.default_rng(zlib.crc32(dataset_name.encode()))
    records = get_records(dataset_name, data_root, rng, num_workers)

    missing = [r for r in records if not os.path.exists(r["src"])]
    if missing:
        print(f"WARNING: skipping {len(missing)} images missing on disk, e.g. {missing[0]['src']}")
        records = [r for r in records if os.path.exists(r["src"])]

    out_dir = os.path.join(out_root, dataset_name)
    img_name2label, img_name2attributes, jobs = {}, {}, []
    category2label = {}
    for r in records:
        label = str(r["label"])
        category = sanitize(label)
        if category2label.setdefault(category, label) != label:
            raise ValueError(f"Labels '{label}' and '{category2label[category]}' both map to folder '{category}'")
        key = f"{category}/{r['name']}"
        if key in img_name2label:
            raise ValueError(f"Duplicate image name '{key}'")
        img_name2label[key] = label
        img_name2attributes[key] = r["attrs"]
        jobs.append((key, r["src"], os.path.join(out_dir, "images", key + ".png"), r.get("crop"), overwrite))

    if not dry_run:
        for category in category2label:
            os.makedirs(os.path.join(out_dir, "images", category), exist_ok=True)

        failed = {}
        with mp.Pool(num_workers) as pool:
            for key, err in tqdm.tqdm(pool.imap_unordered(save_png, jobs, chunksize=32), total=len(jobs)):
                if err is not None:
                    failed[key] = err
        if failed:
            print(f"WARNING: failed to convert {len(failed)} images, dropping them, e.g. {next(iter(failed.items()))}")
            for key in failed:
                del img_name2label[key]
                del img_name2attributes[key]

    task_key = "uco3d" if dataset_name.startswith("uco3d_") else dataset_name
    stats = {
        "dataset": dataset_name,
        "task": TASKS[task_key] + (f" ({dataset_name[len('uco3d_'):]})" if task_key == "uco3d" else ""),
        "num_images": len(img_name2label),
        "num_categories": len(set(img_name2label.values())),
        "attributes": sorted({attr for attrs in img_name2attributes.values() for attr in attrs}),
    }
    print(json.dumps(stats, indent=4))

    if not dry_run:
        with open(os.path.join(out_dir, "img_name2label.json"), "w") as f:
            json.dump(img_name2label, f, indent=4)
        with open(os.path.join(out_dir, "img_name2attributes.json"), "w") as f:
            json.dump(img_name2attributes, f)
        with open(os.path.join(out_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=4)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DATA_ROOT, help="Directory containing the original datasets")
    parser.add_argument("--out_root", type=str, default=OUT_ROOT, help="Directory to save the processed datasets")
    parser.add_argument("--datasets", type=str, nargs="+", default=["all"],
                        help="Datasets to process: any of BASE_DATASETS, 'uco3d_<category>', 'uco3d' (every category) or 'all'")
    parser.add_argument("--num_workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())),
                        help="Number of processes used to convert images")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Re-convert images that already exist")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Only build labels/stats, don't write anything")
    args = parser.parse_args()

    dataset_names = []
    for name in args.datasets:
        if name in ["all", "uco3d"]:
            _, cat2super = list_uco3d_categories(args.data_root)
            uco3d_datasets = [f"uco3d_{category}" for category in cat2super]
            dataset_names += (BASE_DATASETS if name == "all" else []) + uco3d_datasets
        else:
            dataset_names.append(name)

    for dataset_name in dataset_names:
        process_dataset(dataset_name, args.data_root, args.out_root, args.num_workers, args.overwrite, args.dry_run)
