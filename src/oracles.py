"""Oracles answering a pairwise query (i, l): 1 if the two images belong to the same cluster, else 0.

    ground_truth           the true labels (simulated human)
    internvl, qwen         local vision LLMs via transformers (GPU if available); weights from HF Hub
    openai, gemini         hosted APIs; keys from OPENAI_API_KEY / GOOGLE_API_KEY in the environment

LLM answers are cached per (dataset, provider, model) so repeated runs never re-ask a pair.
"""
import io
import os
import time
import base64

from .io_utils import read_json, write_json
from .reid_datasets import image_path

ORACLES = ["ground_truth", "internvl", "qwen", "openai", "gemini"]

DEFAULT_MODELS = {
    "internvl": "OpenGVLab/InternVL3_5-30B-A3B-HF",
    "qwen": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "openai": "gpt-4o",
    "gemini": "gemini-3.6-flash",
}

# "Do these two images show ___?" -- what "same cluster" means for each dataset
SAME_CLUSTER = {
    "cars": "a car of the exact same make, model, AND model year (e.g. a 2012 Honda Civic Sedan vs. a 2012 "
            "Honda Civic Coupe, or the same model in two different years, would NOT count), not just the "
            "same general vehicle type or body style",
    "aircraft": "an aircraft of the exact same variant (the same manufacturer, family, AND variant "
                "designation -- e.g. a Boeing 737-800 vs. a 737-900 would NOT count), not just the same "
                "aircraft family",
    "cub": "a bird of the exact same species",
    "nabirds": "a bird of the exact same species AND the same plumage class (e.g. breeding male vs. "
               "female/juvenile of one species would NOT count)",
    "inat": "an organism of the exact same species",
    "ox_flower": "a flower of the exact same species",
    "food": "the exact same kind of dish",
    "plant_village": "a leaf of the same crop with the same disease (or both healthy)",
    "stanford_products": "the exact same product listing (the same specific item for sale, identifiable "
                         "by its own distinct design), not just the same product category",
    "imat_products": "products of the exact same fine-grained product type (the same specific "
                     "style/cut/silhouette), not just the same broad category such as 'dress' or 'shoes'",
    "vehicle_reid": "the exact same PHYSICAL vehicle (match details like the license plate, damage, dirt "
                    "or accessories, not just the same make/model/color)",
    "veri": "the exact same PHYSICAL vehicle (match details like the license plate, damage, dirt or "
            "accessories, not just the same make/model/color)",
    "more": "the exact same PHYSICAL motorcycle (match its own specific details, not just the same model)",
    "met": "the exact same artwork or museum exhibit",
    "happy_whale": "the exact same individual whale or dolphin",
    "sea_turtle": "the exact same individual sea turtle",
    "lynx": "the exact same individual lynx",
    "giraffe_zebra": "the exact same individual animal",
    "wild_track": "the exact same person",
}


def oracle_prompt(dataset):
    if dataset.startswith("uco3d_"):
        obj = dataset[len("uco3d_"):]
        target = (f"the exact same PHYSICAL {obj} (the same individual {obj} -- match its own details like "
                  f"color, pattern, wear or accessories, not just the same style, brand or model)")
    else:
        target = SAME_CLUSTER[dataset]
    return f"Do these two images show {target}? Answer with exactly one word: \"yes\" or \"no\"."


class GroundTruthOracle:
    def __init__(self, gt):
        self.gt = gt

    def __call__(self, i, l):
        return int(self.gt[i] == self.gt[l])


def parse_yes_no(text):
    t = text.strip().lower().strip(".! ")
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    if ("yes" in t) != ("no" in t):
        return int("yes" in t)
    raise ValueError(f"No clear yes/no in the LLM's answer {text!r}")


# ---------------------------------------------------------------------------
# Vision-LLM calls: (model, prompt, image path 1, image path 2) -> answer text
# ---------------------------------------------------------------------------

IMAGE_MAX_SIZE = 512  # both images are downsized to at most this, bounding cost/latency


def load_image(path):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((IMAGE_MAX_SIZE, IMAGE_MAX_SIZE), Image.LANCZOS)
    return img


_hf_models = {}


def ask_hf_vlm(model, prompt, path1, path2):
    """InternVL / Qwen-VL through transformers' native classes (no trust_remote_code), loaded once."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if model not in _hf_models:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Loading {model} on {device}...")
        vlm = AutoModelForImageTextToText.from_pretrained(model, dtype=dtype, low_cpu_mem_usage=True)
        _hf_models[model] = (vlm.eval().to(device), AutoProcessor.from_pretrained(model), device)
    vlm, processor, device = _hf_models[model]

    messages = [{"role": "user", "content": [
        {"type": "image", "image": load_image(path1)},
        {"type": "image", "image": load_image(path2)},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(device)
    inputs.pop("token_type_ids", None)  # produced by Qwen's processor, not accepted by generate()
    out = vlm.generate(**inputs, max_new_tokens=5, do_sample=False)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def ask_openai(model, prompt, path1, path2):
    from openai import OpenAI

    def data_url(path):
        buf = io.BytesIO()
        load_image(path).save(buf, format="JPEG")
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

    response = OpenAI().chat.completions.create(model=model, max_tokens=5, messages=[{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": data_url(path1)}},
        {"type": "image_url", "image_url": {"url": data_url(path2)}},
    ]}])
    return response.choices[0].message.content


def ask_gemini(model, prompt, path1, path2):
    from google import genai
    return genai.Client().models.generate_content(
        model=model, contents=[prompt, load_image(path1), load_image(path2)]).text


ASK = {"internvl": ask_hf_vlm, "qwen": ask_hf_vlm, "openai": ask_openai, "gemini": ask_gemini}


class LLMOracle:
    def __init__(self, dataset, img_names, provider, model, processed_root, cache_path, max_retries=3):
        self.dataset, self.img_names, self.processed_root = dataset, img_names, processed_root
        self.provider, self.model = provider, model or DEFAULT_MODELS[provider]
        self.prompt = oracle_prompt(dataset)
        self.cache_path, self.max_retries = cache_path, max_retries
        self.cache = read_json(cache_path) if os.path.exists(cache_path) else {}

    def __call__(self, i, l):
        names = sorted([self.img_names[i], self.img_names[l]])
        key = "|".join(names)
        if key not in self.cache:
            paths = [image_path(self.dataset, name, self.processed_root) for name in names]
            for attempt in range(self.max_retries):
                try:
                    self.cache[key] = parse_yes_no(ASK[self.provider](self.model, self.prompt, *paths))
                    break
                except Exception as e:
                    print(f"  {self.provider} call failed ({attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"{self.provider} gave no usable answer for {names}")
            write_json(self.cache_path, self.cache)
        return self.cache[key]


def make_oracle(cfg, cands, processed_root, cache_dir):
    if cfg.oracle == "ground_truth":
        return GroundTruthOracle(cands.gt)
    model = cfg.oracle_model or DEFAULT_MODELS[cfg.oracle]
    cache_path = os.path.join(cache_dir, f"oracle_{cfg.oracle}_{model.replace('/', '_')}.json")
    return LLMOracle(cands.dataset, cands.img_names, cfg.oracle, model, processed_root, cache_path)
