import os
import torch
import random
from typing import List

# Import model and constants from city_gpt
from city_gpt import (
    GPTLanguageModel,
    MODEL_PATH,
    device,
    country_to_flag,
    FEATURE_CLASS_EMOJI,
    _CLASS_FALLBACK,
    FIELD_SEP,
    SOS_TOKEN,
    EOS_TOKEN
)

# Cache the loaded model and vocab
_CACHED_MODEL = None
_CACHED_STOI = None
_CACHED_ITOS = None

def load_model():
    global _CACHED_MODEL, _CACHED_STOI, _CACHED_ITOS
    
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL, _CACHED_STOI, _CACHED_ITOS
        
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}. Please train the model first.")
        
    ckpt = torch.load(MODEL_PATH, map_location=device)
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    
    model = GPTLanguageModel(len(stoi)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    _CACHED_MODEL = model
    _CACHED_STOI = stoi
    _CACHED_ITOS = itos
    
    return model, stoi, itos

def build_prompt(country_code: str, feature_cls: str, feature_code: str, name_prefix: str = "") -> str:
    """Build the prompt string expected by the model."""
    flag = country_to_flag(country_code) if country_code else ""
    
    cls_emoji = ""
    if feature_cls:
        cls_emoji = FEATURE_CLASS_EMOJI.get(feature_cls, _CLASS_FALLBACK)
        
    prompt = f"{flag}{cls_emoji}"
    
    if feature_code:
        prompt += feature_code
        
    if country_code or feature_cls or feature_code:
        prompt += FIELD_SEP
        
    prompt += name_prefix.lower()
    return prompt

@torch.no_grad()
def generate_names(
    prompt_str: str, 
    num_names: int = 1, 
    max_new_tokens: int = 60,
    temperature: float = 1.0, 
    top_k: int = None
) -> List[str]:
    """Generate multiple city names from a given prompt."""
    model, stoi, itos = load_model()
    
    encode = lambda s: [stoi[ch] for ch in s if ch in stoi]
    decode = lambda ids: "".join(itos[i] for i in ids)
    
    sos_id = stoi[SOS_TOKEN]
    eos_id = stoi[EOS_TOKEN]
    
    results = []
    
    # We will run generation in a batch for efficiency if possible, 
    # but the current generate() is designed for a single or pre-batched sequence.
    # We'll construct a batch of size `num_names` containing the prompt.
    prompt_ids = [sos_id] + encode(prompt_str)
    
    # Duplicate the prompt `num_names` times to create a batch
    ctx = torch.tensor(
        [prompt_ids for _ in range(num_names)],
        dtype=torch.long, device=device
    )
    
    out = model.generate(ctx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    
    for i in range(num_names):
        # Decode output
        generated_ids = out[i].tolist()
        
        # Remove prompt from output
        generated_ids = generated_ids[len(prompt_ids):]
        
        # Truncate at EOS token
        if eos_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(eos_id)]
            
        name = decode(generated_ids)
        results.append(name.title()) # Capitalize like a proper name
        
    return results

def get_country_options():
    """Return a curated list of common country codes for the dropdown."""
    return [
        "US", "CN", "IN", "MX", "ID", "RU", "PK", "BR", "JP", "DE",
        "FR", "GB", "IT", "CA", "AU", "ES", "ZA", "NG", "KR", "AR"
    ]

def get_feature_class_options():
    """Return the available feature classes with descriptions."""
    return {
        "P": "Populated Places (P)",
        "H": "Water/Hydrographic (H)",
        "S": "Spots/Structures (S)",
        "T": "Terrain/Mountains (T)",
        "A": "Administrative (A)",
        "L": "Parks/Areas (L)",
        "V": "Vegetation/Forests (V)",
        "R": "Roads (R)",
        "U": "Undersea (U)"
    }

def get_feature_code_options():
    """Return common feature codes."""
    return [
        "PPL", "STM", "HLL", "MT", "FRM", "LK", "SCH", "CH", "HTL", 
        "ISL", "BLDG", "VAL", "PT", "DAM", "PRK", "BAY", "CNL"
    ]
