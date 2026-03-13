import os
import json
from typing import List, Optional

class AnswerCacheSystem:
    def __init__(self, model_id: str, cache_dir: str = "cache"):
        self.model_id = model_id.replace("/", "__")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def push(self, sample_idx: str, answers: List[str]):
        cache_path = os.path.join(self.cache_dir, f"sample_{sample_idx}.json")
        with open(cache_path, 'w+', encoding='utf-8') as fout:
            json.dump({"answers": answers}, fout)

    def get(self, sample_idx: str) -> Optional[List[str]]:
        if not self.exist(sample_idx):
            return None
        cache_path = os.path.join(self.cache_dir, f"sample_{sample_idx}.json")
        with open(cache_path, encoding='utf-8') as fin:
            data = json.load(fin)

        return data['answers']

    def exist(self, sample_idx) -> bool:
        cache_path = os.path.join(self.cache_dir, f"sample_{sample_idx}.json")
        return os.path.exists(cache_path)

def get_cache_id(video_name: str) -> str:
    cache_id = video_name[:4]
    if cache_id == '0036':
        cache_id = "_".join(video_name.split("_")[:2])
    return cache_id