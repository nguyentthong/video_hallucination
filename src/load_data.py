from loguru import logger
import os
import glob
import json
from typing import List, Dict, Any

def load_benchmark(video_dir: str, data_mode: str, questions_dir: str) -> List[Dict[str, Any]]:
    if data_mode == "all":
        json_files = glob.glob(os.path.join(questions_dir, "**", "*.json"), recursive=True)
        logger.info(f"Found {len(json_files)} JSON files in '{questions_dir}'")
    else:
        json_file = os.path.join(questions_dir, data_mode)
        logger.info(f"Loading specific data mode: '{json_file}'")
        json_files = [json_file]

    results = []
    for json_file in json_files:
        with open(json_file, encoding='utf-8') as fin:
            data = json.load(fin)
            data['video_path'] = str(os.path.join(video_dir, data['video_name']))
            results.append(data)

    return results