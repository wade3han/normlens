import json
import os
from typing import List, Dict, Any

import requests
from tqdm import tqdm


def download_file(remote_file_path: str, filename: str) -> None:
    if os.path.exists(filename):
        print(f"File {filename} already exists.")
    else:
        try:
            r = requests.get(remote_file_path, timeout=10)
            with open(filename, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            print(e, f"Can't download file from {remote_file_path}")


def download_datas_from_jsonl(jsonl_path: str, local_dir: str) -> List[Dict[str, Any]]:
    """
    Example of single instance:
        {"question_id": 924,
         "image": "https://storage.googleapis.com/ai2-mosaic-public/projects/normlens/image/3612252751541471262.jpg",
         "text": "have a barbecue with friends and family",
         "answer_judgment": [2, 2, 2, 2],
         "answer_explanation": ["You cannot have a barbecue while studying in your room.",
                                "You would not be able to barbecue inside of a bedroom",
                                "You can't have a bbq with friends and family inside a enclosed space like a bedroom.",
                                "You can't barbecue from your bed"],
         "image_src": "sherlock",
         "caption": "the person is studying for a test"}

    Note that 0 refers to Wrong., 1 refers to Ok., and 2 refers to Impossible.
    """
    datas = []
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            data = json.loads(line.strip())
            image_file_path = data['image']
            download_file(image_file_path, f'{local_dir}/image/{image_file_path.split("/")[-1]}')
            datas.append(data)

    return datas


remote_dir = 'https://storage.googleapis.com/ai2-mosaic-public/projects/normlens/data'
local_dir = '.'
os.makedirs(f'{local_dir}/image', exist_ok=True)

high_agreement_jsonl = f'{remote_dir}/high_agreement.jsonl'
mid_agreement_jsonl = f'{remote_dir}/mid_agreement.jsonl'

# download jsonl files
download_file(high_agreement_jsonl, f'{local_dir}/high_agreement.jsonl')
download_file(mid_agreement_jsonl, f'{local_dir}/mid_agreement.jsonl')

# read jsonl files
high_agreement_data = download_datas_from_jsonl(f'{local_dir}/high_agreement.jsonl', f'{local_dir}/image')
mid_agreement_data = download_datas_from_jsonl(f'{local_dir}/mid_agreement.jsonl', f'{local_dir}/image')

# sanity check
assert len(high_agreement_data) == 934
assert len(mid_agreement_data) == 1049
