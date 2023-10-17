import argparse
import json
import os
from pathlib import Path

from llama_index import GPTVectorStoreIndex
from llama_index.readers.schema.base import ImageDocument
from tqdm import tqdm

if __name__ == '__main__':
    # turn text captions into gpt embeddings, and store them
    parser = argparse.ArgumentParser()
    parser.add_argument('--datatype', type=str, choices=['sherlock', 'coco', 'narratives'])
    parser.add_argument('--root-dir', type=str, required=True)
    args = parser.parse_args()

    documents = []
    if args.datatype == 'sherlock':
        datapath = f'{args.root_dir}/sherlock_dataset/sherlock_train_v1_1.json'
        vis_root = args.root_dir
        with open(datapath, 'r') as f:
            data = json.load(f)
            for d in tqdm(data):
                image_url = d['inputs']['image']['url']
                input_path_split = image_url.split('/')
                if input_path_split[-3] == 'vcr1images':
                    input_path_simple = input_path_split[-3] + "/" + input_path_split[-2] + "/" + input_path_split[-1]
                    image_path = os.path.join(vis_root, input_path_simple)
                else:
                    input_path_simple = input_path_split[-2] + "/" + input_path_split[-1]
                    image_path = os.path.join(vis_root, "vg/images", input_path_simple)

                if not os.path.exists(image_path):
                    print(image_path)
                    continue

                instance_id = d['instance_id']
                clue = d['inputs']['clue']
                inference = d['targets']['inference']

                documents.append(
                    ImageDocument(image=str(image_path),
                                  text=inference,
                                  doc_id=instance_id, ))
    elif args.datatype == 'narratives':
        datapath = f'{args.root_dir}/openimages_localized_narratives/open_images_train_v6_captions.jsonl'
        image_dir = f'{args.root_dir}/image-captioning/openimages_v6_images'
        with open(datapath, 'r') as f:
            all_lines = f.readlines()
            for line in tqdm(all_lines):
                data = json.loads(line.strip())
                image_id = data['image_id']
                caption = data['caption']

                image_path = Path(image_dir) / f'{image_id}.jpg'
                if not image_path.exists():
                    print(image_path)
                    continue

                documents.append(
                    ImageDocument(image=str(image_path),
                                  text=caption,
                                  doc_id=image_id, ))
    elif args.datatype == 'coco':
        datapath = f'{args.root_dir}/coco/dataset_coco.json'
        image_dir = f'{args.root_dir}/coco'
        with open(datapath, 'r') as f:
            data = json.load(f)
            images = data['images']
            for image in tqdm(images):
                filepath = image['filepath']
                filename = image['filename']
                image_path = Path(image_dir) / filepath / filename
                if not image_path.exists():
                    print(image_path)
                    continue

                sentences = image['sentences']
                for sentence in sentences:
                    caption = sentence['raw']
                    img_id = sentence['imgid']
                    sent_id = sentence['sentid']
                    doc_id = f'{img_id}_{sent_id}'
                    documents.append(
                        ImageDocument(image=str(image_path),
                                      text=caption,
                                      doc_id=doc_id, ))
    else:
        raise NotImplementedError

    index = GPTVectorStoreIndex.from_documents(documents)
    persist_dir = Path(datapath).parent / f'{args.datatype}_index'
    persist_dir.mkdir(exist_ok=True, parents=True)
    index.storage_context.persist(persist_dir=str(persist_dir))
    print('done')
