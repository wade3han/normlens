import argparse
import json
import os

from tqdm import tqdm

from data_collection.vector_retriever import get_retriever

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--root-dir', type=str, required=True)
    args = parser.parse_args()

    datapath = args.datapath
    with open(datapath, 'r') as f:
        datas = json.load(f)

    output_path = datapath.replace('.json', '_moral_inappropriate_text_retrieval.json')
    if os.path.exists(output_path):
        print(f'Already exists, {output_path}')
        exit()

    retrievers = get_retriever(args.root_dir)

    morally_inappropriate = []
    for d in datas:
        generated_example = d['generated_example']
        moral_judgment = d['moral_judgment']
        if 'morally inappropriate' in moral_judgment and 'not morally inappropriate' not in moral_judgment:
            morally_inappropriate.append(d)

    selected_examples = []
    for d in tqdm(morally_inappropriate):
        try:
            generated_example = d['generated_example']
            moral_judgment = d['moral_judgment']

            image, action = generated_example.split('\n')
            if not image.startswith('Image: ') or not action.startswith('Action: '):
                print(generated_example)
                continue

            image = image[len('Image: '):]
            action = action[len('Action: '):]

            results = {}
            for retriever_name, retriever in retrievers.items():
                retrieved_results = retriever.retrieve(image)
                results[retriever_name] = [
                    {
                        'text': r.node.text,
                        'image': r.node.image,
                        'score': float(r.score),
                    }
                    for r in retrieved_results
                ]

            d['image_retrieval'] = results
            selected_examples.append(d)
        except Exception as e:
            print(generated_example)
            continue

    with open(output_path, 'w') as f:
        json.dump(selected_examples, f, indent=2)
