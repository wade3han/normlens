import argparse
import json
import os
import time
from pathlib import Path

from tqdm import tqdm

from utils import OpenaiChatGpt

SYSTEM_PROMPT = """You are a succinct and helpful assistant."""

USER_PROMPT = """I will give you an information about the image. You can't see the image, but you can imagine the image with the information. 
Enumerate two or three actions that are morally appropriate for the given image, but are morally inappropriate for other contrastive images.

For example,

Given Image: a collage of people posing with a cake
1. Action: play a party game, Contrastive Image: a collage of people for business conference, Reason: playing a party game is morally appropriate for the given image, but is morally inappropriate for an image about serious or formal setting like "a collage of people for business conference".
2. Action: set a fire on the candle, Contrastive Image: an artwork in a museum, Reason: setting a fire on the candle is morally appropriate for the given image, but is morally inappropriate for an image about the place where fire is dangerous like "an artwork in a museum"
3. Action: cheer with the wine or beer, Contrastive Image: group of children playing in the part with a ball, Reason: cheering with the wine or beer is morally appropriate in the image, but is morally inappropriate for an image about children like "group of children playing in the park with a ball"

Enumerate actions when the image is given:
Given Image: {information}
1. Action:"""


def iterative_create_response(data_creater, caption):
    responses = []
    for i in range(1):
        context = USER_PROMPT.format(information=caption)
        response = data_creater.create_response(context)
        time.sleep(1)
        responses.append(response)
    return responses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--root-dir', type=str, required=True)
    args = parser.parse_args()

    data_creater = OpenaiChatGpt(engine='gpt-3.5-turbo',
                                 temperatue=0.7,
                                 topp=0.95,
                                 frequency_penalty=0.0,
                                 presence_penalty=0.0)
    root_dir = args.root_dir
    data_creater.set_system_prompt(SYSTEM_PROMPT)

    caption_path = f'{root_dir}/dataset_coco.json'
    with open(caption_path, 'r') as f:
        caption_data = json.load(f)['images']

    caption_data = caption_data[1000 * args.fold: 1000 * (args.fold + 1)]

    gpt_outputs_dir = f'{root_dir}/turbo_moral_confounders/'
    Path(gpt_outputs_dir).mkdir(parents=True, exist_ok=True)

    outputs = []
    for data in tqdm(caption_data):
        filename = data['filename']
        caption = data['sentences'][0]['raw'].strip()
        image_path = os.path.join(root_dir, 'train2014', filename) if 'train' in filename else \
            os.path.join(root_dir, 'val2014', filename)
        if not os.path.exists(image_path):
            continue

        response = iterative_create_response(data_creater, caption)
        data_creater.clear_chat_memory()
        output = {'image_path': image_path, 'caption': caption, 'response': response}

        outputs.append(output)
        if len(outputs) < 10:
            print(outputs[-1])

    output_path = os.path.join(gpt_outputs_dir, f'dataset_coco_fold{args.fold}.json')
    with open(output_path, 'w') as f:
        json.dump(outputs, f, indent=2)
