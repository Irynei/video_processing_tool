import json
import argparse
from PIL import Image
from typing import List
from collections import Counter
import cv2
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
import torchvision.models as torchvision_models


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """ Resize, normalize and convert to tensor """
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transformed_frame = transform(Image.fromarray(frame))
    return torch.unsqueeze(transformed_frame, 0)


def predict_frame_label(
    model: nn.Module, processed_frame: torch.Tensor, idx2label: List[str]
) -> str:
    """ Get top-1 prediction label given a frame and a model """
    out = model(processed_frame)
    label = idx2label[int(out.argmax())]
    return label


def get_torchvision_model_inference(model_name: str, pretrained: bool = True) -> nn.Module:
    """ Get pretrained model ready for inference """
    try:
        model = getattr(torchvision_models, model_name)
    except AttributeError:
        raise AttributeError(f'Model {model_name} not found')
    model_instance = model(pretrained=pretrained)
    model_instance.eval()
    return model_instance


def get_idx_label_mapping(class_index_mapping_file: str = 'imagenet_class_index.json') -> List[str]:
    """ Get list with class names (labels) """
    with open(class_index_mapping_file) as f:
        class_idx = json.load(f)
    return [class_idx[str(k)][1] for k in range(len(class_idx))]


def process_video(video_source: str, model: nn.Module) -> Counter:
    """
    Predict ImageNet class for every frame and collect most common classes
    """
    label_counts = Counter()
    idx2label = get_idx_label_mapping()
    video_cap = cv2.VideoCapture(video_source)
    frames_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Video contains {frames_count} frames. Start processing..')
    for _ in tqdm(range(frames_count)):
        ret, frame = video_cap.read()
        if not ret:
            break
        processed_frame = preprocess_frame(frame)
        label = predict_frame_label(model, processed_frame, idx2label)
        label_counts[label] += 1
    video_cap.release()
    return label_counts


def main():
    parser = argparse.ArgumentParser(description='Process video and collect most common classes')
    parser.add_argument('-v', '--video_path', default='video.mp4', help='Path to the video file')
    parser.add_argument(
        '-m',
        '--model',
        default='densenet121',
        help='Pretrained ImageNet model name from torchvision module',
    )
    parser.add_argument(
        '-k', '--top_k', default=3, type=int, help='How many most common classes to return'
    )
    args = parser.parse_args()
    model = get_torchvision_model_inference(args.model)
    results = process_video(args.video_path, model)
    print(f'Most common classes: {dict(results.most_common(args.top_k))}')


if __name__ == '__main__':
    main()
