import copy
from typing import Sequence, Optional
import os

import pandas as pd
import torch
from skimage.io import imread, imshow
from skimage.segmentation import slic
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms

import tqdm

from approximators.base import powerset


def get_superpixels(image: np.ndarray, n_segments=10, exact_n=True, n_segments_iter: int = None,
                    max_iter=10):
    if n_segments_iter is None:
        n_segments_iter = n_segments
    segments_slic = slic(image, n_segments=n_segments_iter, compactness=10, sigma=1, start_label=0)
    n_superpixels = len(np.unique(segments_slic))
    if not exact_n or n_superpixels == n_segments:
        return n_superpixels, segments_slic
    if n_superpixels < n_segments and max_iter > 0:
        n_superpixels, segments_slic = get_superpixels(
            image=image, n_segments=n_segments, exact_n=True, n_segments_iter=n_segments_iter + 1,
            max_iter=max_iter - 1)
    if n_superpixels >= n_segments:
        segments_slic = np.clip(segments_slic, a_min=1, a_max=n_segments - 1)
        return n_segments, segments_slic
    raise RuntimeError(f"Could not find a set of superpixels with {n_segments} segments.")


def mask_superpixels(image: np.ndarray, superpixel_mask: np.ndarray, coalition: Sequence,
                     return_image: bool = True, pbar=None):
    coalition = set(coalition)
    image_replaced = np.zeros_like(image)
    image_replaced[:, :] = [127, 127, 127]
    for super_pixel_id in coalition:
        segment_mask = np.where(superpixel_mask == super_pixel_id)
        image_replaced[segment_mask] = image[segment_mask]
    if return_image:
        image_replaced = Image.fromarray(image_replaced)
    if pbar is not None:
        pbar.update(1)
    return copy.deepcopy(image_replaced)


def mask_image(image: np.ndarray):
    masked_image = image.copy()
    masked_image[:, :] = [127, 127, 127]
    return masked_image


def get_masked_images_from_coalitions(image: np.ndarray, coalitions: list,
                                      superpixel_mask: Optional[np.ndarray] = None, pbar=None):
    if superpixel_mask is None:
        _, superpixel_mask = get_superpixels(image=image)
    images = [mask_superpixels(image=image, superpixel_mask=superpixel_mask, coalition=coalition,
                               pbar=pbar)
              for coalition in coalitions]
    images = np.array(images, dtype=object)
    return images


def eval_complete(file_path, n_players=14):
    img = imread(file_path, as_gray=False)

    tensor_transform = transforms.ToTensor()
    img_tensor = Image.fromarray(img)
    img_tensor = tensor_transform(img_tensor)

    # create superpixels
    n_superpixels, superpixels = get_superpixels(image=img, n_segments=n_players)
    print("Number of superpixels found:", n_superpixels)

    # get coalitions
    coalitions = [subset for subset in powerset(range(n_superpixels))]
    coalitions_lists = [list(subset) for subset in coalitions]

    # create masked images
    with tqdm.tqdm(total=2 ** n_players, desc="Generating Masks") as pbar:
        masked_images = get_masked_images_from_coalitions(image=img, superpixel_mask=superpixels,
                                                          coalitions=coalitions_lists, pbar=pbar)

    # init model
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    # init inference
    preprocess = weights.transforms()

    # apply inference preprocessing transforms
    batch = preprocess(img_tensor).unsqueeze(0)

    # predict
    prediction = model(batch).squeeze(0).softmax(0)
    original_class_id = prediction.argmax().item()
    original_score = prediction[original_class_id].item()
    category_name = weights.meta["categories"][original_class_id]
    print(f"{category_name}: {original_score}")

    # convert images to big tensor
    image_tensors = []
    with tqdm.tqdm(total=2 ** n_players, desc="Creating Torch Tensors") as pbar:
        for image_masked in masked_images:
            image_tensors.append(preprocess(tensor_transform(image_masked)).unsqueeze(0))
            pbar.update(1)
        image_tensors = torch.cat(image_tensors, dim=0)

    # predict masks
    predictions = []
    batch_size = 25
    with tqdm.tqdm(total=2 ** n_players, desc="Running the Model") as pbar:
        for i in range(0, len(image_tensors), batch_size):
            batch = image_tensors[i:i + batch_size]
            prediction_batch = model(batch).softmax(1)
            score_batch = prediction_batch[:, original_class_id].tolist()
            predictions.extend(score_batch)
            pbar.update(batch_size)

    predictions = np.asarray(predictions)
    return coalitions, predictions


def pre_compute_game(n_players, data_folder, n_images=100):
    file_names = os.listdir(data_folder)
    save_folder = os.path.join("games", "data", "image_classifier", str(n_players))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    save_files = os.listdir(save_folder)
    count = 0
    for i, file_name in enumerate(file_names):
        count += 1
        if count > n_images:
            break
        file_id = file_name.split('.')[0]
        if file_id + '.csv' in save_files:
            continue
        print(f"\nPrecomputing {file_name} no. {count}\n")
        file_path = os.path.join(data_folder, file_name)
        try:
            coalitions, predictions = eval_complete(file_path=file_path, n_players=n_players)
        except RuntimeError:
            print(f"Stopped Computation because the subset size could not be found for "
                  f"{file_name} and {n_players}.")
            continue
        except Exception as error:
            print(f"Something happended: {error}")
            continue
        storage_data = []
        for subset, prediction in zip(coalitions, predictions):
            subset_key = 's'
            for player in subset:
                subset_key += str(player)
            storage_data.append({"set": subset_key, "value": prediction})
        save_path = os.path.join(save_folder, file_id + '.csv')
        pd.DataFrame(storage_data).to_csv(save_path, index=False)


if __name__ == "__main__":
    pre_compute_game(n_players=14, data_folder="games/data/images")
