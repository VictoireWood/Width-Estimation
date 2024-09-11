import torch
from tqdm import tqdm

from torch.utils.data import DataLoader

def inference(model:torch.nn.Module, test_dl:DataLoader, range_threshold, test_img_num, device):
    tqdm_bar = tqdm(range(test_img_num), ncols=100, desc="")
    model.eval()

    count_valid_recall = torch.zeros(len(range_threshold))
    valid_recall_percentage = torch.zeros(len(range_threshold))
    range_threshold_tensor = torch.tensor(range_threshold)

    valid_heigths = torch.zeros(test_img_num, len(range_threshold),dtype=torch.bool)

    with torch.no_grad():
        # tqdm_bar = tqdm(range(), ncols=100, desc="")
        # for images, heights_gt in test_dl:
        for query_i, (images,heights_gt) in enumerate(test_dl):
            images = images.to(device)
            heights_gt = torch.tensor(heights_gt).to(device)
            recall_heights_range = torch.zeros(test_img_num, len(range_threshold))
            # with torch.autocast(device):
            heights_pred = model(images)
            distances = abs(heights_pred - heights_gt)
            range_threshold_tensor = range_threshold_tensor.to(distances.device)
            valid_heigths[query_i,:] = distances < range_threshold_tensor

            tqdm_bar.set_description(f"{query_i:.5d}")
            _ = tqdm_bar.refresh()
            _ = tqdm_bar.update()

        count_valid_recall = torch.count_nonzero(valid_heigths, dim=0)
        valid_recall_percentage = 100 * count_valid_recall / test_img_num

    val_recall_str = ", ".join([f'LR@{N}: {acc:.2f}' for N, acc in zip(range_threshold, valid_recall_percentage)])
    return val_recall_str, valid_recall_percentage