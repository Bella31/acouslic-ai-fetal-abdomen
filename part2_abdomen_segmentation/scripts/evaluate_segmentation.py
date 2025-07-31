import os
import torch
import pandas as pd
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from src.model_unet import UNet
from src.utils import SegmentationDataset, get_segmentation_transforms, hausdorff_distance, normalized_ac_error, dice_coefficient
from torch.utils.data import DataLoader
import argparse

def evaluate(model, loader, device, spacing_mm):
    dice_list, hd_list, nae_list = [], [], []
    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            preds_bin = (preds > 0.5).cpu().numpy().astype(np.uint8)
            truths = masks.cpu().numpy().astype(np.uint8)

            for p, t in zip(preds_bin, truths):
                dice_list.append(dice_coefficient(torch.tensor(p), torch.tensor(t)))
                hd_list.append(hausdorff_distance(p[0], t[0]))
                nae_list.append(normalized_ac_error(p[0], t[0], spacing_mm))

    return dice_list, hd_list, nae_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_images", type=str, required=True)
    parser.add_argument("--test_masks", type=str, required=True)
    parser.add_argument("--scan_dir", type=str, required=True)
    args = parser.parse_args()

    # Read pixel spacing from first MHA
    first_scan = sorted(os.listdir(args.scan_dir))[0]
    spacing_mm = sitk.ReadImage(os.path.join(args.scan_dir, first_scan)).GetSpacing()[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # DataLoader
    image_transform, mask_transform = get_segmentation_transforms()
    test_dataset = SegmentationDataset(args.test_images, args.test_masks, image_transform, mask_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluate
    dice_list, hd_list, nae_list = evaluate(model, test_loader, device, spacing_mm)

    # Summarize
    df = pd.DataFrame({"Dice": dice_list, "Hausdorff": hd_list, "NAE_AC": nae_list})
    print(df.agg(["mean", "std"]).T)

    # Compute extra scores
    df["HD_score"] = 1 / (1 + df["Hausdorff"])
    df["AC_score"] = 1 - df["NAE_AC"]
    df["Combined"] = (df["Dice"] + df["HD_score"] + df["AC_score"]) / 3
    summary = df[["Dice","HD_score","AC_score","Combined"]].agg(["mean","std"]).T
    print("\nSummary:")
    print(summary)
