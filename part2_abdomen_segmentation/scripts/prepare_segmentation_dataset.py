import os
import shutil
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
import argparse

def prepare_segmentation_dataset(frames_folder1, frames_folder2, mask_dir, output_img_dir, output_mask_dir):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # Gather all frame paths
    frame_paths = []
    for folder in [frames_folder1, frames_folder2]:
        for fname in os.listdir(folder):
            if fname.endswith(".jpg"):
                frame_paths.append(os.path.join(folder, fname))

    # Process each frame
    for frame_path in tqdm(frame_paths, desc="Processing frames"):
        fname = os.path.basename(frame_path)
        if "_" not in fname:
            continue  # skip bad format
        name_no_ext = os.path.splitext(fname)[0]
        scan_id, frame_str = name_no_ext.rsplit("_", 1)
        frame_idx = int(frame_str)
        mha_path = os.path.join(mask_dir, f"{scan_id}.mha")

        try:
            # Copy image
            shutil.copy(frame_path, os.path.join(output_img_dir, fname))

            # Read mask volume and extract frame
            mask_volume = sitk.GetArrayFromImage(sitk.ReadImage(mha_path))
            mask_frame = mask_volume[frame_idx]

            # Save binary mask as jpg (1=abdomen, 0=background)
            mask_img = Image.fromarray((mask_frame == 1).astype("uint8") * 255)
            mask_img.convert("L").save(os.path.join(output_mask_dir, f"{name_no_ext}.jpg"))

        except Exception as e:
            print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare segmentation dataset (images + masks) for U-Net")
    parser.add_argument("--frames_folder1", type=str, required=True, help="Path to optimal frames folder")
    parser.add_argument("--frames_folder2", type=str, required=True, help="Path to suboptimal frames folder")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to directory containing MHA masks")
    parser.add_argument("--output_img_dir", type=str, required=True, help="Output directory for images")
    parser.add_argument("--output_mask_dir", type=str, required=True, help="Output directory for masks")
    args = parser.parse_args()

    prepare_segmentation_dataset(
        args.frames_folder1, args.frames_folder2, args.mask_dir, args.output_img_dir, args.output_mask_dir
    )
