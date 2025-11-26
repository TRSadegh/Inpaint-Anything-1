import torch
import sys
import argparse
import numpy as np
import json
from pathlib import Path
from matplotlib import pyplot as plt

# Assuming these utilities and core functions exist in your environment
from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points


def setup_args(parser):
    # Changed from --input_img to --input_dir to support folder processing
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the directory containing images to process (e.g., .png, .jpg)")
    # Changed from --coords_file to --coco_json_file to support annotation input
    parser.add_argument("--coco_json_file", type=str, required=True,
                        help="Path to the COCO format JSON annotation file.")
    
    parser.add_argument("--dilate_kernel_size", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="./output")

    parser.add_argument(
        "--sam_model_type", type=str, default="vit_l",
        choices=['vit_h', 'vit_l', 'vit_b', 'vit_t']
    )
    parser.add_argument("--sam_ckpt", type=str, default="./pretrained_models/sam_vit_l_0b3195.pth")

    parser.add_argument(
        "--lama_config",
        type=str,
        default="./lama/configs/prediction/default.yaml",
    )
    parser.add_argument("--lama_ckpt", type=str, default="./pretrained_models/big-lama")


def extract_car_center_points(coco_path, image_name):
    """
    Extracts the center points of bounding boxes for 'car' (category_id=2) 
    from a COCO JSON file for a specific image.
    """
    CAR_CATEGORY_ID = 2
    
    try:
        with open(coco_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: COCO JSON file not found at {coco_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON file at {coco_path}")
        return []

    # 1. Find the image ID based on the file name
    image_id = None
    for img_info in data.get('images', []):
        if img_info.get('file_name') == image_name:
            image_id = img_info['id']
            break

    if image_id is None:
        print(f"  Warning: Image '{image_name}' not found in COCO JSON 'images' section.")
        return []

    # 2. Extract center points for category ID 2 (Car)
    car_center_points = []
    
    for ann in data.get('annotations', []):
        if ann.get('image_id') == image_id and ann.get('category_id') == CAR_CATEGORY_ID:
            # bbox is [x, y, w, h]
            x, y, w, h = ann['bbox']
            center_x = x + w / 2
            center_y = y + h / 2
            car_center_points.append([center_x, center_y])
    
    return car_center_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_folder = Path(args.input_dir)
    if not input_folder.is_dir():
        print(f"Error: Input path '{args.input_dir}' is not a valid directory.")
        sys.exit(1)

    # Find all supported image files in the directory
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(input_folder.glob(ext))

    if not image_files:
        print(f"No images found in {input_folder}. Exiting.")
        sys.exit(0)

    print(f"Found {len(image_files)} images to process from {args.input_dir}.")

    # --- Main Loop: Iterate over each image file ---
    for current_img_path in image_files:
        print(f"\n---Starting processing for image: {current_img_path.name} ---")

        # 1. Extract coordinates specific to this image from COCO JSON
        coordinates_list = extract_car_center_points(args.coco_json_file, current_img_path.name)
        
        if not coordinates_list:
            print(f"  No 'car' (category ID 2) annotations found for {current_img_path.name}. Skipping.")
            continue

        print(f"  Found {len(coordinates_list)} car center points to process simultaneously.")

        # Prepare output directory
        img_stem = current_img_path.stem
        base_out_dir = Path(args.output_dir) / img_stem
        base_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the extracted coordinates for documentation
        coords_out_p = base_out_dir / f"{img_stem}_coords.txt"
        with open(coords_out_p, 'w') as f:
            for x, y in coordinates_list:
                f.write(f"{x} {y}\n")
        print(f"  Saved {len(coordinates_list)} coordinates to: {coords_out_p}")


        # 2. Load the original image once
        img_original = load_img_to_array(str(current_img_path))
        img_h, img_w = img_original.shape[:2]
        
        # Initialize the combined mask array (must be 0-255 uint8 for saving)
        combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        point_labels = [1] # always foreground point

        # --- Mask Aggregation Loop: Generate and combine masks for ALL points ---
        for i, latest_coords in enumerate(coordinates_list):
            
            # Predict masks with SAM using the original image as context
            masks, _, _ = predict_masks_with_sam(
                img_original,
                [latest_coords],
                point_labels,
                model_type=args.sam_model_type,
                ckpt_p=args.sam_ckpt,
                device=device,
            )
            
            # Use only mask index 2 (as in the original script)
            target_idx = 2
            if target_idx >= len(masks):
                print(f"    Warning: Point {i+1} only returned {len(masks)} masks. Skipping this point.")
                continue

            # 3. Apply Dilation
            current_mask = masks[target_idx]
            if args.dilate_kernel_size is not None:
                current_mask = dilate_mask(current_mask, args.dilate_kernel_size)
            
            # 4. Combine the mask using logical OR (union)
            # We use max since masks are 0 (background) or 1 (mask content)
            # Note: current_mask here is a boolean/float array from sam_segment/dilate_mask
            combined_mask = np.maximum(combined_mask, current_mask * 255)
            
        print(f"  Successfully combined {len(coordinates_list)} masks.")
        
        # --- Visualization for Combined Mask ---
        dpi = plt.rcParams['figure.dpi']
        plt.figure(figsize=(img_w/dpi/0.77, img_h/dpi/0.77))
        plt.imshow(img_original)
        plt.axis('off')
        # Show all points on the original image
        show_points(plt.gca(), coordinates_list, [1] * len(coordinates_list), size=(img_w*0.04)**2)
        show_mask(plt.gca(), combined_mask, random_color=False)
        
        combined_mask_p = base_out_dir / "combined_mask.png"
        plt.savefig(combined_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # 5. Single Inpainting Step
        print("  Starting single inpainting process...")
        img_inpainted = inpaint_img_with_lama(
            img_original, combined_mask.astype(np.uint8), args.lama_config, args.lama_ckpt, device=device
        )
        
        # 6. Save Final Output
        inpainted_p = base_out_dir / "inpainted_final.png"
        save_array_to_img(img_inpainted, inpainted_p)

        print(f"Finished processing. Final inpainted image saved to: {inpainted_p}")

    print("\nAll images processed successfully.")