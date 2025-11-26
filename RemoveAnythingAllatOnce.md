# Remove Anything All at Once

This script is an enhanced version of the original `remove_anything.py` that processes **multiple images in batch** and uses **COCO format annotations** to automatically detect and remove cars from images.

## Overview

The `remove_anything_all_at_once.py` script combines the power of Segment Anything Model (SAM) and LaMa inpainting to automatically remove cars from multiple images using COCO format annotations. It processes entire folders of images efficiently by combining all detected objects into a single mask before inpainting.

## Key Features

- **Batch Processing**: Processes entire folders of images automatically
- **COCO Integration**: Uses COCO format annotations to detect cars
- **Efficient Mask Combination**: Combines multiple object masks before inpainting
- **Automated Workflow**: No manual coordinate input required

## How It Works

### 1. Input Processing
- Reads a folder of images (PNG, JPG, JPEG)
- Uses COCO JSON annotations to find car bounding boxes
- Calculates center points of each car for segmentation

### 2. Mask Generation
- For each image, extracts all car coordinates from annotations
- Uses SAM to generate individual masks for each car
- Combines all masks using logical OR operation

### 3. Single Inpainting
- Performs one inpainting operation using the combined mask
- More efficient than sequential inpainting operations
- Maintains better image quality

## Installation

Make sure you have the required dependencies:

```bash
pip install torch numpy matplotlib pathlib json
```

You'll also need the following modules in your environment:
- `sam_segment`
- `lama_inpaint` 
- `utils`

## Usage

### Basic Command

```bash
python remove_anything_all_at_once.py \
    --input_dir ./images_folder \
    --coco_json_file ./annotations.json \
    --output_dir ./results \
    --sam_ckpt ./pretrained_models/sam_vit_l_0b3195.pth \
    --lama_ckpt ./pretrained_models/big-lama
```

### Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--input_dir` | str | ✓ | - | Path to directory containing images |
| `--coco_json_file` | str | ✓ | - | Path to COCO format JSON annotation file |
| `--output_dir` | str | ✗ | `./output` | Output directory for results |
| `--dilate_kernel_size` | int | ✗ | `20` | Kernel size for mask dilation |
| `--sam_model_type` | str | ✗ | `vit_l` | SAM model type (vit_h/vit_l/vit_b/vit_t) |
| `--sam_ckpt` | str | ✗ | `./pretrained_models/sam_vit_l_0b3195.pth` | Path to SAM checkpoint |
| `--lama_config` | str | ✗ | `./lama/configs/prediction/default.yaml` | LaMa config file |
| `--lama_ckpt` | str | ✗ | `./pretrained_models/big-lama` | Path to LaMa checkpoint |

## COCO Annotation Format

The script expects a COCO format JSON file with the following structure:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 2,
      "bbox": [100, 200, 150, 80]
    }
  ],
  "categories": [
    {
      "id": 2,
      "name": "car"
    }
  ]
}
```

**Note**: The script specifically looks for `category_id = 2` (cars). Modify the `CAR_CATEGORY_ID` variable if needed.

## Output Structure

For each processed image, the script creates:

```
output/
└── image_name/
    ├── image_name_coords.txt      # List of extracted coordinates
    ├── combined_mask.png          # Visualization with all points and mask
    └── inpainted_final.png        # Final result with cars removed
```

### Output Files

1. **`{image_name}_coords.txt`**: Contains extracted center coordinates
   ```
   750.5 400.2
   920.1 350.8
   ```

2. **`combined_mask.png`**: Visualization showing:
   - Original image
   - All detected points marked
   - Combined mask overlay

3. **`inpainted_final.png`**: Final inpainted image with all cars removed

## Code Structure

### Key Functions

#### `setup_args(parser)`
Configures command-line arguments for batch processing and COCO integration.

#### `extract_car_center_points(coco_path, image_name)`
- Reads COCO JSON file
- Finds image ID by filename
- Extracts bounding boxes for cars (category_id=2)
- Calculates center points: `[x + w/2, y + h/2]`

#### Main Processing Loop
```python
for current_img_path in image_files:
    # 1. Extract coordinates from COCO
    coordinates_list = extract_car_center_points(args.coco_json_file, current_img_path.name)
    
    # 2. Generate and combine masks
    combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for coords in coordinates_list:
        masks = predict_masks_with_sam(img_original, [coords], [1], ...)
        current_mask = masks[2]  # Use mask index 2
        combined_mask = np.maximum(combined_mask, current_mask * 255)
    
    # 3. Single inpainting operation
    img_inpainted = inpaint_img_with_lama(img_original, combined_mask, ...)
```

## Key Differences from Original

| Feature | Original `remove_anything.py` | Enhanced Version |
|---------|-------------------------------|------------------|
| Input | Single image | Batch processing (folder) |
| Coordinates | Manual input/clicking | COCO annotations |
| Processing | One object at a time | All objects simultaneously |
| Masks | Individual processing | Combined mask approach |
| Efficiency | Multiple inpainting passes | Single inpainting pass |

## Performance Benefits

1. **Efficiency**: Single inpainting operation vs multiple passes
2. **Quality**: Avoids quality degradation from sequential inpainting
3. **Automation**: No manual coordinate input required
4. **Scalability**: Processes entire datasets automatically

## Error Handling

The script includes robust error handling for:
- Invalid input directories
- Missing COCO JSON files
- JSON parsing errors
- Images not found in annotations
- Insufficient mask generation

## Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended)
- Pretrained SAM and LaMa models
- COCO format annotation file

## Example Workflow

1. Prepare your images in a folder
2. Create/obtain COCO format annotations
3. Download pretrained models
4. Run the script:

```bash
python remove_anything_all_at_once.py \
    --input_dir ./car_images \
    --coco_json_file ./car_annotations.json \
    --output_dir ./results
```

5. Check results in the output directory

## Troubleshooting

**No cars found**: Check that your COCO file uses `category_id = 2` for cars

**CUDA out of memory**: Use a smaller SAM model (`vit_b` or `vit_t`)

**Poor quality results**: Increase `dilate_kernel_size` or use higher quality SAM model (`vit_h`)

## License

This project follows the same license as the original Inpaint-Anything repository.