import json
import numpy as np
import cv2
import os
from typing import List, Tuple, Dict, Any

def compute_center_from_bbox(bbox: List[float]) -> Tuple[float, float]:
    """
    Compute center point from bounding box [x, y, width, height]
    """
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    return center_x, center_y

def compute_center_from_polygon(segmentation: List[List[float]]) -> Tuple[float, float]:
    """
    Compute center point from polygon segmentation
    """
    if not segmentation or not segmentation[0]:
        return 0.0, 0.0
    
    # Take the first polygon if multiple exist
    polygon = segmentation[0]
    
    # Convert to numpy array and reshape to (n, 2) points
    points = np.array(polygon).reshape(-1, 2)
    
    # Compute centroid
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])
    
    return float(center_x), float(center_y)

def compute_center_from_mask(mask_path: str) -> Tuple[float, float]:
    """
    Compute center point from binary mask image
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0.0, 0.0
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0, 0.0
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Compute moments
    M = cv2.moments(largest_contour)
    
    if M["m00"] == 0:
        return 0.0, 0.0
    
    center_x = M["m10"] / M["m00"]
    center_y = M["m01"] / M["m00"]
    
    return float(center_x), float(center_y)

def process_coco_annotations(annotation_file: str, output_file: str, target_category_id: int = 2):
    """
    Process COCO format annotations and extract center points for specific category
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    center_points = []
    
    # Process annotations
    for ann in data.get('annotations', []):
        if ann.get('category_id') == target_category_id:
            image_id = ann.get('image_id')
            
            # Try to compute center from different annotation types
            center_x, center_y = 0.0, 0.0
            
            if 'bbox' in ann:
                center_x, center_y = compute_center_from_bbox(ann['bbox'])
            elif 'segmentation' in ann:
                center_x, center_y = compute_center_from_polygon(ann['segmentation'])
            
            center_points.append({
                'image_id': image_id,
                'center_x': center_x,
                'center_y': center_y,
                'annotation_id': ann.get('id', -1)
            })
    
    # Save to text file
    with open(output_file, 'w') as f:
        f.write("image_id,center_x,center_y,annotation_id\n")
        for point in center_points:
            f.write(f"{point['image_id']},{point['center_x']:.2f},{point['center_y']:.2f},{point['annotation_id']}\n")
    
    print(f"Saved {len(center_points)} center points to {output_file}")
    return center_points

def process_custom_annotations(annotation_file: str, output_file: str, target_category_id: int = 2):
    """
    Process custom format annotations (assuming JSON with bbox format)
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    center_points = []
    
    # Assuming data structure like: {"images": [{"annotations": [...]}]}
    for image_data in data.get('images', []):
        image_name = image_data.get('name', 'unknown')
        
        for ann in image_data.get('annotations', []):
            if ann.get('category_id') == target_category_id:
                if 'bbox' in ann:
                    center_x, center_y = compute_center_from_bbox(ann['bbox'])
                    center_points.append({
                        'image_name': image_name,
                        'center_x': center_x,
                        'center_y': center_y
                    })
    
    # Save to text file
    with open(output_file, 'w') as f:
        f.write("image_name,center_x,center_y\n")
        for point in center_points:
            f.write(f"{point['image_name']},{point['center_x']:.2f},{point['center_y']:.2f}\n")
    
    print(f"Saved {len(center_points)} center points to {output_file}")
    return center_points

def main():
    # Configuration
    annotation_file = "/home/sadegh/tomRobotics/Inpaint-Anything/01_Cam1_MorningPeak/instances_remapped.json"
    output_file = "/home/sadegh/tomRobotics/Inpaint-Anything/center_points/center_points_category_2.txt"
    target_category_id = 2
    
    # Check if annotation file exists
    if not os.path.exists(annotation_file):
        print(f"Error: Annotation file not found: {annotation_file}")
        return
    
    try:
        # Try COCO format first
        center_points = process_coco_annotations(annotation_file, output_file, target_category_id)
    except (KeyError, json.JSONDecodeError) as e:
        print(f"COCO format failed: {e}")
        try:
            # Try custom format
            center_points = process_custom_annotations(annotation_file, output_file, target_category_id)
        except Exception as e:
            print(f"Custom format failed: {e}")
            print("Please check your annotation file format and update the code accordingly.")

if __name__ == "__main__":
    main()