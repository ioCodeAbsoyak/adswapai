"""AdSwapAI R&D, 2025-04-28: convert a VGG Image Annotator (VIA) polygon CSV export into a COCO json.

Input : via_annotations.csv (one row per polygon, region_shape_attributes holds all_points_x/y)
        images/ (the annotated frames; only needed to read the image size)
Output: converted_dataset_real.json with a single category "pitch side billboards"
"""
import csv
import json
import os
from PIL import Image

# Settings
CSV_FILE = "via_annotations.csv"             # VIA csv export
IMAGE_ROOT = "images"                        # folder with the annotated frames (see docs/assets.md)
OUTPUT_JSON = "converted_dataset_real.json"  # COCO json to write

# COCO template
coco_output = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "pitch side billboards", "supercategory": "none"}]
}

image_id_map = {}
annotation_id = 1
image_id_counter = 1

with open(CSV_FILE, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        filename = row['filename']
        image_path = os.path.join(IMAGE_ROOT, filename)

        # Register the image once
        if filename not in image_id_map:
            if not os.path.exists(image_path):
                print(f"WARNING: {image_path} not found, skipping")
                continue

            img = Image.open(image_path)
            width, height = img.size

            coco_output['images'].append({
                "id": image_id_counter,
                "file_name": filename,
                "width": width,
                "height": height
            })
            image_id_map[filename] = image_id_counter
            image_id_counter += 1

        image_id = image_id_map[filename]

        # Skip rows without a region
        if row['region_shape_attributes'] and len(row['region_shape_attributes']) > 10:
            region = json.loads(row['region_shape_attributes'].replace('\"\"', '\"'))
            all_points_x = region['all_points_x']
            all_points_y = region['all_points_y']

            segmentation = []
            for x, y in zip(all_points_x, all_points_y):
                segmentation.append(x)
                segmentation.append(y)

            # Bounding box from the polygon
            x_min = min(all_points_x)
            y_min = min(all_points_y)
            width_box = max(all_points_x) - x_min
            height_box = max(all_points_y) - y_min

            coco_output['annotations'].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [segmentation],
                "area": width_box * height_box,
                "bbox": [x_min, y_min, width_box, height_box],
                "iscrowd": 0
            })

            annotation_id += 1

with open(OUTPUT_JSON, 'w') as jsonfile:
    json.dump(coco_output, jsonfile, indent=2)

print(f"COCO json written: {OUTPUT_JSON} ({len(coco_output['images'])} images, {len(coco_output['annotations'])} annotations)")
