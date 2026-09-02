# merge_coco_jsons.py
"""
Merge two COCO-format annotation JSON files into a single JSON.
Usage:
    python3 merge_coco_jsons.py old.json new.json output.json
"""
import json
import argparse

def merge_coco_jsons(old_json_path, new_json_path, output_json_path):
    # Load both JSON files
    with open(old_json_path, 'r', encoding='utf-8') as f:
        old_coco = json.load(f)
    with open(new_json_path, 'r', encoding='utf-8') as f:
        new_coco = json.load(f)

    # Initialize merged structure
    merged = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    # Merge categories
    category_map = {}
    next_cat_id = 1
    for cat in old_coco.get('categories', []):
        merged['categories'].append(cat.copy())
        category_map[cat['id']] = cat['id']
        next_cat_id = max(next_cat_id, cat['id'] + 1)
    for cat in new_coco.get('categories', []):
        if cat['name'] not in {c['name'] for c in merged['categories']}:
            new_id = next_cat_id
            category_map[cat['id']] = new_id
            merged['categories'].append({**cat, 'id': new_id})
            next_cat_id += 1
        else:
            # map to existing
            existing = next(c for c in merged['categories'] if c['name'] == cat['name'])
            category_map[cat['id']] = existing['id']

    # Merge images
    image_map = {}
    next_img_id = 1
    for img in old_coco.get('images', []):
        merged['images'].append(img.copy())
        image_map[img['id']] = img['id']
        next_img_id = max(next_img_id, img['id'] + 1)
    for img in new_coco.get('images', []):
        if img['file_name'] not in {i['file_name'] for i in merged['images']}:
            new_id = next_img_id
            image_map[img['id']] = new_id
            merged['images'].append({**img, 'id': new_id})
            next_img_id += 1
        else:
            existing = next(i for i in merged['images'] if i['file_name'] == img['file_name'])
            image_map[img['id']] = existing['id']

    # Merge annotations
    next_ann_id = 1
    for ann in old_coco.get('annotations', []):
        merged['annotations'].append(ann.copy())
        next_ann_id = max(next_ann_id, ann['id'] + 1)
    for ann in new_coco.get('annotations', []):
        # remap ids
        img_id = image_map.get(ann['image_id'])
        cat_id = category_map.get(ann['category_id'])
        if img_id is None or cat_id is None:
            continue
        merged['annotations'].append({
            **ann,
            'id': next_ann_id,
            'image_id': img_id,
            'category_id': cat_id
        })
        next_ann_id += 1

    # Copy optional fields
    for key in ('info', 'licenses'):
        if key in old_coco:
            merged[key] = old_coco[key]
        elif key in new_coco:
            merged[key] = new_coco[key]

    # Write out merged JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Merged JSON saved to {output_json_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge two COCO JSON annotation files'
    )
    parser.add_argument(
        'old_json',
        help='Path to the old COCO JSON file'
    )
    parser.add_argument(
        'new_json',
        help='Path to the new COCO JSON file'
    )
    parser.add_argument(
        'output_json',
        help='Path where merged JSON will be saved'
    )
    args = parser.parse_args()
    merge_coco_jsons(args.old_json, args.new_json, args.output_json)
