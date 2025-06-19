# Debug val dataset
import json
import os

# Check val annotations vs val images
with open('data/val/val_data.json', 'r', encoding='utf-8') as f:
    val_data = json.load(f)

val_annotations = val_data['annotations']
val_image_ids = set(ann['image_id'] for ann in val_annotations)

print(f"Val annotations: {len(val_annotations)}")
print(f"Unique image IDs in val: {len(val_image_ids)}")

# Check actual val images
val_img_folder = 'data/val/val-images'
val_files = os.listdir(val_img_folder)
val_actual_ids = set()

for filename in val_files:
    if filename.endswith('.jpg'):
        img_id = int(filename.replace('.jpg', ''))
        val_actual_ids.add(img_id)

print(f"Actual val images: {len(val_actual_ids)}")
print(f"Missing images: {len(val_image_ids - val_actual_ids)}")

if len(val_image_ids - val_actual_ids) > 0:
    missing = list(val_image_ids - val_actual_ids)[:5]
    print(f"Sample missing IDs: {missing}")