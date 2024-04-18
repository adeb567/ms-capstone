import json

with open('/home/amitabha/capstone/instance_version/Annotations/instances_val_trashcan.json', 'r') as f:
    json_data = json.load(f)

for annotations in json_data["annotations"]:
    image = next((image for image in json_data["images"] if image["id"] == annotations["image_id"]), None)
    '''
    image = None
    for im in json_data["images"]:
        if im["id"] == annotations["image_id"]:
            image = im
            break
    '''
    original_width, original_height = image["width"], image["height"]

    print(annotations["image_id"], image["id"])

    new_width, new_height = 256, 256

    width_scale = new_width / original_width
    height_scale = new_height / original_height

    x, y, width, height = annotations['bbox']
    x_resized = x * width_scale
    y_resized = y * height_scale
    width_resized = width * width_scale
    height_resized = height * height_scale

    segmentation = annotations['segmentation'][0]
    segmentation_resized = [(x * width_scale, y * height_scale) for x, y in zip(segmentation[::2], segmentation[1::2])]

    annotations['bbox'] = [x_resized, y_resized, width_resized, height_resized]
    annotations['segmentation'] = [segmentation_resized]
    annotations['area'] = annotations['area'] * width_scale * height_scale

with open('/home/amitabha/capstone/instance_version/Annotations/instances_val_trashcan_funie.json', 'w') as f:
    json.dump(json_data, f, indent=4)

