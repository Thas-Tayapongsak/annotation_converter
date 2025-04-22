import json, cv2

def process_coco(dst_path, src_dataset, json_path, verbose=True):
    """
    Load COCO JSON data and create directory structure

    Parameters
    ----------
    dst_path : Path
        Destination path for directory structure
    src_dataset : str
        Name of the source dataset
    json_path : Path
        Path to the COCO JSON file
    verbose : bool, optional
        Print progress messages, True by default

    Returns
    -------
    bool
        True if the dataset is split into train, validation, and test sets
    Path
        Destination path for split datasets
    list
        List of images in the dataset
    list
        List of categories in the dataset
    list
        List of annotations in the dataset

    """
    # Create directory structure based on the dataset

    if json_path.parent.name != src_dataset:
        split_name = json_path.parent.name
        key = split_name
    else: 
        split_name = ''
        key = 'all'

    if verbose:
        print(f"Processing JSON file: {json_path.name} in {json_path.parent.name}")

    # Load COCO JSON data
    with open(json_path, 'r') as f:
        coco_json = json.load(f)
        missing_keys = [key for key in ['categories', 'images', 'annotations'] if key not in coco_json]
        if missing_keys:
            raise ValueError(f"Invalid COCO JSON format in {json_path}. Missing keys: {', '.join(missing_keys)}")
        categories = coco_json['categories']
        images = coco_json['images']
        annotations = coco_json['annotations']

    return key, images, categories, annotations

def find_contours(sub_mask):
    gray = cv2.cvtColor(sub_mask, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

def process_bin(dst_path, images_path, masks_path, split, verbose=True):
    if split.is_dir():
        split_name = split.name
        key = split_name
    else:
        split_name = ''
        key = 'all'

    images = []
    for image in (images_path / split_name).iterdir():
        if verbose:
            print(f"\rProcessing image file: {image.name} in {key}", end='')
        height, width, _ = cv2.imread(image).shape
        images.append({
            'id': len(images),
            'file_name': image.name,
            'width' : width,
            'height' : height
        })

    categories = []
    annotations = []
    for category in (masks_path).iterdir():
        if category.is_dir():
            category_id = len(categories) + 1

            categories.append({
                'id': category_id,
                'name': category.name,
                'supercategory': category.name
            })

            for mask in (masks_path / category / split_name).iterdir():
                mask_image_name = mask.name

                for image in images:
                    if image.get('file_name') == mask_image_name:
                        image_id = image.get('id')

                contours = find_contours(cv2.imread(mask))
                for contour in contours:
                    segmentation = contour.flatten().tolist()

                    if not segmentation:
                        raise ValueError(f"Segmentation data missing for {category} binary mask {mask_image_name}")
                    elif len(segmentation) % 2 != 0:
                        segmentation.pop()

                    x_points = segmentation[::2]
                    y_points = segmentation[1::2]
                    x = int(min(x_points))
                    y = int(min(y_points))
                    width = int(max(x_points) - min(x_points))
                    height = int(max(y_points) - min(y_points))

                    annotations.append({
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "bbox": [
                            x,
                            y,
                            width,
                            height
                        ],
                        "area": width*height,
                        "iscrowd": 0,
                        "category_id": category_id,
                        "segmentation": [
                            segmentation
                        ]
                    })

    if verbose:
        print()

    return key, images, categories, annotations

def process_yolo():
    pass