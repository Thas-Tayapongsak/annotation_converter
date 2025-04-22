from processor import (
    process_coco,
    process_bin,
)

def from_bin(opt, verbose=True):

    src_path = opt.get('src_path')
    dst_path = opt.get('dst_path')

    # find images and masks directory
    images_path = src_path / 'images'
    masks_path = src_path / 'masks'
    if not images_path.is_dir():
        raise FileNotFoundError(f"No images directory found in {src_path}")
    if not masks_path.is_dir():
        raise FileNotFoundError(f"No masks directory found in {src_path}")

    # check if images are split into train, validation, and test sets, and create directory structure
    splits = []
    for split in images_path.iterdir(): 
        key, images, categories, annotations = process_bin(dst_path, images_path, masks_path, split, verbose)
        splits.append({key: {'images': images, 'categories': categories, 'annotations': annotations}})
        if not split.is_dir():
            break

    coco_dict = {
        'options': opt,
        'splits': splits
    }

    return coco_dict

def from_coco(opt, verbose=True):
    src_path = opt.get('src_path')
    dst_path = opt.get('dst_path')
    src_dataset = opt.get('src_dataset')

    #find COCO json files
    json_paths = list(src_path.rglob('*.json')) 
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in {src_path}")

    splits = []
    for json_path in json_paths:
        key, images, categories, annotations = process_coco(dst_path, src_dataset, json_path, verbose)
        splits.append({key: {'images': images, 'categories': categories, 'annotations': annotations}})

    coco_dict = {
        'options': opt,
        'splits': splits
    }

    return coco_dict

def from_yolo(opt, verbose=True):
    pass

__all__ = [
    'from_bin',
    'from_coco',
    'from_yolo'
]