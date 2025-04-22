from pathlib import Path
from shutil import copy
import yaml

def validate_options(opt, verbose=True):
    """
    Validate selected options

    Parameters
    ----------
    opt : dict
        A dictionary containing the selected options
    verbose : bool, optional
        Print progress messages, True by default
    
    Returns
    -------
    Path
        Source path for the dataset
    Path
        Destination path for the dataset
    str
        Name of the source dataset
    str
        Task to perform

    """
    # Define source and destination paths, dataset name, and task
    src_path = Path(opt.get('src_path', ''))
    dst_path = Path(opt.get('dst_path', ''))
    src_dataset = opt.get('src_dataset', '')
    src_format = opt.get('src_format', '')
    dst_format = opt.get('dst_format', '')
    task = opt.get('task', '')

    # Print initial information if verbose is True
    if verbose:
        print(f"Starting conversion from {src_format} to {dst_format} format.")
        print(f"Source path: {src_path}")
        print(f"Destination path: {dst_path}")
        print(f"Dataset: {src_dataset}")
        print(f"Source format: {src_format}")
        print(f"Task: {task}")

    # Check if source and destination paths exist
    if not src_path.exists():
        raise FileNotFoundError(f"Source path {src_path} does not exist.")
    if not dst_path.exists():
        dst_path.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Created destination path: {dst_path}")
    
    # Check task and format compatibility
    if task not in ['detect', 'segment']:
        raise ValueError(f"Invalid task {task}. Task must be 'detect' or 'segment'.")
    elif src_format == 'bin' and task != 'segment':
        raise ValueError(f"Invalid task {task} for source format 'bin'. Task must be 'segment'.")
     
    return opt

def copy_images(src_path, dst_path, images, verbose=True):
    for image in images:
        try:
            src_image_path = src_path / image['file_name']
            copy(src_image_path, dst_path)
            if verbose:
                print(f"\r...Copying image file #{image['id']}: {image['file_name']}    ", end='')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Image file {image['file_name']} not found in {src_path}.") from e
        except Exception as e:
            print(f"Error copying image file {image['file_name']} to {dst_path}: {e}")
            return
    if verbose:
            print()

def write_yolo_yaml(dst_path, src_dataset, src_split, categories):
    """
    Write dataset information to YAML file

    Parameters
    ----------
    dst_path : Path
        Destination path to write the YAML file
    src_dataset : str
        Name of the source dataset
    src_split : bool
        True if the dataset is split into train, validation, and test sets
    categories : list
        List of categories in the dataset

    """
    # Write categories and split paths to YAML file
    try:
        with open(dst_path / f'{src_dataset}.yaml', 'w') as f:
            if src_split:
                yaml.dump({
                        'train': '../train/images',
                        'val': '../val/images',
                        'test': '../test/images',
                        'names': {cat['id'] - 1: cat['name'] for cat in categories}
                    }, f)
            else:
                yaml.dump({
                        'names': {cat['id'] - 1: cat['name'] for cat in categories}
                    }, f)
    except Exception as e:
        print(f"Error creating YAML file for dataset {src_dataset}: {e}")

def initialize_yolo_labels(dst_path, images, verbose=True):
    """
    Create empty YOLO label files for each image

    Parameters
    ----------
    dst_path : Path
        Destination path for YOLO label files
    src_split : bool
        True if the dataset is split into train, validation, and test sets
    images : list
        List of images in the dataset
    verbose : bool, optional
        Print progress messages, True by default

    """
    # Create empty YOLO txt files for each image
    yolo_txt_path = ''
    for image in images:
        try:
            image_name = image['file_name'].stem
            image_id = image['id']
            yolo_txt_path = dst_path / 'labels' / f'{image_name}.txt'
            yolo_txt_path.touch(exist_ok=True)
            if verbose:
                print(f"\r...Creating YOLO label file for image #{image_id}: {image_name}    ", end='')
        except Exception as e:
            print(f"Error creating YOLO label file for image #{image_id}: {e}")
    if verbose:
        print()

