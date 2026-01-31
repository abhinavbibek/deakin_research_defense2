import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob

def resize_image(path):
    with Image.open(path) as img:
        img = img.convert('RGB')
        img = img.resize((32, 32), Image.BICUBIC)
        return np.array(img, dtype=np.uint8)

def process_folder(folder_path, is_train=True):
    print(f"Scanning {folder_path}...")
    images = []
    labels = []
    
    # Check if folder has class subdirectories (standard ImageFolder format)
    classes = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    
    if len(classes) > 0:
        print(f"Found {len(classes)} class folders.")
        # Try to sort numerically if possible (00000, 00001...)
        try:
            classes.sort(key=lambda x: int(x))
        except:
            pass
            
        for class_name in tqdm(classes):
            try:
                class_id = int(class_name)
            except ValueError:
                continue # Skip non-integer folders if any
            
            class_dir = os.path.join(folder_path, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        img_np = resize_image(img_path)
                        images.append(img_np)
                        labels.append(class_id)
                    except Exception as e:
                        pass
    else:
        # Flat directory (common for GTSRB Test)
        print("No class subdirectories found. Assuming flat image directory.")
        if is_train:
            print("WARNING: Train directory is flat! Cannot infer labels without subdirectories.")
            return None, None
            
        # For Test data, we need annotations. 
        # Checking for common annotation files in parent dir
        parent_dir = os.path.dirname(folder_path)
        csv_files = glob.glob(os.path.join(parent_dir, "*.csv")) + glob.glob(os.path.join(folder_path, "*.csv"))
        
        gt_file = None
        for f in csv_files:
            if "test" in f.lower() or "gt" in f.lower():
                gt_file = f
                break
        
        if gt_file:
            print(f"Found annotation file: {gt_file}")
            import csv
            with open(gt_file, 'r') as f:
                reader = csv.reader(f, delimiter=';' if gt_file.endswith('.csv') else ',')
                # Try to skip header if it exists
                header = next(reader, None)
                # Check if header is actually a header
                if header and not header[0].lower().endswith(('.png', '.jpg', 'ppm')):
                    pass # It was a header
                else:
                    # It was data, reset
                    f.seek(0)
                    reader = csv.reader(f, delimiter=';' if gt_file.endswith('.csv') else ',')

                for row in reader:
                    # GTSRB csv format: Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
                    # If simplified format, might differ. Assuming standard or flexible
                    try:
                        img_name = row[0] 
                        class_id = int(row[-1]) # Assuming class is last column
                        
                        img_path = os.path.join(folder_path, img_name)
                        if os.path.exists(img_path):
                            img_np = resize_image(img_path)
                            images.append(img_np)
                            labels.append(class_id)
                    except Exception:
                        continue
        else:
            print("WARNING: No CSV annotation file found for flat Test directory!")
            print("Creating dummy labels (0) just to allow loading (Accuracy will be meaningless).")
            # Fallback: Load images with dummy label 0
            for img_name in tqdm(os.listdir(folder_path)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):
                    img_path = os.path.join(folder_path, img_name)
                    img_np = resize_image(img_path)
                    images.append(img_np)
                    labels.append(0)

    return np.array(images), np.array(labels)

def save_pickle(images, labels, path):
    print(f"Saving {path}...")
    data = {"images": images, "labels": labels}
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {path} | Shape: {images.shape}")

def prepare_from_local(source_root, destination_root):
    print(f"Converting local dataset from {source_root} to {destination_root}...")
    
    if not os.path.exists(destination_root):
        os.makedirs(destination_root)
        
    # Paths in the existing dataset
    train_dir = os.path.join(source_root, "train_images")
    test_dir = os.path.join(source_root, "test_images") # Or "val_images"?
    
    # Process Train
    if os.path.exists(train_dir):
        images, labels = process_folder(train_dir, is_train=True)
        if images is not None and len(images) > 0:
            save_pickle(images, labels, os.path.join(destination_root, "train.pkl"))
    else:
        print(f"Error: {train_dir} not found!")

    # Process Test
    if os.path.exists(test_dir):
        images, labels = process_folder(test_dir, is_train=False)
        if images is not None and len(images) > 0:
            save_pickle(images, labels, os.path.join(destination_root, "test.pkl"))
    else:
        print(f"Error: {test_dir} not found!")

if __name__ == "__main__":
    # Source: The path you showed in the screenshot
    SOURCE = "/home/dgxuser10/cryptonym/data/GTSRB_dataset"
    
    # Destination: Where the config expects the pickles
    DEST = " /home/dgxuser10/cryptonym/data/gtsrb"
   
    prepare_from_local(SOURCE, DEST)
