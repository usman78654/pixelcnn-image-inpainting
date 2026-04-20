import os
import shutil
from pathlib import Path
import random

# Configuration
DATA_DIR = './data'
TRAIN_OCC_DIR = os.path.join(DATA_DIR, 'train', 'occluded_images')
TRAIN_ORIG_DIR = os.path.join(DATA_DIR, 'train', 'original_images')
VAL_OCC_DIR = os.path.join(DATA_DIR, 'val', 'occluded_images')
VAL_ORIG_DIR = os.path.join(DATA_DIR, 'val', 'original_images')
VAL_SPLIT_RATIO = 0.15  # 15% for validation
RANDOM_SEED = 42


def create_validation_split():
    """
    Creates a validation set by moving pairs of occluded/original images
    from the training set to a new validation directory.
    """
    # Set seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Create validation directories
    os.makedirs(VAL_OCC_DIR, exist_ok=True)
    os.makedirs(VAL_ORIG_DIR, exist_ok=True)
    
    # Get list of occluded images (sorted for consistency)
    occ_files = sorted([f for f in os.listdir(TRAIN_OCC_DIR) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Determine validation set size
    n_total = len(occ_files)
    n_val = int(n_total * VAL_SPLIT_RATIO)
    
    print(f"Total training pairs: {n_total}")
    print(f"Moving {n_val} pairs to validation set ({VAL_SPLIT_RATIO*100:.0f}%)")
    
    # Randomly select files for validation
    val_files = random.sample(occ_files, n_val)
    
    moved_count = 0
    for occ_filename in val_files:
        # Derive original filename (remove 'occluded_' prefix)
        if occ_filename.startswith('occluded_'):
            orig_filename = occ_filename.replace('occluded_', '', 1)
        else:
            orig_filename = occ_filename
        
        occ_src = os.path.join(TRAIN_OCC_DIR, occ_filename)
        orig_src = os.path.join(TRAIN_ORIG_DIR, orig_filename)
        
        # Check if both files exist
        if not os.path.exists(orig_src):
            print(f"Warning: Original file not found for {occ_filename}, skipping")
            continue
        
        # Move files to validation directories
        occ_dst = os.path.join(VAL_OCC_DIR, occ_filename)
        orig_dst = os.path.join(VAL_ORIG_DIR, orig_filename)
        
        shutil.move(occ_src, occ_dst)
        shutil.move(orig_src, orig_dst)
        moved_count += 1
    
    print(f"\nValidation split created successfully!")
    print(f"Moved {moved_count} pairs to validation set")
    print(f"Remaining training pairs: {len(os.listdir(TRAIN_OCC_DIR))}")
    print(f"\nDirectories created:")
    print(f"  - {VAL_OCC_DIR}")
    print(f"  - {VAL_ORIG_DIR}")


if __name__ == '__main__':
    create_validation_split()
