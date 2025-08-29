#!/usr/bin/env python3
"""
HM3D Floor Top-down Views Web Crawler
Downloads and processes top-down floor views from HM3D dataset
"""

import os
import json
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm

def create_directories(save_dir):
    """Create necessary directories"""
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'full').mkdir(parents=True, exist_ok=True)
    return save_dir

def get_scene_names(scene_dir):
    """Get scene names from directory or use default list"""
    scene_dir = Path(scene_dir).expanduser()
    
    if scene_dir.exists():
        scene_names = [name for name in os.listdir(scene_dir) if name[0] != "."]
        print(f"Found {len(scene_names)} scenes in {scene_dir}")
    else:
        # Fallback: use a sample of known HM3D scene names
        print(f"Directory {scene_dir} not found, using sample scene names")
        scene_names = [
            "00034-6imZUJGRUq4",
            # "00009-vLpv2VX547B",
            # "00016-qk9eeNeR4vX", 
            # "00017-Pa4fRMbTKUY",
            # "00025-A6YmRgkwdYB",
            # "00041-GpnQXhI5hVd",
            # "00065-RPmz2sHmrrY",
            # "00086-s9hNv5qa6Lo",
            # "00103-8LtY6oXvdNp"
        ]
        print(f"Using {len(scene_names)} sample scenes")
    
    return scene_names

def download_image(url, save_path, max_retries=3):
    """Download image from URL with error handling"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30, stream=True)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                print(f"HTTP {response.status_code} for {url}")
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    
    return False

def download_topdown_images(scene_names, save_dir):
    """Download top-down floor images for all scenes"""
    root = "https://habitatwebsite.s3.amazonaws.com/website-visualization/"
    successful_downloads = []
    
    print("Downloading top-down floor images...")
    for scene_name in tqdm(scene_names, desc="Downloading"):
        image_url = f"{root}{scene_name}/topdown_floors.png"
        save_path = save_dir / 'full' / f"{scene_name}.png"
        
        if save_path.exists():
            print(f"Skipping {scene_name} - already exists")
            successful_downloads.append(scene_name)
            continue
            
        if download_image(image_url, save_path):
            successful_downloads.append(scene_name)
        else:
            print(f"Failed to download {scene_name}")
    
    print(f"Successfully downloaded {len(successful_downloads)}/{len(scene_names)} images")
    return successful_downloads

def separate_floors(img):
    """Separate floors from combined top-down view image"""
    img_row_indices = []
    flag_within_floor = False
    
    for i in range(img.shape[0]):
        row = img[i, :]
        # Check if row has significant content (not mostly empty/black)
        num_empty_pixel_ratio = np.mean(np.all(row == 0, axis=-1))
        
        # Start of a floor
        if np.any(row) and not flag_within_floor:
            if num_empty_pixel_ratio < 0.90:
                flag_within_floor = True
                start_ind = i
        
        # End of a floor (empty row)
        if not np.any(row) and flag_within_floor:
            end_ind = i
            img_row_indices.append((start_ind, end_ind))
            flag_within_floor = False
    
    # Handle case where last floor goes to end of image
    if flag_within_floor:
        img_row_indices.append((start_ind, img.shape[0]))
    
    return img_row_indices

def process_floor_images(scene_names, save_dir):
    """Process downloaded images to separate individual floors"""
    print("Processing floor images...")
    floor_count_summary = []
    
    for scene_name in tqdm(scene_names, desc="Processing"):
        image_path = save_dir / "full" / f"{scene_name}.png"
        
        if not image_path.exists():
            print(f"Skipping {scene_name} - image not found")
            continue
            
        try:
            # Load and convert image
            img = Image.open(image_path)
            img = np.array(img)
            
            # Handle different image formats
            if len(img.shape) == 2:  # Grayscale
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 4:  # RGBA
                img = img[:, :, :3]  # Remove alpha channel
            
            # Separate floors
            floor_indices = separate_floors(img)
            
            if not floor_indices:
                print(f"No floors detected in {scene_name}")
                continue
            
            # Save individual floor images
            for floor_idx, (start_ind, end_ind) in enumerate(floor_indices):
                floor_img = img[start_ind:end_ind, :]
                
                # Remove empty columns to clean up the image
                non_empty_cols = np.any(np.any(floor_img, axis=0), axis=-1)
                if np.any(non_empty_cols):
                    floor_img = floor_img[:, non_empty_cols, :]
                
                # Save floor image
                floor_img_pil = Image.fromarray(floor_img.astype(np.uint8))
                floor_save_path = save_dir / f"{scene_name}_floor_{floor_idx}.png"
                floor_img_pil.save(floor_save_path)
            
            floor_count_summary.append((scene_name, len(floor_indices)))
            
        except Exception as e:
            print(f"Error processing {scene_name}: {e}")
    
    return floor_count_summary

def display_summary(floor_count_summary, save_dir):
    """Display summary of processed floors"""
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    
    total_scenes = len(floor_count_summary)
    total_floors = sum(count for _, count in floor_count_summary)
    
    print(f"Scenes processed: {total_scenes}")
    print(f"Total floors extracted: {total_floors}")
    print(f"Average floors per scene: {total_floors/total_scenes:.2f}")
    
    print("\nFloor count by scene:")
    for scene_name, count in sorted(floor_count_summary):
        print(f"  {scene_name}: {count} floors")
    
    print(f"\nAll files saved to: {save_dir}")
    print(f"Full images in: {save_dir}/full/")
    print(f"Individual floors in: {save_dir}/")

def main():
    """Main execution function"""
    # Configuration
    scene_dir = "data/HM3D"  # Original scene directory
    save_dir = "topdown"  # Where to save downloaded images
    
    print("HM3D Floor Top-down Views Crawler")
    print("="*40)
    
    # Create directories
    save_dir = create_directories(save_dir)
    
    # Get scene names
    scene_names = get_scene_names(scene_dir)
    
    # Download top-down images
    successful_downloads = download_topdown_images(scene_names, save_dir)
    
    if not successful_downloads:
        print("No images downloaded successfully. Exiting.")
        return
    
    # Process images to separate floors
    floor_count_summary = process_floor_images(successful_downloads, save_dir)
    
    # Display summary
    display_summary(floor_count_summary, save_dir)

if __name__ == "__main__":
    main()
