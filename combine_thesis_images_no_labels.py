
import os
from PIL import Image, ImageDraw, ImageFont

def combine_images():
    # Paths
    path_cat = "/home/zzn/qfl_tq/datasets/low_shot/grumpy_cat/0.jpg"
    path_obama = "/home/zzn/qfl_tq/LD-Diffusion-quantum-v3/temp_real_obama/0.jpg"
    # Panda path
    path_panda = "/home/zzn/qfl_tq/LD-Diffusion-quantum-v3/panda_32_real/temp_real_32x32/0.jpg"
    
    output_path = "/home/zzn/qfl_tq/LD-Diffusion-quantum-v3/docs/thesis_figure_datasets.png"
    
    # Load images
    try:
        img_cat = Image.open(path_cat).convert("RGB")
        print(f"Loaded Grumpy Cat: {img_cat.size}")
    except Exception as e:
        print(f"Error loading Cat: {e}")
        return

    try:
        img_obama = Image.open(path_obama).convert("RGB")
        print(f"Loaded Obama: {img_obama.size}")
    except Exception as e:
        print(f"Error loading Obama: {e}")
        return

    try:
        img_panda = Image.open(path_panda).convert("RGB")
        print(f"Loaded Panda: {img_panda.size}")
    except Exception as e:
        print(f"Error loading Panda: {e}")
        return

    # Target height
    target_height = 128
    
    # Resize keeping aspect ratio
    def resize_to_height(img, height):
        ratio = height / img.height
        width = int(img.width * ratio)
        return img.resize((width, height), Image.Resampling.LANCZOS)

    img_cat_r = resize_to_height(img_cat, target_height)
    img_obama_r = resize_to_height(img_obama, target_height)
    img_panda_r = resize_to_height(img_panda, target_height)

    # Padding
    padding = 10
    total_width = img_cat_r.width + img_obama_r.width + img_panda_r.width + 2 * padding
    
    # Create new image (white background) - REMOVED extra height for labels
    combined = Image.new("RGB", (total_width, target_height), (255, 255, 255))
    
    # Paste WITHOUT labels
    current_x = 0
    
    # Cat
    combined.paste(img_cat_r, (current_x, 0))
    current_x += img_cat_r.width + padding
    
    # Obama
    combined.paste(img_obama_r, (current_x, 0))
    current_x += img_obama_r.width + padding
    
    # Panda
    combined.paste(img_panda_r, (current_x, 0))
    
    # Save
    combined.save(output_path)
    print(f"Saved combined image to {output_path}")

if __name__ == "__main__":
    combine_images()
