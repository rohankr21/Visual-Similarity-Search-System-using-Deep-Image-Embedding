"""
reorganize_data.py
Moves Fashion-MNIST data to backup, then organizes Myntra real photos
into data/deepfashion/<category>/ structure.
No changes needed to any existing scripts after this.
"""

import os
import shutil
import zipfile
import pandas as pd
from tqdm import tqdm

# ── Config ────────────────────────────────────────
ZIP_PATH       = "data/raw_myntra/fashion-product-images-small.zip"
IMAGES_DIR     = "data/raw_myntra/images"
DEEPFASHION    = "data/deepfashion"
BACKUP_DIR     = "data/fashionmnist_backup"

# How many images per category to copy (keeps things manageable)
# 300 × ~10 main categories = ~3000 images, fast extraction on CPU
LIMIT_PER_CAT  = 300

# Only keep these clean top-level categories (avoids 100s of tiny classes)
KEEP_CATEGORIES = {
    "Tshirts", "Shirts", "Jeans", "Trousers", "Dresses",
    "Tops", "Shorts", "Jackets", "Sweaters", "Shoes",
    "Sandals", "Heels", "Handbags", "Watches", "Sunglasses"
}
# ─────────────────────────────────────────────────


def step1_backup_fashionmnist():
    """Move current deepfashion folder to backup."""
    if os.path.exists(DEEPFASHION):
        if os.path.exists(BACKUP_DIR):
            shutil.rmtree(BACKUP_DIR)
        shutil.move(DEEPFASHION, BACKUP_DIR)
        print(f"✅ Backed up Fashion-MNIST → {BACKUP_DIR}")
    else:
        print("No existing deepfashion folder found, skipping backup.")
    os.makedirs(DEEPFASHION, exist_ok=True)


def step2_extract_styles_csv():
    """Extract styles.csv from zip."""
    print("Extracting styles.csv from zip...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        # It appears twice in zip, extract the root one
        z.extract('styles.csv', path='data/raw_myntra/')
    print("✅ styles.csv extracted.")


def step3_load_and_filter_styles():
    """Load styles.csv and filter to our chosen categories."""
    df = pd.read_csv(
        "data/raw_myntra/styles.csv",
        on_bad_lines='skip'   # some rows have formatting issues
    )
    print(f"Total products in styles.csv: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # The category column is called 'articleType'
    df = df[df['articleType'].isin(KEEP_CATEGORIES)].copy()
    print(f"After filtering to {len(KEEP_CATEGORIES)} categories: {len(df)} products")
    print(df['articleType'].value_counts().to_string())
    return df


def step4_copy_images(df):
    """Copy images into data/deepfashion/<category>/ folders."""
    print(f"\nCopying up to {LIMIT_PER_CAT} images per category...")

    counts = {cat: 0 for cat in KEEP_CATEGORIES}
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        category  = row['articleType']
        image_id  = str(row['id'])
        src_path  = os.path.join(IMAGES_DIR, f"{image_id}.jpg")

        # Skip if already at limit for this category
        if counts[category] >= LIMIT_PER_CAT:
            continue

        # Skip if image file doesn't exist
        if not os.path.exists(src_path):
            skipped += 1
            continue

        # Create category folder and copy
        dest_folder = os.path.join(DEEPFASHION, category)
        os.makedirs(dest_folder, exist_ok=True)
        dest_path = os.path.join(dest_folder, f"{image_id}.jpg")
        shutil.copy2(src_path, dest_path)
        counts[category] += 1

    print(f"\n✅ Copy complete. Skipped {skipped} missing files.")
    print("\nFinal image counts per category:")
    total = 0
    for cat, count in sorted(counts.items()):
        if count > 0:
            print(f"  {cat:<20} {count}")
            total += count
    print(f"  {'TOTAL':<20} {total}")
    return total


def step5_verify():
    """Quick sanity check — open one image from each category."""
    from PIL import Image
    print("\nVerifying images (opening one per category)...")
    for cat in sorted(os.listdir(DEEPFASHION)):
        cat_dir = os.path.join(DEEPFASHION, cat)
        if not os.path.isdir(cat_dir):
            continue
        images = os.listdir(cat_dir)
        if not images:
            continue
        sample = os.path.join(cat_dir, images[0])
        img = Image.open(sample).convert('RGB')
        print(f"  {cat:<20} {img.size} {img.mode} ✅")


if __name__ == "__main__":
    print("=" * 50)
    print("  DATA REORGANIZATION SCRIPT")
    print("=" * 50)

    step1_backup_fashionmnist()
    step2_extract_styles_csv()
    df = step3_load_and_filter_styles()
    total = step4_copy_images(df)
    step5_verify()

    print("\n" + "=" * 50)
    print(f"  DONE — {total} real product images ready")
    print("  Next steps:")
    print("  1. python -m src.extract")
    print("  2. python -m src.classify")
    print("  3. python app.py")
    print("=" * 50)