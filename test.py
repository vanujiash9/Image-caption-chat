import os

def count_images(folder):
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    return len([f for f in os.listdir(folder) if f.lower().endswith(image_exts)])

if __name__ == "__main__":
    folders = {
        "Train": "data/train/train-images",
        "Validation": "data/val/val-images",
        "Test": "data/test/test-images"
    }

    for name, path in folders.items():
        try:
            count = count_images(path)
            print(f"{name}: {count} ảnh")
        except FileNotFoundError:
            print(f" Không tìm thấy thư mục: {path}")
