import os
import random
import shutil

def split_dataset(base_dir, test_ratio=0.2):
    categories = ["wake-word", "not-wake-word"]
    for category in categories:
        src_dir = os.path.join(base_dir, category)
        test_dir = os.path.join(base_dir, "test", category)
        os.makedirs(test_dir, exist_ok=True)

        files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        num_test = int(len(files) * test_ratio)
        test_files = random.sample(files, num_test)

        for f in test_files:
            src_file = os.path.join(src_dir, f)
            dst_file = os.path.join(test_dir, f)
            if not os.path.exists(dst_file):
                shutil.copy(src_file, dst_file)
                print(f"Copied {f} → {test_dir}")

base_dir = "/Users/james/Documents/Projects/Voice_Assistant/wake-word-training/ww_datasets/tars/"  # <-- change this to your dataset root
split_dataset(base_dir, test_ratio=0.2)