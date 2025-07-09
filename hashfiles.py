import os
import sys
import hashlib
import shutil

def is_text_file(file_path, blocksize=512):
    try:
        with open(file_path, 'rb') as f:
            block = f.read(blocksize)
            if b'\0' in block:
                return False
            block.decode('utf-8')
            return True
    except:
        return False

def compute_md5(file_path):
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    return md5.hexdigest()

def process_files(source_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)

            if is_text_file(file_path):
                md5_hash = compute_md5(file_path)
                dest_path = os.path.join(dest_dir, f"{md5_hash}.txt")

                if os.path.exists(dest_path):
                    print(f"MD5 exists, skipping: {file_path}")
                    continue

                shutil.copy2(file_path, dest_path)
                print(f"Copied text file: {file_path} -> {dest_path}")
            else:
                print(f"Skipped binary file: {file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <source_dir> <destination_dir>")
        sys.exit(1)

    source_directory = sys.argv[1]
    destination_directory = sys.argv[2]

    process_files(source_directory, destination_directory)

