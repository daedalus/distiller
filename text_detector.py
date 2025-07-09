import os
import shutil
import string
import mimetypes

try:
    import chardet
    _chardet_available = True
except ImportError:
    _chardet_available = False


class FileClassifier:
    def __init__(self, blocksize=1024):
        self.blocksize = blocksize
        self.text_chars = bytes(string.printable, 'ascii')

    def is_text_by_decode(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                block = f.read(self.blocksize)
                block.decode('utf-8')
            return True
        except UnicodeDecodeError:
            return False

    def is_text_by_chardet(self, filepath):
        if not _chardet_available:
            return False
        with open(filepath, 'rb') as f:
            raw = f.read(self.blocksize)
            result = chardet.detect(raw)
            return result['encoding'] is not None

    def is_text_by_control_chars(self, filepath):
        with open(filepath, 'rb') as f:
            block = f.read(self.blocksize)
            if not block:
                return True
            nontext = block.translate(None, self.text_chars)
            return len(nontext) / len(block) < 0.30

    def is_text_by_mime(self, filepath):
        mime, _ = mimetypes.guess_type(filepath)
        return mime is not None and mime.startswith('text/')

    def is_text_combined(self, filepath):
        results = [
            self.is_text_by_decode(filepath),
            self.is_text_by_control_chars(filepath),
            self.is_text_by_mime(filepath),
        ]
        if _chardet_available:
            results.append(self.is_text_by_chardet(filepath))
        return results.count(True) > len(results) // 2

    def filter_text_files(self, filepaths, method="decode"):
        checkers = {
            "decode": self.is_text_by_decode,
            "chardet": self.is_text_by_chardet,
            "control": self.is_text_by_control_chars,
            "mime": self.is_text_by_mime,
            "combined": self.is_text_combined,
        }
        if method not in checkers:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(checkers.keys())}")
        checker = checkers[method]
        return [f for f in filepaths if checker(f)]

    def filter_text_files_in_dir(self, directory, method="decode"):
        files = [os.path.join(directory, f) for f in os.listdir(directory)
                 if os.path.isfile(os.path.join(directory, f))]
        return self.filter_text_files(files, method=method)

    def sort_files_by_type(self, directory, method="combined", text_subdir="txt", bin_subdir="bin"):
        os.makedirs(os.path.join(directory, text_subdir), exist_ok=True)
        os.makedirs(os.path.join(directory, bin_subdir), exist_ok=True)

        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

        checkers = {
            "decode": self.is_text_by_decode,
            "chardet": self.is_text_by_chardet,
            "control": self.is_text_by_control_chars,
            "mime": self.is_text_by_mime,
            "combined": self.is_text_combined,
        }

        if method not in checkers:
            raise ValueError(f"Unknown method '{method}'")

        check = checkers[method]

        for f in files:
            full_path = os.path.join(directory, f)
            size = os.path.getsize(full_path)
            is_text = check(full_path)

            if is_text:
                dest = os.path.join(directory, text_subdir, f)
                label = "TEXT"
            else:
                dest = os.path.join(directory, bin_subdir, f)
                label = "BINARY"

            print(f"[{label}] {f} ({size} bytes) → {dest}")
            shutil.move(full_path, dest)

import sys
fc = FileClassifier()
fc.sort_files_by_type(sys.argv[1], method="combined")

