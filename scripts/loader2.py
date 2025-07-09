import os
import hashlib
import sys
import sqlite3
import zlib
from bitarray import bitarray


def sanitize(lines, reps):
    for rep in reps:
        lines = lines.replace(rep,"")
    return lines


class BloomFilter:
    def __init__(self, size=100_000_000, hash_count=8):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def _hashes(self, item):
        hashes = []
        for i in range(self.hash_count):
            h = hashlib.sha256((str(i) + item).encode()).hexdigest()
            idx = int(h, 16) % self.size
            hashes.append(idx)
        return hashes

    def add(self, item):
        for idx in self._hashes(item):
            self.bit_array[idx] = 1

    def __contains__(self, item):
        return all(self.bit_array[idx] for idx in self._hashes(item))

bloom = BloomFilter(size=100_000_000, hash_count=6)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # or another model


reps = open(".reps.txt","r").readlines()

db = sys.argv[1]
directory = sys.argv[2]

conn = sqlite3.connect(db)

total = 0
for dirpath, dirnames, filenames in os.walk(directory):
    for filename in filenames:
        file = os.path.join(dirpath, filename)
        if os.path.isfile(file):
            print(f"Reading file: {file} ", end="")
            with open(file, "r",errors="ignore") as fp:
                lines = fp.read()
                if lines in bloom:
                    continue
                tokens = tokenizer.encode(lines)
                bloom.add(lines)
                lines = sanitize(lines, reps)
                lines = lines.encode("utf8")
                data = zlib.compress(lines,level=9)
                tok = len(tokens)
                if tok > 20:
                    total += tok
                    try:
                        conn.execute("insert into texts (word,data,tok) values (?,?,?) ", ("archive",data,tok,))
                        print(f"Inserted {tok} tokens, total: {total}\n")
                    except:
                        print("Error...")
                        pass
conn.commit()
