import hashlib
import sys
import sqlite3
import zlib
from bitarray import bitarray

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

conn = sqlite3.connect("words/data.db")

for line in sys.stdin:
    line = line.strip() #.encode("utf8")
    tokens = tokenizer.encode(line)
    if line in bloom:
        continue
    bloom.add(line)
    line = line.encode("utf8")
    data = zlib.compress(line,level=9)
    tok = len(tokens)
    try:
        conn.execute("insert into texts (word,data,tok) values (?,?,?) ", ("archive",data,tok,))
        print(f"Inserted {tok} tokens")
    except:
        pass

conn.commit()
