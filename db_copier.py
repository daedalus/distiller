import hashlib
import sys
import sqlite3
import zlib
from bitarray import bitarray

class BloomFilter:
    def __init__(self, power2=27, hash_count=8):
        # size = 2^power bits
        self.size = (1 << power2) - 1  # bitmask for indexing
        self.bit_array = bitarray(self.size + 1)  # allocate full array size
        self.bit_array.setall(0)
        self.hash_count = hash_count

    def _hashes(self, item):
        hashes = []
        prefix = hashlib.sha256(item.encode()).digest()
        for i in range(self.hash_count):
            i_bytes = str(i).encode()
            h_bytes = hashlib.sha256(prefix + i_bytes).digest()
            idx = int.from_bytes(h_bytes, 'big') & self.size  # bitmask instead of modulo
            hashes.append(idx)
        return hashes

    def add(self, item):
        for idx in self._hashes(item):
            self.bit_array[idx] = 1

    def __contains__(self, item):
        return all(self.bit_array[idx] for idx in self._hashes(item))

bloom = BloomFilter(power2=27, hash_count=6)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # or another model

connS = sqlite3.connect("words/data1.db")
connD = sqlite3.connect("words/data.db")

for row in connS.execute("select data from texts where word != 'archive'"):
    data = row[0]
    if data:
        text = zlib.decompress(data).decode("utf8")
        if text not in bloom:
            bloom.add(text)
            tokens = tokenizer.encode(text)
            tok = len(tokens)
            try:
                connD.execute("insert into texts (word,data,tok) values (?,?,?) ", ("archive",data,tok,))
                print(f"Inserted {tok} tokens")
            except:
                pass
connD.commit()
