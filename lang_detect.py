
#@title BLOOMFILTER

# ========== Bloom filter ==========
from bitarray import bitarray
import hashlib

class BloomFilter:
    def __init__(self, size=1_000_000, hash_count=5):
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

bloom = BloomFilter(size=1_000_000_000, hash_count=6)

from langdetect import detect
import sys

LANGS = ['en','es','pt','it','fr','undetected']

FPS = {}

for lang in LANGS:
    FPS[lang] = open(f"{lang}.txt","a")

for line in sys.stdin:
    ls = line.strip()

    if ls in bloom: continue
    bloom.add(ls)

    print(ls)
    try:
        lang=detect(ls)
    except:
        lang = 'undetected'
    print(f"[{lang}] || {ls}")
    if lang in LANGS:
        FPS[lang].write(ls + "\n")
        FPS[lang].flush()
    else:
        FPS[lang] = open(f"{lang}.txt","a")
        FPS[lang].write(ls + "\n")
        FPS[lang].flush()
        LANGS.append(lang)
 
for lang in LANGS:
    FPS[lang].close()
