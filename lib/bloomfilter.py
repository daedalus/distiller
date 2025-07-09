import hashlib
import mmh3
from bitarray import bitarray

class BloomFilter:
    def __init__(self, power2=27, hash_count=8):
        self.size = (1 << power2) - 1
        self.bit_array = bitarray(self.size + 1)
        self.bit_array.setall(0)
        self.hash_count = hash_count

    def _hashes(self, item):
        hashes = []
        prefix = hashlib.sha256(item.encode()).digest()
        for i in range(self.hash_count):
            i_bytes = str(i).encode()
            idx = mmh3.hash(prefix + i_bytes, 0, signed=False) & self.size
            hashes.append(idx)
        return hashes

    def add(self, item):
        for idx in self._hashes(item):
            self.bit_array[idx] = 1

    def __contains__(self, item):
        return all(self.bit_array[idx] for idx in self._hashes(item))