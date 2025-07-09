import hashlib
from bitarray import bitarray

class BloomFilter:
    def __init__(self, power2=27, hash_count=8):
        self.size = (1 << power2) - 1
        self.bit_array = bitarray(self.size + 1)
        self.bit_array.setall(0)
        self.hash_count = hash_count
        self.slice_size = 256 // hash_count

    def _hashes(self, item):
        hashes = []
        prefix = hashlib.sha256(item.encode()).digest()
        iprefix = int.from_bytes(prefix)

        for i in range(self.hash_count):
            shift = (self.hash_count - 1 - i) * self.slice_size
            idx = (iprefix >> shift) & self.size
            hashes.append(idx)
        return hashes

    def add(self, item):
        for idx in self._hashes(item):
            self.bit_array[idx] = 1

    def __contains__(self, item):
        return all(self.bit_array[idx] for idx in self._hashes(item))
