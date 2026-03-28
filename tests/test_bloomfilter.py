from llm_distiller.adapters.bloomfilter import BloomFilter


def test_bloomfilter_initialization() -> None:
    bf = BloomFilter(size=1000, hash_count=3)
    assert bf.size == 1000
    assert bf.hash_count == 3


def test_bloomfilter_add_and_check() -> None:
    bf = BloomFilter(size=1000, hash_count=3)
    bf.add("test_item")
    assert "test_item" in bf


def test_bloomfilter_false_positive() -> None:
    bf = BloomFilter(size=100, hash_count=1)
    bf.add("definitely_not_present")
    assert "something_else" not in bf


def test_bloomfilter_multiple_items() -> None:
    bf = BloomFilter(size=1000, hash_count=3)
    items = ["item1", "item2", "item3", "item4", "item5"]
    for item in items:
        bf.add(item)
    for item in items:
        assert item in bf


def test_bloomfilter_default_params() -> None:
    bf = BloomFilter()
    assert bf.size == 100000
    assert bf.hash_count == 3
