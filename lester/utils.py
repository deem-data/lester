import zlib


def hash_str(value):
    return hex(zlib.crc32(str.encode(value)))
