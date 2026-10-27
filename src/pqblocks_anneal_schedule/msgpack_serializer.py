import msgpack
from typing import List, Tuple

def schedule_to_msgpack(schedule: List[Tuple[int, float, float]]) -> bytes:
    """Serialize a schedule to MessagePack bytes."""
    return msgpack.packb(schedule, use_bin_type=True)

def schedule_from_msgpack(data: bytes) -> List[Tuple[int, float, float]]:
    """Deserialize a schedule from MessagePack bytes."""
    unpacked = msgpack.unpackb(data, raw=False)
    # Reconstruct exact types
    return [(int(step), float(T), float(G)) for step, T, G in unpacked]
