from __future__ import annotations

import random


def simulate_message_latency(configured_delay_ms: int) -> dict[str, float]:
    """Generate demo processing and transport timing without sleeping or networking."""
    processing_ms = round(random.uniform(4.0, 16.0), 1)
    delay_jitter = random.uniform(0.0, max(1.0, configured_delay_ms * 0.08))
    transport_ms = round(configured_delay_ms + delay_jitter, 1)
    return {
        "processing_ms": processing_ms,
        "transport_ms": transport_ms,
        "total_ms": round(processing_ms + transport_ms, 1),
    }
