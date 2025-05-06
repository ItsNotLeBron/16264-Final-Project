# chat_interface_demo.py

import os
from pathlib import Path
from datetime import datetime, timedelta

import ChatModule.dispatcher as dispatcher
from MemoryModule.memory_module import MemoryModule
from InferenceModule.inference_module import InferenceModule, Place


def main():
    # Ensure logs folder next to this script
    script_dir = Path(__file__).parent
    logs_dir = script_dir / "memory_logs_chat_demo"

    # Initialize MemoryModule and InferenceModule with manual zones
    mem = MemoryModule(storage_dir=str(logs_dir))
    zones = [
        Place("bedside", (37.7749, -122.4194), radius_m=5.0),
        Place("desk",    (37.7755, -122.4185), radius_m=5.0)
    ]
    inf = InferenceModule(mem, manual_zones=zones, cluster_radius_m=5.0, freshness_seconds=600)

    dispatcher.mem = mem
    dispatcher.inf = inf

    # Simulate some sightings for "laptop"
    now = datetime.now()
    mem.store_sighting({
        "label": "laptop",
        "track_id": 1,
        "bbox": (0, 0, 0, 0),  # dummy values
        "timestamp": now - timedelta(hours=1),
        "location": zones[0].center
    })
    mem.store_sighting({
        "label": "laptop",
        "track_id": 1,
        "bbox": (0, 0, 0, 0),
        "timestamp": now - timedelta(minutes=30),
        "location": zones[1].center
    })
    # Train time model so ChatGPT can predict
    inf.train_time_model("laptop")

    # Test various chat prompts
    prompts = [
        "Where was laptop in the past 2 hours?",
        "Where is laptop now?",
        "Explain where laptop likely is right now."
    ]

    for prompt in prompts:
        print(f"You: {prompt}")
        reply = dispatcher.chat_with_bot(prompt)
        print(f"Bot: {reply}\n")

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in env
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable before running this demo.")
        exit(1)
    main()
