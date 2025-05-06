# 16264 Final Project: Humanoid Object Memory System

> A modular pipeline for capturing, remembering, and inferring the location of personal objects.


## Overview

This goal of this project was to provide a modular object memory system for humanoid robots, enabling them to detect and remember the locations of personal items over time. It captures live video, logs each sighting (with timestamp and location), and builds a temporal model of where objects tend to be. When queried—either programmatically or via a ChatGPT interface—it can answer questions like:

- "Where was my phone yesterday?"
- "Where is my laptop most likely right now?"


## Motivation

As humanoid robots evolve into fully functional assistants, they’ll need to handle the small, valuable items humans rely on every day. This system addresses three core needs:

- **Elder care & Assistive robotics:** In environments where users depend on robots for support, reliably finding keys, phones, or medication can be critical for safety and independence.
- **Object permanence:** True object permanence—remembering where an item was last seen—prevents unnecessary searching and frustration for both humans and robots (minus the frustration for robots).
- **Personalized intelligence:** By learning individual routines and habits, robots can anticipate where items are likely to be, making interactions more seamless and human-like.


## Architecture

```mermaid
flowchart TD
  subgraph Perception
    A[CameraModule] --> |frames| B[AnalyzerModule]
  end
  subgraph Memory
    B --> |sightings| C[MemoryModule]
  end
  subgraph Reasoning
    C --> |logs & cache| D[InferenceModule]
  end
  subgraph Chat
    U[User] -->|"Where is my laptop?"| E[Chat Interface]
    E --> |function call| D
    D --> |prediction / explanation| E
    E --> |"Your laptop was last seen at ..."| U
  end
  ```
### Pipeline Overview

1. **CameraModule**
   - **Role**: Independently captures video frames from a USB or IP camera at a configurable frame rate. 

2. **AnalyzerModule** *(black box)*
   - **Role**: Processes each frame to detect and track objects of interest, emitting standardized `Sighting` events.  
   - **Input**: Raw frames from the CameraModule.  
   - **Output**: `Sighting` objects containing:  
     - `label`: object class or user-defined name  
     - `track_id`: persistent instance identifier  
     - `bbox`: bounding-box coordinates  
     - `timestamp`: capture time  
     - `location`: GPS coordinates 
   - **Note**: Implementation details (e.g., YOLOv5 + DeepSORT, or alternative pipelines) will be discussed later.

3. **MemoryModule**
   - **Role**: Persists and indexes every sighting for later retrieval and visualization.  
   - **Features**:
     - **Durable logs**: Appends each event as a CSV line in `memory_logs/<label>.txt`.
     - **In-memory cache**: Mirrors logs for fast querying without disk reads.
   - **APIs**:
     - `store_sighting(event)`: write to disk & cache.
     - `get_last_seen(label)`: fetch most recent event.
     - `get_sightings(label, since)`: fetch full or time-filtered history.
     - `annotate_frame(frame, label)`: draw the chosen bounding box on an image copy.

4. **InferenceModule**
   - **Role**: Builds an item location habit model from logged sightings and answers high-level queries.  
   - **Components**:
     - **ZoneModel**: Groups GPS points into named zones (user-defined).
     - **TimeModel**: Computes `P(in zone │ at hour)` histograms and stores zone centroids.
   - **APIs**:
     - `train_time_model(label)`: cluster past data and build probability tables.
     - `predict_location(label, at_time)`: return recent sighting if fresh; otherwise return most probable zone centroid.
     - `explain_prediction(label, at_time)`: produce a human-readable rationale leveraging time-bin probabilities.

5. **Chat Interface**
   - **Role**: Exposes the InferenceModule to end users via natural-language conversation.  
   - **Mechanism**:
     - Defines JSON-schema functions (`get_history`, `last_seen`, `predict_location`, `explain_prediction`).
     - Uses OpenAI function-calling to map user queries to API calls.
     - Renders function outputs into clear replies (e.g., “Your laptop was last seen at your desk at 3:14 PM.”).




## What Works & What Doesn’t

- **CameraModule**: Fully implemented and tested ([see here](https://github.com/ItsNotLeBron/16264-Final-Project/blob/dd017c8c859c14de151a01c8684dd9290a43ab67/src/CameraModule/test_camera.py#L6)).
- **MemoryModule**: Fully implemented and tested; logs and in-memory queries work as expected ([see here](https://github.com/ItsNotLeBron/16264-Final-Project/blob/dd017c8c859c14de151a01c8684dd9290a43ab67/src/MemoryModule/test_memory.py#L10)).
- **InferenceModule**: Fully implemented and tested; Model training, prediction, and explanation have been validated ([see here](https://github.com/ItsNotLeBron/16264-Final-Project/blob/dd017c8c859c14de151a01c8684dd9290a43ab67/src/test_inference.py#L11)).
- **Chat Interface**: Almost fully implemented with a few caveats; End-to-end function-calling integration tested with live OpenAI calls ([see here](https://github.com/ItsNotLeBron/16264-Final-Project/blob/dd017c8c859c14de151a01c8684dd9290a43ab67/src/test_chat_interface.py#L12)).
- **AnalyzerModule**: *Not yet implemented.* I realized as I tried every option I could find that accurately identifying **unique** object instances goes beyond simple classification. It may requires custom per-item models which is a significant challenge and breaks the simplicity aspect that I was trying to achieve. Currently exploring user-provided image sets and few-shot approaches as potential solutions.

Once the AnalyzerModule is complete and integrated, the full pipeline should operate seamlessly since everything else has been validated.


## Module Examples

Below are brief pointers to demo scripts and sample outputs for each working module (ran in root):

### CameraModule Demo
```bash
python src/CameraModule/test_camera_module.py 
```
<img width="1580" alt="image" src="https://github.com/user-attachments/assets/52fd5b51-b121-4f3e-9224-094d74c14edc" />

Image displays working camera stream.

### MemoryModule Demo

```bash
python src/MemoryModule/test_memory_module.py
# Last seen: {'label': 'test_object', 'timestamp': datetime.datetime(2025, 5, 6, 7, 12, 56, 334499), 'track_id': 2, 'bbox': (100, 40, 80, 90), 'location': (37.7751, -122.4196)}
# All sightings:
#  {'label': 'test_object', 'timestamp': datetime.datetime(2025, 5, 6, 7, 12, 54, 334499), 'track_id': 1, 'bbox': (10, 20, 50, 60), 'location': (37.7749, -122.4194)}
#  {'label': 'test_object', 'timestamp': datetime.datetime(2025, 5, 6, 7, 12, 55, 334499), 'track_id': 1, 'bbox': (15, 25, 50, 60), 'location': (37.775, -122.4195)}
#  {'label': 'test_object', 'timestamp': datetime.datetime(2025, 5, 6, 7, 12, 56, 334499), 'track_id': 2, 'bbox': (100, 40, 80, 90), 'location': (37.7751, -122.4196)}
Wrote /home/itsnotlebron/humanoids/16264-Final-Project/src/MemoryModule/annotated_demo.png with the last bbox and location annotation
```

### InferenceModule Demo

This module combines with the Memory module in order to function:

```
python src/test_inference_module.py
# At 2025-05-06 02:30:00 → Predicted location: (37.7755, -122.4185)
# Explanation: Your test_object was seen -25200 seconds ago at location (37.7755, -122.4185). Returning that location.

# At 2025-05-06 03:00:00 → Predicted location: (37.7755, -122.4185)
# Explanation: Your test_object was seen -23400 seconds ago at location (37.7755, -122.4185). Returning that location.

# At 2025-05-06 09:45:00 → Predicted location: (37.7755, -122.4185)
# Explanation: No recent sightings. Between hour 9 and 10, your test_object was in 'desk' 100.0% of the time (centroid at (37.7755, -122.4185)).

# At 2025-05-06 12:00:00 → Predicted location: (37.7755, -122.4185)
# Explanation: No data for hour 12. Unable to infer test_object location.
```

### ChatModule

```
python src/test_chat_interface.py

# prints
# You: Where was laptop in the past 2 hours?
# Function call: get_history with args: {'label': 'laptop', 'since': '2024-06-15T08:17:45.009Z'}
# Bot: [{"label": "laptop", "timestamp": "2025-05-06 03:37:06.202176", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 04:07:06.202176", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 03:41:05.355798", "track_id": 1, "bbox": # [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 04:11:05.355798", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 03:44:55.764200", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-# # 06 04:14:55.764200", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 03:45:20.728420", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 04:15:20.728420", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, # {"label": "laptop", "timestamp": "2025-05-06 03:46:15.863246", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 04:16:15.863246", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 03:48:45.153003", "track_id": 1, "bbox": [0, 0, # 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 04:18:45.153003", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 03:50:19.204582", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 # 04:20:19.204582", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 03:51:04.355388", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 04:21:04.355388", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, # {"label": "laptop", "timestamp": "2025-05-06 04:04:07.460756", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 04:34:07.460756", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 04:09:13.988963", "track_id": 1, "bbox": [0, 0, # 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 04:39:13.988963", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 04:17:55.229261", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 # 04:47:55.229261", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 04:21:50.092834", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 04:51:50.092834", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, # {"label": "laptop", "timestamp": "2025-05-06 04:22:33.603052", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 04:52:33.603052", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 04:23:07.450930", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 04:53:07.450930", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 04:24:00.284924", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 04:54:00.284924", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 04:29:31.303334", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 04:59:31.303334", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 04:32:14.793098", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 05:02:14.793098", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 04:34:19.333044", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 05:04:19.333044", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 04:37:59.683538", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 05:07:59.683538", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 04:40:39.969013", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 05:10:39.969013", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 04:45:41.280834", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 05:15:41.280834", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 04:45:55.109329", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 05:15:55.109329", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 04:49:07.271194", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 05:19:07.271194", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 04:54:28.310505", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 05:24:28.310505", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 04:59:14.383807", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 05:29:14.383807", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 05:00:32.878129", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 05:30:32.878129", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}, {"label": "laptop", "timestamp": "2025-05-06 06:18:39.416236", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7749, -122.4194]}, {"label": "laptop", "timestamp": "2025-05-06 06:48:39.416236", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}]

# You: Where is laptop now?
# Function call: last_seen with args: {'label': 'laptop'}
# Bot: {"label": "laptop", "timestamp": "2025-05-06 06:48:39.416236", "track_id": 1, "bbox": [0, 0, 0, 0], "location": [37.7755, -122.4185]}

# You: Explain where laptop likely is right now.
# Function call: predict_location with args: {'label': 'laptop'}
# Bot: [37.7755, -122.4185]

```

I mentioned above that there were a few caveats. THose had to do with the natural language. In the above example we prompted chatgpt by asking `Where was laptop in the past 2 hours?`. but if you instead ask `Where was MY laptop in the past 2 hours?` we get:

```
python src/test_chat_interface.py
# You: Where was my laptop in the past 2 hours?
# Function call: get_history with args: {'label': 'my_laptop', 'since': '2024-06-15T13:10:00Z'}
# Bot: []

# You: Where is my laptop now?
# Function call: last_seen with args: {'label': 'my_laptop'}
# Bot: null

# You: Explain where my laptop likely is right now.
# Function call: predict_location with args: {'label': 'my_laptop'}
# Bot: [37.7749, -122.4194]
```

There is no clear way (at least for me) to cleanly resolve this. Also notice the date in the readout. The year is 2024 and that is because ChatGPT is frozen in time. This is another bug that can't be easily fixed that leads to very bad outcomes.


## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourname/humanoid-object-memory.git
   cd humanoid-object-memory
   ```
2. Create & activate a virtual environment
```
  python3 -m venv venv
source venv/bin/activate
```
3. Install dependencies
```
pip install -r requirements.txt
```

4. Create a .env file to place Chatgpt api key.
```
OPENAI_API_KEY=<my api key>
```

## Running

Once the setup is complete you can now run the test scripts. 

## Recreating this Project

Copy the structure and code...

