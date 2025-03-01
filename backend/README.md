# Brainrot Detector

This backend service provides an API to serve a PyTorch model for inference.

## Prerequisites

- Python 3.8+
- pip (Python package installer)

## Setup Instructions

1. Install the required Python packages:

```bash
make setup
```

2. Place the PyTorch model file (`.pt` file) in the `models/` directory.

3. Start the backend service:

```bash
make run
```

The backend service will start on `http://localhost:8000`.
