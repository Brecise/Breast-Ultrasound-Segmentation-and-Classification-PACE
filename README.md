# PACE 2025 – Breast Cancer Detection System

 **Final Submission Deadline**: 24th August 2025 (23:59 GMT)

##  Project Overview
This Docker container runs a multi-task deep learning model for breast ultrasound image analysis that can perform:
- **Segmentation** – Generates binary masks for ultrasound images
- **Classification** – Classifies ultrasound images into predefined categories

##  Project Structure
```
.
├── Dockerfile
├── requirements.txt
├── main.py                 # Main inference script
├── model.py               # Model architecture
├── checkpoints/           # Model weights directory
│   └── best_model.pth    # Trained model checkpoint
└── tools/                # Preprocessing and postprocessing utilities
    ├── preprocess.py
    └── postprocess.py
```

##  Quick Start

### Prerequisites
- Docker 20.10.0 or higher
- Git LFS (for model weights)
- NVIDIA Container Toolkit (for GPU support)

### 1. Install Docker
First, install Docker Desktop (available for Windows, macOS, and Linux):
[Download Docker](https://www.docker.com/get-started/)

Verify Docker is installed:
```bash
docker --version
```

### 2. Build the Docker Image
```bash
docker build -t pace2025-breast-cancer .
```

### 3. Run the Docker Container

#### For Segmentation Task (GPU)
```bash
docker run --gpus all --rm \
  -v $(pwd)/input:/input:ro \
  -v $(pwd)/output:/output \
  -it pace2025-breast-cancer python main.py -i /input -o /output -t seg -d gpu
```

#### For Classification Task (GPU)
```bash
docker run --gpus all --rm \
  -v $(pwd)/input:/input:ro \
  -v $(pwd)/output:/output \
  -it pace2025-breast-cancer python main.py -i /input -o /output -t cls -d gpu
```

#### For CPU-only Systems
```bash
docker run --rm \
  -v $(pwd)/input:/input:ro \
  -v $(pwd)/output:/output \
  -it pace2025-breast-cancer python main.py -i /input -o /output -t seg -d cpu
```

##  Command-Line Arguments
| Argument | Description | Options | Default |
|----------|-------------|----------|---------|
| `-i, --input` | Path to input directory containing images | Directory path | Required |
| `-o, --output` | Path to output directory | Directory path | Required |
| `-t, --task` | Task to perform | `seg` (segmentation), `cls` (classification) | `seg` |
| `-d, --device` | Device to use | `cpu`, `gpu` | `gpu` |

##  Output Structure

### Segmentation Task Output
```
output_directory/
└── segmentation/
    ├── PACE_00001_000_BUS_mask.png
    ├── PACE_00002_001_BRE_mask.png
    └── ...
```

### Classification Task Output
```
output_directory/
└── classification/
    └── predictions.csv    # Contains image_id, label columns
```

Example `predictions.csv`:
```csv
image_id,label
PACE_00001_000_BUS,Normal
PACE_00002_001_BRE,Benign
```

##  Submission Details
- **Team Name**: MedViewPro
- **Team Members**: 
  - Goodness Nwokebu (Team Lead)
  - Kevin Obote (Data Scientist)
  - Paul Ndirangu (ML Engineer)
- **Model Type**: Multi-task CNN with attention
- **Input Format**: PNG ultrasound images
- **Output Formats**: 
  - Segmentation: PNG masks
  - Classification: CSV file with predictions

##  Contact
For any questions, please contact: paulmwaura254@gmail.com

---
*This submission is for the PACE 2025 Challenge. All rights reserved.*