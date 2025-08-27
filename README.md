# Breast Cancer Detection System

## Project Overview
This Docker container runs a multi-task deep learning model for breast ultrasound image analysis that can perform:
- **Segmentation** – Generates binary masks for ultrasound images
- **Classification** – Classifies ultrasound images into predefined categories
- **Both** – Performs both segmentation and classification in one go

## Project Structure
```
.
├── Dockerfile
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py           # Main inference script
│   ├── model.py          # Model architecture
│   ├── preprocess.py     # Image preprocessing
│   ├── postprocess.py    # Postprocessing utilities
│   └── schemas.py        # Data models
├── checkpoints/          # Model weights directory
│   └── best_model.pt     # Trained model checkpoint
└── test_data/
    ├── input/           # Input images
    └── output/          # Output directory for results
```

## Quick Start

### Prerequisites
- Docker 20.10.0 or higher
- 4GB+ free disk space
- 8GB+ RAM recommended

### 1. Build the Docker Image
```bash
docker build -t breast-cancer-detection .
```

### 2. Prepare Data
Create input and output directories:
```bash
mkdir -p test_data/input test_data/output
```

Place your ultrasound images in the `test_data/input` directory.

### 3. Run the Container

#### For CPU (Default)
```bash
docker run -v $(pwd)/test_data/input:/app/test_data/input \
           -v $(pwd)/test_data/output:/app/test_data/output \
           -p 8000:8000 \
           breast-cancer-detection \
           --input /app/test_data/input \
           --output /app/test_data/output \
           --device cpu
```

### Command-Line Arguments
| Argument | Description | Options | Default |
|----------|-------------|----------|---------|
| `-i, --input` | Path to input directory containing images | Directory path | Required |
| `-o, --output` | Path to output directory | Directory path | Required |
| `-m, --model` | Path to model checkpoint | File path | `app/checkpoints/best_model.pt` |
| `-t, --task` | Task to perform | `seg`, `cls`, `both` | `both` |
| `-d, --device` | Device to use | `cpu`, `cuda` | `cpu` |

## Output Structure

### Segmentation Output
```
output_directory/
└── segmentation/
    ├── PACE_00001_000_BUS_mask.png
    ├── PACE_00002_001_BRE_mask.png
    └── ...
```

### Classification Output
```
output_directory/
└── classification/
    └── predictions.csv    # Contains image_id, label, confidence
```

### Combined Output (default)
```
output_directory/
├── segmentation/
│   └── ...
└── classification/
    └── predictions.csv
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support
For support, please open an issue in the project repository.