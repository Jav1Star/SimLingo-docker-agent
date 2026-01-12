---
license: other
task_categories:
- visual-question-answering
- robotics
language:
- en
tags:
- AutonomousDriving
- VQA
- Commentary
- VLA
---



# SimLingo Dataset

## Overview
SimLingo-Data is a large-scale autonomous driving CARLA 2.0 dataset containing sensor data, action labels, a wide range of simulator state information, and language labels for VQA, commentary and instruction following. The driving data is collected with the privileged rule-based expert [PDM-Lite](https://github.com/OpenDriveLab/DriveLM/tree/DriveLM-CARLA/pdm_lite). 

## Dataset Statistics
- **Large-scale dataset**: 3,308,315 total samples (note: these are not from unique routes as the provided CARLA route files are limited)
- **Diverse Scenarios:** Covers 38 complex scenarios, including urban traffic, participants violating traffic rules, and high-speed highway driving
- **Focused Evaluation:** Short routes with 1 scenario (62.1%) or 3 scenarios (37.9%) per route
- **Data Types**: RGB images (.jpg), LiDAR point clouds (.laz), Sensor measurements (.json.gz), Bounding boxes (.json.gz), Language annotations (.json.gz)

## Dataset Structure
The dataset is organized hierarchically with the following main components:
- `data/`: Raw sensor data (RGB, LiDAR, measurements, bounding boxes)
- `commentary/`: Natural language descriptions of driving decisions
- `dreamer/`: Instruction following data with multiple instruction/action pairs per sample
- `drivelm/`: VQA data, based on DriveLM

### Data Details
- **RGB Images**: 1024x512 front-view camera image
- **Augmented RGB Images**: 1024x512 front-view camera image with a random shift and orientation offset of the camera
- **LiDAR**: Point cloud data saved in LAZ format
- **Measurements**: Vehicle state, simulator state, and sensor readings in JSON format
- **Bounding Boxes**: Detailed information about each object in the scene.
- **Commentary, Dreamer, VQA**: Language annotations

## Usage
This dataset is chunked into groups of multiple routes for efficient download and processing.

### Download the whole dataset using git with Git LFS

```bash
# Clone the repository
git clone https://huggingface.co/datasets/RenzKa/simlingo

# Navigate to the directory
cd simlingo

# Pull the LFS files
git lfs pull
```

### Download a single file with wget

```bash
# Download individual files (replace with actual file URLs from Hugging Face)
wget https://huggingface.co/datasets/RenzKa/simlingo/resolve/main/[filename].tar.gz
```

### Extract to a single directory - please specify the location where you want to store the dataset
```bash
# Create output directory
mkdir -p database/simlingo

# Extract all archives to the same directory
for file in *.tar.gz; do
    echo "Extracting $file to database/simlingo/..."
    tar -xzf "$file" -C database/simlingo/
done
```


## License
Please refer to the license file for usage terms and conditions.


## Citation
If you use this dataset in your research, please cite:
```bibtex
@inproceedings{renz2025simlingo,
  title={SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment},
  author={Renz, Katrin and Chen, Long and Arani, Elahe and Sinavski, Oleg},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025},
}
@inproceedings{sima2024drivelm,
  title={DriveLM: Driving with Graph Visual Question Answering},
  author={Chonghao Sima and Katrin Renz and Kashyap Chitta and Li Chen and Hanxue Zhang and Chengen Xie and Jens Beißwenger and Ping Luo and Andreas Geiger and Hongyang Li},
  booktitle={European Conference on Computer Vision},
  year={2024},
}
```
