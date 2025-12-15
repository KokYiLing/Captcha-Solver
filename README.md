# CAPTCHA Solver: Exact Template Matching

## Overview
This project implements a CAPTCHA Solver using Exact Template Matching. The solver is designed to process grayscale CAPTCHA images, segment individual characters, and recognize them using a dictionary-based template matching approach. 

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip 

### Installation

1. **Clone the repository** (or download the project):
```bash
git clone https://github.com/KokYiLing/Captcha-Solver.git
cd Captcha-Solver
```

2. **Create a virtual environment**:
```bash
python3 -m venv venv
```

3. **Activate the virtual environment**:
- On macOS/Linux:
```bash
source venv/bin/activate
```
- On Windows:
```bash
venv\Scripts\activate
```

4. **Install required dependencies**:
```bash
pip install -r requirements.txt
```

### Running the Script

#### Command-Line Usage:
```bash
python Captcha_Solver.py --training_dir ./sampleCaptchas/training --input_path ./sampleCaptchas/input/input00.jpg --output_path ./output.txt
```

**Arguments:**
- `--training_dir`: Path to the training directory containing labeled CAPTCHA images
- `--input_path`: Path to the input CAPTCHA image file to solve
- `--output_path`: Path to save the solved CAPTCHA result

#### Using as a Library:
```python
from Captcha_Solver import Captcha

# Initialize and train the solver
solver = Captcha(training_dir='./sampleCaptchas/training')

# Solve a CAPTCHA
result = solver(
   im_path='sampleCaptchas/input/input00.txt',
   save_path='output00.txt'
)
```

## Problem Framing
Here are some observations about the problem:

### Image Characteristics:
- Images are 30 pixels in height and 60 pixels in width.
- Characters are spaced 1-2 pixels apart.
- Images are grayscale, where each pixel's RGB values are identical.
- High pixel values represent noise, while low pixel values represent character pixels.
- Digits are consistently 10 pixels tall.

### Key Insights:
- The problem can be broken into two phases: **Training** and **Inference**.
- Training involves creating a dictionary that maps segmented character images to their corresponding labels.
- Inference involves preprocessing new input images, segmenting characters, and recognizing them using the dictionary.

## Solution Approach

### Phase 1: Supervised Training

#### Input Data:
- **Labeled Data**: A directory containing `output*.txt` files with the correct labels for each CAPTCHA.
- **Input Data**: Corresponding `input*.txt` files containing the CAPTCHA images.

#### Steps:
1. **Preprocessing**: Convert grayscale images into binary images (binarization) where:
   - `1` represents ink (character pixels).
   - `0` represents the background.
2. **Segmentation**: Identify and extract individual characters from the binary image using column-wise sums to detect character boundaries.
3. **Dictionary Mapping**: Flatten each segmented character image into a hashable key and map it to its corresponding label.

#### Output:
- A dictionary (`self.templates`) mapping each unique character image to its label.

### Phase 2: Inference

#### Input:
- A new CAPTCHA image to solve.

#### Steps:
1. **Preprocessing**: Binarize the input image.
2. **Segmentation**: Extract individual characters from the binary image.
3. **Recognition**: For each segmented character:
   - Look up the character in the dictionary.
   - If no match is found, return `'?'`.
4. **Joining**: Combine recognized characters into the final CAPTCHA string.

#### Output:
- The recognized CAPTCHA string.

## Code Structure

### Key Components

#### Captcha Class:
Handles training, preprocessing, segmentation, and recognition.

**Key methods:**
- `_load_image`: Loads grayscale images from `.txt` files.
- `_preprocess`: Converts grayscale images to binary.
- `_segment_characters`: Segments binary images into individual characters.
- `_build_templates`: Builds the dictionary of character templates during training.
- `_recognize`: Recognizes a single character using the dictionary.

#### Command-Line Interface:
The script accepts the following arguments:
- `--training_dir`: Path to the training directory containing labeled data.
- `--input_path`: Path to the input CAPTCHA file.
- `--output_path`: Path to save the solved CAPTCHA result.


## Example Workflow

### Training

#### Input:
Training directory containing:
- `input01.txt`, `input02.txt`, ..., `inputN.txt`: CAPTCHA images.
- `output01.txt`, `output02.txt`, ..., `outputN.txt`: Corresponding labels.

#### Process:
1. Load and preprocess each `input*.txt` file.
2. Segment characters and map them to labels from `output*.txt`.

#### Output:
- A dictionary mapping character images to their labels.

### Inference

#### Input:
- A new CAPTCHA image (e.g., `input06.txt`).

#### Process:
1. Preprocess the image.
2. Segment characters.
3. Recognize each character using the dictionary.

#### Output:
- The recognized CAPTCHA string (e.g., `12345`).
