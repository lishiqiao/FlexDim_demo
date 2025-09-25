
# Flexi_demo

## Overview
This repository provides a demo implementation of our proposed hyperspectral light-field reconstruction framework.  
It includes:
- A simplified testing pipeline,
- An example dataset (25-view hyperspectral data of one scene),
- Sample input and output results.

This demo is provided to allow reviewers to reproduce the basic workflow and verify the feasibility of our method. The full dataset and complete reconstruction code will be released upon acceptance.

---

## Installation

Clone the repository and install the required Python dependencies:

```bash
git clone https://github.com/lishiqiao/Flexi_demo.git
cd Flexi_demo
pip install -r requirements.txt
```

---

## Dataset

To run the demo, please download the `data/` folder (example scene with 25 views) from the following Google Drive link and place it under the project root directory:

👉 https://drive.google.com/drive/folders/15bdm__k6pzH18y-VnNB9X6QTfNbZETrT?usp=drive_link

After downloading, the project directory should look like this:

```
Flexi_demo/
├── architecture/             # network architectures
├── check_point/              # pretrained model checkpoints
├── data/                     # downloaded dataset (example scene with 25 views)
├── hsi_dataset_25.py         # dataset loader
├── model_0611.py             # model definition
├── test.py                   # demo testing script
├── utils.py                  # utility functions
├── input_rendered_rgb.png    # example input (rendered RGB)
├── output_rendered_rgb.png   # example reconstructed RGB
├── output_channels.png       # example reconstructed spectral channels
└── requirements.txt          # dependencies
```

**Important:** make sure the `data/` folder is placed directly under the repository root (i.e., `Flexi_demo/data/...`) before running the demo.

---

## Running the Demo

After downloading the dataset and confirming files are in place, run:

```bash
python test.py
```

What `test.py` does (typical behavior of the demo script):
1. Loads the example dataset from `data/`.
2. Loads a provided pretrained model from `check_point/` (if available).
3. Runs the reconstruction pipeline.

---

## Example Results


**Input (rendered RGB):**

<img src="input_rendered_rgb.png" width="480">

**Output (reconstructed RGB):**

<img src="output_rendered_rgb.png" width="480">

**Output (reconstructed spectral channels):**

<img src="output_channels.png" width="480">

---


---

## Contact
If you run into any issues running the demo or accessing the dataset, please open an issue on the repository or contact the corresponding author.
