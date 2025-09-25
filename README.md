
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
4. Saves reconstructed results to the project directory (or `results/` if configured).

If the script requires a specific checkpoint name or data path, please edit the relevant path variables at the top of `test.py` (or pass them as command-line arguments if the script supports it).

---

## Example Results

**Input (rendered RGB):**

<img src="input_rendered_rgb.png" width="480">

**Output (reconstructed RGB):**

<img src="output_rendered_rgb.png" width="480">

**Output (reconstructed spectral channels):**

<img src="output_channels.png" width="480">

---

## Notes for Reviewers / Users
- This repository contains a working demo and one example dataset for testing, as requested in the review process.  
- The full dataset and complete codebase (training scripts, full datasets, additional examples) will be released upon acceptance.  
- If you encounter missing files (e.g., images or `data/`), please ensure you downloaded the `data/` folder from the provided Google Drive link and that all image files are pushed to the repository.

---

## Troubleshooting & Common Commands

If your README images do not display on GitHub, ensure the files are actually committed and pushed to the repository root. Example commands:

```bash
# add images (run in repository root)
git add input_rendered_rgb.png output_rendered_rgb.png output_channels.png

# add README if not added yet
git add README.md

# commit and push
git commit -m "Add README and example images"
git push origin master
```

If your Google Drive link is not accessible to reviewers, set the folder share settings to: **Anyone with the link → Viewer**.

---

## Contact
If you run into any issues running the demo or accessing the dataset, please open an issue on the repository or contact the corresponding author.
