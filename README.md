# FlexDim_demo

## Overview
This repository provides a demo implementation of the proposed hyperspectral light-field reconstruction framework.  
It includes:
- A simplified pipeline for testing,
- An example dataset (25-view hyperspectral data of one scene),
- Sample input and output results.

This demo allows reviewers to reproduce the workflow and verify the feasibility of our method.  
The complete dataset and full codebase will be released after acceptance.

---

## Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/lishiqiao/Flexi_demo.git
cd Flexi_demo
pip install -r requirements.txt

Dataset

To run the demo, please download the data/ folder from the following Google Drive link and place it under the project root directory:
https://drive.google.com/drive/folders/15bdm__k6pzH18y-VnNB9X6QTfNbZETrT
After downloading, the project directory should look like this:

Flexi_demo/
├── architecture/
├── check_point/
├── data/                # downloaded dataset
├── hsi_dataset_25.py
├── model_0611.py
├── test.py
├── utils.py
├── input_rendered_rgb.png
├── output_rendered_rgb.png
├── output_channels.png
└── requirements.txt

Running the Demo

After downloading the dataset, run the demo with:
python test.py

Example Results

Input (rendered RGB):

<img src="input_rendered_rgb.png" width="400">

Output (rendered RGB):

<img src="output_rendered_rgb.png" width="400">

Output (spectral channels):

<img src="output_channels.png" width="400">
