# Flexi_demo

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


python test.py
