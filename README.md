# Colorization Dataset Processing

This repository provides utility scripts for preprocessing datasets used in image colorization tasks. It helps prepare grayscale input and color target images for deep learning models.

## Features

- Resize and normalize images
- Convert RGB images to Lab color space (L and ab channels)
- Save processed images in structured folders (e.g., `input/` and `target/`)
- Batch processing support

## Technologies Used

- **Python 3**
- **OpenCV (cv2)**
- **NumPy**

## Use Case

This project is ideal for preparing training data for colorization models such as CNNs, GANs, or autoencoders. It ensures consistency and structure in datasets.
