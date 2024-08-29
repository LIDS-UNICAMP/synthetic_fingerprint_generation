
# Synthetic Latent Fingerprint Dataset Generation

## Overview

This repository contains the code and related resources for generating synthetic latent fingerprint datasets. The code is part of the project presented in the paper titled "Improving Local Latent Fingerprint Representations under Data Constraints". This project aims to enhance latent fingerprint identification by developing efficient minutiae descriptors using self-supervised learning without relying on private datasets.

## Installation

### Requirements

- Python 3.8+
- Required Python libraries:
  - OpenCV
  - NumPy
  - Pillow
  - scikit-image
  - wsq

## Usage

### Command-Line Interface

The script \`generate_synthetic_latents.py\` provides a command-line interface to generate synthetic latent fingerprints.

#### Example

\`\`\`bash
python generate_synthetic_latents.py <input_folder> <num_synthetic_images> <output_folder>
\`\`\`

- \`<input_folder>\`: Path to the folder containing original fingerprint images in \`.png\` format.
- \`<num_synthetic_images>\`: Number of synthetic latents to generate per original fingerprint.
- \`<output_folder>\`: Path to the folder where the generated images will be saved.

#### Arguments

- **images**: Folder containing fingerprint images.
- **n**: Number of synthetic latents to generate per reference image.
- **output**: Output folder to save the generated images.

### Example Command

\`\`\`bash
python generate_synthetic_latents.py ./data/fingerprints 10 ./output/synthetic_latents
\`\`\`

This command will generate 10 synthetic latent images for each fingerprint in the \`./data/fingerprints\` folder and save them in the \`./output/synthetic_latents\` directory.

## Contact

For any questions or inquiries, please contact André Nóbrega at andreigor008@gmail.com.
