# EvoJulia
# Evolutionary Fractal Engine
**GPU-Accelerated Generative Art using Genetic Algorithms and Simulated Annealing**

## Overview
This project is a high-performance Python engine that uses evolutionary algorithms to mathematically reconstruct target images using purely layered fractal geometry (specifically, multi-layered Julia Sets). 

Instead of traditional neural networks, this engine utilizes a custom **Genetic Algorithm** optimized for hardware acceleration. By bypassing traditional complex number math in favor of 4D matrix algebra on the GPU, the engine evaluates hundreds of millions of pixels per second, evolving complex visual structures in a fraction of the time it would take on a standard CPU.

## Core Features & Architecture

### 1. GPU Acceleration via CuPy Broadcasting
The heaviest computational bottleneck in fractal generation is calculating the escape times for millions of pixels across multiple layers. 
* Instead of nested loops, the `evaluator.py` utilizes **CuPy** to shift the workload to the GPU's VRAM.
* Genotypes are structured into 4D matrices `(Batch Size, Layers, Y-Axis, X-Axis)`, allowing CuPy to utilize array broadcasting. This processes entire batches of the population simultaneously.

### 2. Simulated Annealing (Adaptive Mutation)
To balance *exploration* (finding the general shape) with *exploitation* (fine-tuning micro-details), the engine utilizes a cooling mutation schedule.
* The algorithm begins with high mutation rates and ranges (e.g., 50%) to prevent premature convergence.
* As generations progress, a mathematical cooling factor dynamically scales the mutation parameters down to microscopic fractions (< 1%). This ensures the algorithm perfectly locks into the global minimum (lowest Mean Squared Error) without destroying highly fit genotypes in the late stages.

### 3. Dual-Resolution Pipeline
Fractals possess infinite detail, but rendering high resolutions during the evolutionary loop is computationally wasteful.
* **Training Phase:** Evaluates the population rapidly at a low resolution (e.g., 400x400) using a lower iteration cap.
* **4K Finisher:** Once the evolutionary loop concludes, the engine decouples the resolution variables and renders the winning genotype at 5x the resolution (2000x2000) with a massively increased iteration depth, revealing microscopic fractal tendrils that were hidden during training.

### 4. Histogram Color Matching
The underlying Julia Sets generate grayscale maps based on escape-time algorithms. To map the target image's color palette onto the mathematical structure, the final output is converted to a 3-channel RGB canvas. Using `scikit-image`, the algorithm extracts the exact color histogram of the original image and applies it to the fractal's brightness map.

### 5. Automated Timelapse Generation
Because the learning process is as important as the final output, the engine automatically samples the fittest genotype at regular intervals during the training loop. Using `imageio`, it stitches these NumPy arrays into a smooth timelapse GIF, physically demonstrating the algorithm "learning" the target shape. Outputs are automatically organized into timestamped directories.

## The Genotype Structure
Each "individual" in the population consists of `N` mathematical layers. Each layer contains 5 distinct floating-point variables representing a unique Julia Set:
* `c_real`: The real component of the complex constant.
* `c_imag`: The imaginary component of the complex constant.
* `x_offset` / `y_offset`: The cartesian coordinates for the camera pan.
* `zoom`: The scale of the viewport.

## Tech Stack & Dependencies
* **NumPy:** CPU array management and initial DNA generation.
* **CuPy:** GPU-accelerated 32-bit floating-point matrix operations.
* **OpenCV (`cv2`):** High-speed image reading, resizing, and color space conversions.
* **scikit-image:** Advanced histogram color matching.
* **imageio:** Memory-safe timeline and GIF generation.
* **psutil & tqdm:** System resource management and CLI progress tracking.

## Usage
The engine is designed to be headless and configurable for streamlined testing. 

1. Edit the parameters in `config.py` (Target Image, Population Size, Generation count, Mutation parameters).
2. Execute the engine: 
    Run 'run.bat'
