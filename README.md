# CNN-vs-transfer-learning-emotion-recognition

## Overview

This project investigates facial expression classification using the FER2013 dataset. It compares custom convolutional neural network (CNN) architectures with established transfer-learning models, examining how architectural design choices and training strategies influence classification performance.

Originally developed from work undertaken during my Open University studies, this independent project extends the investigation into a comparative evaluation of custom CNN architectures, pretrained models, training strategies and performance measures.

The experiments consider not only overall classification accuracy, but also class-level performance, the effects of class imbalance and computational requirements.

## Research Aim

The aim of this project is to evaluate how architectural design choices and selected training strategies influence the predictive performance and computational efficiency of deep-learning models for facial expression classification.

The investigation addresses the following research question:

> How do architectural design choices and training strategies influence the performance of deep-learning models for facial expression classification on the FER2013 dataset?

## Dataset

This project uses the FER2013 (Facial Expression Recognition 2013) dataset, originally released through the Kaggle *Challenges in Representation Learning: Facial Expression Recognition Challenge*.

The dataset contains 35,887 grayscale facial images, each measuring 48 × 48 pixels, labelled across seven expression categories:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

The original dataset partitions were retained:

| Partition | Images | Project use |
|---|---:|---|
| Training | 28,709 | Model training |
| PublicTest | 3,589 | Validation |
| PrivateTest | 3,589 | Testing |

The dataset has an uneven class distribution, with Disgust substantially underrepresented compared with categories such as Happy. This makes class-level evaluation particularly important.

### Dataset citation

Goodfellow, I. J., Erhan, D., Carrier, P. L., et al. (2013). *Challenges in Representation Learning: A Report on Three Machine Learning Contests.* arXiv:1307.0414.

Dataset: [FER2013 – Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)

### Dataset availability

The FER2013 CSV files are not included in this repository. Obtain the dataset from Kaggle and follow the dataset source's terms.

The raw and processed dataset files are excluded from version control. You will need to provide the dataset locally and run the relevant data-preparation workflow before running notebooks that depend on the processed files.

## Methodology

The investigation compared four custom CNN architectures with three pretrained transfer-learning architectures.

### Custom CNN architectures

Four custom CNN designs were evaluated, incorporating variations in:

- Convolutional depth and filter capacity
- Data augmentation
- Regularisation
- Residual connections

### Transfer-learning architectures

The following pretrained architectures were evaluated using ImageNet weights:

- VGG16
- ResNet50
- InceptionV3

Experiments investigated frozen feature extractors, partial fine-tuning and alternative classification heads.

### Training strategies

The experiments considered several approaches to model training:

- Standard categorical cross-entropy
- Class weighting
- Focal loss
- Combined class weighting and focal loss
- Data augmentation, where applicable

Transfer learning used a two-stage experimental process. Nine initial architecture and classification-head configurations were compared, with three selected configurations progressing to further training-strategy experiments.

In total, **34 model configurations** were evaluated, each with five successful runs using different random seeds.

### Evaluation

Models were assessed using a range of complementary measures:

- Accuracy
- Balanced accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Confusion matrices
- Training and evaluation time
- An investigation-specific composite ranking

Using multiple measures helped identify differences that overall accuracy alone could conceal, particularly for underrepresented emotion categories.

## Technologies and environment

The project uses:

- Python
- TensorFlow and Keras
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Jupyter Notebook

The experiments were developed and run in a Docker-based environment with GPU acceleration. The repository includes a `Dockerfile` and `requirements.txt` to support environment setup.

GPU support depends on the host system, drivers and Docker/NVIDIA configuration. Training is resource-intensive, and execution time or results may differ on other hardware or software versions.

## Results

### Overall comparison

VGG16-based transfer-learning configurations performed most strongly in the overall comparison. VGG16 configurations occupied the first eight positions in the composite ranking, while the highest-ranked custom CNN was `CNN_Model_Residual_fnloss`, which ranked ninth.

The highest-ranked configuration overall was:

**`VGG16_Unfreeze4_Weighted`**

Its reported results included:

| Measure | Result |
|---|---:|
| Mean validation accuracy | 0.5852 |
| Mean test accuracy | 0.5596 |
| Mean balanced accuracy | 0.5540 |
| Macro precision* | 0.5488 |
| Macro recall* | 0.5846 |
| Macro F1* | 0.5579 |

*The macro-averaged classification-report measures are from the selected best run, rather than averages across the five successful runs. Accuracy and balanced accuracy are repeated-run means.*

### Key findings

- **Transfer learning:** VGG16-based configurations generally outperformed the custom CNNs and the other transfer-learning architectures under the experimental conditions.
- **Architecture matters:** The effects of architectural changes varied between models. Greater depth or filter capacity did not automatically result in better performance.
- **Class weighting:** Class weighting improved balanced accuracy and Macro F1 for the selected transfer-learning configurations, but reduced these measures for the custom CNNs.
- **Focal loss:** Its effects were mixed, and combining it with class weighting did not consistently provide additional improvements.
- **Class-level performance:** Happy and Surprise were generally recognised more successfully than Disgust and Fear. Difficulties with individual categories remained even in stronger configurations.
- **Evaluation measures:** No single model performed best across every measure, demonstrating the importance of considering aggregate and class-level performance together.
- **Computational performance:** VGG16 configurations generally offered a favourable balance between predictive performance and recorded computational time.

## Limitations

The findings should be interpreted within the scope of the experiments conducted.

- Only the FER2013 dataset was evaluated; performance on independent datasets was not established.
- The dataset contains low-resolution grayscale images and uneven class representation.
- Five successful runs provide some indication of run-to-run variability, but additional repetitions would strengthen the comparisons.
- The experimental design does not isolate every individual architectural or training-strategy effect.
- Test-derived measures contributed to model selection and the overall ranking. The test partition was not used for gradient updates, but it was not an entirely independent final assessment.
- Recorded computational times included model creation, compilation, fitting, evaluation and, where applicable, failed out-of-memory attempts and retries. Peak memory, energy consumption and inference latency were not systematically measured.

Importantly, the models classify images according to FER2013's predefined expression labels. These predictions should not be interpreted as definitive assessments of a person's actual emotional or psychological state.

## Repository contents

The GitHub repository contains the project code, notebooks, reports and supporting materials. Its visible top-level structure includes:

| Path | Contents |
|---|---|
| `data/processed/` | Directory for processed data-related files |
| `notebooks/` | Jupyter notebooks |
| `reports/` | Project reports and related materials |
| `results/` | Experimental results and supporting outputs |
| `scripts/` | Supporting scripts |
| `src/` | Python source code |
| `.gitignore` | Files and directories excluded from version control |
| `Dockerfile` | Docker environment definition |
| `requirements.txt` | Python package requirements |
| `README.md` | Project overview and documentation |

The raw and processed FER2013 CSV files and trained model files are excluded from version control. Their absence from the repository does not mean they were not used during the experiments; they were part of the local working environment.

## Jupyter working environment

During the experiments, the project was run in a Jupyter environment whose file browser showed the following folders:

- `confusion_matrices`
- `data`
- `histories`
- `models`
- `plots`
- `reports`
- `results`
- `src`

The Jupyter file browser also showed the notebooks `CNN_all.ipynb`, `Transfer_part1.ipynb`, `Transfer_part2.ipynb` and `All_Plots.ipynb`, along with `check_environment.py`, `Dockerfile` and `requirements.txt`.

This describes the working environment used during the experiments; it is not a representation of the GitHub repository's directory structure. When running the notebooks in another Jupyter environment, check their file paths and ensure that the required data, output folders and supporting files are available in the locations expected by the notebook code.

## Reproducibility

The repository provides notebooks and supporting code for data preparation, model construction, training, evaluation and plot generation. Reproducing the experiments also requires the FER2013 dataset, a compatible software environment and sufficient compute resources.

### 1. Obtain the dataset

Download FER2013 from the [Kaggle dataset page](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge).

The dataset files are not committed to this repository. Place the downloaded files in the location expected by the data-loading and preprocessing code.

### 2. Prepare the data

The processed dataset files are also excluded from version control. Run the relevant data-preparation workflow in the notebooks or supporting source code to generate the files required by the training and evaluation notebooks.

Check the file paths in the relevant notebook cells before running them. The exact local paths and required execution order depend on the data-preparation workflow.

### 3. Set up the environment

The project was developed in a Docker-based Jupyter environment. The repository includes a `Dockerfile` and a `requirements.txt` file.

The `Dockerfile` defines the container environment, while `requirements.txt` lists Python package requirements. TensorFlow installation and GPU support depend on the host operating system, hardware, drivers and Docker/NVIDIA configuration. A conventional Python installation may require environment-specific adjustments.

### 4. Run the notebooks

Open the project in Jupyter and run the notebooks as needed:

1. `CNN_all.ipynb` — custom CNN experiments.
2. `Transfer_part1.ipynb` — first-stage transfer-learning experiments.
3. `Transfer_part2.ipynb` — second-stage transfer-learning experiments.
4. `All_Plots.ipynb` — plot generation.

The notebooks may depend on data, results or model files created by earlier cells or other notebooks. Follow the notebook execution order and check its paths and prerequisites before running a full experiment.

Training is resource-intensive and may take substantially longer on different hardware. Results may also vary with software versions, hardware, random seeds and other execution details. The reported results are the outcomes of the recorded experimental runs, not guaranteed results from every rerun.

## Further work

Potential extensions include:

- Evaluating selected models on independent facial-expression datasets.
- Investigating alternative pretrained architectures and training strategies.
- Examining class-balanced sampling and alternative loss functions.
- Increasing the number of independent experimental runs.
- Separating training, evaluation and inference costs in computational comparisons.
- Investigating the generalisability of models under different image and recording conditions.

## Author

Independent data science project developed from work originally undertaken during BSc Data Science studies with The Open University.
