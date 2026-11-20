# CNN-vs-transfer-learning-emotion-recognition

## Overview

This project investigates facial emotion recognition using the FER2013 dataset. It compares custom convolutional neural network (CNN) architectures with established transfer-learning models, examining how architectural design choices and training strategies influence classification performance.

Originally developed from work undertaken during my Open University studies, this independent project extends the investigation into a comparative evaluation of custom CNN architectures, pretrained models, training strategies and performance measures.

The experiments consider not only overall classification accuracy, but also class-level performance, the effects of class imbalance and computational requirements.

## Research Aim

The aim of this project is to evaluate how architectural design choices and selected training strategies influence the predictive performance and computational efficiency of deep-learning models for facial emotion recognition.

The investigation addresses the following research question:

> How do architectural design choices and training strategies influence the performance of deep-learning models for facial emotion recognition on the FER2013 dataset?

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

## Methodology

The investigation compared four custom CNN architectures with three pretrained transfer-learning architectures:

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

## Technologies

The project uses:

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Jupyter Notebook

The experiments were developed and run in a Docker-based environment with GPU acceleration.

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

*Macro-averaged classification-report measures are from the selected best run, rather than averages across the five successful runs. Accuracy and balanced accuracy are repeated-run means.*

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

## Repository Contents

The repository is intended to contain the relevant project materials, including:

- Technical project report
- Jupyter notebooks
- Python source code
- Model implementation and training code
- Evaluation and comparison code
- Supporting plots and figures
- Experimental results and supplementary material
- References and supporting documentation

See the repository folders for the individual files and supporting materials.

## Reproducibility

The project uses TensorFlow/Keras and a GPU-enabled Docker environment.

To reproduce the experiments, the required Python environment, dataset and supporting files must be available. The notebooks and source code provide the implementation details for data preparation, model construction, training and evaluation.

The FER2013 dataset should be obtained from its original Kaggle source rather than assumed to be included with the source code.

## Further Work

Potential extensions include:

- Evaluating selected models on independent facial-expression datasets.
- Investigating alternative pretrained architectures and training strategies.
- Examining class-balanced sampling and alternative loss functions.
- Increasing the number of independent experimental runs.
- Separating training, evaluation and inference costs in computational comparisons.
- Investigating the generalisability of models under different image and recording conditions.

## Author

Independent data science project developed from work originally undertaken during BSc Data Science studies with The Open University.
