Yoga Pose Classification
========================

This project focuses on the classification of yoga poses using the Yoga-82 dataset and the Kaggle Yoga Poses dataset. The main goal is to implement and evaluate hierarchical classification models on these datasets, particularly using DenseNet-201 variants. The hierarchical classification aims to improve performance by leveraging the hierarchical nature of the pose labels.

Dataset Preparation:
--------------------

Yoga-82 Dataset:
1. Download the Yoga-82 dataset from: https://sites.google.com/view/yoga-82/home
2. Place the downloaded images in the directory: data\Yoga-82\YOGA_downloads
3. Code for training with the Yoga-82 dataset is available in: _1_Yoga-82_DenseNet.ipynb

Kaggle Yoga Pose Dataset:
1. Download the Kaggle Yoga Pose dataset (5 poses) from: https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset
2. Place the downloaded images in the directory: data\kaggle_yogaposes
3. Code for training with the Kaggle dataset is available in: _1_Kaggle_DenseNet.ipynb

Models:
-------
This project uses three variants of the DenseNet-201 model for hierarchical classification of yoga poses:
1. DenseNet-201 Variant 1
2. DenseNet-201 Variant 2
3. DenseNet-201 Variant 3

Experiments:
------------
Multiple experiments are conducted to compare the effectiveness of hierarchical classification using the above models. Performance metrics such as classification accuracy and loss are analyzed.

Conclusion:
-----------
The results indicate that hierarchical classification improves classification accuracy for complex datasets like Yoga-82. However, there are limitations in dataset quality and model performance, which offer areas for future improvements.
