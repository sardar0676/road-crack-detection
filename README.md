# Road Crack Detection with Deep Learning

A binary image classifier that predicts whether a **concrete / pavement surface image contains a crack** (`Cracked`) or not (`Non-Cracked`). It uses transfer learning with **MobileNetV2** (ImageNet weights) and was trained in a Kaggle Notebook on the **SDNET** (Structural Defects Network) dataset.

> **Scope note:** SDNET consists of close-up, 256×256 photos of concrete surfaces (bridge decks, walls, pavements). The model is a screening aid for close-up surface images. It has **not** been validated on wide street photos, asphalt roads, or other cameras, and its output should be reviewed by a person.

## Results

Evaluated once on a held-out **test set of 4,176 images** (1,255 cracked, 2,921 non-cracked) that played no role in training or tuning.

| Metric | Threshold 0.131 (high recall) | Threshold 0.5 (default) |
|---|---|---|
| Accuracy | 63.4% | 87.3% |
| Precision | 44.8% | 83.2% |
| Recall | 94.6% | 72.5% |
| F1-score | 0.608 | 0.775 |
| Missed cracks (false negatives) | 68 of 1,255 (5.4%) | 345 of 1,255 (27.5%) |
| False alarms (false positives) | 1,462 of 2,921 | 184 of 2,921 |

**ROC-AUC: 0.909** (independent of the threshold).

**How to read this:** the model clearly learned to detect cracks, but it cannot give high recall *and* few false alarms at the same time. The decision threshold lets you choose the trade-off:

- Low threshold (0.131, chosen on validation data to reach about 95% recall): catches almost every crack but flags about half of the clean images.
- Default threshold (0.5): alerts are trustworthy, but about 27% of cracks are missed.
- An intermediate value such as 0.30 is a reasonable compromise (on validation data: about 82% recall, 64% precision).

Training curves show no strong overfitting (training accuracy about 87.6%, validation accuracy about 87.2%), and both losses were still decreasing at epoch 20, so longer fine-tuning would likely improve the results.

## Dataset

- **SDNET** (Structural Defects Network), obtained from Kaggle. The dataset is **not** included in this repository.
- Images used: 28,484 (20,000 non-cracked, 8,484 cracked), capped at 20,000 per class.
- Split: about 70% train / 15% validation / 15% test, stratified by class. When file names reveal the source concrete panel, whole panels are kept in one split to prevent data leakage.

## Method

1. **Preprocessing:** decode as RGB, resize to 224×224, `MobileNetV2.preprocess_input` (pixels scaled to [-1, 1]).
2. **Augmentation (training data only):** horizontal flip, small rotation (about ±11°), zoom ±10%, contrast ±10%.
3. **Model:** MobileNetV2 (no top) → GlobalAveragePooling → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.3) → Dense(1, sigmoid).
4. **Training:** Adam, binary cross-entropy, class weights for the imbalance, batch size 32, EarlyStopping / ModelCheckpoint / ReduceLROnPlateau. Stage 1 trains the new head; stage 2 fine-tunes the last 30 MobileNetV2 layers at a learning rate of 1e-5 (20 epochs in total).
5. **Threshold:** chosen on the validation set (highest precision with recall of at least 95%), never on the test set.

## Repository contents

| File | Description |
|---|---|
| `road_crack_detection.ipynb` | The full Kaggle notebook (EDA, training, evaluation, prediction function) |
| `README.md` | This file |

Optional (if you upload them): `road_crack_detection.keras` (trained model), `class_mapping.json`, `preprocessing_config.json`, `training_history.json`.

## How to run

1. Create a new notebook on [Kaggle](https://www.kaggle.com), then **File → Import Notebook** and upload `road_crack_detection.ipynb`.
2. Add the SDNET dataset via **+ Add Input**.
3. In the notebook settings, turn **GPU** and **Internet** on (Internet is needed once to download the ImageNet weights).
4. Click **Run All**. For a quick trial, set `MAX_IMAGES_PER_CLASS = 2000` in the configuration cell.

## Predicting on a new image

After training, the notebook defines `predict_road_crack(image_path)`:

```python
predict_road_crack("/kaggle/working/test_road.jpg")
# Prediction: CRACKED
# Confidence: 96.45%      (illustrative output)
```

It converts the image to RGB, resizes it to 224×224, applies the same preprocessing as in training, and shows the image with the predicted class and confidence. The decision threshold can be changed per call, e.g. `predict_road_crack(path, threshold=0.3)`.

## Limitations

- Trained and tested only on SDNET (concrete surfaces, close-up images); performance on other data is unknown.
- The test set contains 30% cracked images; in real inspections cracks are usually rarer, so precision would be lower in practice.
- The model is not a replacement for human inspection.

## Possible improvements

Longer fine-tuning with more unfrozen layers, a larger backbone (e.g. EfficientNet), probability calibration, Grad-CAM visual explanations, and a segmentation model if crack location or width is needed.

## Acknowledgements

SDNET dataset (Structural Defects Network) and the MobileNetV2 architecture (Sandler et al., 2018), used via TensorFlow/Keras.
