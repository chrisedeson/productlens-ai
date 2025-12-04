# ProductLens AI - Machine Learning Pipeline

This document describes the end-to-end ML pipeline for training and deploying the product image classification model.

## Overview

The ML pipeline consists of four main stages:

1. **Data Cleaning** - Prepare product descriptions from raw data
2. **Image Scraping** - Download product images from the web
3. **Model Training** - Train a CNN classifier
4. **Model Deployment** - Deploy to Hugging Face Hub and serve predictions

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Cleaning  │───▶│  Image Scraping │───▶│  Model Training │───▶│   Deployment    │
│                 │    │                 │    │                 │    │                 │
│ - Filter noise  │    │ - DuckDuckGo    │    │ - Data Pipeline │    │ - HuggingFace   │
│ - Remove dupes  │    │ - Deduplication │    │ - Augmentation  │    │ - Render API    │
│ - Validate text │    │ - Quality check │    │ - CNN Training  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Directory Structure

```
ml/
├── __init__.py
├── config.py              # MLConfig dataclass
├── data/
│   ├── __init__.py
│   ├── data_cleaner.py    # DataCleaner class
│   └── image_scraper.py   # ImageScraper class
├── training/
│   ├── __init__.py
│   ├── data_pipeline.py   # DataPipeline class
│   └── cnn_trainer.py     # CNNTrainer class
├── inference/
│   ├── __init__.py
│   └── image_classifier.py # ImageClassifier class
└── evaluation/
    └── __init__.py        # Metrics and evaluation
```

## Stage 1: Data Cleaning

### Purpose
Clean and prepare product descriptions from the UCI Online Retail dataset.

### Class: `DataCleaner`

```python
from ml.data.data_cleaner import DataCleaner

cleaner = DataCleaner(
    min_desc_length=10,
    max_desc_length=200,
    min_word_count=2
)

result = cleaner.clean("data/raw_products.csv")
print(f"Cleaned {result.valid_products} products")
```

### Cleaning Pipeline

1. **Text Validation**
   - Remove empty/null descriptions
   - Filter descriptions that are too short (<10 chars) or too long (>200 chars)
   - Require minimum word count (≥2 words)

2. **Pattern Removal**
   - Remove stock codes accidentally in descriptions
   - Filter test entries and placeholders
   - Remove non-ASCII characters

3. **Normalization**
   - Convert to uppercase for consistency
   - Normalize whitespace
   - Strip leading/trailing spaces

4. **Deduplication**
   - Remove duplicate stock codes
   - Keep most informative description when duplicates exist

### Output
- `CleaningResult` dataclass with statistics
- CSV file with cleaned data

---

## Stage 2: Image Scraping

### Purpose
Download product images for training data using web search.

### Class: `ImageScraper`

```python
from ml.data.image_scraper import ImageScraper

scraper = ImageScraper(
    images_per_product=10,
    target_size=(224, 224),
    max_concurrent=5
)

await scraper.scrape_product_images(
    product_list=products,
    output_dir="data/images"
)
```

### Scraping Pipeline

1. **Search Query Generation**
   - Combine product description with category hints
   - Add "product photo" suffix for better results

2. **Image Download**
   - Use DuckDuckGo Image Search (no API key required)
   - Concurrent downloads with rate limiting
   - Retry logic for failed downloads

3. **Quality Filtering**
   - Minimum resolution check (64x64)
   - Format validation (JPEG, PNG)
   - Perceptual hash for duplicate detection

4. **Preprocessing**
   - Resize to target size (224x224)
   - Convert to RGB
   - Save in consistent format

### Directory Structure
```
data/images/
├── 22384_LUNCH_BAG_PINK_POLKADOT/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── 22423_REGENCY_CAKESTAND/
│   └── ...
└── ...
```

---

## Stage 3: Model Training

### Data Pipeline

```python
from ml.training.data_pipeline import DataPipeline
from ml.config import MLConfig

config = MLConfig(
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.2
)

pipeline = DataPipeline(config)
train_ds, val_ds = pipeline.create_datasets("data/images")
```

### Data Augmentation

Applied during training to improve generalization:

| Augmentation | Range |
|--------------|-------|
| Random Flip | Horizontal |
| Random Rotation | ±10% |
| Random Zoom | ±10% |
| Random Contrast | ±10% |
| Random Brightness | ±10% |

### CNN Architecture

```python
from ml.training.cnn_trainer import CNNTrainer

trainer = CNNTrainer(config)
model = trainer.build_model(num_classes=10)
```

**Architecture Details:**

```
Input (224, 224, 3)
    │
    ▼
┌─────────────────┐
│  Data Augment   │  (during training only)
└────────┬────────┘
         │
    ▼    ▼    ▼
┌─────────────────┐
│ EfficientNetV2B0│  (pretrained, frozen)
│   Feature       │
│   Extraction    │
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│ GlobalAvgPool2D │
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│ Dropout (0.3)   │
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│ Dense (256)     │
│ BatchNorm + ReLU│
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│ Dropout (0.2)   │
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│ Dense (N)       │
│ Softmax         │
└────────┬────────┘
         │
    ▼
Output (N classes)
```

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 50 | Maximum training epochs |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 0.001 | Initial learning rate |
| `early_stopping_patience` | 10 | Epochs without improvement |
| `reduce_lr_patience` | 5 | LR reduction patience |
| `reduce_lr_factor` | 0.5 | LR reduction factor |

### Callbacks

1. **EarlyStopping** - Stop when validation loss plateaus
2. **ReduceLROnPlateau** - Reduce learning rate on plateau
3. **ModelCheckpoint** - Save best model by validation accuracy
4. **TensorBoard** - Training visualization

### Training Command

```python
history = trainer.train(
    train_dataset=train_ds,
    val_dataset=val_ds,
    epochs=50
)

trainer.save_model("models/product_classifier.keras")
```

---

## Stage 4: Deployment

### Model Upload to Hugging Face

```python
from ml.training.cnn_trainer import CNNTrainer

trainer = CNNTrainer(config)
trainer.load_model("models/product_classifier.keras")
trainer.upload_to_huggingface(
    repo_id="username/productlens-cnn-model",
    token="hf_xxx"
)
```

### Uploaded Files
- `model.keras` - Full Keras model
- `class_names.json` - Class name mapping
- `config.json` - Model configuration

### Inference

```python
from ml.inference.image_classifier import ImageClassifier

classifier = ImageClassifier(
    model_source="huggingface",
    model_id="chrisedeson/productlens-cnn-model"
)

predictions = classifier.classify(
    image_path="product.jpg",
    top_k=5
)
```

### Memory Optimization

For memory-constrained environments (e.g., Render free tier with 512MB):

```bash
# Disable CNN loading at startup
export DISABLE_CNN=true
```

The image classification endpoint will return a graceful error message when disabled.

---

## Evaluation Metrics

### Classification Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| Top-5 Accuracy | Correct class in top 5 predictions |
| Precision | Per-class precision (weighted) |
| Recall | Per-class recall (weighted) |
| F1-Score | Harmonic mean of precision/recall |

### Confusion Matrix

Generated during evaluation to identify commonly confused classes.

---

## Best Practices

### Data Quality

1. **Minimum Images**: At least 10 images per class
2. **Diversity**: Include varied backgrounds and angles
3. **Balance**: Aim for balanced class distribution
4. **Validation**: Manually verify a sample of scraped images

### Training

1. **Transfer Learning**: Start with pretrained weights
2. **Fine-tuning**: Optionally unfreeze later layers
3. **Regularization**: Use dropout and data augmentation
4. **Monitoring**: Watch for overfitting via TensorBoard

### Deployment

1. **Model Size**: Keep under 500MB for free tier hosting
2. **Lazy Loading**: Only load model when needed
3. **Caching**: Cache model in memory after first load
4. **Fallback**: Provide graceful degradation when model unavailable

---

## Troubleshooting

### Out of Memory During Training

```python
config = MLConfig(
    batch_size=16,  # Reduce batch size
    mixed_precision=True  # Enable mixed precision
)
```

### Poor Classification Accuracy

1. Check class imbalance
2. Increase training data
3. Try fine-tuning base model
4. Adjust learning rate

### Slow Inference

1. Use TensorFlow Lite for edge deployment
2. Enable GPU acceleration
3. Batch predictions when possible
