# ProductLens AI

An intelligent e-commerce product recommendation system with OCR capabilities and CNN-based image classification.

## Project Overview

ProductLens AI combines multiple AI/ML technologies to deliver a comprehensive product intelligence platform:

1. **Vector-Based Product Recommendations** - Natural language queries matched against product embeddings in Pinecone
2. **OCR Text Extraction** - Tesseract-powered handwritten text recognition for product queries
3. **CNN Image Classification** - Custom-trained neural network for product image classification

## Tech Stack

### Backend
- **Flask 3.0** - Python web framework
- **Pinecone** - Vector database for semantic search
- **OpenAI** - text-embedding-3-small for embeddings, GPT-3.5-turbo for LLM responses
- **TensorFlow/Keras** - CNN model training and inference
- **Tesseract OCR** - Handwritten text extraction

### Frontend
- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling

## Project Structure

```
productlens-ai/
├── backend/
│   ├── app.py                    # Flask application factory
│   ├── config.py                 # Configuration management
│   ├── requirements.txt          # Python dependencies
│   ├── data/
│   │   ├── raw/                  # Original dataset
│   │   ├── cleaned/              # Processed dataset
│   │   └── images/               # Scraped training images
│   ├── models/                   # Data models
│   ├── routes/                   # API endpoints
│   ├── services/                 # Business logic
│   ├── trained_models/           # Saved CNN models
│   ├── templates/                # HTML templates
│   ├── tests/                    # Unit tests
│   └── utils/                    # Helper functions
├── frontend/                     # Next.js application
├── reports/                      # Module documentation
└── notebooks/                    # Jupyter notebooks
```

## API Endpoints

### 1. Product Recommendation
```
POST /api/recommendations
Content-Type: application/json

{
  "query": "I need a gift for someone who loves gardening"
}
```

### 2. OCR Product Search
```
POST /api/ocr
Content-Type: multipart/form-data

image: <handwritten text image>
```

### 3. Image Classification
```
POST /api/classify
Content-Type: multipart/form-data

image: <product image>
```

## Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js 18+
- Tesseract OCR

### Backend Setup

1. Create virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Run the server:
```bash
python app.py
```

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Configure environment:
```bash
cp .env.example .env.local
# Edit with backend URL
```

3. Run development server:
```bash
npm run dev
```

## CNN Model Classes

The CNN model is trained to classify 10 product categories:

| Stock Code | Product Description |
|------------|---------------------|
| 22384 | LUNCH BAG PINK POLKADOT |
| 22727 | ALARM CLOCK BAKELIKE PINK |
| 22112 | CHOCOLATE HOT WATER BOTTLE |
| 23298 | SPOTTY BUNTING |
| 20726 | LUNCH BAG CARS BLUE |
| 21034 | VINTAGE SEASIDE JIGSAW PUZZLES |
| 21931 | JUMBO STORAGE BAG SKULLS |
| 22139 | RETROSPOT TEA SET CERAMIC 11 PC |
| 22077 | SIX RIBBONS RUSTIC CHARM |
| 22423 | REGENCY CAKESTAND 3 TIER |

## Dataset

The original dataset contains 541,910 e-commerce transactions with the following fields:
- InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

## Author

Christopher Edeson
