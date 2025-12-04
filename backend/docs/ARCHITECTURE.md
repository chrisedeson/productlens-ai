# ProductLens AI - System Architecture

This document describes the overall architecture of the ProductLens AI system.

## High-Level Overview

ProductLens AI is a multi-modal e-commerce product recommendation system with three main capabilities:

1. **Semantic Text Search** - Natural language product discovery
2. **Handwriting OCR** - Shopping list digitization
3. **Image Recognition** - Visual product identification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ProductLens AI                                  │
├─────────────────┬───────────────────┬───────────────────┬───────────────────┤
│   Text Search   │   Handwriting OCR │  Image Recognition│    Hybrid Mode    │
│   (OpenAI +     │   (Tesseract +    │  (TensorFlow      │   (All combined)  │
│    Pinecone)    │    GPT-4 Vision)  │   CNN)            │                   │
└─────────────────┴───────────────────┴───────────────────┴───────────────────┘
```

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                  Frontend                                    │
│                           (Next.js / TypeScript)                             │
│                          Deployed on: Vercel                                 │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │ HTTPS
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                                  Backend                                      │
│                            (Flask / Python)                                   │
│                          Deployed on: Render                                  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                           API Layer (app.py)                            │  │
│  │  - REST endpoints          - Request validation    - Response schemas   │  │
│  │  - CORS handling           - Error handling        - Health checks      │  │
│  └────────────────────────────────────┬────────────────────────────────────┘  │
│                                       │                                       │
│  ┌────────────────────────────────────┴─────────────────────────────────────┐ │
│  │                         Service Layer                                    │ │
│  │                                                                          │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │ │
│  │  │  Search Services │  │   OCR Services   │  │   ML Services    │      │ │
│  │  │                  │  │                  │  │                  │      │ │
│  │  │ - Embedding      │  │ - OCR Service    │  │ - Image          │      │ │
│  │  │ - Vector Store   │  │   (Tesseract +   │  │   Classifier     │      │ │
│  │  │ - LLM            │  │    OpenAI)       │  │   (TensorFlow)   │      │ │
│  │  │ - Recommendation │  │                  │  │                  │      │ │
│  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘      │ │
│  │           │                     │                     │                │ │
│  └───────────┼─────────────────────┼─────────────────────┼────────────────┘ │
│              │                     │                     │                  │
└──────────────┼─────────────────────┼─────────────────────┼──────────────────┘
               │                     │                     │
               ▼                     ▼                     ▼
┌──────────────────────┐ ┌──────────────────┐ ┌──────────────────────────────┐
│      OpenAI API      │ │   Tesseract OCR  │ │      Hugging Face Hub        │
│  - text-embedding-3  │ │   (Local)        │ │  - CNN Model (309MB)         │
│  - GPT-4             │ │                  │ │  - Class Names               │
│  - GPT-4 Vision      │ │                  │ │                              │
└──────────────────────┘ └──────────────────┘ └──────────────────────────────┘
               │
               ▼
┌──────────────────────┐
│     Pinecone DB      │
│  - Vector Index      │
│  - Product Metadata  │
└──────────────────────┘
```

## Directory Structure

```
productlens-ai/
├── backend/
│   ├── app.py                    # Main Flask application
│   ├── core/                     # Core utilities
│   │   ├── __init__.py
│   │   ├── base_service.py       # Abstract base service
│   │   └── exceptions.py         # Custom exceptions
│   ├── schemas/                  # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── requests.py           # Request validation
│   │   └── responses.py          # Response serialization
│   ├── services/                 # Business logic
│   │   ├── search/               # Search services
│   │   │   ├── embedding_service.py
│   │   │   ├── vector_service.py
│   │   │   ├── llm_service.py
│   │   │   └── recommendation_service.py
│   │   └── ocr/                  # OCR services
│   │       └── ocr_service.py
│   ├── ml/                       # Machine learning
│   │   ├── config.py             # ML configuration
│   │   ├── data/                 # Data processing
│   │   │   ├── data_cleaner.py
│   │   │   └── image_scraper.py
│   │   ├── training/             # Model training
│   │   │   ├── data_pipeline.py
│   │   │   └── cnn_trainer.py
│   │   ├── inference/            # Model inference
│   │   │   └── image_classifier.py
│   │   └── evaluation/           # Model evaluation
│   ├── data/                     # Data files
│   ├── docs/                     # Documentation
│   ├── tests/                    # Test suite
│   ├── Dockerfile
│   ├── Makefile
│   └── requirements.txt
│
└── frontend/
    ├── app/                      # Next.js app directory
    ├── components/               # React components
    ├── lib/                      # Utilities
    └── public/                   # Static assets
```

## Core Components

### 1. Base Service Pattern

All services inherit from `BaseService`:

```python
class BaseService(ABC):
    """Abstract base class for all services."""
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        pass
    
    def validate(self, **kwargs):
        """Validate input parameters."""
        pass
    
    def _handle_error(self, error: Exception, operation: str):
        """Standardized error handling."""
        pass
    
    def _log_operation(self, operation: str, **context):
        """Structured logging."""
        pass
```

### 2. Exception Hierarchy

```
ProductLensError (base)
├── ValidationError
│   └── InvalidInputError
├── ServiceError
│   ├── ServiceInitializationError
│   └── ServiceUnavailableError
├── ConfigurationError
├── ModelError
│   ├── ModelLoadError
│   └── ModelInferenceError
├── ExternalServiceError
│   ├── OpenAIError
│   ├── PineconeError
│   └── HuggingFaceError
└── DataError
    ├── DataCleaningError
    └── DataIngestionError
```

### 3. Pydantic Schemas

Request and response validation using Pydantic v2:

```python
# Request validation
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=50)
    include_metadata: bool = Field(default=True)

# Response serialization
class SearchResponse(BaseModel):
    success: bool
    query: str
    recommendations: List[ProductResult]
    explanation: str
```

## API Endpoints

### Search

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/recommend` | POST | Semantic product search |
| `/api/batch-recommend` | POST | Batch search multiple queries |

### OCR

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ocr` | POST | Extract text from image |
| `/api/ocr/shopping-list` | POST | Parse handwritten shopping list |

### Image Classification

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/classify` | POST | Classify product image |

### Utility

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/` | GET | API info |

## Data Flow

### 1. Text Search Flow

```
User Query
    │
    ▼
┌───────────────────┐
│ EmbeddingService  │
│ (OpenAI)          │
│                   │
│ Query → Vector    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ VectorService     │
│ (Pinecone)        │
│                   │
│ Vector → Matches  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ LLMService        │
│ (OpenAI GPT-4)    │
│                   │
│ Generate          │
│ Explanation       │
└─────────┬─────────┘
          │
          ▼
     Response
```

### 2. OCR Flow

```
Handwritten Image
    │
    ▼
┌───────────────────┐
│ Image Preprocess  │
│ - Grayscale       │
│ - Contrast        │
│ - Denoise         │
└─────────┬─────────┘
          │
          ├────────────────────┐
          ▼                    ▼
┌───────────────────┐  ┌───────────────────┐
│ Tesseract OCR     │  │ OpenAI Vision     │
│ (Local)           │  │ (API)             │
└─────────┬─────────┘  └─────────┬─────────┘
          │                      │
          ▼                      ▼
┌───────────────────────────────────────────┐
│         Select Best Result                │
│         (by confidence/quality)           │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
          Extracted Text
```

### 3. Image Classification Flow

```
Product Image
    │
    ▼
┌───────────────────┐
│ Image Preprocess  │
│ - Resize 224x224  │
│ - Normalize       │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ CNN Model         │
│ (EfficientNetV2)  │
│                   │
│ From: HuggingFace │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Softmax Output    │
│                   │
│ Top-K Classes     │
│ with Confidence   │
└─────────┬─────────┘
          │
          ▼
     Predictions
```

## External Services

### OpenAI API

| Service | Model | Purpose |
|---------|-------|---------|
| Embeddings | text-embedding-3-small | Convert text to vectors |
| Chat | GPT-4-turbo | Generate recommendations |
| Vision | GPT-4-vision | OCR for handwriting |

### Pinecone

- **Index Type**: Cosine similarity
- **Dimensions**: 1536 (OpenAI embeddings)
- **Metadata**: Stock code, description, price, country

### Hugging Face Hub

- **Repository**: `chrisedeson/productlens-cnn-model`
- **Model Size**: ~309MB
- **Format**: Keras (.keras)

## Deployment

### Frontend (Vercel)

- **URL**: https://productlens-ai.vercel.app
- **Build**: `npm run build`
- **Environment**: `NEXT_PUBLIC_API_URL`

### Backend (Render)

- **URL**: https://productlens-ai-backend.onrender.com
- **Runtime**: Docker
- **Memory**: 512MB (free tier)
- **Environment Variables**:
  - `OPENAI_API_KEY`
  - `PINECONE_API_KEY`
  - `PINECONE_INDEX_NAME`
  - `PINECONE_HOST`
  - `HF_TOKEN`
  - `DISABLE_CNN` (optional)

## Security

### API Security

- CORS configured for frontend domain
- Input validation on all endpoints
- Rate limiting (TODO)

### Secrets Management

- Environment variables for API keys
- No secrets in code or repository
- Render/Vercel secret management

## Monitoring

### Health Checks

```bash
curl https://productlens-ai-backend.onrender.com/api/health
```

### Logging

- Structured logging with timestamps
- Operation tracking
- Error stack traces

## Performance Considerations

### Memory Optimization

1. **Lazy Model Loading**: CNN loaded on first request
2. **Disable CNN**: `DISABLE_CNN=true` for low-memory environments
3. **Streaming**: Use streaming responses for large results (TODO)

### Caching

1. **Embedding Cache**: Cache common query embeddings (TODO)
2. **Model Cache**: Model loaded once at startup

### Scaling

1. **Horizontal**: Multiple Render instances
2. **Vertical**: Upgrade to larger instance

## Future Improvements

1. **Redis Caching** - Cache embeddings and results
2. **Rate Limiting** - Protect against abuse
3. **Async Processing** - Background jobs for heavy operations
4. **Model Quantization** - Reduce CNN model size
5. **A/B Testing** - Compare model versions
6. **Analytics** - Track search quality metrics
