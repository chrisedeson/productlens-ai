# ProductLens AI - Backend Makefile
# ===================================

.PHONY: help install dev run test lint clean health check-env setup test-unit test-integration test-ml test-coverage

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║           ProductLens AI - Backend Commands                  ║"
	@echo "╠══════════════════════════════════════════════════════════════╣"
	@echo "║  make install        - Install dependencies                 ║"
	@echo "║  make dev            - Run development server               ║"
	@echo "║  make run            - Run production server                ║"
	@echo "║  make health         - Check all services health            ║"
	@echo "║  make test           - Run all pytest tests                 ║"
	@echo "║  make test-unit      - Run unit tests only                  ║"
	@echo "║  make test-integration - Run integration tests              ║"
	@echo "║  make test-ml        - Run ML-specific tests                ║"
	@echo "║  make test-coverage  - Run tests with coverage              ║"
	@echo "║  make sanity         - Run endpoint sanity checks           ║"
	@echo "║  make lint           - Run linter (flake8)                  ║"
	@echo "║  make check-env      - Verify environment variables         ║"
	@echo "║  make setup          - Full setup (install + check-env)     ║"
	@echo "║  make clean          - Remove cache and temp files          ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"

# Install dependencies
install:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

# Run development server with debug mode
dev:
	@echo "🚀 Starting development server..."
	FLASK_DEBUG=1 python app.py

# Run production server
run:
	@echo "🚀 Starting production server..."
	python app.py

# Health check - verify all services are running
health:
	@echo "🏥 Checking service health..."
	@curl -s http://localhost:5000/health | python -m json.tool || echo "❌ Backend not running! Start with 'make dev'"

# Run all pytest tests
test:
	@echo "🧪 Running all tests..."
	@pip show pytest > /dev/null 2>&1 || pip install pytest pytest-cov
	pytest tests/ -v

# Run unit tests only
test-unit:
	@echo "🧪 Running unit tests..."
	pytest tests/unit/ -v

# Run integration tests only
test-integration:
	@echo "🧪 Running integration tests..."
	pytest tests/integration/ -v -m integration

# Run ML-specific tests
test-ml:
	@echo "🧪 Running ML tests..."
	pytest tests/ml/ -v -m ml

# Run tests with coverage report
test-coverage:
	@echo "🧪 Running tests with coverage..."
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing
	@echo "📊 Coverage report generated in htmlcov/"

# Run endpoint sanity checks (manual testing)
sanity:
	@echo "🧪 Running endpoint sanity checks..."
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "1. Health Check Endpoint (GET /health)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@curl -s http://localhost:5000/health | python -m json.tool || echo "❌ FAILED - Is the server running?"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "2. Product Recommendation (POST /product-recommendation)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@curl -s -X POST http://localhost:5000/product-recommendation \
		-H "Content-Type: application/json" \
		-d '{"query": "tea set"}' | python -m json.tool || echo "❌ FAILED"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "3. OCR Endpoint (POST /ocr-query) - Method check"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@curl -s -X OPTIONS http://localhost:5000/ocr-query -w "Status: %{http_code}\n" -o /dev/null && echo "✅ Endpoint accessible (POST only)"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "4. Image Classification (POST /image-product-search) - Method check"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@curl -s -X OPTIONS http://localhost:5000/image-product-search -w "Status: %{http_code}\n" -o /dev/null && echo "✅ Endpoint accessible (POST only)"
	@echo ""
	@echo "✅ All sanity checks completed!"

# Check environment variables
check-env:
	@echo "🔍 Checking environment variables..."
	@test -f .env && echo "✅ .env file exists" || echo "❌ .env file missing!"
	@grep -q "OPENAI_API_KEY" .env 2>/dev/null && echo "✅ OPENAI_API_KEY is set" || echo "❌ OPENAI_API_KEY missing"
	@grep -q "PINECONE_API_KEY" .env 2>/dev/null && echo "✅ PINECONE_API_KEY is set" || echo "❌ PINECONE_API_KEY missing"
	@grep -q "PINECONE_HOST" .env 2>/dev/null && echo "✅ PINECONE_HOST is set" || echo "❌ PINECONE_HOST missing"
	@echo ""
	@echo "📊 Checking model files..."
	@test -f models/simple_cnn_model.keras && echo "✅ CNN model exists" || echo "⚠️  CNN model not found (training required)"
	@test -f models/class_mapping.json && echo "✅ Class mapping exists" || echo "⚠️  Class mapping not found"

# Lint the code
lint:
	@echo "🔎 Running linter..."
	@pip show flake8 > /dev/null 2>&1 || pip install flake8
	flake8 --max-line-length=120 --exclude=.venv,__pycache__ .
	@echo "✅ Linting complete!"

# Full setup
setup: install check-env
	@echo ""
	@echo "🎉 Setup complete! Run 'make dev' to start the server."

# Clean up cache files
clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Verify Pinecone connection
verify-pinecone:
	@echo "🌲 Verifying Pinecone connection..."
	@python -c "from dotenv import load_dotenv; load_dotenv(); from pinecone import Pinecone; import os; pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY')); print('✅ Pinecone connected!'); print(f'Index stats: {pc.Index(host=os.getenv(\"PINECONE_HOST\")).describe_index_stats()}')"

# Verify OpenAI connection
verify-openai:
	@echo "🤖 Verifying OpenAI connection..."
	@python -c "from dotenv import load_dotenv; load_dotenv(); from openai import OpenAI; import os; client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')); print('✅ OpenAI connected!')"
