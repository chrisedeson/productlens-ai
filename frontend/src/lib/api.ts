/**
 * API Service for ProductLens AI Frontend
 * Handles communication with the Flask backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

export interface Product {
  stock_code: string;
  description: string;
  unit_price: number;
  country: string;
}

export interface RecommendationResponse {
  products: Product[];
  response: string;
}

export interface OCRResponse {
  products: Product[];
  response: string;
  extracted_text: string;
}

export interface ImageSearchResponse {
  products: Product[];
  response: string;
  predicted_class: string;
  predictions: Array<{
    class_name: string;
    stock_code: string;
    description: string;
    confidence: number;
  }>;
}

export interface HealthResponse {
  status: string;
  services: {
    recommendation: boolean;
    ocr: boolean;
    cnn: boolean;
  };
}

/**
 * Get product recommendations based on natural language query
 */
export async function getRecommendations(query: string): Promise<RecommendationResponse> {
  const response = await fetch(`${API_BASE_URL}/product-recommendation`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    throw new Error(`Failed to get recommendations: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Process handwritten query image using OCR
 */
export async function processOCRQuery(imageFile: File): Promise<OCRResponse> {
  const formData = new FormData();
  formData.append('handwritten_query', imageFile);

  const response = await fetch(`${API_BASE_URL}/ocr-query`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Failed to process OCR query: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Search for products using image detection
 */
export async function searchByImage(imageFile: File): Promise<ImageSearchResponse> {
  const formData = new FormData();
  formData.append('product_image', imageFile);

  const response = await fetch(`${API_BASE_URL}/image-product-search`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Failed to search by image: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Check backend health status
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }

  return response.json();
}
