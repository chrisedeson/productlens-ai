# ProductLens AI - Frontend

A modern, responsive Next.js frontend for the ProductLens AI e-commerce product recommendation system.

## Features

- **Natural Language Search** - Search for products using conversational queries
- **OCR Search** - Upload handwritten shopping lists for automatic text extraction
- **Image Search** - Upload product images for CNN-powered category detection
- **Modern UI** - Glass morphism effects, smooth animations, and dark mode support
- **Responsive Design** - Works seamlessly on desktop and mobile devices

## Tech Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Font**: Geist (Vercel)

## Getting Started

### Prerequisites

- Node.js 18+ 
- Backend API running (see `/backend` folder)

### Installation

```bash
npm install
```

### Environment Setup

Copy the example environment file:

```bash
cp .env.example .env.local
```

Update `.env.local` with your backend API URL:

```env
NEXT_PUBLIC_API_URL=http://localhost:5000
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Deploy on Vercel

### One-Click Deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/your-repo/productlens-ai)

### Manual Deployment

1. Install Vercel CLI:
   ```bash
   npm i -g vercel
   ```

2. Deploy:
   ```bash
   vercel
   ```

3. Set environment variables in Vercel dashboard:
   - `NEXT_PUBLIC_API_URL`: Your deployed backend URL

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | Yes |

## Project Structure

```
src/
├── app/
│   ├── globals.css      # Global styles with CSS variables
│   ├── layout.tsx       # Root layout
│   └── page.tsx         # Main page with tab navigation
├── components/
│   ├── TextSearch.tsx   # Natural language search
│   ├── OCRSearch.tsx    # OCR image upload
│   ├── ImageSearch.tsx  # CNN image classification
│   └── ProductCard.tsx  # Product result cards
└── ...
```

## Backend Setup

The frontend requires the ProductLens AI backend to be running. See the `/backend` folder for setup instructions.

For production deployment, ensure:
1. Backend is deployed and accessible
2. Backend CORS is configured for your Vercel domain
3. `NEXT_PUBLIC_API_URL` points to your deployed backend

## License

MIT

