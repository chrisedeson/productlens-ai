import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  reactCompiler: true,
  images: {
    // Allow data URLs for uploaded image previews
    dangerouslyAllowSVG: true,
    remotePatterns: [],
    // Optimize images for production
    unoptimized: process.env.NODE_ENV === 'development',
  },
  // Enable static export optimization
  output: 'standalone',
  // Disable x-powered-by header for security
  poweredByHeader: false,
  // Compress responses
  compress: true,
};

export default nextConfig;
