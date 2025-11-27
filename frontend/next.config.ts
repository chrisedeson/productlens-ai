import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  reactCompiler: true,
  images: {
    // Allow data URLs for uploaded image previews
    dangerouslyAllowSVG: true,
    remotePatterns: [],
  },
};

export default nextConfig;
