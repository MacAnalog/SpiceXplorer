/** @type {import('next').NextConfig} */
const BACKEND = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

const nextConfig = {
  typedRoutes: true,
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${BACKEND}/api/:path*` },
    ];
  },
};

export default nextConfig;
