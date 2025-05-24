// // @type {import('next').NextConfig}
// const nextConfig = {
//   reactStrictMode: true,
//   async rewrites() {
//     return [
//       {
//         source: '/api/:path*',
//         destination: 'http://localhost:4000/api/:path*',
//       },
//     ];
//   },
//   images: {
//     domains: [],
//   },
//   experimental: {
//     allowedDevOrigins: ["10.127.127.1"] // Add your network origin here
//   }
// };

// module.exports = nextConfig;
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [{
      source: '/api/:path*',
      // destination: 'http://localhost:3001/api/:path*'
      destination: 'http://127.0.0.1:3001/api/:path*'  // Changed here
    }]
  }
  // experimental: {
  //   allowedDevOrigins: ["localhost"]
  // }
}

module.exports = nextConfig