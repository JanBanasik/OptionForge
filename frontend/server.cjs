const express = require("express");
const { createProxyMiddleware } = require("http-proxy-middleware");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;
const BACKEND = process.env.BACKEND_URL || "http://localhost:8000";

// Proxy /api requests to backend (preserving full path)
app.use(
  createProxyMiddleware({
    target: BACKEND,
    changeOrigin: true,
    pathFilter: "/api",
  }),
);

// Serve static frontend build
app.use(express.static(path.join(__dirname, "dist")));

// SPA fallback — send index.html for any non-API, non-static route
app.use((req, res, next) => {
  if (req.path.startsWith("/api")) return next();
  // Only apply to GET requests that don't match static files
  if (req.method === "GET" && !req.path.includes(".")) {
    return res.sendFile(path.join(__dirname, "dist", "index.html"));
  }
  next();
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Frontend server on :${PORT}, proxying /api → ${BACKEND}`);
});
