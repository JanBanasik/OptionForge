import type { PricingRequest, PricingResponse, VisualizationResponse } from "../types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

async function post<T>(path: string, body: PricingRequest): Promise<T> {
  const res = await fetch(`${BASE}/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export function fetchPrice(req: PricingRequest): Promise<PricingResponse> {
  return post<PricingResponse>("/price", req);
}

export function fetchVisualization(req: PricingRequest): Promise<VisualizationResponse> {
  return post<VisualizationResponse>("/visualization", req);
}
