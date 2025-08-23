const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function _post(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

export function predictRegression({ tag, framework, X }) {
  return _post("/predict/regression", { tag, framework, X });
}

export function predictClassification({ tag, framework, X }) {
  return _post("/predict/classification", { tag, framework, X });
}
