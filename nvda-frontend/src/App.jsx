import { useMemo, useState } from "react";
import { predictRegression } from "./api";

const defaultMatrix = `0.12,0.03,0.45,0.20
0.10,0.04,0.44,0.18
0.08,0.05,0.46,0.22`;

function parseInput(text, mode) {
  let X;
  const trimmed = text.trim();
  try {
    if (trimmed.startsWith("[")) {
      X = JSON.parse(trimmed);
    } else {
      X = trimmed
        .split(/\r?\n/)
        .map((row) =>
          row
            .split(/[,\s]+/)
            .filter(Boolean)
            .map((v) => Number(v))
        )
        .filter((r) => r.length > 0);
    }
  } catch {
    throw new Error("Failed to parse input as JSON or CSV.");
  }
  if (!Array.isArray(X) || X.length === 0 || !Array.isArray(X[0])) {
    throw new Error("Input must be a 2D array [T,F] or CSV rows.");
  }
  if (mode === "last") return [X[X.length - 1]];
  return X;
}

export default function App() {
  const [framework, setFramework] = useState("lgbm"); // lgbm | lstm | bilstm
  const [tag, setTag] = useState("B");                // A | B | AFF
  const [mode, setMode] = useState("auto");           // auto | seq | last
  const [raw, setRaw] = useState(defaultMatrix);
  const [loading, setLoading] = useState(false);
  const [regResult, setRegResult] = useState(null);
  const [error, setError] = useState("");

  const helper = useMemo(() => {
    return framework === "lgbm"
      ? "LightGBM: you can paste [T,F]; server uses the last row."
      : "LSTM/BiLSTM: paste a full sequence [T,F] (e.g., 60×F).";
  }, [framework]);

  async function onPredict() {
    setError("");
    setLoading(true);
    setRegResult(null);
    try {
      const X = parseInput(
        raw,
        mode === "auto" ? (framework === "lgbm" ? "last" : "seq") : mode
      );
      // -- Only call REGRESSION now --
      const reg = await predictRegression({ tag, framework, X });
      setRegResult(reg);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 980, margin: "2rem auto", fontFamily: "Inter, system-ui, Arial" }}>
      <h1>NVDA Forecast — React Client</h1>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div>
          <label style={{ display: "block", fontWeight: 600 }}>Framework</label>
          <select value={framework} onChange={(e) => setFramework(e.target.value)}>
            <option value="lgbm">LightGBM</option>
            <option value="lstm">LSTM</option>
            <option value="bilstm">BiLSTM+Attention</option>
          </select>
        </div>
        <div>
          <label style={{ display: "block", fontWeight: 600 }}>Tag / View</label>
          <select value={tag} onChange={(e) => setTag(e.target.value)}>
            <option value="A">A (NVDA only)</option>
            <option value="B">B (All)</option>
            <option value="AFF">AFF (Affiliates)</option>
          </select>
        </div>
      </div>

      <div style={{ marginTop: 12 }}>
        <label style={{ display: "block", fontWeight: 600 }}>Input mode</label>
        <div style={{ display: "flex", gap: 16, marginTop: 4 }}>
          <label><input type="radio" name="mode" value="auto" checked={mode==="auto"} onChange={()=>setMode("auto")} /> Auto</label>
          <label><input type="radio" name="mode" value="seq" checked={mode==="seq"} onChange={()=>setMode("seq")} /> Sequence [T,F]</label>
          <label><input type="radio" name="mode" value="last" checked={mode==="last"} onChange={()=>setMode("last")} /> Last step [1,F]</label>
        </div>
        <small style={{ color: "#555" }}>{helper}</small>
      </div>

      <div style={{ marginTop: 12 }}>
        <label style={{ display: "block", fontWeight: 600 }}>Paste CSV or JSON matrix</label>
        <textarea
          value={raw}
          onChange={(e) => setRaw(e.target.value)}
          rows={8}
          style={{ width: "100%", fontFamily: "monospace" }}
          placeholder='CSV rows or JSON e.g. [[0.1,0.2,0.3,0.4],[...]]'
        />
      </div>

      <button
        onClick={onPredict}
        disabled={loading}
        style={{
          marginTop: 12, padding: "0.6rem 1rem",
          background: "#7e3ff2", color: "white",
          border: "none", borderRadius: 8, cursor: "pointer", fontWeight: 700
        }}
      >
        {loading ? "Predicting…" : "Predict (Regression)"}
      </button>

      {error && (
        <div style={{ marginTop: 12, color: "crimson" }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Only the Regression card is shown */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 16, marginTop: 12 }}>
        <div style={{ padding: 16, border: "1px solid #eee", borderRadius: 10 }}>
          <h3>Regression</h3>
          {regResult ? (
            <ul>
              <li><b>Framework:</b> {regResult.framework}</li>
              <li><b>Tag:</b> {regResult.tag}</li>
              <li><b>Prediction:</b> {regResult.y_pred}</li>
              {regResult.scaled && <li><b>Note:</b> {regResult.note}</li>}
            </ul>
          ) : (
            <p>No result yet.</p>
          )}
        </div>
      </div>

      <p style={{ marginTop: 12, color: "#666" }}>
        Tip: For LightGBM, the server uses the last row of your matrix. For LSTM/BiLSTM, it uses the full sequence.
      </p>
    </div>
  );
}
