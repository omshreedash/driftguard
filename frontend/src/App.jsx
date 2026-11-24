import React, { useState, useEffect, useCallback, useRef } from "react";
import axios from "axios";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend
} from "recharts";
import {
  Activity, AlertTriangle, CheckCircle, Play, Zap, RotateCcw, RefreshCw, Wifi, WifiOff, X
} from "lucide-react";

const RAW_API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const API = RAW_API.trim().replace(/\/+$/, "");

const api = axios.create({
  baseURL: API,
  timeout: 45000,
});

function formatApiError(err) {
  if (err?.code === "ECONNABORTED") {
    return `Request to ${API} timed out. If the backend is on Render's free tier, it may still be waking up—please retry in a few seconds.`;
  }
  if (err?.response) {
    const detail = err.response.data?.detail;
    if (err.response.status === 404) {
      return `Backend returned 404 (${detail || "Not Found"}) at ${API}. Verify VITE_API_URL points to your deployed FastAPI service.`;
    }
    return `Backend error (${err.response.status}): ${detail || err.message}`;
  }
  const isHostedNonLocal =
    typeof window !== "undefined" &&
    window.location.hostname !== "localhost" &&
    window.location.hostname !== "127.0.0.1" &&
    API.includes("localhost");

  if (isHostedNonLocal) {
    return "VITE_API_URL is not configured for production (currently pointing to http://localhost:8000). Set VITE_API_URL to your Render backend URL in Vercel Environment Variables and redeploy.";
  }
  return `Unable to reach backend at ${API}. If hosted on Render free tier, the service may be waking up (30–50s).`;
}

function buildSimulatedPayloads(isDrifted, count = 30) {
  return Array.from({ length: count }).map(() =>
    isDrifted
      ? {
        amount: +(320 + Math.random() * 250).toFixed(2),
        distance_from_home: +(45 + Math.random() * 40).toFixed(2),
        hour_of_day: Math.floor(Math.random() * 5),
        tx_frequency_24h: Math.floor(6 + Math.random() * 6),
      }
      : {
        amount: +(20 + Math.random() * 110).toFixed(2),
        distance_from_home: +(5 + Math.random() * 20).toFixed(2),
        hour_of_day: Math.floor(9 + Math.random() * 12),
        tx_frequency_24h: Math.floor(1 + Math.random() * 3),
      }
  );
}

export default function App() {
  const [report, setReport] = useState(null);
  const [selectedFeature, setSelectedFeature] = useState("amount");
  const [simulating, setSimulating] = useState(null); // "normal" | "drift" | null
  const [retrainingReq, setRetrainingReq] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [connectionState, setConnectionState] = useState("connecting"); // "connecting" | "connected" | "error"
  const [errorMessage, setErrorMessage] = useState(null);
  const isFetchingRef = useRef(false);

  const fetchReport = useCallback(async () => {
    if (isFetchingRef.current) return;
    isFetchingRef.current = true;
    try {
      const res = await api.get("/drift-report?window=100");
      setReport(res.data);
      setConnectionState("connected");
      setErrorMessage(null);
    } catch (err) {
      console.error("API error:", err);
      setConnectionState("error");
      setErrorMessage(formatApiError(err));
    } finally {
      isFetchingRef.current = false;
    }
  }, []);

  useEffect(() => {
    let active = true;

    const poll = async () => {
      if (!active) return;
      await fetchReport();
    };

    const initialTimer = setTimeout(poll, 0);
    const interval = setInterval(poll, 3000);

    return () => {
      active = false;
      clearTimeout(initialTimer);
      clearInterval(interval);
    };
  }, [fetchReport]);

  const simulateBatch = async (isDrifted) => {
    setSimulating(isDrifted ? "drift" : "normal");
    setErrorMessage(null);
    const payloads = buildSimulatedPayloads(isDrifted, 30);

    try {
      try {
        await api.post("/predict-batch", payloads);
      } catch (batchErr) {
        if (batchErr?.response?.status === 404) {
          // Fallback for backends running before /predict-batch is deployed
          for (let i = 0; i < payloads.length; i += 5) {
            const chunk = payloads.slice(i, i + 5);
            await Promise.all(chunk.map((payload) => api.post("/predict", payload)));
          }
        } else {
          throw batchErr;
        }
      }
      await fetchReport();
    } catch (err) {
      console.error("Simulation error:", err);
      setConnectionState("error");
      setErrorMessage(formatApiError(err));
    } finally {
      setSimulating(null);
    }
  };

  const triggerRetrain = async () => {
    setRetrainingReq(true);
    setErrorMessage(null);
    try {
      await api.post("/retrain");
      await fetchReport();
    } catch (err) {
      console.error("Retrain error:", err);
      setConnectionState("error");
      setErrorMessage(formatApiError(err));
    } finally {
      setRetrainingReq(false);
    }
  };

  const resetLogs = async () => {
    setResetting(true);
    setErrorMessage(null);
    try {
      await api.post("/reset");
      await fetchReport();
    } catch (err) {
      console.error("Reset error:", err);
      setConnectionState("error");
      setErrorMessage(formatApiError(err));
    } finally {
      setResetting(false);
    }
  };

  const statusBadge = {
    HEALTHY: { color: "bg-emerald-500/20 text-emerald-400 border-emerald-500/40", icon: CheckCircle },
    WARNING: { color: "bg-amber-500/20 text-amber-400 border-amber-500/40", icon: AlertTriangle },
    CRITICAL_DRIFT: { color: "bg-rose-500/20 text-rose-400 border-rose-500/40", icon: AlertTriangle },
    INSUFFICIENT_DATA: { color: "bg-slate-800/80 text-slate-300 border-slate-700", icon: Activity },
  };

  const currentStatus = report?.status || "INSUFFICIENT_DATA";
  const StatusIcon = statusBadge[currentStatus]?.icon || Activity;
  const activeFeatureData = report?.features?.[selectedFeature];
  const isBusy = Boolean(simulating) || retrainingReq || resetting;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 px-4 py-5 sm:px-6 sm:py-6 lg:p-8 font-sans">
      <div className="max-w-6xl mx-auto space-y-5 sm:space-y-6">
        <header className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 border-b border-slate-800 pb-5 sm:pb-6">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="text-xl sm:text-2xl font-bold tracking-tight flex items-center gap-2">
                <Activity className="text-indigo-400 shrink-0" size={22} />
                <span>DriftGuard MLOps Monitor</span>
              </h1>
              <span className="text-xs font-mono bg-indigo-500/20 text-indigo-300 px-2 py-0.5 rounded border border-indigo-500/30">
                {report?.model_version || "v1.0"}
              </span>
              <span
                className={`inline-flex items-center gap-1.5 text-xs px-2 py-0.5 rounded border ${connectionState === "connected"
                    ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                    : connectionState === "connecting"
                      ? "bg-amber-500/10 text-amber-300 border-amber-500/30"
                      : "bg-rose-500/10 text-rose-400 border-rose-500/30"
                  }`}
              >
                {connectionState === "error" ? <WifiOff size={12} /> : <Wifi size={12} />}
                {connectionState === "connected"
                  ? "API Connected"
                  : connectionState === "connecting"
                    ? "Connecting..."
                    : "API Offline"}
              </span>
            </div>
            <p className="text-slate-400 text-xs sm:text-sm mt-1.5">
              Real-time Kolmogorov-Smirnov statistical drift detection & zero-downtime model hot-swapping
            </p>
          </div>

          <div className="grid grid-cols-2 sm:flex sm:flex-wrap gap-2.5 sm:gap-3 w-full lg:w-auto shrink-0">
            <button
              type="button"
              onClick={() => simulateBatch(false)}
              disabled={isBusy}
              className="flex items-center justify-center gap-2 px-3 sm:px-4 py-2.5 sm:py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 active:scale-[0.98] text-xs sm:text-sm font-medium transition disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
            >
              <Play size={15} className={simulating === "normal" ? "animate-pulse" : ""} />
              {simulating === "normal" ? "Sending..." : "Send Normal (30)"}
            </button>
            <button
              type="button"
              onClick={() => simulateBatch(true)}
              disabled={isBusy}
              className="flex items-center justify-center gap-2 px-3 sm:px-4 py-2.5 sm:py-2 rounded-lg bg-rose-600 hover:bg-rose-500 active:scale-[0.98] text-xs sm:text-sm font-medium transition disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
            >
              <Zap size={15} className={simulating === "drift" ? "animate-pulse" : ""} />
              {simulating === "drift" ? "Injecting..." : "Inject Drift (30)"}
            </button>
            <button
              type="button"
              onClick={triggerRetrain}
              disabled={isBusy || report?.is_retraining || (report?.sample_count || 0) < 20}
              className="flex items-center justify-center gap-2 px-3 sm:px-4 py-2.5 sm:py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 active:scale-[0.98] text-xs sm:text-sm font-medium transition disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
            >
              <RefreshCw
                size={15}
                className={report?.is_retraining || retrainingReq ? "animate-spin" : ""}
              />
              {report?.is_retraining || retrainingReq ? "Retraining..." : "Retrain & Swap"}
            </button>
            <button
              type="button"
              onClick={resetLogs}
              disabled={isBusy}
              className="flex items-center justify-center gap-2 px-3 sm:px-4 py-2.5 sm:py-2 rounded-lg bg-slate-800 hover:bg-slate-700 active:scale-[0.98] text-xs sm:text-sm font-medium transition disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
            >
              <RotateCcw size={15} className={resetting ? "animate-spin" : ""} />
              {resetting ? "Resetting..." : "Reset"}
            </button>
          </div>
        </header>

        {errorMessage && (
          <div
            role="alert"
            className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-xl bg-rose-950/50 border border-rose-500/40 text-rose-200 text-xs sm:text-sm"
          >
            <div className="flex items-start gap-2.5 min-w-0">
              <AlertTriangle size={18} className="text-rose-400 shrink-0 mt-0.5" />
              <p className="break-words">{errorMessage}</p>
            </div>
            <div className="flex items-center gap-2 shrink-0 self-end sm:self-center">
              <button
                type="button"
                onClick={fetchReport}
                className="px-3 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 border border-rose-500/40 text-xs font-medium transition"
              >
                Retry
              </button>
              <button
                type="button"
                onClick={() => setErrorMessage(null)}
                aria-label="Dismiss error"
                className="p-1.5 rounded-lg hover:bg-rose-500/20 text-rose-300 transition"
              >
                <X size={16} />
              </button>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
          <div className={`p-4 rounded-xl border min-w-0 ${statusBadge[currentStatus].color}`}>
            <div className="text-xs uppercase opacity-75">System Health</div>
            <div className="text-lg sm:text-xl font-bold mt-1 flex items-center gap-2">
              <StatusIcon size={20} className="shrink-0" />
              <span className="truncate">{currentStatus.replace("_", " ")}</span>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 min-w-0">
            <div className="text-xs text-slate-400 uppercase">Window Sample Size</div>
            <div className="text-xl sm:text-2xl font-bold mt-1">{report?.sample_count || 0} / 100</div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 min-w-0">
            <div className="text-xs text-slate-400 uppercase">Drifted Features</div>
            <div className="text-xl sm:text-2xl font-bold mt-1">{report?.drifted_features ?? 0} / 4</div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 min-w-0">
            <div className="text-xs text-slate-400 uppercase">Live Fraud Rate</div>
            <div className="text-xl sm:text-2xl font-bold mt-1 text-indigo-400">
              {report?.predicted_fraud_rate ?? 0}%
            </div>
          </div>
        </div>

        {report?.status !== "INSUFFICIENT_DATA" && report?.features ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 sm:p-5 space-y-3 min-w-0">
              <h2 className="font-semibold text-slate-200 mb-2">Feature Drift Metrics</h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-1 gap-3">
                {Object.entries(report.features).map(([name, stats]) => (
                  <button
                    key={name}
                    type="button"
                    onClick={() => setSelectedFeature(name)}
                    className={`w-full text-left p-3.5 rounded-lg border transition ${selectedFeature === name
                        ? "bg-indigo-950/50 border-indigo-500"
                        : "bg-slate-950/60 border-slate-800 hover:border-slate-700"
                      }`}
                  >
                    <div className="flex justify-between items-center gap-2">
                      <span className="font-mono text-sm font-medium truncate">{name}</span>
                      <span
                        className={`text-xs px-2 py-0.5 rounded font-semibold shrink-0 ${stats.drifted
                            ? "bg-rose-500/20 text-rose-400"
                            : "bg-emerald-500/20 text-emerald-400"
                          }`}
                      >
                        {stats.drifted ? "DRIFTED" : "OK"}
                      </span>
                    </div>
                    <div className="text-xs text-slate-400 mt-2 flex flex-wrap justify-between gap-x-3 gap-y-1">
                      <span>KS Stat: {stats.ks_statistic}</span>
                      <span>p-val: {stats.p_value}</span>
                    </div>
                    <div className="text-xs text-slate-500 mt-1">
                      Mean: {stats.ref_mean} (Ref) → {stats.curr_mean} (Live)
                    </div>
                  </button>
                ))}
              </div>
            </div>

            <div className="lg:col-span-2 min-w-0 bg-slate-900 border border-slate-800 rounded-xl p-4 sm:p-6">
              <div className="mb-4">
                <h2 className="font-semibold text-base sm:text-lg">
                  Distribution Shift:{" "}
                  <span className="font-mono text-indigo-400 break-all">{selectedFeature}</span>
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Comparing training baseline density vs. live production sliding window
                </p>
              </div>

              <div className="h-60 sm:h-72 lg:h-80 w-full min-w-0">
                <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                  <AreaChart
                    data={activeFeatureData?.histogram || []}
                    margin={{ top: 8, right: 8, left: -12, bottom: 0 }}
                  >
                    <XAxis dataKey="bin" stroke="#64748b" fontSize={11} minTickGap={16} />
                    <YAxis stroke="#64748b" fontSize={11} width={44} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#0f172a",
                        borderColor: "#1e293b",
                        borderRadius: "0.5rem",
                        fontSize: "12px",
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: "12px" }} />
                    <Area
                      type="monotone"
                      dataKey="baseline"
                      name="Training Baseline"
                      stroke="#10b981"
                      fill="#10b981"
                      fillOpacity={0.2}
                    />
                    <Area
                      type="monotone"
                      dataKey="production"
                      name="Production Traffic"
                      stroke="#f43f5e"
                      fill="#f43f5e"
                      fillOpacity={0.25}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 sm:p-12 text-center text-slate-400 text-sm sm:text-base">
            {connectionState === "connecting" ? (
              <span>Connecting to DriftGuard backend...</span>
            ) : (
              <span>
                Click <strong>&quot;Send Normal (30)&quot;</strong> above to populate the sliding window
                and compute statistical drift metrics.
                {(report?.sample_count || 0) > 0 && (
                  <span className="block text-xs text-slate-500 mt-2">
                    ({report.sample_count} / 15 minimum samples currently logged)
                  </span>
                )}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
