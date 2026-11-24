import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend
} from "recharts";
import { Activity, AlertTriangle, CheckCircle, Play, Zap, RotateCcw, RefreshCw } from "lucide-react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [report, setReport] = useState(null);
  const [selectedFeature, setSelectedFeature] = useState("amount");
  const [simulating, setSimulating] = useState(false);

  const fetchReport = async () => {
    try {
      const res = await axios.get(`${API}/drift-report?window=100`);
      setReport(res.data);
    } catch (err) {
      console.error("API error:", err);
    }
  };

  useEffect(() => {
    fetchReport();
    const interval = setInterval(fetchReport, 3000);
    return () => clearInterval(interval);
  }, []);

  const simulateBatch = async (isDrifted) => {
    setSimulating(true);
    const requests = Array.from({ length: 30 }).map(() => {
      const payload = isDrifted
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
          };
      return axios.post(`${API}/predict`, payload);
    });

    await Promise.all(requests);
    await fetchReport();
    setSimulating(false);
  };

  const triggerRetrain = async () => {
    await axios.post(`${API}/retrain`);
    setTimeout(fetchReport, 1000);
  };

  const resetLogs = async () => {
    await axios.post(`${API}/reset`);
    fetchReport();
  };

  const statusBadge = {
    HEALTHY: { color: "bg-emerald-500/20 text-emerald-400 border-emerald-500/40", icon: CheckCircle },
    WARNING: { color: "bg-amber-500/20 text-amber-400 border-amber-500/40", icon: AlertTriangle },
    CRITICAL_DRIFT: { color: "bg-rose-500/20 text-rose-400 border-rose-500/40", icon: AlertTriangle },
    INSUFFICIENT_DATA: { color: "bg-slate-700 text-slate-300 border-slate-600", icon: Activity },
  };

  const currentStatus = report?.status || "INSUFFICIENT_DATA";
  const StatusIcon = statusBadge[currentStatus]?.icon || Activity;
  const activeFeatureData = report?.features?.[selectedFeature];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8 font-sans">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex flex-wrap justify-between items-center gap-4 border-b border-slate-800 pb-6">
          <div>
            <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
              <Activity className="text-indigo-400" /> DriftGuard MLOps Monitor
              <span className="text-xs font-mono bg-indigo-500/20 text-indigo-300 px-2 py-0.5 rounded border border-indigo-500/30">
                {report?.model_version || "v1.0"}
              </span>
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Real-time Kolmogorov-Smirnov statistical drift detection & zero-downtime model hot-swapping
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              onClick={() => simulateBatch(false)}
              disabled={simulating}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-sm font-medium transition disabled:opacity-50"
            >
              <Play size={16} /> Send Normal (30)
            </button>
            <button
              onClick={() => simulateBatch(true)}
              disabled={simulating}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-sm font-medium transition disabled:opacity-50"
            >
              <Zap size={16} /> Inject Drift (30)
            </button>
            <button
              onClick={triggerRetrain}
              disabled={report?.is_retraining || (report?.sample_count || 0) < 20}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-sm font-medium transition disabled:opacity-50"
            >
              <RefreshCw size={16} className={report?.is_retraining ? "animate-spin" : ""} />
              {report?.is_retraining ? "Retraining..." : "Retrain & Swap"}
            </button>
            <button
              onClick={resetLogs}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-sm font-medium transition"
            >
              <RotateCcw size={16} /> Reset
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className={`p-4 rounded-xl border ${statusBadge[currentStatus].color}`}>
            <div className="text-xs uppercase opacity-75">System Health</div>
            <div className="text-xl font-bold mt-1 flex items-center gap-2">
              <StatusIcon size={20} /> {currentStatus.replace("_", " ")}
            </div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
            <div className="text-xs text-slate-400 uppercase">Window Sample Size</div>
            <div className="text-2xl font-bold mt-1">{report?.sample_count || 0} / 100</div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
            <div className="text-xs text-slate-400 uppercase">Drifted Features</div>
            <div className="text-2xl font-bold mt-1">{report?.drifted_features ?? 0} / 4</div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
            <div className="text-xs text-slate-400 uppercase">Live Fraud Rate</div>
            <div className="text-2xl font-bold mt-1 text-indigo-400">
              {report?.predicted_fraud_rate ?? 0}%
            </div>
          </div>
        </div>

        {report?.status !== "INSUFFICIENT_DATA" && report?.features ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
              <h2 className="font-semibold text-slate-200 mb-3">Feature Drift Metrics</h2>
              {Object.entries(report.features).map(([name, stats]) => (
                <div
                  key={name}
                  onClick={() => setSelectedFeature(name)}
                  className={`p-3.5 rounded-lg border cursor-pointer transition ${
                    selectedFeature === name
                      ? "bg-indigo-950/50 border-indigo-500"
                      : "bg-slate-950/60 border-slate-800 hover:border-slate-700"
                  }`}
                >
                  <div className="flex justify-between items-center">
                    <span className="font-mono text-sm font-medium">{name}</span>
                    <span
                      className={`text-xs px-2 py-0.5 rounded font-semibold ${
                        stats.drifted
                          ? "bg-rose-500/20 text-rose-400"
                          : "bg-emerald-500/20 text-emerald-400"
                      }`}
                    >
                      {stats.drifted ? "DRIFTED" : "OK"}
                    </span>
                  </div>
                  <div className="text-xs text-slate-400 mt-2 flex justify-between">
                    <span>KS Stat: {stats.ks_statistic}</span>
                    <span>p-val: {stats.p_value}</span>
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    Mean: {stats.ref_mean} (Ref) → {stats.curr_mean} (Live)
                  </div>
                </div>
              ))}
            </div>

            <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-6">
              <div className="mb-4">
                <h2 className="font-semibold text-lg">
                  Distribution Shift: <span className="font-mono text-indigo-400">{selectedFeature}</span>
                </h2>
                <p className="text-xs text-slate-400">
                  Comparing training baseline density vs. live production sliding window
                </p>
              </div>

              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={activeFeatureData?.histogram || []}>
                    <XAxis dataKey="bin" stroke="#64748b" fontSize={12} />
                    <YAxis stroke="#64748b" fontSize={12} />
                    <Tooltip
                      contentStyle={{ backgroundColor: "#0f172a", borderColor: "#1e293b" }}
                    />
                    <Legend />
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
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center text-slate-400">
            Click <strong>"Send Normal (30)"</strong> above to populate the sliding window and compute statistical drift metrics.
          </div>
        )}
      </div>
    </div>
  );
}