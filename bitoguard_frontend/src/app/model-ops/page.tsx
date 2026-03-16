"use client"

import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"
import { AlertTriangle, CheckCircle, XCircle, Info } from "lucide-react"
import { FEATURE_ZH } from "@/lib/labels"

interface DriftFeature {
  feature: string
  zero_rate_delta: number
  mean_rel_change: number | null
  std_rel_change: number | null
}

interface DriftResult {
  snapshot_from: string
  snapshot_to: string
  drifted_features: DriftFeature[]
  total_checked: number
  total_drifted: number
  health_ok: boolean
}

interface ThresholdRow {
  threshold: number
  precision: number
  recall: number
  f1: number
}

interface ScenarioRow {
  scenario: string
  count: number
  precision: number
  recall: number
}

interface CalibrationBin {
  mean_predicted: number
  fraction_positive: number
}

interface FeatureImportanceRow {
  feature: string
  importance_gain: number
  importance_pct: number
}

interface OofMetrics {
  auc: number
  pr_auc: number
}

interface OraclePrecisionAtK {
  hits: number
  k: number
  precision: number
  lift: number
}

interface ModelMetrics {
  model_version: string
  holdout_rows: number
  holdout_positives: number
  holdout_negatives: number
  precision: number
  recall: number
  f1: number
  fpr: number
  average_precision: number
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number }
  precision_at_k: Record<string, number>
  recall_at_k: Record<string, number>
  calibration: { brier_score: number; n_bins: number; bins: CalibrationBin[] }
  feature_importance_top20: FeatureImportanceRow[]
  threshold_sensitivity: ThresholdRow[]
  scenario_breakdown: ScenarioRow[]
  pr_curve?: { precision: number[]; recall: number[]; thresholds: number[] }
  // OOF stacker fields (true, leakage-free generalization metrics)
  oof_metrics?: { catboost: OofMetrics; lgbm: OofMetrics; xgboost?: OofMetrics; extratrees?: OofMetrics; randomforest?: OofMetrics; stacker: OofMetrics }
  oracle_precision_at_k?: Record<string, OraclePrecisionAtK>
  dataset_stats?: { users: number; positives: number; positive_rate: number; features: number }
}

// ── SVG PR-Curve component ──────────────────────────────────────────────────
function PrCurveChart({ precision, recall }: { precision: number[]; recall: number[] }) {
  const W = 360
  const H = 200
  const PAD = { top: 12, right: 16, bottom: 36, left: 40 }
  const iW = W - PAD.left - PAD.right
  const iH = H - PAD.top - PAD.bottom

  // Build polyline points (recall on x, precision on y)
  const pts = recall.map((r, i) => ({
    x: PAD.left + r * iW,
    y: PAD.top + (1 - precision[i]) * iH,
  }))

  const pathD = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ")
  // Filled area under curve
  const areaD = `${pathD} L${PAD.left + iW},${PAD.top + iH} L${PAD.left},${PAD.top + iH} Z`

  const xTicks = [0, 0.25, 0.5, 0.75, 1.0]
  const yTicks = [0, 0.25, 0.5, 0.75, 1.0]

  return (
    <svg width={W} height={H} style={{ overflow: "visible" }}>
      {/* Grid lines */}
      {yTicks.map((t) => (
        <line
          key={`gy-${t}`}
          x1={PAD.left} y1={PAD.top + (1 - t) * iH}
          x2={PAD.left + iW} y2={PAD.top + (1 - t) * iH}
          stroke="#f3f4f6" strokeWidth={1}
        />
      ))}
      {xTicks.map((t) => (
        <line
          key={`gx-${t}`}
          x1={PAD.left + t * iW} y1={PAD.top}
          x2={PAD.left + t * iW} y2={PAD.top + iH}
          stroke="#f3f4f6" strokeWidth={1}
        />
      ))}
      {/* Filled area */}
      <path d={areaD} fill="#5c6bc0" fillOpacity={0.08} />
      {/* Curve */}
      <path d={pathD} fill="none" stroke="#5c6bc0" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
      {/* Random baseline */}
      <line
        x1={PAD.left} y1={PAD.top + iH}
        x2={PAD.left + iW} y2={PAD.top}
        stroke="#e5e7eb" strokeWidth={1} strokeDasharray="4 4"
      />
      {/* Axes */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + iH} stroke="#d1d5db" strokeWidth={1} />
      <line x1={PAD.left} y1={PAD.top + iH} x2={PAD.left + iW} y2={PAD.top + iH} stroke="#d1d5db" strokeWidth={1} />
      {/* Y ticks & labels */}
      {yTicks.map((t) => (
        <text key={`yl-${t}`} x={PAD.left - 6} y={PAD.top + (1 - t) * iH + 3.5}
          textAnchor="end" fontSize={9} fill="#9ca3af">{t.toFixed(2)}</text>
      ))}
      {/* X ticks & labels */}
      {xTicks.map((t) => (
        <text key={`xl-${t}`} x={PAD.left + t * iW} y={PAD.top + iH + 14}
          textAnchor="middle" fontSize={9} fill="#9ca3af">{t.toFixed(2)}</text>
      ))}
      {/* Axis labels */}
      <text x={PAD.left + iW / 2} y={H - 2} textAnchor="middle" fontSize={10} fill="#6b7280">Recall</text>
      <text x={8} y={PAD.top + iH / 2} textAnchor="middle" fontSize={10} fill="#6b7280"
        transform={`rotate(-90,8,${PAD.top + iH / 2})`}>Precision</text>
    </svg>
  )
}

// ── Metric card ────────────────────────────────────────────────────────────
function MetricCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-white rounded-xl border border-[#e5e7eb] px-4 py-3 shadow-sm">
      <p className="text-[11px] text-[#9ca3af] font-semibold uppercase tracking-wider">{label}</p>
      <p className="text-[18px] font-semibold text-[#1a1d2e] mt-0.5 leading-tight">{value}</p>
      {sub && <p className="text-[11px] text-[#9ca3af] mt-0.5">{sub}</p>}
    </div>
  )
}

// ── Horizontal bar ─────────────────────────────────────────────────────────
function HBar({ pct, color, label, value }: { pct: number; color: string; label: string; value: string }) {
  return (
    <div className="space-y-0.5">
      <div className="flex justify-between text-[11px]">
        <span className="text-[#6b7280] truncate max-w-[200px]" title={label}>{label}</span>
        <span className="font-mono font-semibold text-[#1a1d2e]">{value}</span>
      </div>
      <div className="h-2 bg-[#f3f4f6] rounded-full overflow-hidden">
        <div className="h-2 rounded-full transition-all" style={{ width: `${Math.min(100, pct)}%`, backgroundColor: color }} />
      </div>
    </div>
  )
}

export default function ModelOpsPage() {
  const { data: metrics, isLoading, error } = useQuery({
    queryKey: ["modelMetrics"],
    queryFn: () => api.getModelMetrics() as Promise<ModelMetrics>,
    staleTime: 300_000,
  })
  const { data: drift } = useQuery({
    queryKey: ["driftMetrics"],
    queryFn: () => api.getDriftMetrics() as Promise<DriftResult>,
    refetchInterval: 300_000,
    refetchIntervalInBackground: false,
  })

  // Detect suspiciously perfect metrics (leakage signal)
  const leakageSuspected = metrics != null &&
    metrics.average_precision > 0.95 &&
    metrics.confusion_matrix.fn === 0 &&
    metrics.confusion_matrix.fp < 10

  // Non-zero feature importance only
  const nonZeroFeatures = metrics?.feature_importance_top20.filter(f => f.importance_pct > 0) ?? []
  const topFeatureMax = nonZeroFeatures[0]?.importance_pct ?? 100

  return (
    <div className="max-w-[1000px] mx-auto space-y-4">
      <div>
        <h1 className="text-[22px] font-semibold text-[#1a1d2e]">模型指標</h1>
        <p className="text-[13px] text-[#9ca3af] mt-0.5">風險模型驗證結果、PR 曲線與特徵漂移健康狀態</p>
      </div>

      {isLoading && (
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-24 bg-[#f3f4f6] rounded-xl animate-pulse" />
          ))}
        </div>
      )}
      {error && (
        <div className="flex items-center gap-2 text-[#e53935] bg-[#fff5f5] border border-[#ef9a9a] rounded-xl px-4 py-3">
          <XCircle size={14} />
          <span className="text-[13px]">無法載入模型指標，請確認後端服務是否正常運行</span>
        </div>
      )}

      {metrics && (
        <div className="space-y-4">
          {/* ── Leakage warning ── */}
          {leakageSuspected && (
            <div className="flex items-start gap-3 bg-[#fff8e1] border border-[#ffe082] rounded-xl px-4 py-3">
              <AlertTriangle size={15} className="text-[#f59e0b] mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-[13px] font-semibold text-[#b45309]">疑似特徵洩漏 — Holdout 指標偏高</p>
                <p className="text-[11px] text-[#92400e] mt-0.5">
                  Holdout PR-AUC 達 {metrics.average_precision.toFixed(4)} 且 FN=0，可能因 peer-percentile 特徵在整體資料集計算時洩漏至驗證集。
                  請參考 OOF Stacker 的真實泛化指標{metrics.oof_metrics ? ` (PR-AUC = ${metrics.oof_metrics.stacker.pr_auc.toFixed(4)})` : ""}。
                </p>
              </div>
            </div>
          )}

          {/* ── OOF Stacker metrics (true generalization, leakage-free) ── */}
          {metrics.oof_metrics && (
            <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-[#e5e7eb] flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-[#5c6bc0]" />
                <h2 className="text-[14px] font-semibold text-[#1a1d2e]">OOF 堆疊模型 — 真實泛化指標</h2>
                <span className="ml-auto text-[11px] text-[#9ca3af]">5-fold StratifiedGroupKFold, leakage-free</span>
              </div>
              <div className={`p-4 grid gap-4 ${
                [metrics.oof_metrics?.xgboost, metrics.oof_metrics?.extratrees, metrics.oof_metrics?.randomforest].filter(Boolean).length === 3
                  ? "grid-cols-6"
                  : [metrics.oof_metrics?.xgboost, metrics.oof_metrics?.extratrees, metrics.oof_metrics?.randomforest].filter(Boolean).length === 2
                    ? "grid-cols-5"
                    : [metrics.oof_metrics?.xgboost, metrics.oof_metrics?.extratrees, metrics.oof_metrics?.randomforest].filter(Boolean).length === 1
                      ? "grid-cols-4"
                      : "grid-cols-3"
              }`}>
                {(["catboost", "lgbm",
                   ...(metrics.oof_metrics?.xgboost      ? ["xgboost"]      : []),
                   ...(metrics.oof_metrics?.extratrees   ? ["extratrees"]   : []),
                   ...(metrics.oof_metrics?.randomforest ? ["randomforest"] : []),
                   "stacker"] as const).map((branch) => {
                  const m = metrics.oof_metrics![branch as keyof typeof metrics.oof_metrics]
                  if (!m) return null
                  const isStacker = branch === "stacker"
                  const BRANCH_LABEL: Record<string, string> = {
                    catboost: "CatBoost", lgbm: "LightGBM",
                    xgboost: "XGBoost", extratrees: "ExtraTrees",
                    randomforest: "RandomForest", stacker: "Stacker (meta)",
                  }
                  return (
                    <div key={branch} className={`rounded-lg p-3 ${isStacker ? "bg-[#eef0fa]" : "bg-[#f9fafb]"}`}>
                      <p className="text-[11px] font-semibold text-[#6b7280] uppercase tracking-wider mb-2">
                        {BRANCH_LABEL[branch] ?? branch}
                      </p>
                      <div className="space-y-1.5">
                        <div className="flex justify-between items-baseline">
                          <span className="text-[11px] text-[#9ca3af]">ROC-AUC</span>
                          <span className={`text-[15px] font-semibold ${isStacker ? "text-[#5c6bc0]" : "text-[#1a1d2e]"}`}>
                            {m.auc.toFixed(4)}
                          </span>
                        </div>
                        <div className="h-1.5 bg-[#e5e7eb] rounded-full overflow-hidden">
                          <div className="h-1.5 rounded-full" style={{ width: `${m.auc * 100}%`, backgroundColor: isStacker ? "#5c6bc0" : "#9ca3af" }} />
                        </div>
                        <div className="flex justify-between items-baseline mt-1.5">
                          <span className="text-[11px] text-[#9ca3af]">PR-AUC</span>
                          <span className={`text-[15px] font-semibold ${isStacker ? "text-[#5c6bc0]" : "text-[#1a1d2e]"}`}>
                            {m.pr_auc.toFixed(4)}
                          </span>
                        </div>
                        <div className="h-1.5 bg-[#e5e7eb] rounded-full overflow-hidden">
                          <div className="h-1.5 rounded-full" style={{ width: `${Math.min(100, m.pr_auc / 0.5 * 100)}%`, backgroundColor: isStacker ? "#5c6bc0" : "#9ca3af" }} />
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
              {metrics.dataset_stats && (
                <div className="px-4 pb-3 flex gap-6 text-[12px] text-[#9ca3af]">
                  <span>{metrics.dataset_stats.users?.toLocaleString()} 使用者</span>
                  <span>{metrics.dataset_stats.positives?.toLocaleString()} 正樣本 ({((metrics.dataset_stats.positive_rate ?? 0) * 100).toFixed(1)}%)</span>
                  <span>{metrics.dataset_stats.features} 特徵</span>
                </div>
              )}
            </div>
          )}

          {/* ── Oracle P@K ── */}
          {metrics.oracle_precision_at_k && Object.keys(metrics.oracle_precision_at_k).length > 0 && (
            <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-[#e5e7eb]">
                <h2 className="text-[14px] font-semibold text-[#1a1d2e]">Oracle Precision@K (線上評估)</h2>
                <p className="text-[11px] text-[#9ca3af] mt-0.5">依模型分數排序後，前 K 名使用者中確認為可疑的比例</p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-[13px]">
                  <thead>
                    <tr className="bg-[#f9fafb] border-b border-[#e5e7eb]">
                      {["K", "命中", "Precision", "Lift vs Random", "精準度"].map((h) => (
                        <th key={h} className="px-4 py-2 text-left text-[11px] font-semibold text-[#9ca3af] uppercase tracking-wider">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(metrics.oracle_precision_at_k).map(([k, v]) => (
                      <tr key={k} className="border-b border-[#f3f4f6] hover:bg-[#f9fafb]">
                        <td className="px-4 py-2 font-mono font-semibold text-[#5c6bc0]">{v.k}</td>
                        <td className="px-4 py-2 font-mono text-[#43a047] font-semibold">{v.hits}/{v.k}</td>
                        <td className="px-4 py-2 font-mono">{(v.precision * 100).toFixed(1)}%</td>
                        <td className="px-4 py-2">
                          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold bg-[#eef0fa] text-[#5c6bc0]">
                            {v.lift.toFixed(1)}×
                          </span>
                        </td>
                        <td className="px-4 py-2 w-40">
                          <div className="h-1.5 bg-[#f3f4f6] rounded-full overflow-hidden">
                            <div className="h-1.5 rounded-full" style={{ width: `${v.precision * 100}%`, backgroundColor: v.precision >= 0.9 ? "#43a047" : v.precision >= 0.7 ? "#fb8c00" : "#e53935" }} />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ── Core KPI cards ── */}
          <div className="grid grid-cols-3 gap-3 md:grid-cols-6">
            {[
              { label: "PR-AUC", value: metrics.average_precision.toFixed(4) },
              { label: "Precision", value: metrics.precision.toFixed(4) },
              { label: "Recall",   value: metrics.recall.toFixed(4) },
              { label: "F1 Score", value: metrics.f1.toFixed(4) },
              { label: "FPR",      value: metrics.fpr.toFixed(5) },
              { label: "Brier",    value: metrics.calibration?.brier_score.toFixed(5) ?? "—" },
            ].map(({ label, value }) => (
              <MetricCard key={label} label={label} value={value} />
            ))}
          </div>

          {/* ── Holdout summary + Confusion matrix ── */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm p-4">
              <h2 className="text-[14px] font-semibold text-[#1a1d2e] mb-3">Holdout 集統計</h2>
              <div className="space-y-2 text-[13px]">
                {[
                  { label: "總樣本", value: (metrics.holdout_rows ?? 0).toLocaleString() },
                  { label: "正樣本 (可疑)", value: (metrics.holdout_positives ?? 0).toLocaleString(), color: "#e53935" },
                  { label: "負樣本 (正常)", value: (metrics.holdout_negatives ?? 0).toLocaleString(), color: "#43a047" },
                  { label: "模型版本", value: metrics.model_version, mono: true },
                ].map(({ label, value, color, mono }) => (
                  <div key={label} className="flex justify-between items-center py-1 border-b border-[#f3f4f6]">
                    <span className="text-[#6b7280]">{label}</span>
                    <span className={`font-semibold ${mono ? "font-mono text-[11px]" : ""}`} style={color ? { color } : {}}>
                      {value}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm p-4">
              <h2 className="text-[14px] font-semibold text-[#1a1d2e] mb-3">混淆矩陣</h2>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { label: "True Negative",  short: "TN", value: metrics.confusion_matrix.tn, color: "#43a047", bg: "bg-green-50" },
                  { label: "False Positive", short: "FP", value: metrics.confusion_matrix.fp, color: "#fb8c00", bg: "bg-orange-50" },
                  { label: "False Negative", short: "FN", value: metrics.confusion_matrix.fn, color: "#e53935", bg: "bg-red-50" },
                  { label: "True Positive",  short: "TP", value: metrics.confusion_matrix.tp, color: "#5c6bc0", bg: "bg-[#eef0fa]" },
                ].map(({ label, short, value, color, bg }) => (
                  <div key={short} className={`${bg} rounded-lg px-3 py-2.5 text-center`}>
                    <p className="text-[10px] text-[#9ca3af] font-semibold">{short}</p>
                    <p className="text-[22px] font-semibold leading-tight" style={{ color }}>{value.toLocaleString()}</p>
                    <p className="text-[10px] text-[#9ca3af] mt-0.5">{label}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* ── PR Curve + Feature Importance ── */}
          <div className="grid grid-cols-5 gap-4">
            {/* PR Curve */}
            {metrics.pr_curve && metrics.pr_curve.precision.length > 2 && (
              <div className="col-span-2 bg-white rounded-xl border border-[#e5e7eb] shadow-sm p-4">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-[14px] font-semibold text-[#1a1d2e]">PR 曲線</h2>
                  <span className="text-[11px] font-mono font-semibold text-[#5c6bc0]">
                    AUC = {metrics.average_precision.toFixed(4)}
                  </span>
                </div>
                <PrCurveChart
                  precision={metrics.pr_curve.precision}
                  recall={metrics.pr_curve.recall}
                />
              </div>
            )}

            {/* Feature Importance */}
            <div className={`${metrics.pr_curve && metrics.pr_curve.precision.length > 2 ? "col-span-3" : "col-span-5"} bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden`}>
              <div className="px-4 py-3 border-b border-[#e5e7eb] flex items-center justify-between">
                <div>
                  <h2 className="text-[14px] font-semibold text-[#1a1d2e]">特徵重要度</h2>
                  <p className="text-[11px] text-[#9ca3af] mt-0.5">LightGBM gain-based (有效特徵)</p>
                </div>
                {nonZeroFeatures.length < metrics.feature_importance_top20.length && (
                  <div className="flex items-center gap-1 text-[11px] text-[#9ca3af]">
                    <Info size={11} />
                    <span>{metrics.feature_importance_top20.length - nonZeroFeatures.length} 個特徵重要度為零</span>
                  </div>
                )}
              </div>
              <div className="p-4 space-y-2.5">
                {nonZeroFeatures.length === 0 ? (
                  <p className="text-[13px] text-[#9ca3af] text-center py-4">無有效特徵重要度資料</p>
                ) : (
                  nonZeroFeatures.map((row) => (
                    <HBar
                      key={row.feature}
                      label={FEATURE_ZH[row.feature] ?? row.feature}
                      pct={(row.importance_pct / topFeatureMax) * 100}
                      color="#5c6bc0"
                      value={`${row.importance_pct.toFixed(2)}%`}
                    />
                  ))
                )}
              </div>
            </div>
          </div>

          {/* ── P@K / R@K ── */}
          {metrics.precision_at_k && Object.keys(metrics.precision_at_k).length > 0 && (
            <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-[#e5e7eb]">
                <h2 className="text-[14px] font-semibold text-[#1a1d2e]">Precision@K / Recall@K</h2>
                <p className="text-[11px] text-[#9ca3af] mt-0.5">稽核容量固定時，前 K 名使用者的精準度與召回率</p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-[13px]">
                  <thead>
                    <tr className="bg-[#f9fafb] border-b border-[#e5e7eb]">
                      {["K", "Precision@K", "Recall@K", "Precision Bar"].map((h) => (
                        <th key={h} className="px-4 py-2 text-left text-[11px] font-semibold text-[#9ca3af] uppercase tracking-wider">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.keys(metrics.precision_at_k).map((k) => {
                      const p = metrics.precision_at_k[k] ?? 0
                      return (
                        <tr key={k} className="border-b border-[#f3f4f6] hover:bg-[#f9fafb]">
                          <td className="px-4 py-2 font-mono font-semibold text-[#5c6bc0]">{k.replace("P@", "")}</td>
                          <td className="px-4 py-2 font-mono">{p.toFixed(4)}</td>
                          <td className="px-4 py-2 font-mono">{(metrics.recall_at_k?.[k.replace("P@", "R@")] ?? 0).toFixed(4)}</td>
                          <td className="px-4 py-2 w-40">
                            <div className="h-1.5 bg-[#f3f4f6] rounded-full overflow-hidden">
                              <div className="h-1.5 rounded-full bg-[#5c6bc0]" style={{ width: `${p * 100}%` }} />
                            </div>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ── Threshold Sensitivity ── */}
          {metrics.threshold_sensitivity.length > 0 && (
            <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-[#e5e7eb]">
                <h2 className="text-[14px] font-semibold text-[#1a1d2e]">閾值敏感度分析</h2>
                <p className="text-[11px] text-[#9ca3af] mt-0.5">不同決策閾值下的 Precision / Recall / F1 變化</p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-[13px]">
                  <thead>
                    <tr className="bg-[#f9fafb] border-b border-[#e5e7eb]">
                      {["閾值", "Precision", "Recall", "F1", "F1 Bar"].map((h) => (
                        <th key={h} className="px-4 py-2 text-left text-[11px] font-semibold text-[#9ca3af] uppercase tracking-wider">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.threshold_sensitivity.map((row) => (
                      <tr key={row.threshold} className="border-b border-[#f3f4f6] hover:bg-[#f9fafb]">
                        <td className="px-4 py-2 font-mono font-semibold text-[#5c6bc0]">{row.threshold.toFixed(2)}</td>
                        <td className="px-4 py-2 font-mono">{row.precision.toFixed(4)}</td>
                        <td className="px-4 py-2 font-mono">{row.recall.toFixed(4)}</td>
                        <td className="px-4 py-2 font-mono">{row.f1.toFixed(4)}</td>
                        <td className="px-4 py-2 w-32">
                          <div className="h-1.5 bg-[#f3f4f6] rounded-full overflow-hidden">
                            <div className="h-1.5 rounded-full bg-[#43a047]" style={{ width: `${row.f1 * 100}%` }} />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ── Scenario breakdown ── */}
          {metrics.scenario_breakdown.length > 0 && (
            <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-[#e5e7eb]">
                <h2 className="text-[14px] font-semibold text-[#1a1d2e]">情境別表現</h2>
              </div>
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="bg-[#f9fafb] border-b border-[#e5e7eb]">
                    {["情境", "樣本數", "Precision", "Recall"].map((h) => (
                      <th key={h} className="px-4 py-2 text-left text-[11px] font-semibold text-[#9ca3af] uppercase tracking-wider">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {metrics.scenario_breakdown.map((row) => (
                    <tr key={row.scenario} className="border-b border-[#f3f4f6] hover:bg-[#f9fafb]">
                      <td className="px-4 py-2 font-semibold capitalize">{row.scenario}</td>
                      <td className="px-4 py-2 font-mono">{row.count.toLocaleString()}</td>
                      <td className="px-4 py-2 font-mono">{row.precision.toFixed(4)}</td>
                      <td className="px-4 py-2 font-mono">{row.recall.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* ── Feature drift ── */}
      {drift && (
        <div className={`rounded-xl border shadow-sm overflow-hidden ${drift.health_ok ? "border-[#e5e7eb] bg-white" : "border-[#ef9a9a] bg-[#fff5f5]"}`}>
          <div className="px-4 py-3 border-b border-[#e5e7eb] flex items-center justify-between">
            <div>
              <h2 className="text-[14px] font-semibold text-[#1a1d2e]">特徵漂移健康狀態</h2>
              <p className="text-[11px] text-[#9ca3af] mt-0.5">{drift.snapshot_from} → {drift.snapshot_to}</p>
            </div>
            <span className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold ${drift.health_ok ? "bg-[#e8f5e9] text-[#2e7d32]" : "bg-[#ffebee] text-[#c62828]"}`}>
              {drift.health_ok ? <CheckCircle size={11} /> : <XCircle size={11} />}
              {drift.health_ok ? "HEALTHY" : `${drift.total_drifted} DRIFTED`}
            </span>
          </div>
          <div className="p-4 flex gap-6 text-[13px]">
            <div>
              <p className="text-[11px] text-[#9ca3af] uppercase tracking-wider">檢查特徵數</p>
              <p className="text-[20px] font-semibold text-[#1a1d2e]">{drift.total_checked}</p>
            </div>
            <div>
              <p className="text-[11px] text-[#9ca3af] uppercase tracking-wider">漂移特徵數</p>
              <p className={`text-[20px] font-semibold ${drift.total_drifted > 0 ? "text-[#e53935]" : "text-[#43a047]"}`}>
                {drift.total_drifted}
              </p>
            </div>
          </div>
          {drift.drifted_features.length > 0 && (
            <div className="border-t border-[#e5e7eb] overflow-x-auto">
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="bg-[#f9fafb]">
                    {["特徵", "零值率變化", "均值相對變化", "標準差相對變化"].map((h) => (
                      <th key={h} className="px-4 py-2 text-left text-[11px] font-semibold text-[#9ca3af] uppercase tracking-wider">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {drift.drifted_features.map((row) => (
                    <tr key={row.feature} className="border-t border-[#f3f4f6] hover:bg-[#fff8f8]">
                      <td className="px-4 py-2 font-semibold text-[#e53935]" title={row.feature}>
                        {FEATURE_ZH[row.feature] ?? row.feature}
                      </td>
                      <td className="px-4 py-2 font-mono">{(row.zero_rate_delta * 100).toFixed(2)}pp</td>
                      <td className="px-4 py-2 font-mono">
                        {row.mean_rel_change != null ? `${(row.mean_rel_change * 100).toFixed(1)}%` : "—"}
                      </td>
                      <td className="px-4 py-2 font-mono">
                        {row.std_rel_change != null ? `${(row.std_rel_change * 100).toFixed(1)}%` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
