"use client"

import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"

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

interface ModelMetrics {
  model_version: string
  precision: number
  recall: number
  f1: number
  fpr: number
  average_precision: number
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number }
  threshold_sensitivity: ThresholdRow[]
  scenario_breakdown: ScenarioRow[]
}

export default function ModelOpsPage() {
  const { data: metrics, isLoading, error } = useQuery({
    queryKey: ["modelMetrics"],
    queryFn: () => api.getModelMetrics() as Promise<ModelMetrics>,
  })

  return (
    <div className="max-w-[960px] mx-auto space-y-4">
      <div>
        <h1 className="text-[22px] font-semibold text-[#1a1d2e]">模型指標</h1>
        <p className="text-[13px] text-[#9ca3af] mt-0.5">檢視風險模型的驗證結果與閾值敏感度分析</p>
      </div>

      {isLoading && <div className="text-[#9ca3af] text-center py-8">載入中...</div>}
      {error && <div className="text-[#e53935] text-center py-8">無法載入模型指標，請確認後端服務是否正常運行</div>}

      {metrics && (
        <div className="space-y-4">
          {/* 版本與核心指標 */}
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: "模型版本", value: metrics.model_version },
              { label: "Precision", value: metrics.precision.toFixed(4) },
              { label: "Recall", value: metrics.recall.toFixed(4) },
              { label: "F1 Score", value: metrics.f1.toFixed(4) },
              { label: "FPR (偽陽率)", value: metrics.fpr.toFixed(4) },
              { label: "Average Precision", value: metrics.average_precision.toFixed(4) },
            ].map(({ label, value }) => (
              <div key={label} className="bg-white rounded-xl border border-[#e5e7eb] px-4 py-3 shadow-sm">
                <p className="text-[11px] text-[#9ca3af] font-semibold uppercase tracking-wider">{label}</p>
                <p className="text-[18px] font-semibold text-[#1a1d2e] mt-0.5">{value}</p>
              </div>
            ))}
          </div>

          {/* 混淆矩陣 */}
          <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-[#e5e7eb]">
              <h2 className="text-[14px] font-semibold text-[#1a1d2e]">混淆矩陣</h2>
            </div>
            <div className="p-4 grid grid-cols-4 gap-3">
              {[
                { label: "True Negative (TN)", value: metrics.confusion_matrix.tn, color: "#43a047" },
                { label: "False Positive (FP)", value: metrics.confusion_matrix.fp, color: "#fb8c00" },
                { label: "False Negative (FN)", value: metrics.confusion_matrix.fn, color: "#fb8c00" },
                { label: "True Positive (TP)", value: metrics.confusion_matrix.tp, color: "#5c6bc0" },
              ].map(({ label, value, color }) => (
                <div key={label} className="rounded-lg border border-[#e5e7eb] px-4 py-3 text-center">
                  <p className="text-[11px] text-[#9ca3af] font-semibold">{label}</p>
                  <p className="text-[24px] font-semibold mt-0.5" style={{ color }}>{value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* 閾值敏感度 */}
          {metrics.threshold_sensitivity.length > 0 && (
            <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-[#e5e7eb]">
                <h2 className="text-[14px] font-semibold text-[#1a1d2e]">閾值敏感度分析</h2>
              </div>
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="bg-[#f9fafb] border-b border-[#e5e7eb]">
                    {["閾值", "Precision", "Recall", "F1"].map((h) => (
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
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* 情境分析 */}
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
                      <td className="px-4 py-2 font-semibold">{row.scenario}</td>
                      <td className="px-4 py-2 font-mono">{row.count}</td>
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
    </div>
  )
}
