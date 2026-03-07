"use client"

import { Suspense, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"
import { useRouter, useSearchParams } from "next/navigation"
import { ErrorBanner } from "@/components/ErrorBanner"
import { ALERT_STATUS_ZH, RECOMMENDED_ACTION_ZH, RISK_LEVEL_ZH } from "@/lib/labels"

function ReportPageContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [manualSelectedAlert, setManualSelectedAlert] = useState<string>("")

  const { data: alertsData, isLoading: isAlertsLoading, error: alertsError } = useQuery({
    queryKey: ["alerts"],
    queryFn: () => api.getAlerts({ page_size: 200 }),
    select: (d) => d.items,
  })

  const alertIds = alertsData?.map((a) => a.alert_id) ?? []
  const requestedAlertId = searchParams.get("alertId") ?? ""
  const selectedAlert =
    (manualSelectedAlert && alertIds.includes(manualSelectedAlert) && manualSelectedAlert)
    || (requestedAlertId && alertIds.includes(requestedAlertId) && requestedAlertId)
    || alertIds[0]
    || ""

  const { data: report, isLoading, error: reportError } = useQuery({
    queryKey: ["report", selectedAlert],
    queryFn: () => api.getAlertReport(selectedAlert),
    enabled: !!selectedAlert,
  })

  const maxAbsImpact = Math.max(...(report?.shap_top_factors.map((factor) => Math.abs(factor.impact)) ?? [0]))

  return (
    <div className="max-w-[900px] mx-auto space-y-4">
      <div>
        <h1 className="text-[22px] font-semibold text-[#1a1d2e]">風險診斷報告</h1>
      </div>
      {alertsError && <ErrorBanner message={alertsError instanceof Error ? alertsError.message : "無法載入警示清單"} />}
      {reportError && <ErrorBanner message={reportError instanceof Error ? reportError.message : "無法載入風險診斷報告"} />}
      <div className="bg-white rounded-xl border border-[#e5e7eb] p-4 shadow-sm">
        <label className="text-[11px] font-semibold text-[#9ca3af] uppercase tracking-wider block mb-1">切換警示報告</label>
        <p className="text-[12px] text-[#6b7280] mb-3">
          選擇要查看的警示案件，系統將顯示對應診斷報告。
        </p>
        <select
          value={selectedAlert}
          onChange={(e) => {
            const alertId = e.target.value
            setManualSelectedAlert(alertId)
            router.replace(`/alerts/report?alertId=${alertId}`, { scroll: false })
          }}
          className="border border-[#e5e7eb] rounded-lg px-3 py-2 text-[13px] text-[#1a1d2e] bg-white focus:outline-none min-w-[220px]"
          disabled={isAlertsLoading || alertIds.length === 0}
        >
          {alertIds.length === 0 && <option value="">沒有可用警示</option>}
          {(alertsData ?? []).map((alert) => (
            <option key={alert.alert_id} value={alert.alert_id}>
              {[
                alert.user_id,
                RISK_LEVEL_ZH[String(alert.risk_level ?? "")] ?? String(alert.risk_level ?? "未知風險"),
                ALERT_STATUS_ZH[alert.status] ?? alert.status,
                alert.alert_id,
              ].join(" / ")}
            </option>
          ))}
        </select>
      </div>
      {isLoading && <div className="text-[#9ca3af] text-center py-8">載入中...</div>}
      {!isAlertsLoading && !alertsError && alertIds.length === 0 && (
        <div className="text-[#9ca3af] text-center py-8">目前沒有可查看的警示</div>
      )}
      {report && (
        <div className="space-y-4">
          <div className="bg-[#eef0fa] border border-[#c5cae9] rounded-xl px-4 py-3">
            <p className="text-[14px] font-medium text-[#3949ab]">{report.summary_zh}</p>
          </div>
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: "風險等級", value: RISK_LEVEL_ZH[String(report.risk_summary.risk_level ?? "")] ?? (report.risk_summary.risk_level ?? "N/A") },
              { label: "風險分數", value: Number(report.risk_summary.risk_score).toFixed(3) },
              { label: "建議行動", value: RECOMMENDED_ACTION_ZH[report.recommended_action] ?? report.recommended_action },
            ].map(({ label, value }) => (
              <div key={label} className="bg-white rounded-xl border border-[#e5e7eb] px-4 py-3 shadow-sm">
                <p className="text-[11px] text-[#9ca3af] font-semibold uppercase tracking-wider">{label}</p>
                <p className="text-[18px] font-semibold text-[#1a1d2e] mt-0.5">{value}</p>
              </div>
            ))}
          </div>
          {report.shap_top_factors.length > 0 && (
            <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-[#e5e7eb]">
                <h2 className="text-[14px] font-semibold text-[#1a1d2e]">SHAP 重要特徵</h2>
              </div>
              <div className="p-4 space-y-3">
                {report.shap_top_factors.map((f) => (
                  <div key={f.feature} className="grid grid-cols-[220px_minmax(320px,1fr)_80px] items-center gap-4">
                    <span className="text-[13px] text-[#6b7280] leading-5 break-words" title={f.feature_zh}>
                      {f.feature_zh}
                    </span>
                    <div className="relative h-6">
                      <div className="absolute inset-y-0 left-0 right-0 top-1/2 -translate-y-1/2 rounded-full bg-[#f4f6f9]" />
                      <div className="absolute inset-y-0 left-1/2 w-px bg-[#cbd5e1]" />
                      <div className="absolute inset-y-0 left-0 w-1/2 pr-1">
                        {f.impact < 0 && (
                          <div
                            className="ml-auto mt-1 h-4 min-w-0 rounded-full bg-[#43a047]"
                            style={{
                              width: `${maxAbsImpact === 0 ? 0 : (Math.abs(f.impact) / maxAbsImpact) * 100}%`,
                              minWidth: Math.abs(f.impact) > 0 ? "6px" : "0px",
                            }}
                          />
                        )}
                      </div>
                      <div className="absolute inset-y-0 right-0 w-1/2 pl-1">
                        {f.impact > 0 && (
                          <div
                            className="mt-1 h-4 min-w-0 rounded-full bg-[#e53935]"
                            style={{
                              width: `${maxAbsImpact === 0 ? 0 : (Math.abs(f.impact) / maxAbsImpact) * 100}%`,
                              minWidth: Math.abs(f.impact) > 0 ? "6px" : "0px",
                            }}
                          />
                        )}
                      </div>
                    </div>
                    <span className={`text-[12px] font-mono font-semibold w-16 text-right ${f.impact > 0 ? "text-[#e53935]" : "text-[#43a047]"}`}>
                      {f.impact > 0 ? "+" : ""}{f.impact.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function ReportPage() {
  return (
    <Suspense fallback={<div className="text-[#9ca3af] text-center py-8">載入中...</div>}>
      <ReportPageContent />
    </Suspense>
  )
}
