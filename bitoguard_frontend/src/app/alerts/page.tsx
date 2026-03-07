"use client"

import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { api, type RiskLevel } from "@/lib/api"
import Link from "next/link"
import { AlertTriangle, ChevronRight } from "lucide-react"
import { ErrorBanner } from "@/components/ErrorBanner"
import { ALERT_STATUS_ZH } from "@/lib/labels"

const RISK_COLORS: Record<NonNullable<RiskLevel>, string> = {
  critical: "bg-red-50 text-[#e53935] border-red-200",
  high:     "bg-orange-50 text-[#fb8c00] border-orange-200",
  medium:   "bg-yellow-50 text-[#f59e0b] border-yellow-200",
  low:      "bg-green-50 text-[#43a047] border-green-200",
}

const RISK_ZH: Record<NonNullable<RiskLevel>, string> = {
  critical: "極高風險", high: "高風險", medium: "中風險", low: "低風險",
}

export default function AlertCenterPage() {
  const [filter, setFilter] = useState<string>("all")

  const { data, isLoading, error } = useQuery({
    queryKey: ["alerts", filter],
    queryFn: () => api.getAlerts({ page_size: 200, ...(filter !== "all" ? { risk_level: filter } : {}) }),
  })

  const FILTERS = [
    { value: "all", label: "全部" },
    { value: "critical", label: "極高風險" },
    { value: "high", label: "高風險" },
    { value: "medium", label: "中風險" },
  ]

  return (
    <div className="max-w-[1100px] mx-auto space-y-4">
      <div>
        <h1 className="text-[22px] font-semibold text-[#1a1d2e]">警示中心</h1>
        <p className="text-[13px] text-[#9ca3af] mt-0.5">
          共 {data?.total ?? 0} 筆警示
        </p>
      </div>

      {/* Filter pills */}
      <div className="flex gap-2">
        {FILTERS.map((f) => (
          <button
            key={f.value}
            onClick={() => setFilter(f.value)}
            className={`px-4 py-1.5 rounded-full text-[13px] font-medium border transition-colors ${
              filter === f.value
                ? "bg-[#5c6bc0] text-white border-[#5c6bc0]"
                : "bg-white text-[#6b7280] border-[#e5e7eb] hover:border-[#9ca3af]"
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {error && <ErrorBanner message={error instanceof Error ? error.message : "無法載入警示資料"} />}
      {isLoading && <div className="text-[#9ca3af] py-8 text-center">載入中...</div>}

      {/* Table */}
      {!isLoading && data && (
        <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
          <table className="w-full text-[13px]">
            <thead className="bg-[#f4f6f9]">
              <tr>
                {["警示 ID", "帳戶", "風險等級", "風險分數", "狀態", "建立時間", ""].map((h) => (
                  <th key={h} className="text-left px-4 py-3 font-semibold text-[#6b7280] text-[11px] uppercase tracking-wider">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.items.map((alert) => (
                <tr key={alert.alert_id} className="border-t border-[#f4f6f9] hover:bg-[#fafbfc] transition-colors">
                  <td className="px-4 py-3 font-mono text-[12px] text-[#6b7280]">{alert.alert_id}</td>
                  <td className="px-4 py-3 font-medium text-[#1a1d2e]">{alert.user_id}</td>
                  <td className="px-4 py-3">
                    {alert.risk_level && (
                      <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold border ${RISK_COLORS[alert.risk_level]}`}>
                        <AlertTriangle size={10} />
                        {RISK_ZH[alert.risk_level]}
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 font-mono text-[#1a1d2e]">
                    <span className={`font-bold ${
                      typeof alert.risk_score === "number" && alert.risk_score >= 80
                        ? "text-[#e53935]"
                        : typeof alert.risk_score === "number" && alert.risk_score >= 60
                          ? "text-[#fb8c00]"
                          : typeof alert.risk_score === "number" && alert.risk_score >= 35
                            ? "text-[#f59e0b]"
                          : "text-[#1a1d2e]"
                    }`}>
                      {typeof alert.risk_score === "number" ? alert.risk_score.toFixed(2) : "—"}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className="px-2 py-0.5 rounded text-[11px] bg-[#f4f6f9] text-[#6b7280]">
                      {ALERT_STATUS_ZH[alert.status] ?? alert.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-[#9ca3af] text-[12px]">
                    {new Date(alert.created_at).toLocaleDateString("zh-TW")}
                  </td>
                  <td className="px-4 py-3">
                    <Link
                      href={`/alerts/report?alertId=${alert.alert_id}`}
                      className="inline-flex items-center gap-1 text-[#5c6bc0] hover:text-[#3949ab] text-[12px] font-medium"
                    >
                      診斷 <ChevronRight size={14} />
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {data.items.length === 0 && (
            <div className="text-center text-[#9ca3af] py-12">目前沒有符合條件的警示</div>
          )}
        </div>
      )}

      {!isLoading && !error && !data && (
        <div className="text-center text-[#9ca3af] py-12">目前無法取得警示資料</div>
      )}
    </div>
  )
}
