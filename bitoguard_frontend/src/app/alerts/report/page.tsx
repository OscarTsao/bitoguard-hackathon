"use client"

import { Suspense, useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { api } from "@/lib/api"
import { useSearchParams } from "next/navigation"
import Link from "next/link"
import { ChevronLeft, AlertTriangle, Clock, CheckCircle, XCircle, TrendingUp, Shield } from "lucide-react"
import type { ReactNode } from "react"
import { ErrorBanner } from "@/components/ErrorBanner"
import {
  ALERT_STATUS_ZH, RECOMMENDED_ACTION_ZH, RISK_LEVEL_ZH,
  DECISION_ZH, DECISION_COLOR, TIMELINE_TYPE_ZH, CASE_STATUS_ZH, GRAPH_EVIDENCE_ZH,
} from "@/lib/labels"

const RISK_COLORS: Record<string, string> = {
  critical: "bg-red-50 text-[#e53935] border-red-200",
  high:     "bg-orange-50 text-[#fb8c00] border-orange-200",
  medium:   "bg-yellow-50 text-[#f59e0b] border-yellow-200",
  low:      "bg-green-50 text-[#43a047] border-green-200",
}

const TIMELINE_ICON: Record<string, ReactNode> = {
  login:  <Shield size={12} />,
  trade:  <TrendingUp size={12} />,
  crypto: <TrendingUp size={12} />,
  fiat:   <TrendingUp size={12} />,
}

interface TimelineEntry {
  time: string
  type: string
  amount: number | null
}

function RiskBadge({ level }: { level: string | null | undefined }) {
  if (!level) return null
  return (
    <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold border ${RISK_COLORS[level] ?? ""}`}>
      <AlertTriangle size={10} />
      {RISK_LEVEL_ZH[level] ?? level}
    </span>
  )
}

function DecisionPanel({
  alertId,
  allowedDecisions,
  caseStatus,
  latestDecision,
}: {
  alertId: string
  allowedDecisions: string[]
  caseStatus: string | null | undefined
  latestDecision: string | null | undefined
}) {
  const queryClient = useQueryClient()
  const [note, setNote] = useState("")
  const [submitted, setSubmitted] = useState(false)

  const { mutate, isPending, error } = useMutation({
    mutationFn: (decision: string) =>
      api.postDecision(alertId, { decision, actor: "analyst", note }),
    onSuccess: () => {
      setSubmitted(true)
      setNote("")
      queryClient.invalidateQueries({ queryKey: ["report", alertId] })
    },
  })

  if (submitted || (latestDecision && !allowedDecisions.length)) {
    return (
      <div className="flex items-center gap-2 text-[#43a047] text-[13px]">
        <CheckCircle size={16} />
        決策已記錄：{DECISION_ZH[latestDecision ?? ""] ?? latestDecision ?? "已處理"}
      </div>
    )
  }

  if (!allowedDecisions.length) {
    return (
      <p className="text-[12px] text-[#9ca3af]">
        案件狀態：{CASE_STATUS_ZH[caseStatus ?? ""] ?? caseStatus ?? "N/A"}
      </p>
    )
  }

  return (
    <div className="space-y-3">
      {latestDecision && (
        <p className="text-[12px] text-[#6b7280]">
          最新決策：{DECISION_ZH[latestDecision] ?? latestDecision}
        </p>
      )}
      <div className="flex flex-wrap gap-2">
        {allowedDecisions.map((d) => (
          <button
            key={d}
            onClick={() => mutate(d)}
            disabled={isPending}
            className={`px-3 py-1.5 rounded-lg text-[12px] font-semibold border transition-colors disabled:opacity-50 ${DECISION_COLOR[d] ?? "bg-gray-50 text-gray-700 border-gray-300 hover:bg-gray-100"}`}
          >
            {DECISION_ZH[d] ?? d}
          </button>
        ))}
      </div>
      <textarea
        value={note}
        onChange={(e) => setNote(e.target.value)}
        placeholder="備註（可選）"
        rows={2}
        className="w-full border border-[#e5e7eb] rounded-lg px-3 py-2 text-[12px] text-[#1a1d2e] resize-none focus:outline-none focus:border-[#9ca3af]"
      />
      {error && <p className="text-[11px] text-[#e53935]">{error instanceof Error ? error.message : "提交失敗"}</p>}
    </div>
  )
}

function TimelinePanel({ events }: { events: TimelineEntry[] }) {
  if (!events.length) {
    return <p className="text-[12px] text-[#9ca3af]">無近期活動紀錄</p>
  }
  return (
    <div className="space-y-2">
      {events.map((ev, i) => (
        <div key={i} className="flex items-start gap-3">
          <div className="mt-0.5 w-5 h-5 rounded-full bg-[#f4f6f9] border border-[#e5e7eb] flex items-center justify-center text-[#6b7280] flex-shrink-0">
            {TIMELINE_ICON[ev.type] ?? <Clock size={10} />}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-[12px] font-medium text-[#1a1d2e]">
                {TIMELINE_TYPE_ZH[ev.type] ?? ev.type}
              </span>
              {ev.amount != null && (
                <span className="text-[11px] text-[#6b7280]">
                  {ev.amount.toLocaleString("zh-TW", { maximumFractionDigits: 4 })}
                </span>
              )}
            </div>
            <p className="text-[11px] text-[#9ca3af]">
              {new Date(ev.time).toLocaleString("zh-TW")}
            </p>
          </div>
        </div>
      ))}
    </div>
  )
}

function ReportPageContent() {
  const searchParams = useSearchParams()
  const alertId = searchParams.get("alertId") ?? ""

  const { data: report, isLoading, error } = useQuery({
    queryKey: ["report", alertId],
    queryFn: () => api.getAlertReport(alertId),
    enabled: !!alertId,
    staleTime: 30_000,
  })

  const maxAbsImpact = Math.max(
    ...(report?.shap_top_factors.map((f) => Math.abs(f.impact)) ?? [0])
  )

  const riskLevel = report?.risk_summary.risk_level ?? report?.alert?.risk_level

  return (
    <div className="max-w-[900px] mx-auto space-y-4">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2">
        <Link
          href="/alerts"
          className="inline-flex items-center gap-1 text-[12px] text-[#5c6bc0] hover:text-[#3949ab] font-medium"
        >
          <ChevronLeft size={14} />
          返回警示中心
        </Link>
        {alertId && (
          <>
            <span className="text-[#d1d5db]">/</span>
            <span className="text-[12px] text-[#9ca3af] font-mono">{alertId}</span>
          </>
        )}
      </div>

      {!alertId && (
        <div className="text-[#9ca3af] text-center py-12">
          請從警示中心點擊「診斷」以查看報告
        </div>
      )}

      {alertId && error && (
        <ErrorBanner message={error instanceof Error ? error.message : "無法載入報告"} />
      )}

      {isLoading && (
        <div className="text-[#9ca3af] text-center py-8">載入中...</div>
      )}

      {report && (
        <div className="space-y-4">
          {/* Alert header card */}
          <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm p-4">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <h1 className="text-[20px] font-semibold text-[#1a1d2e]">
                    用戶 {report.user_id}
                  </h1>
                  <RiskBadge level={String(riskLevel ?? "")} />
                </div>
                <div className="flex items-center gap-4 text-[12px] text-[#6b7280]">
                  <span>
                    風險分數：
                    <span className={`font-bold ml-1 ${
                      (report.risk_summary.risk_score ?? 0) >= 80 ? "text-[#e53935]"
                      : (report.risk_summary.risk_score ?? 0) >= 50 ? "text-[#fb8c00]"
                      : "text-[#1a1d2e]"
                    }`}>
                      {Number(report.risk_summary.risk_score).toFixed(2)}
                    </span>
                  </span>
                  <span>
                    建議：
                    <span className="font-medium text-[#1a1d2e] ml-1">
                      {RECOMMENDED_ACTION_ZH[report.recommended_action] ?? report.recommended_action}
                    </span>
                  </span>
                  {report.alert && (
                    <span>
                      狀態：
                      <span className="font-medium text-[#1a1d2e] ml-1">
                        {ALERT_STATUS_ZH[report.alert.status] ?? report.alert.status}
                      </span>
                    </span>
                  )}
                </div>
              </div>
              <Link
                href={`/users?userId=${report.user_id}`}
                className="text-[12px] text-[#5c6bc0] hover:text-[#3949ab] font-medium whitespace-nowrap"
              >
                查看用戶全貌 →
              </Link>
            </div>
          </div>

          {/* Summary */}
          <div className="bg-[#eef0fa] border border-[#c5cae9] rounded-xl px-4 py-3">
            <p className="text-[14px] font-medium text-[#3949ab]">{report.summary_zh}</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Left column */}
            <div className="space-y-4">
              {/* SHAP factors */}
              {report.shap_top_factors.length > 0 ? (
                <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
                  <div className="px-4 py-3 border-b border-[#e5e7eb]">
                    <h2 className="text-[14px] font-semibold text-[#1a1d2e]">SHAP 特徵重要性</h2>
                    <p className="text-[11px] text-[#9ca3af] mt-0.5">紅色＝增加風險，綠色＝降低風險</p>
                  </div>
                  <div className="p-4 space-y-3">
                    {report.shap_top_factors.map((f) => (
                      <div key={f.feature} className="grid grid-cols-[1fr_140px_56px] items-center gap-2">
                        <span className="text-[12px] text-[#6b7280] leading-5 truncate" title={f.feature}>
                          {f.feature_zh}
                        </span>
                        <div className="relative h-5">
                          <div className="absolute inset-y-0 left-0 right-0 top-1/2 -translate-y-1/2 rounded-full bg-[#f4f6f9]" />
                          <div className="absolute inset-y-0 left-1/2 w-px bg-[#cbd5e1]" />
                          <div className="absolute inset-y-0 left-0 w-1/2 pr-1">
                            {f.impact < 0 && (
                              <div
                                className="ml-auto mt-0.5 h-4 rounded-full bg-[#43a047]"
                                style={{ width: `${maxAbsImpact === 0 ? 0 : (Math.abs(f.impact) / maxAbsImpact) * 100}%`, minWidth: "4px" }}
                              />
                            )}
                          </div>
                          <div className="absolute inset-y-0 right-0 w-1/2 pl-1">
                            {f.impact > 0 && (
                              <div
                                className="mt-0.5 h-4 rounded-full bg-[#e53935]"
                                style={{ width: `${maxAbsImpact === 0 ? 0 : (Math.abs(f.impact) / maxAbsImpact) * 100}%`, minWidth: "4px" }}
                              />
                            )}
                          </div>
                        </div>
                        <span className={`text-[11px] font-mono font-semibold text-right ${f.impact > 0 ? "text-[#e53935]" : "text-[#43a047]"}`}>
                          {f.impact > 0 ? "+" : ""}{f.impact.toFixed(3)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm p-4">
                  <h2 className="text-[14px] font-semibold text-[#1a1d2e] mb-2">SHAP 特徵重要性</h2>
                  <p className="text-[12px] text-[#9ca3af]">無 SHAP 特徵資料（需產生 SHAP 報告）</p>
                </div>
              )}

              {/* Rule hits */}
              <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
                <div className="px-4 py-3 border-b border-[#e5e7eb]">
                  <h2 className="text-[14px] font-semibold text-[#1a1d2e]">規則觸發</h2>
                </div>
                <div className="p-4">
                  {Array.isArray(report.rule_hits) && report.rule_hits.length > 0 ? (
                    <div className="space-y-1">
                      {(report.rule_hits as string[]).map((hit, i) => (
                        <div key={i} className="flex items-center gap-2">
                          <XCircle size={12} className="text-[#e53935] flex-shrink-0" />
                          <span className="text-[12px] text-[#1a1d2e]">{hit}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-[12px] text-[#9ca3af]">無規則觸發</p>
                  )}
                </div>
              </div>
            </div>

            {/* Right column */}
            <div className="space-y-4">
              {/* Case decision */}
              <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
                <div className="px-4 py-3 border-b border-[#e5e7eb]">
                  <h2 className="text-[14px] font-semibold text-[#1a1d2e]">案件決策</h2>
                  {report.case && (
                    <p className="text-[11px] text-[#9ca3af] mt-0.5">
                      案件 {report.case.case_id.slice(-8)} ·
                      {CASE_STATUS_ZH[report.case.status] ?? report.case.status}
                    </p>
                  )}
                </div>
                <div className="p-4">
                  <DecisionPanel
                    alertId={alertId}
                    allowedDecisions={report.allowed_decisions as string[]}
                    caseStatus={report.case?.status}
                    latestDecision={report.case?.latest_decision}
                  />
                </div>
              </div>

              {/* Timeline */}
              <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
                <div className="px-4 py-3 border-b border-[#e5e7eb]">
                  <h2 className="text-[14px] font-semibold text-[#1a1d2e]">近期活動時間軸</h2>
                </div>
                <div className="p-4">
                  <TimelinePanel events={(report.timeline_summary as TimelineEntry[]) ?? []} />
                </div>
              </div>

              {/* Graph evidence */}
              {report.graph_evidence && (
                <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
                  <div className="px-4 py-3 border-b border-[#e5e7eb]">
                    <h2 className="text-[14px] font-semibold text-[#1a1d2e]">圖譜風險指標</h2>
                  </div>
                  <div className="p-4">
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(report.graph_evidence as unknown as Record<string, number>)
                        .filter(([, v]) => v != null)
                        .map(([k, v]) => (
                          <div key={k} className="flex items-center justify-between">
                            <span className="text-[11px] text-[#6b7280]">{GRAPH_EVIDENCE_ZH[k] ?? k.replace(/_/g, " ")}</span>
                            <span className={`text-[12px] font-semibold ${v > 0 ? "text-[#e53935]" : "text-[#9ca3af]"}`}>
                              {v}
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
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
