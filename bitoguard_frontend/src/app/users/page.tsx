"use client"

import { Suspense, useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"
import { useSearchParams } from "next/navigation"
import { ErrorBanner } from "@/components/ErrorBanner"
import { CASE_STATUS_ZH, DECISION_ZH, RISK_LEVEL_ZH } from "@/lib/labels"

const USER_FIELD_ZH: Record<string, string> = {
  user_id: "用戶 ID", kyc_level: "KYC 等級", occupation: "職業",
  income_source: "收入來源", monthly_income_twd: "月收入 (TWD)",
  created_at: "開戶日期", country: "國家", is_vip: "VIP 狀態",
  account_age_days: "帳戶天數",
}

const DIRECTION_ZH: Record<string, string> = {
  deposit: "入金", withdrawal: "出金",
}

const RISK_COLOR: Record<string, string> = {
  critical: "text-[#e53935]", high: "text-[#fb8c00]",
  medium: "text-[#f59e0b]", low: "text-[#43a047]",
}

function walletRoleLabel(direction: unknown): string {
  if (direction === "withdrawal") return "目的錢包"
  if (direction === "deposit") return "來源錢包"
  return "交易對象錢包"
}

function formatDate(v: unknown): string {
  if (!v) return "—"
  try { return new Date(String(v)).toLocaleDateString("zh-TW") } catch { return String(v) }
}

function UsersPageContent() {
  const searchParams = useSearchParams()
  const urlUserId = searchParams.get("userId") ?? ""
  const [selectedUserId, setSelectedUserId] = useState<string>(urlUserId)

  const { data: alertsData, isLoading: isAlertsLoading, error: alertsError } = useQuery({
    queryKey: ["alerts", "all"],
    queryFn: () => api.getAllAlerts({ page_size: 200 }),
    select: (d) => d.items,
    staleTime: 60_000,
  })

  const userIds = useMemo(
    () => [...new Set(alertsData?.map((a) => a.user_id) ?? [])],
    [alertsData],
  )
  const activeUserId =
    (selectedUserId && selectedUserId)
    || urlUserId
    || userIds[0]
    || ""

  const { data: profile, isLoading, error: profileError } = useQuery({
    queryKey: ["user360", activeUserId],
    queryFn: () => api.getUser360(activeUserId) as Promise<Record<string, unknown>>,
    enabled: !!activeUserId,
    staleTime: 60_000,
  })

  const user = profile?.user as Record<string, unknown> | undefined
  const prediction = profile?.latest_prediction as Record<string, unknown> | null | undefined
  const cases = profile?.cases as Record<string, unknown>[] | undefined
  const loginEvents = profile?.recent_login_events as Record<string, unknown>[] | undefined
  const cryptoTxns = profile?.recent_crypto_transactions as Record<string, unknown>[] | undefined

  return (
    <div className="max-w-[960px] mx-auto space-y-4">
      <div>
        <h1 className="text-[22px] font-semibold text-[#1a1d2e]">用戶全貌</h1>
        <p className="text-[13px] text-[#9ca3af] mt-0.5">查看用戶基本資訊、風險預測與近期交易紀錄</p>
      </div>
      {alertsError && <ErrorBanner message={alertsError instanceof Error ? alertsError.message : "無法載入用戶清單"} />}
      {profileError && <ErrorBanner message={profileError instanceof Error ? profileError.message : "無法載入用戶資料"} />}

      <div className="bg-white rounded-xl border border-[#e5e7eb] p-4 shadow-sm">
        <label className="text-[11px] font-semibold text-[#9ca3af] uppercase tracking-wider block mb-1">選擇用戶</label>
        <select
          value={activeUserId}
          onChange={(e) => setSelectedUserId(e.target.value)}
          className="border border-[#e5e7eb] rounded-lg px-3 py-2 text-[13px] text-[#1a1d2e] bg-white focus:outline-none min-w-[220px]"
          disabled={isAlertsLoading || userIds.length === 0}
        >
          {userIds.length === 0 && <option value="">沒有可用用戶</option>}
          {userIds.map((id) => <option key={id} value={id}>{id}</option>)}
        </select>
      </div>

      {isLoading && <div className="text-[#9ca3af] text-center py-8">載入中...</div>}
      {!isAlertsLoading && !alertsError && userIds.length === 0 && (
        <div className="text-[#9ca3af] text-center py-8">目前沒有可查看的用戶資料</div>
      )}

      {profile && (
        <div className="space-y-4">
          {/* 基本資訊 */}
          {user && (
            <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-[#e5e7eb]">
                <h2 className="text-[14px] font-semibold text-[#1a1d2e]">基本資訊</h2>
              </div>
              <div className="p-4 grid grid-cols-3 gap-4">
                {Object.entries(user).map(([k, v]) => (
                  <div key={k}>
                    <p className="text-[11px] text-[#9ca3af] font-semibold uppercase tracking-wider">
                      {USER_FIELD_ZH[k] ?? k}
                    </p>
                    <p className="text-[13px] text-[#1a1d2e] mt-0.5 truncate">
                      {k.endsWith("_at") || k === "created_at" ? formatDate(v) : String(v ?? "—")}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 最新風險預測 */}
          {prediction && (
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-white rounded-xl border border-[#e5e7eb] px-4 py-3 shadow-sm">
                <p className="text-[11px] text-[#9ca3af] font-semibold uppercase tracking-wider">風險等級</p>
                <p className={`text-[18px] font-semibold mt-0.5 ${RISK_COLOR[String(prediction.risk_level ?? "")] ?? "text-[#1a1d2e]"}`}>
                  {RISK_LEVEL_ZH[String(prediction.risk_level ?? "")] ?? String(prediction.risk_level ?? "N/A")}
                </p>
              </div>
              <div className="bg-white rounded-xl border border-[#e5e7eb] px-4 py-3 shadow-sm">
                <p className="text-[11px] text-[#9ca3af] font-semibold uppercase tracking-wider">風險分數</p>
                <p className={`text-[18px] font-semibold mt-0.5 font-mono ${
                  typeof prediction.risk_score === "number" && Number(prediction.risk_score) >= 80 ? "text-[#e53935]"
                  : typeof prediction.risk_score === "number" && Number(prediction.risk_score) >= 50 ? "text-[#fb8c00]"
                  : "text-[#1a1d2e]"
                }`}>
                  {typeof prediction.risk_score === "number" ? Number(prediction.risk_score).toFixed(2) : "—"}
                </p>
              </div>
              <div className="bg-white rounded-xl border border-[#e5e7eb] px-4 py-3 shadow-sm">
                <p className="text-[11px] text-[#9ca3af] font-semibold uppercase tracking-wider">快照日期</p>
                <p className="text-[18px] font-semibold text-[#1a1d2e] mt-0.5">{formatDate(prediction.snapshot_date)}</p>
              </div>
            </div>
          )}

          {/* 案件紀錄 */}
          {cases && cases.length > 0 && (
            <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-[#e5e7eb]">
                <h2 className="text-[14px] font-semibold text-[#1a1d2e]">案件紀錄</h2>
              </div>
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="bg-[#f9fafb] border-b border-[#e5e7eb]">
                    {["案件 ID", "狀態", "最新決定", "建立時間"].map((h) => (
                      <th key={h} className="px-4 py-2 text-left text-[11px] font-semibold text-[#9ca3af] uppercase tracking-wider">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {cases.map((c, i) => (
                    <tr key={i} className="border-b border-[#f3f4f6] hover:bg-[#f9fafb]">
                      <td className="px-4 py-2 font-mono">{String(c.case_id ?? "—")}</td>
                      <td className="px-4 py-2">{CASE_STATUS_ZH[String(c.status ?? "")] ?? String(c.status ?? "—")}</td>
                      <td className="px-4 py-2">{DECISION_ZH[String(c.latest_decision ?? "")] ?? String(c.latest_decision ?? "—")}</td>
                      <td className="px-4 py-2 text-[#9ca3af]">{formatDate(c.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* 近期登入 */}
          {loginEvents && loginEvents.length > 0 && (
            <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-[#e5e7eb]">
                <h2 className="text-[14px] font-semibold text-[#1a1d2e]">近期登入紀錄</h2>
              </div>
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="bg-[#f9fafb] border-b border-[#e5e7eb]">
                    {["時間", "IP", "裝置 ID", "國家"].map((h) => (
                      <th key={h} className="px-4 py-2 text-left text-[11px] font-semibold text-[#9ca3af] uppercase tracking-wider">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {loginEvents.map((e, i) => (
                    <tr key={i} className="border-b border-[#f3f4f6] hover:bg-[#f9fafb]">
                      <td className="px-4 py-2 text-[#9ca3af]">{formatDate(e.occurred_at)}</td>
                      <td className="px-4 py-2 font-mono">{String(e.ip_address ?? "—")}</td>
                      <td className="px-4 py-2 font-mono text-[12px]">{String(e.device_id ?? "—")}</td>
                      <td className="px-4 py-2">{String(e.ip_country ?? "—")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* 近期虛幣交易 */}
          {cryptoTxns && cryptoTxns.length > 0 && (
            <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-[#e5e7eb]">
                <h2 className="text-[14px] font-semibold text-[#1a1d2e]">近期虛幣交易</h2>
              </div>
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="bg-[#f9fafb] border-b border-[#e5e7eb]">
                    {["時間", "方向", "金額 (TWD)", "來源 / 目的錢包"].map((h) => (
                      <th key={h} className="px-4 py-2 text-left text-[11px] font-semibold text-[#9ca3af] uppercase tracking-wider">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {cryptoTxns.map((t, i) => (
                    <tr key={i} className="border-b border-[#f3f4f6] hover:bg-[#f9fafb]">
                      <td className="px-4 py-2 text-[#9ca3af]">{formatDate(t.occurred_at)}</td>
                      <td className="px-4 py-2">{DIRECTION_ZH[String(t.direction ?? "")] ?? String(t.direction ?? "—")}</td>
                      <td className="px-4 py-2 font-mono">{typeof t.amount_twd_equiv === "number" ? t.amount_twd_equiv.toFixed(2) : String(t.amount_twd_equiv ?? "—")}</td>
                      <td className="px-4 py-2 font-mono text-[12px] truncate max-w-[240px]" title={String(t.counterparty_wallet_id ?? "—")}>
                        {t.counterparty_wallet_id
                          ? `${walletRoleLabel(t.direction)}：${String(t.counterparty_wallet_id)}`
                          : "—"}
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

export default function UsersPage() {
  return (
    <Suspense fallback={<div className="text-[#9ca3af] text-center py-8">載入中...</div>}>
      <UsersPageContent />
    </Suspense>
  )
}
