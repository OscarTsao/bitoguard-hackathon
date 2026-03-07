"use client"

import { useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"
import { ErrorBanner } from "@/components/ErrorBanner"
import { CASE_STATUS_ZH } from "@/lib/labels"

function walletRoleLabel(direction: unknown): string {
  if (direction === "withdrawal") return "目的錢包"
  if (direction === "deposit") return "來源錢包"
  return "交易對象錢包"
}

export default function UsersPage() {
  const [selectedUserId, setSelectedUserId] = useState<string>("")

  const { data: alertsData, isLoading: isAlertsLoading, error: alertsError } = useQuery({
    queryKey: ["alerts"],
    queryFn: () => api.getAlerts({ page_size: 200 }),
    select: (d) => d.items,
  })

  const userIds = useMemo(
    () => [...new Set(alertsData?.map((a) => a.user_id) ?? [])],
    [alertsData],
  )
  const activeUserId =
    (selectedUserId && userIds.includes(selectedUserId) && selectedUserId)
    || userIds[0]
    || ""

  const { data: profile, isLoading, error: profileError } = useQuery({
    queryKey: ["user360", activeUserId],
    queryFn: () => api.getUser360(activeUserId) as Promise<Record<string, unknown>>,
    enabled: !!activeUserId,
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
                    <p className="text-[11px] text-[#9ca3af] font-semibold uppercase tracking-wider">{k}</p>
                    <p className="text-[13px] text-[#1a1d2e] mt-0.5 truncate">{String(v ?? "—")}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 最新風險預測 */}
          {prediction && (
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: "風險等級", value: String(prediction.risk_level ?? "N/A") },
                { label: "風險分數", value: typeof prediction.risk_score === "number" ? Number(prediction.risk_score).toFixed(3) : String(prediction.risk_score ?? "N/A") },
                { label: "快照日期", value: String(prediction.snapshot_date ?? "N/A") },
              ].map(({ label, value }) => (
                <div key={label} className="bg-white rounded-xl border border-[#e5e7eb] px-4 py-3 shadow-sm">
                  <p className="text-[11px] text-[#9ca3af] font-semibold uppercase tracking-wider">{label}</p>
                  <p className="text-[18px] font-semibold text-[#1a1d2e] mt-0.5">{value}</p>
                </div>
              ))}
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
                      <td className="px-4 py-2">{String(c.latest_decision ?? "—")}</td>
                      <td className="px-4 py-2 text-[#9ca3af]">{String(c.created_at ?? "—")}</td>
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
                      <td className="px-4 py-2 text-[#9ca3af]">{String(e.occurred_at ?? "—")}</td>
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
                      <td className="px-4 py-2 text-[#9ca3af]">{String(t.occurred_at ?? "—")}</td>
                      <td className="px-4 py-2">{String(t.direction ?? "—")}</td>
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
