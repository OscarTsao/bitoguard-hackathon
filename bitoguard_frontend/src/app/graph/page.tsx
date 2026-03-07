"use client"

import { useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { api, type GraphNode, type GraphEdge } from "@/lib/api"
import { GraphCanvas } from "@/components/graph/GraphCanvas"
import { Monitor, Landmark, Wallet, Globe, AlertTriangle, ShieldAlert } from "lucide-react"
import { ErrorBanner } from "@/components/ErrorBanner"

// ── Constants ──────────────────────────────────────────────
const ALL_RELATIONS = [
  "uses_device", "uses_bank_account", "owns_wallet",
  "crypto_transfer_to_wallet", "login_from_ip",
] as const

const RELATION_ZH: Record<string, string> = {
  uses_device: "使用裝置",
  uses_bank_account: "綁定銀行帳戶",
  owns_wallet: "持有錢包",
  crypto_transfer_to_wallet: "虛幣轉帳",
  login_from_ip: "登入來源 IP",
}

const TYPE_ZH: Record<string, string> = {
  device: "裝置", bank_account: "銀行帳戶", wallet: "錢包", ip: "登入 IP", user: "帳戶",
}

const NODE_COLORS: Record<string, string> = {
  focus: "#5c6bc0", blacklist: "#e53935", high_risk: "#fb8c00",
  device: "#00acc1", bank_account: "#43a047", wallet: "#8e24aa",
  ip: "#90a4ae", user: "#78909c",
}

// ── Helper components ──────────────────────────────────────
function MetricCard({ label, value, icon: Icon, danger }: {
  label: string; value: number; icon: React.ElementType; danger?: boolean
}) {
  return (
    <div className="bg-white rounded-xl border border-[#e5e7eb] px-4 py-3 flex items-center gap-3 shadow-sm">
      <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${danger && value > 0 ? "bg-red-50" : "bg-[#f4f6f9]"}`}>
        <Icon size={16} className={danger && value > 0 ? "text-[#e53935]" : "text-[#5c6bc0]"} />
      </div>
      <div>
        <p className="text-[11px] text-[#9ca3af] font-medium">{label}</p>
        <p className={`text-[20px] font-semibold leading-tight ${danger && value > 0 ? "text-[#e53935]" : "text-[#1a1d2e]"}`}>
          {value}
        </p>
      </div>
    </div>
  )
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5 text-[12px] text-[#6b7280]">
      <span className="inline-block w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
      {label}
    </span>
  )
}

function buildNarrative(focusUserId: string, nodes: GraphNode[]): string {
  const entityCounts: Record<string, number> = {}
  const neighborUsers: GraphNode[] = []
  for (const node of nodes) {
    if (node.is_focus) continue
    if (node.type === "user") neighborUsers.push(node)
    else entityCounts[node.type] = (entityCounts[node.type] ?? 0) + 1
  }
  const parts: string[] = []
  for (const etype of ["device", "bank_account", "wallet", "ip"] as const) {
    const cnt = entityCounts[etype] ?? 0
    if (cnt) {
      const unit = etype === "device" ? "台" : "個"
      parts.push(`${TYPE_ZH[etype]}（${cnt} ${unit}）`)
    }
  }
  const blacklistCnt = neighborUsers.filter((u) => u.is_known_blacklist).length
  const highriskCnt = neighborUsers.filter((u) => u.risk_level === "high" || u.risk_level === "critical").length
  const connStr = parts.length > 0 ? parts.join("、") : "無共享資產"
  return `「${focusUserId}」透過 ${connStr}，與 ${neighborUsers.length} 個其他帳戶形成關聯網路。其中 ${blacklistCnt} 個鄰居為已知黑名單、${highriskCnt} 個被模型判為高風險。`
}

function buildSharedTable(focusNodeId: string, nodes: GraphNode[], edges: GraphEdge[]) {
  const nodeMap = new Map(nodes.map((n) => [n.id, n]))
  const entityUsers: Record<string, string[]> = {}
  for (const edge of edges) {
    const srcNode = nodeMap.get(edge.source)
    const tgtNode = nodeMap.get(edge.target)
    if (!srcNode || !tgtNode) continue
    let entityId: string, userId: string
    if (srcNode.type === "user") { userId = edge.source; entityId = edge.target }
    else { userId = edge.target; entityId = edge.source }
    if (userId === focusNodeId) continue
    if (!entityUsers[entityId]) entityUsers[entityId] = []
    if (!entityUsers[entityId].includes(userId)) entityUsers[entityId].push(userId)
  }
  return Object.entries(entityUsers)
    .filter(([, users]) => users.length > 0)
    .map(([entityId, userIds]) => {
      const entityNode = nodeMap.get(entityId)
      return {
        asset: entityNode?.label ?? entityId,
        assetType: TYPE_ZH[entityNode?.type ?? ""] ?? "未知",
        relatedUsers: userIds.map((uid) => uid.replace("user:", "")).join("、"),
        riskStatus: userIds.map((uid) => {
          const u = nodeMap.get(uid)
          if (u?.is_known_blacklist) return "黑名單"
          if (u?.risk_level === "high" || u?.risk_level === "critical") return "高風險"
          return "正常"
        }).join("、"),
      }
    })
}

// ── Main page ──────────────────────────────────────────────
export default function GraphExplorerPage() {
  const [selectedUser, setSelectedUser] = useState<string>("")
  const [hops, setHops] = useState<1 | 2>(2)
  const [selectedRelations, setSelectedRelations] = useState<string[]>([...ALL_RELATIONS])

  const { data: alertsData, isLoading: isAlertsLoading, error: alertsError } = useQuery({
    queryKey: ["alerts"],
    queryFn: () => api.getAlerts({ page_size: 200 }),
  })

  const userIds = useMemo(
    () => [...new Set(alertsData?.items.map((a) => a.user_id) ?? [])].sort(),
    [alertsData],
  )
  const activeSelectedUser =
    (selectedUser && userIds.includes(selectedUser) && selectedUser)
    || userIds[0]
    || ""

  const { data: graphData, isLoading, error: graphError } = useQuery({
    queryKey: ["graph", activeSelectedUser, hops],
    queryFn: () => api.getUserGraph(activeSelectedUser, hops),
    enabled: !!activeSelectedUser,
  })

  const { filteredNodes, filteredEdges, focusNodeId } = useMemo(() => {
    if (!graphData) return { filteredNodes: [], filteredEdges: [], focusNodeId: "" }
    const fid = `user:${graphData.focus_user_id}`
    const edges = graphData.edges.filter((e) => selectedRelations.includes(e.relation_type))
    const usedIds = new Set([fid, ...edges.flatMap((e) => [e.source, e.target])])
    const nodeMap = new Map(graphData.nodes.map((n) => [n.id, n]))
    const fnodes = [...usedIds].flatMap((id) => nodeMap.has(id) ? [nodeMap.get(id)!] : [])
    return { filteredNodes: fnodes, filteredEdges: edges, focusNodeId: fid }
  }, [graphData, selectedRelations])

  const entityCounts = useMemo(() => {
    const counts = { device: 0, bank_account: 0, wallet: 0, ip: 0 }
    filteredNodes.forEach((n) => { if (!n.is_focus && n.type in counts) counts[n.type as keyof typeof counts]++ })
    return counts
  }, [filteredNodes])

  const blacklistCnt = filteredNodes.filter((n) => n.type === "user" && !n.is_focus && n.is_known_blacklist).length
  const highriskCnt = filteredNodes.filter((n) => n.type === "user" && !n.is_focus && (n.risk_level === "high" || n.risk_level === "critical")).length
  const narrative = activeSelectedUser ? buildNarrative(activeSelectedUser, filteredNodes) : ""
  const sharedRows = hops === 2 ? buildSharedTable(focusNodeId, filteredNodes, filteredEdges) : []

  function toggleRelation(rel: string) {
    setSelectedRelations((prev) =>
      prev.includes(rel) ? prev.filter((r) => r !== rel) : [...prev, rel]
    )
  }

  return (
    <div className="max-w-[1200px] mx-auto space-y-4">
      {/* Header */}
      <div>
        <h1 className="text-[22px] font-semibold text-[#1a1d2e]">關聯圖探索</h1>
        <p className="text-[13px] text-[#9ca3af] mt-0.5">分析帳戶透過共享資產與其他帳戶形成的關聯網路</p>
      </div>
      {alertsError && <ErrorBanner message={alertsError instanceof Error ? alertsError.message : "無法載入關聯圖用戶清單"} />}
      {graphError && <ErrorBanner message={graphError instanceof Error ? graphError.message : "無法載入關聯圖資料"} />}

      {/* Controls */}
      <div className="bg-white rounded-xl border border-[#e5e7eb] p-4 shadow-sm">
        <div className="flex flex-wrap gap-4 items-end">
          {/* User selector */}
          <div className="flex flex-col gap-1">
            <label className="text-[11px] font-semibold text-[#9ca3af] uppercase tracking-wider">查詢帳戶</label>
            <select
              value={activeSelectedUser}
              onChange={(e) => setSelectedUser(e.target.value)}
              className="border border-[#e5e7eb] rounded-lg px-3 py-2 text-[13px] text-[#1a1d2e] bg-white focus:outline-none focus:ring-2 focus:ring-[#5c6bc0] focus:ring-opacity-30 min-w-[160px]"
              disabled={isAlertsLoading || userIds.length === 0}
            >
              {userIds.length === 0 && <option value="">沒有可用帳戶</option>}
              {userIds.map((id) => <option key={id} value={id}>{id}</option>)}
            </select>
          </div>

          {/* Hop toggle */}
          <div className="flex flex-col gap-1">
            <label className="text-[11px] font-semibold text-[#9ca3af] uppercase tracking-wider">關聯層數</label>
            <div className="flex rounded-lg border border-[#e5e7eb] overflow-hidden">
              {([1, 2] as const).map((h) => (
                <button
                  key={h}
                  onClick={() => setHops(h)}
                  className={`px-4 py-2 text-[13px] font-medium transition-colors ${
                    hops === h
                      ? "bg-[#5c6bc0] text-white"
                      : "bg-white text-[#6b7280] hover:bg-[#f4f6f9]"
                  }`}
                >
                  {h}-hop {h === 1 ? "（直接）" : "（間接）"}
                </button>
              ))}
            </div>
          </div>

          {/* Relation type filter */}
          <div className="flex flex-col gap-1">
            <label className="text-[11px] font-semibold text-[#9ca3af] uppercase tracking-wider">關聯類型</label>
            <div className="flex flex-wrap gap-2">
              {ALL_RELATIONS.map((rel) => (
                <button
                  key={rel}
                  onClick={() => toggleRelation(rel)}
                  className={`px-3 py-1.5 rounded-full text-[12px] font-medium border transition-colors ${
                    selectedRelations.includes(rel)
                      ? "bg-[#eef0fa] text-[#5c6bc0] border-[#5c6bc0]"
                      : "bg-white text-[#9ca3af] border-[#e5e7eb] hover:border-[#9ca3af]"
                  }`}
                >
                  {RELATION_ZH[rel]}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {isLoading && (
        <div className="flex items-center justify-center h-24 text-[#9ca3af]">載入中...</div>
      )}

      {!isAlertsLoading && !alertsError && userIds.length === 0 && (
        <div className="text-[#9ca3af] text-center py-8">目前沒有可用的關聯圖帳戶</div>
      )}

      {!isLoading && activeSelectedUser && (
        <>
          {/* Narrative */}
          {narrative && (
            <div className="bg-[#eef0fa] border border-[#c5cae9] rounded-xl px-4 py-3">
              <p className="text-[13px] text-[#3949ab] leading-relaxed">{narrative}</p>
            </div>
          )}

          {/* Metrics */}
          <div className="grid grid-cols-6 gap-3">
            <MetricCard label="裝置" value={entityCounts.device} icon={Monitor} />
            <MetricCard label="銀行帳戶" value={entityCounts.bank_account} icon={Landmark} />
            <MetricCard label="錢包" value={entityCounts.wallet} icon={Wallet} />
            <MetricCard label="登入 IP" value={entityCounts.ip} icon={Globe} />
            <MetricCard label="黑名單鄰居" value={blacklistCnt} icon={AlertTriangle} danger />
            <MetricCard label="高風險鄰居" value={highriskCnt} icon={ShieldAlert} danger />
          </div>

          {/* Legend + Graph */}
          <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-[#e5e7eb] flex flex-wrap gap-4 items-center">
              <span className="text-[12px] font-semibold text-[#9ca3af] uppercase tracking-wider mr-1">圖例</span>
              <LegendDot color={NODE_COLORS.focus} label="調查對象" />
              <LegendDot color={NODE_COLORS.blacklist} label="已知黑名單" />
              <LegendDot color={NODE_COLORS.high_risk} label="高風險帳戶" />
              <LegendDot color={NODE_COLORS.device} label="裝置" />
              <LegendDot color={NODE_COLORS.bank_account} label="銀行帳戶" />
              <LegendDot color={NODE_COLORS.wallet} label="錢包" />
              <LegendDot color={NODE_COLORS.ip} label="登入 IP" />
              <LegendDot color={NODE_COLORS.user} label="其他帳戶" />
            </div>
            <div className="p-4">
              <GraphCanvas nodes={filteredNodes} edges={filteredEdges} focusNodeId={focusNodeId} />
            </div>
          </div>

          {/* 2-hop shared table */}
          {hops === 2 && sharedRows.length > 0 && (
            <div className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-[#e5e7eb]">
                <h2 className="text-[14px] font-semibold text-[#1a1d2e]">共同關聯帳戶（間接關聯分析）</h2>
                <p className="text-[12px] text-[#9ca3af] mt-0.5">透過共享資產與調查對象間接連結的帳戶</p>
              </div>
              <table className="w-full text-[13px]">
                <thead className="bg-[#f4f6f9]">
                  <tr>
                    {["共享資產", "資產類型", "共同關聯帳戶", "風險狀態"].map((h) => (
                      <th key={h} className="text-left px-4 py-2.5 font-semibold text-[#6b7280] text-[11px] uppercase tracking-wider">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sharedRows.map((row, i) => (
                    <tr key={i} className="border-t border-[#f4f6f9] hover:bg-[#fafbfc]">
                      <td className="px-4 py-2.5 font-mono text-[12px] text-[#1a1d2e]">{row.asset}</td>
                      <td className="px-4 py-2.5 text-[#6b7280]">{row.assetType}</td>
                      <td className="px-4 py-2.5 font-mono text-[12px] text-[#1a1d2e]">{row.relatedUsers}</td>
                      <td className="px-4 py-2.5">
                        {row.riskStatus.split("、").map((s, j) => (
                          <span key={j} className={`inline-block mr-1 px-2 py-0.5 rounded-full text-[11px] font-medium ${
                            s === "黑名單" ? "bg-red-50 text-[#e53935]"
                            : s === "高風險" ? "bg-orange-50 text-[#fb8c00]"
                            : "bg-green-50 text-[#43a047]"
                          }`}>{s}</span>
                        ))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Truncation warning */}
          {graphData?.summary.is_truncated && (
            <div className="bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 text-[13px] text-amber-700">
              關聯圖節點數量超過上限，已自動截斷部分資料。
            </div>
          )}
        </>
      )}
    </div>
  )
}
