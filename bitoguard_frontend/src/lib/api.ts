const API_BASE = "/api/backend"

async function parseError(path: string, res: Response): Promise<Error> {
  let detail = ""
  try {
    const payload = await res.json() as { detail?: string; message?: string }
    detail = payload.detail ?? payload.message ?? ""
  } catch {
    detail = ""
  }
  return new Error(`API error ${res.status}: ${path}${detail ? ` (${detail})` : ""}`)
}

function buildUrl(path: string, params?: Record<string, string | number>): string {
  const searchParams = new URLSearchParams()
  if (params) {
    Object.entries(params).forEach(([k, v]) => searchParams.set(k, String(v)))
  }
  const query = searchParams.toString()
  return `${API_BASE}${path}${query ? `?${query}` : ""}`
}

async function get<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const res = await fetch(buildUrl(path, params), { cache: "no-store" })
  if (!res.ok) throw await parseError(path, res)
  return res.json()
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw await parseError(path, res)
  return res.json()
}

// ── Types ──────────────────────────────────────────────────

export type RiskLevel = "low" | "medium" | "high" | "critical" | null

export interface Alert {
  alert_id: string
  user_id: string
  risk_level: RiskLevel
  risk_score?: number | null
  status: string
  created_at: string
}

export interface AlertsResponse {
  items: Alert[]
  total: number
}

export interface GraphNode {
  id: string
  type: "user" | "device" | "bank_account" | "wallet" | "ip"
  label: string
  hop: number
  is_focus: boolean
  risk_level: RiskLevel
  is_known_blacklist: boolean
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  relation_type:
    | "uses_device"
    | "uses_bank_account"
    | "owns_wallet"
    | "crypto_transfer_to_wallet"
    | "login_from_ip"
}

export interface GraphPayload {
  focus_user_id: string
  nodes: GraphNode[]
  edges: GraphEdge[]
  summary: { is_truncated: boolean; node_count: number; edge_count: number }
}

export interface ShapFactor {
  feature: string
  feature_zh: string
  impact: number
}

export interface DiagnosisReport {
  summary_zh: string
  alert: Alert | null
  case: { case_id: string; status: string; latest_decision: string | null; created_at: string } | null
  risk_summary: { risk_level: RiskLevel; risk_score: number }
  shap_top_factors: ShapFactor[]
  rule_hits: unknown[]
  graph_evidence: unknown
  timeline_summary: unknown[]
  recommended_action: string
  allowed_decisions: string[]
  case_actions: unknown[]
}

// ── API functions ──────────────────────────────────────────

export const api = {
  getAlerts: (params?: { risk_level?: string; page_size?: number }) =>
    get<AlertsResponse>("/alerts", params as Record<string, string | number>),

  getAlertReport: (alertId: string) =>
    get<DiagnosisReport>(`/alerts/${alertId}/report`),

  postDecision: (alertId: string, body: { decision: string; actor: string; note: string }) =>
    post(`/alerts/${alertId}/decision`, body),

  getUserGraph: (userId: string, maxHops: 1 | 2) =>
    get<GraphPayload>(`/users/${userId}/graph`, { max_hops: maxHops }),

  getUser360: (userId: string) =>
    get(`/users/${userId}/360`),

  getModelMetrics: () =>
    get("/metrics/model"),
}
