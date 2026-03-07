export const ALERT_STATUS_ZH: Record<string, string> = {
  open: "待處理",
  closed: "已關閉",
  reviewing: "審查中",
  monitoring: "監控中",
  escalated: "已升級",
  confirmed_suspicious: "已確認可疑",
  dismissed_false_positive: "已排除誤報",
}

export const CASE_STATUS_ZH: Record<string, string> = {
  open: "待處理",
  monitoring: "監控中",
  escalated: "已升級",
  closed_confirmed: "已確認結案",
  closed_dismissed: "已排除結案",
}

export const RISK_LEVEL_ZH: Record<string, string> = {
  low: "低風險",
  medium: "中風險",
  high: "高風險",
  critical: "極高風險",
}

export const RECOMMENDED_ACTION_ZH: Record<string, string> = {
  monitor: "持續監控",
  manual_review: "人工複核",
  hold_withdrawal: "暫停出金",
}
