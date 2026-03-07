"use client"

import { useEffect, useRef } from "react"
import type { GraphNode, GraphEdge } from "@/lib/api"

// ── Color mapping ──────────────────────────────────────────
const NODE_COLORS: Record<string, string> = {
  focus:        "#5c6bc0",
  blacklist:    "#e53935",
  high_risk:    "#fb8c00",
  device:       "#00acc1",
  bank_account: "#43a047",
  wallet:       "#8e24aa",
  ip:           "#90a4ae",
  user:         "#78909c",
}

function nodeColor(node: GraphNode): string {
  if (node.is_focus) return NODE_COLORS.focus
  if (node.is_known_blacklist) return NODE_COLORS.blacklist
  if (node.risk_level === "high" || node.risk_level === "critical") return NODE_COLORS.high_risk
  return NODE_COLORS[node.type] ?? "#b0bec5"
}

function edgeColor(target: GraphNode | undefined): string {
  if (!target) return "#cfd8dc"
  if (target.is_known_blacklist) return "#e53935"
  if (target.risk_level === "high" || target.risk_level === "critical") return "#fb8c00"
  return "#cfd8dc"
}

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

interface Props {
  nodes: GraphNode[]
  edges: GraphEdge[]
  focusNodeId: string
}

export function GraphCanvas({ nodes, edges, focusNodeId }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const cyRef = useRef<unknown>(null)

  useEffect(() => {
    if (!containerRef.current || nodes.length === 0) return

    // Dynamically import cytoscape to avoid SSR issues
    import("cytoscape").then((cytoscapeModule) => {
      const cytoscape = cytoscapeModule.default

      // Destroy previous instance
      if (cyRef.current) {
        (cyRef.current as { destroy: () => void }).destroy()
      }

      const nodeMap = new Map(nodes.map((n) => [n.id, n]))

      const cyNodes = nodes.map((node) => ({
        data: {
          id: node.id,
          label: node.is_focus ? `★ ${node.label}` : node.label,
          color: nodeColor(node),
          size: node.is_focus ? 50 : node.type !== "user" ? 32 : 24,
          tooltip: [
            node.is_focus ? "【調查對象】" : "",
            `類型：${TYPE_ZH[node.type] ?? node.type}`,
            `關聯層數：${node.hop}`,
            `風險等級：${node.risk_level ?? "正常"}`,
            `黑名單：${node.is_known_blacklist ? "是" : "否"}`,
          ].filter(Boolean).join("\n"),
          is_focus: String(node.is_focus),
          node_type: node.type,
        },
      }))

      const cyEdges = edges.map((edge) => ({
        data: {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          color: edgeColor(nodeMap.get(edge.target)),
          tooltip: RELATION_ZH[edge.relation_type] ?? edge.relation_type,
        },
      }))

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const cy = (cytoscape as any)({
        container: containerRef.current,
        elements: [...cyNodes, ...cyEdges],
        style: [
          {
            selector: "node",
            style: {
              "background-color": "data(color)",
              "width": "data(size)",
              "height": "data(size)",
              "label": "data(label)",
              "color": "#1a1d2e",
              "font-size": "11px",
              "text-valign": "bottom",
              "text-halign": "center",
              "text-margin-y": 4,
              "text-max-width": "80px",
              "text-wrap": "ellipsis",
              "border-width": 2,
              "border-color": "#ffffff",
              "border-opacity": 0.8,
            },
          },
          {
            selector: "node[is_focus = 'true']",
            style: {
              "border-width": 3,
              "border-color": "#5c6bc0",
              "border-opacity": 0.4,
              "font-weight": "bold",
              "font-size": "13px",
            },
          },
          {
            selector: "edge",
            style: {
              "line-color": "data(color)",
              "target-arrow-color": "data(color)",
              "target-arrow-shape": "triangle",
              "arrow-scale": 0.8,
              "width": 1.5,
              "curve-style": "bezier",
              "opacity": 0.7,
            },
          },
          {
            selector: "node:selected",
            style: { "border-width": 3, "border-color": "#5c6bc0", "border-opacity": 1 },
          },
          {
            selector: ".highlighted",
            style: { "opacity": 1, "width": 3 },
          },
          {
            selector: ".faded",
            style: { "opacity": 0.15 },
          },
        ],
        layout: {
          name: "concentric",
          concentric: (node: { data: (key: string) => string }) => {
            if (node.data("is_focus") === "true") return 3
            if (node.data("node_type") !== "user") return 2
            return 1
          },
          levelWidth: () => 1,
          minNodeSpacing: 50,
          padding: 40,
          animate: true,
          animationDuration: 600,
        } as object,
      })

      // Hover tooltip
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      cy.on("mouseover", "node", (evt: any) => {
        const node = evt.target
        node.tippy?.destroy()
        const tip = document.createElement("div")
        tip.style.cssText = "position:absolute;background:#1a1d2e;color:#fff;padding:6px 10px;border-radius:6px;font-size:12px;pointer-events:none;white-space:pre;z-index:9999;line-height:1.6"
        tip.textContent = node.data("tooltip")
        document.body.appendChild(tip)
        const moveHandler = (e: MouseEvent) => {
          tip.style.left = e.clientX + 12 + "px"
          tip.style.top = e.clientY + 12 + "px"
        }
        window.addEventListener("mousemove", moveHandler)
        node.on("mouseout", () => {
          tip.remove()
          window.removeEventListener("mousemove", moveHandler)
        })
      })

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      cy.on("mouseover", "edge", (evt: any) => {
        const edge = evt.target
        const tip = document.createElement("div")
        tip.style.cssText = "position:absolute;background:#1a1d2e;color:#fff;padding:5px 10px;border-radius:6px;font-size:12px;pointer-events:none;z-index:9999"
        tip.textContent = edge.data("tooltip")
        document.body.appendChild(tip)
        const moveHandler = (e: MouseEvent) => {
          tip.style.left = e.clientX + 12 + "px"
          tip.style.top = e.clientY + 12 + "px"
        }
        window.addEventListener("mousemove", moveHandler)
        edge.on("mouseout", () => {
          tip.remove()
          window.removeEventListener("mousemove", moveHandler)
        })
      })

      cyRef.current = cy
    })

    return () => {
      if (cyRef.current) {
        (cyRef.current as { destroy: () => void }).destroy()
        cyRef.current = null
      }
    }
  }, [nodes, edges, focusNodeId])

  if (nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-[520px] bg-white rounded-xl border border-[#e5e7eb] text-[#9ca3af]">
        目前篩選條件下沒有可顯示的關聯
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className="w-full rounded-xl border border-[#e5e7eb] bg-white"
      style={{ height: "520px" }}
    />
  )
}
