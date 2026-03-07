"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { AlertTriangle, Activity, Network, BarChart2, Users } from "lucide-react"
import { cn } from "@/lib/utils"

const NAV_ITEMS = [
  { href: "/alerts",    label: "警示中心",    icon: AlertTriangle },
  { href: "/alerts/report", label: "風險診斷",  icon: Activity },
  { href: "/users",     label: "用戶全貌",    icon: Users },
  { href: "/graph",     label: "關聯圖探索",  icon: Network },
  { href: "/model-ops", label: "模型指標",    icon: BarChart2 },
]

export function Sidebar() {
  const pathname = usePathname()

  return (
    <aside
      className="fixed top-0 left-0 h-screen bg-white border-r border-[#e5e7eb] flex flex-col z-10"
      style={{ width: "var(--sidebar-width)" }}
    >
      {/* Logo */}
      <div className="px-5 py-5 border-b border-[#e5e7eb]">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-[#5c6bc0] flex items-center justify-center">
            <Network size={16} className="text-white" />
          </div>
          <span className="font-semibold text-[15px] text-[#1a1d2e] tracking-tight">BitoGuard</span>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5">
        <p className="px-2 pb-2 text-[11px] font-semibold text-[#9ca3af] uppercase tracking-wider">分析工具</p>
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || pathname.startsWith(href + "/")
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-2.5 px-2 py-2 rounded-lg text-[13px] font-medium transition-colors",
                active
                  ? "bg-[#eef0fa] text-[#5c6bc0]"
                  : "text-[#6b7280] hover:bg-[#f4f6f9] hover:text-[#1a1d2e]"
              )}
            >
              <Icon size={16} />
              {label}
            </Link>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="px-5 py-3 border-t border-[#e5e7eb]">
        <p className="text-[11px] text-[#9ca3af]">BitoGuard v1.0</p>
      </div>
    </aside>
  )
}
