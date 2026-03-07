import type { Metadata } from "next"
import "./globals.css"
import { Sidebar } from "@/components/Sidebar"
import { Providers } from "@/lib/providers"

export const metadata: Metadata = {
  title: "BitoGuard — 風控分析平台",
  description: "BitoGuard 風險偵測與關聯圖分析",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-TW">
      <body>
        <Providers>
          <Sidebar />
          <main
            className="min-h-screen p-6"
            style={{ marginLeft: "var(--sidebar-width)" }}
          >
            {children}
          </main>
        </Providers>
      </body>
    </html>
  )
}
