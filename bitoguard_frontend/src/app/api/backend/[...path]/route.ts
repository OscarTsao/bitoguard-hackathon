import { NextRequest, NextResponse } from "next/server"

const API_BASE = (process.env.BITOGUARD_INTERNAL_API_BASE ?? "http://127.0.0.1:8001").replace(/\/$/, "")

export const dynamic = "force-dynamic"

async function proxy(request: NextRequest, path: string[]) {
  try {
    const upstreamUrl = new URL(`${API_BASE}/${path.join("/")}`)
    upstreamUrl.search = request.nextUrl.search

    const headers = new Headers()
    const contentType = request.headers.get("content-type")
    if (contentType) {
      headers.set("content-type", contentType)
    }

    const upstreamResponse = await fetch(upstreamUrl, {
      method: request.method,
      headers,
      body: request.method === "GET" || request.method === "HEAD" ? undefined : await request.text(),
      cache: "no-store",
    })

    return new NextResponse(upstreamResponse.body, {
      status: upstreamResponse.status,
      headers: {
        "content-type": upstreamResponse.headers.get("content-type") ?? "application/json; charset=utf-8",
        "cache-control": "no-store",
      },
    })
  } catch {
    return NextResponse.json(
      { message: "Unable to reach internal API" },
      { status: 502, headers: { "cache-control": "no-store" } },
    )
  }
}

export async function GET(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params
  return proxy(request, path)
}

export async function POST(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params
  return proxy(request, path)
}
