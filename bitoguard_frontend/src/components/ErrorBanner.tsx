export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-[13px] text-[#e53935]">
      {message}
    </div>
  )
}
