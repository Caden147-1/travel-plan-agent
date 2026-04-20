"use client"

import { useEffect, useRef, useState } from "react"
import type { FormEvent, KeyboardEvent } from "react"
import { ExternalLink, Loader2, Menu, Plane, Plus, Send, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  reportUrl?: string
}

interface PlanResponse {
  success: boolean
  error?: string
  guidance?: string
  warnings?: string[]
  report_url?: string
  trip_summary?: {
    origin?: string
    destination?: string[]
    duration_days?: number
    budget?: {
      total?: number
      currency?: string
    }
  }
  itinerary_summary?: {
    total_days?: number
    estimated_total_cost?: number
    must_visit_coverage?: string
  }
}

interface HistoryItem {
  id: string
  request: string
  summary?: string
  reportUrl?: string
  createdAt: string
  status?: "pending" | "ready" | "error"
  error?: string
}

const apiBase = process.env.NEXT_PUBLIC_TRAVEL_API_BASE || "http://127.0.0.1:8000"
const historyStorageKey = "travelai-plan-history"

const suggestedPrompts = [
  "I want to travel from Beijing to Chengdu from 6.19 to 6.21 with a budget of 6000 CNY. I love local food.",
  "Plan a 5 day Tokyo trip from Shanghai for 2 people with a 15000 CNY budget.",
  "Help me plan a budget-friendly Paris trip with clean central hotels and local restaurants.",
  "I want a 4 day Chengdu food trip with pandas, Jinli, and relaxed sightseeing.",
]

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const activeHistoryIdRef = useRef<string | null>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isLoading])

  useEffect(() => {
    if (!textareaRef.current) return
    textareaRef.current.style.height = "auto"
    textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`
  }, [input])

  useEffect(() => {
    try {
      const savedHistory = window.localStorage.getItem(historyStorageKey)
      if (!savedHistory) return
      const parsedHistory = JSON.parse(savedHistory) as HistoryItem[]
      if (Array.isArray(parsedHistory)) {
        setHistory(parsedHistory)
      }
    } catch {
      setHistory([])
    }
  }, [])

  function setActiveConversation(id: string | null) {
    activeHistoryIdRef.current = id
    setActiveHistoryId(id)
  }

  function saveHistory(updater: (previous: HistoryItem[]) => HistoryItem[]) {
    setHistory((previous) => {
      const nextHistory = updater(previous).slice(0, 20)
      window.localStorage.setItem(historyStorageKey, JSON.stringify(nextHistory))
      return nextHistory
    })
  }

  async function handleSubmit(e: FormEvent | KeyboardEvent) {
    e.preventDefault()
    const request = input.trim()
    if (!request || isLoading) return
    setInput("")

    const historyId = crypto.randomUUID()
    setActiveConversation(historyId)
    setMessages([
      {
        id: crypto.randomUUID(),
        role: "user",
        content: request,
      },
    ])
    saveHistory((previous) => [
      {
        id: historyId,
        request,
        createdAt: new Date().toISOString(),
        status: "pending",
      },
      ...previous.filter((item) => item.request !== request),
    ])
    setIsLoading(true)

    try {
      const response = await fetch(`${apiBase}/api/plan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: request }),
      })
      const data = (await response.json()) as PlanResponse

      if (!response.ok || !data.success) {
        const guidance = data.guidance || buildFallbackGuidance(data.error)
        saveHistory((previous) =>
          previous.map((item) =>
            item.id === historyId
              ? {
                  ...item,
                  summary: guidance,
                  status: "error",
                  error: data.error || "Planning failed.",
                }
              : item,
          ),
        )
        if (activeHistoryIdRef.current === historyId) {
          setMessages([
            {
              id: crypto.randomUUID(),
              role: "user",
              content: request,
            },
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content: guidance,
            },
          ])
        }
        return
      }

      const summary = formatSummary(data)
      saveHistory((previous) =>
        previous.map((item) =>
          item.id === historyId
            ? {
                ...item,
                summary,
                reportUrl: data.report_url,
                status: "ready",
              }
            : item,
        ),
      )
      if (activeHistoryIdRef.current === historyId) {
        setMessages([
          {
            id: crypto.randomUUID(),
            role: "user",
            content: request,
          },
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: summary,
            reportUrl: data.report_url,
          },
        ])
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error."
      const errorSummary = `The plan could not be generated.\n\n${message}`
      saveHistory((previous) =>
        previous.map((item) =>
          item.id === historyId
            ? {
                ...item,
                summary: errorSummary,
                status: "error",
                error: message,
              }
            : item,
        ),
      )
      if (activeHistoryIdRef.current === historyId) {
        setMessages([
          {
            id: crypto.randomUUID(),
            role: "user",
            content: request,
          },
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: errorSummary,
          },
        ])
      }
    } finally {
      setIsLoading(false)
    }
  }

  function handleHistoryClick(item: HistoryItem) {
    setActiveConversation(item.id)
    const nextMessages: Message[] = [
      {
        id: crypto.randomUUID(),
        role: "user",
        content: item.request,
      },
    ]

    if (getHistoryStatus(item) === "pending") {
      nextMessages.push({
        id: crypto.randomUUID(),
        role: "assistant",
        content: "This report is still generating. You can keep browsing history and come back when it is ready.",
      })
    } else {
      nextMessages.push({
        id: crypto.randomUUID(),
        role: "assistant",
        content: item.summary || "Plan generated.",
        reportUrl: item.reportUrl,
      })
    }

    setMessages(nextMessages)
    setInput("")
    textareaRef.current?.focus()
  }

  function handlePromptClick(prompt: string) {
    setInput(prompt)
    textareaRef.current?.focus()
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      void handleSubmit(e)
    }
  }

  const activeHistoryItem = activeHistoryId ? history.find((item) => item.id === activeHistoryId) : null
  const showActiveLoading = activeHistoryItem?.status === "pending"

  return (
    <div className="flex h-screen bg-background">
      <aside
        className={cn(
          "hidden md:flex flex-col bg-sidebar border-r border-sidebar-border transition-all duration-300",
          sidebarOpen ? "w-72" : "w-0 overflow-hidden",
        )}
      >
        <div className="p-4">
          <Button
            variant="outline"
            className="w-full justify-start gap-2 bg-sidebar-accent hover:bg-sidebar-accent/80 border-sidebar-border text-sidebar-foreground"
            onClick={() => {
              setMessages([])
              setInput("")
              setActiveConversation(null)
            }}
          >
            <Plus className="h-4 w-4" />
            New plan
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto px-4">
          <HistoryList history={history} activeId={activeHistoryId} onSelect={handleHistoryClick} />
        </div>

        <div className="p-4 border-t border-sidebar-border text-xs text-muted-foreground">
          Backend: {apiBase}
        </div>
      </aside>

      {mobileSidebarOpen && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div className="absolute inset-0 bg-background/80 backdrop-blur-sm" onClick={() => setMobileSidebarOpen(false)} />
          <aside className="absolute left-0 top-0 bottom-0 w-72 bg-sidebar border-r border-sidebar-border p-4">
            <div className="flex items-center justify-between mb-4">
              <span className="font-semibold text-sidebar-foreground">TravelAI</span>
              <Button variant="ghost" size="icon" onClick={() => setMobileSidebarOpen(false)}>
                <X className="h-5 w-5" />
              </Button>
            </div>
            <HistoryList
              history={history}
              activeId={activeHistoryId}
              onSelect={(item) => {
                handleHistoryClick(item)
                setMobileSidebarOpen(false)
              }}
            />
          </aside>
        </div>
      )}

      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-14 border-b border-border flex items-center px-4 gap-3">
          <Button variant="ghost" size="icon" className="md:hidden" onClick={() => setMobileSidebarOpen(true)}>
            <Menu className="h-5 w-5" />
          </Button>
          <Button variant="ghost" size="icon" className="hidden md:flex" onClick={() => setSidebarOpen(!sidebarOpen)}>
            <Menu className="h-5 w-5" />
          </Button>
          <div className="flex items-center gap-2">
            <Plane className="h-5 w-5 text-primary" />
            <span className="font-semibold">TravelAI Planner</span>
          </div>
          <span className="text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded">Local</span>
        </header>

        <main className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center p-4">
              <div className="max-w-2xl w-full text-center space-y-6">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-lg bg-primary/10">
                  <Plane className="h-8 w-8 text-primary" />
                </div>
                <div className="space-y-2">
                  <h1 className="text-2xl font-semibold text-balance">Plan a trip from one message</h1>
                  <p className="text-muted-foreground text-balance">
                    Enter your origin, destination, dates, budget, and preferences. The backend will run the router pipeline and create an HTML report.
                  </p>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-left">
                  {suggestedPrompts.slice(0, 4).map((prompt) => (
                    <button
                      key={prompt}
                      onClick={() => handlePromptClick(prompt)}
                      className="rounded-lg border border-border bg-card hover:bg-secondary transition-colors p-4 text-sm"
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto py-6 px-4 space-y-6">
              {messages.map((message) => (
                <div key={message.id} className={cn("flex gap-4", message.role === "user" && "justify-end")}>
                  {message.role === "assistant" && (
                    <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                      <Plane className="h-4 w-4 text-primary-foreground" />
                    </div>
                  )}
                  <div
                    className={cn(
                      "max-w-[82%] rounded-lg px-4 py-3",
                      message.role === "user" ? "bg-primary text-primary-foreground" : "bg-card border border-border",
                    )}
                  >
                    <div className="text-sm whitespace-pre-wrap">{message.content}</div>
                    {message.reportUrl && (
                      <a
                        href={message.reportUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="mt-3 inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
                      >
                        Open report
                        <ExternalLink className="h-4 w-4" />
                      </a>
                    )}
                  </div>
                  {message.role === "user" && (
                    <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-secondary flex items-center justify-center text-sm font-medium">
                      U
                    </div>
                  )}
                </div>
              ))}
              {showActiveLoading && (
                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                    <Plane className="h-4 w-4 text-primary-foreground" />
                  </div>
                  <div className="bg-card border border-border rounded-lg px-4 py-3 text-sm flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Running router pipeline and generating the report...
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </main>

        <div className="border-t border-border p-4">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
            <div className="relative flex items-end gap-2 bg-input rounded-lg border border-border p-2">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Describe your trip..."
                className="flex-1 bg-transparent resize-none outline-none text-sm px-2 py-2 max-h-[180px] min-h-[40px]"
                rows={1}
              />
              <Button type="submit" size="icon" disabled={!input.trim() || isLoading} className="flex-shrink-0 h-10 w-10 rounded-lg">
                {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              </Button>
            </div>
            <p className="text-xs text-center text-muted-foreground mt-2">
              The report is generated locally from your Python router pipeline.
            </p>
          </form>
        </div>
      </div>
    </div>
  )
}

function HistoryList({
  history,
  activeId,
  onSelect,
}: {
  history: HistoryItem[]
  activeId: string | null
  onSelect: (item: HistoryItem) => void
}) {
  return (
    <div>
      <p className="text-xs font-medium text-muted-foreground mb-3">History</p>
      {history.length === 0 ? (
        <div className="rounded-lg border border-sidebar-border bg-background px-3 py-4 text-sm text-muted-foreground">
          No history yet.
        </div>
      ) : (
        <div className="space-y-2">
          {history.map((item) => {
            const status = getHistoryStatus(item)
            return (
              <button
                key={item.id}
                onClick={() => onSelect(item)}
                className={cn(
                  "w-full text-left rounded-lg border px-3 py-2 text-sm text-sidebar-foreground hover:bg-sidebar-accent transition-colors",
                  activeId === item.id ? "border-primary bg-sidebar-accent" : "border-sidebar-border bg-background",
                )}
              >
                <span className="block line-clamp-3">{item.request}</span>
                <span className="mt-2 flex items-center justify-between gap-2 text-xs text-muted-foreground">
                  <span>{formatHistoryDate(item.createdAt)}</span>
                  <HistoryStatusBadge status={status} />
                </span>
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

function HistoryStatusBadge({ status }: { status: "pending" | "ready" | "error" }) {
  if (status === "pending") {
    return (
      <span className="inline-flex items-center gap-1 rounded bg-muted px-1.5 py-0.5">
        <Loader2 className="h-3 w-3 animate-spin" />
        Generating
      </span>
    )
  }

  if (status === "error") {
    return <span className="rounded bg-destructive/10 px-1.5 py-0.5 text-destructive">Failed</span>
  }

  return <span className="rounded bg-primary/10 px-1.5 py-0.5 text-primary">Ready</span>
}

function getHistoryStatus(item: HistoryItem): "pending" | "ready" | "error" {
  if (item.status) return item.status
  if (item.summary || item.reportUrl) return "ready"
  return "pending"
}

function formatHistoryDate(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ""
  return date.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })
}

function validateRequest(request: string): string | null {
  const dateError = validateTravelDates(request)
  if (dateError) return dateError

  const budgetMatch = request.match(/\b(?:budget|预算|费用)\D{0,12}(\d+(?:\.\d+)?)/i)
  if (budgetMatch) {
    const budget = Number(budgetMatch[1])
    if (!Number.isFinite(budget) || budget <= 0) {
      return "I can't create a plan with an invalid budget. Please enter a budget greater than 0."
    }
  }

  const peopleMatch = request.match(/\b(\d+)\s*(?:people|person|travelers|travellers|人)\b/i)
  if (peopleMatch) {
    const people = Number(peopleMatch[1])
    if (!Number.isInteger(people) || people <= 0 || people > 50) {
      return "I can't create a plan with that traveler count. Please enter a number of travelers between 1 and 50."
    }
  }

  const daysMatch = request.match(/\b(\d+)\s*(?:day|days|天)\b/i)
  if (daysMatch) {
    const days = Number(daysMatch[1])
    if (!Number.isInteger(days) || days <= 0 || days > 60) {
      return "I can't create a plan with that trip length. Please enter a duration between 1 and 60 days."
    }
  }

  return null
}

function buildFallbackGuidance(error?: string): string {
  return [
    "I can help turn this into a complete travel request.",
    "",
    error ? `What happened: ${error}` : null,
    "Please include these information points: departure city, destination city or country, travel dates or trip length, number of travelers, total budget and currency, and any hotel, food, transport, attraction, or pace preferences.",
    "",
    "You can reply like this: I want to travel from Shanghai to Paris from 2026-06-10 to 2026-06-15 for 2 people with a budget of 15000 CNY. I prefer clean central hotels, local food, and a relaxed pace.",
  ]
    .filter(Boolean)
    .join("\n")
}

function validateTravelDates(request: string): string | null {
  const range = extractDateRange(request)
  if (!range) return null

  const today = startOfDay(new Date())
  if (range.start.getTime() < today.getTime()) {
    return `I can't create a plan for dates that have already passed. Please choose dates from ${formatIsoDate(today)} onward.`
  }

  if (range.end.getTime() < range.start.getTime()) {
    return "I can't create a plan where the end date is before the start date. Please check the travel dates."
  }

  const maxDate = new Date(today)
  maxDate.setFullYear(maxDate.getFullYear() + 1)
  if (range.start.getTime() > maxDate.getTime()) {
    return "I can't create a plan that far in the future. Please choose dates within the next year."
  }

  return null
}

function extractDateRange(request: string): { start: Date; end: Date } | null {
  const fullDatePattern = /\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b/g
  const fullDates = [...request.matchAll(fullDatePattern)].map((match) =>
    buildDate(Number(match[1]), Number(match[2]), Number(match[3])),
  )

  if (fullDates.length >= 2 && fullDates[0] && fullDates[1]) {
    return { start: fullDates[0], end: fullDates[1] }
  }

  const monthDayPattern = /(?:^|\D)(\d{1,2})[./](\d{1,2})(?=\D|$)/g
  const monthDayDates = [...request.matchAll(monthDayPattern)].map((match) =>
    buildDate(new Date().getFullYear(), Number(match[1]), Number(match[2])),
  )

  if (monthDayDates.length >= 2 && monthDayDates[0] && monthDayDates[1]) {
    return { start: monthDayDates[0], end: monthDayDates[1] }
  }

  return null
}

function buildDate(year: number, month: number, day: number): Date | null {
  const date = new Date(year, month - 1, day)
  if (date.getFullYear() !== year || date.getMonth() !== month - 1 || date.getDate() !== day) {
    return null
  }
  return startOfDay(date)
}

function startOfDay(date: Date): Date {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate())
}

function formatIsoDate(date: Date): string {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, "0")
  const day = String(date.getDate()).padStart(2, "0")
  return `${year}-${month}-${day}`
}

function formatSummary(data: PlanResponse): string {
  const trip = data.trip_summary || {}
  const destinations = trip.destination?.length ? trip.destination.join(", ") : "destination"
  const budget = trip.budget?.total ? `${trip.budget.total} ${trip.budget.currency || "CNY"}` : "not specified"
  const warnings = data.warnings?.length ? `\n\nWarnings:\n${data.warnings.slice(0, 3).map((item) => `- ${item}`).join("\n")}` : ""

  return [
    "Plan generated.",
    "",
    `Route: ${trip.origin || "Origin"} -> ${destinations}`,
    `Days: ${trip.duration_days || data.itinerary_summary?.total_days || "unknown"}`,
    `Budget: ${budget}`,
    data.report_url ? "The HTML report is ready." : "The pipeline finished, but no report URL was returned.",
    warnings,
  ].filter(Boolean).join("\n")
}
