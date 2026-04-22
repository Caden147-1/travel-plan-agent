"use client"

import { useEffect, useRef, useState } from "react"
import type { FormEvent, KeyboardEvent } from "react"
import { ExternalLink, Loader2, Menu, Plane, Plus, Send, Trash2, X } from "lucide-react"
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
      source_total?: number
      source_currency?: string
      conversion_rate_to_cny?: number
      conversion_note?: string
    }
  }
  itinerary_summary?: {
    total_days?: number
    estimated_total_cost?: number
    must_visit_coverage?: string
  }
}

interface GuidanceResponse {
  success: boolean
  ready?: boolean
  guidance?: string
  error?: string
}

interface HistoryItem {
  id: string
  request: string
  messages?: Message[]
  summary?: string
  reportUrl?: string
  createdAt: string
  updatedAt?: string
  status?: "pending" | "ready" | "error" | "cancelled"
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

const thinkingSteps = [
  { afterSeconds: 0, text: "Understanding your route, dates, budget, and preferences" },
  { afterSeconds: 4, text: "Checking the request for missing or conflicting details" },
  { afterSeconds: 8, text: "Core precheck: if flight + hotel + three meals exceed 70% of budget, requirements will be lowered" },
  { afterSeconds: 14, text: "Searching flights, hotels, food, and attractions" },
  { afterSeconds: 24, text: "The default plan may exceed the budget, switching to a lower-cost option" },
  { afterSeconds: 36, text: "The default plan exceeded the budget, preparing an updated plan" },
  { afterSeconds: 52, text: "Live search and itinerary generation are still running; report writing is usually quick" },
  { afterSeconds: 90, text: "Still waiting for flight, hotel, places search, or the budget retry to return" },
]

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [loadingHistoryId, setLoadingHistoryId] = useState<string | null>(null)
  const [thinkingStepIndex, setThinkingStepIndex] = useState(0)
  const [thinkingElapsedSeconds, setThinkingElapsedSeconds] = useState(0)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const activeHistoryIdRef = useRef<string | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const loadingHistoryIdRef = useRef<string | null>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isLoading, thinkingStepIndex])

  useEffect(() => {
    if (!isLoading) {
      setThinkingStepIndex(0)
      setThinkingElapsedSeconds(0)
      return
    }

    const startedAt = Date.now()
    const interval = window.setInterval(() => {
      const elapsedSeconds = Math.floor((Date.now() - startedAt) / 1000)
      setThinkingElapsedSeconds(elapsedSeconds)
      setThinkingStepIndex(getThinkingStepIndex(elapsedSeconds))
    }, 500)

    return () => window.clearInterval(interval)
  }, [isLoading])

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
        setHistory(parsedHistory.map(normalizeHistoryItem))
      }
    } catch {
      setHistory([])
    }
  }, [])

  function setActiveConversation(id: string | null) {
    activeHistoryIdRef.current = id
    setActiveHistoryId(id)
  }

  function setLoadingConversation(id: string | null) {
    loadingHistoryIdRef.current = id
    setLoadingHistoryId(id)
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
    if (isLoading) {
      handleStopThinking()
      return
    }

    const request = input.trim()
    if (!request) return
    setInput("")

    const historyId = activeHistoryIdRef.current || crypto.randomUUID()
    const existingHistory = history.find((item) => item.id === historyId)
    const previousMessages = existingHistory?.messages?.length
      ? existingHistory.messages
      : messages.length
        ? messages
        : []
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: request,
    }
    const submittedMessages = [...previousMessages, userMessage]
    const plannerRequest = buildPlannerRequest(submittedMessages)
    const requestValidationError = validateRequest(plannerRequest)

    setActiveConversation(historyId)
    setMessages(submittedMessages)
    saveHistory((previous) => {
      const now = new Date().toISOString()
      const existing = previous.find((item) => item.id === historyId)
      const nextItem: HistoryItem = {
        ...existing,
        id: historyId,
        request: existing?.request || request,
        messages: submittedMessages,
        createdAt: existing?.createdAt || now,
        updatedAt: now,
        status: "pending",
        error: undefined,
      }
      return [nextItem, ...previous.filter((item) => item.id !== historyId)]
    })
    setIsLoading(true)
    setLoadingConversation(historyId)
    setThinkingStepIndex(0)
    setThinkingElapsedSeconds(0)
    const abortController = new AbortController()
    abortControllerRef.current = abortController

    try {
      if (requestValidationError) {
        const guidance = buildFallbackGuidance(requestValidationError)
        const nextMessages: Message[] = [
          ...submittedMessages,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: guidance,
          },
        ]
        saveHistory((previous) =>
          previous.map((item) =>
            item.id === historyId
              ? {
                  ...item,
                  messages: nextMessages,
                  summary: guidance,
                  status: "error",
                  error: requestValidationError,
                  updatedAt: new Date().toISOString(),
                }
              : item,
          ),
        )
        if (activeHistoryIdRef.current === historyId) {
          setMessages(nextMessages)
        }
        return
      }

      const guidanceResponse = await fetch(`${apiBase}/api/guidance`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: plannerRequest }),
        signal: abortController.signal,
      })
      const guidanceData = (await guidanceResponse.json()) as GuidanceResponse

      if (!guidanceResponse.ok || !guidanceData.success) {
        throw new Error(guidanceData.error || "Request intake failed.")
      }

      if (!guidanceData.ready) {
        const guidance = guidanceData.guidance || buildFallbackGuidance()
        const nextMessages: Message[] = [
          ...submittedMessages,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: guidance,
          },
        ]
        saveHistory((previous) =>
          previous.map((item) =>
            item.id === historyId
              ? {
                  ...item,
                  messages: nextMessages,
                  summary: guidance,
                  status: "error",
                  error: guidanceData.error || "The request needs more details.",
                  updatedAt: new Date().toISOString(),
                }
              : item,
          ),
        )
        if (activeHistoryIdRef.current === historyId) {
          setMessages(nextMessages)
        }
        return
      }

      const response = await fetch(`${apiBase}/api/plan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: plannerRequest }),
        signal: abortController.signal,
      })
      const data = (await response.json()) as PlanResponse

      if (!response.ok || !data.success) {
        const guidance = data.guidance || buildFallbackGuidance(data.error)
        const nextMessages: Message[] = [
          ...submittedMessages,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: guidance,
          },
        ]
        saveHistory((previous) =>
          previous.map((item) =>
            item.id === historyId
              ? {
                  ...item,
                  messages: nextMessages,
                  summary: guidance,
                  status: "error",
                  error: data.error || "Planning failed.",
                  updatedAt: new Date().toISOString(),
                }
              : item,
          ),
        )
        if (activeHistoryIdRef.current === historyId) {
          setMessages(nextMessages)
        }
        return
      }

      const summary = formatSummary(data)
      const nextMessages: Message[] = [
        ...submittedMessages,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: summary,
          reportUrl: data.report_url,
        },
      ]
      saveHistory((previous) =>
        previous.map((item) =>
          item.id === historyId
            ? {
                ...item,
                messages: nextMessages,
                summary,
                reportUrl: data.report_url,
                status: "ready",
                updatedAt: new Date().toISOString(),
              }
            : item,
        ),
      )
      if (activeHistoryIdRef.current === historyId) {
        setMessages(nextMessages)
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        const cancelledSummary = "Stopped. You can revise the request and send it again when you are ready."
        const nextMessages: Message[] = [
          ...submittedMessages,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: cancelledSummary,
          },
        ]
        saveHistory((previous) =>
          previous.map((item) =>
            item.id === historyId
              ? {
                  ...item,
                  messages: nextMessages,
                  summary: cancelledSummary,
                  status: "cancelled",
                  error: undefined,
                  updatedAt: new Date().toISOString(),
                }
              : item,
          ),
        )
        if (activeHistoryIdRef.current === historyId) {
          setMessages(nextMessages)
        }
        return
      }

      const message = error instanceof Error ? error.message : "Unknown error."
      const errorSummary = `The plan could not be generated.\n\n${message}`
      const nextMessages: Message[] = [
        ...submittedMessages,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: errorSummary,
        },
      ]
      saveHistory((previous) =>
        previous.map((item) =>
          item.id === historyId
            ? {
                ...item,
                messages: nextMessages,
                summary: errorSummary,
                status: "error",
                error: message,
                updatedAt: new Date().toISOString(),
              }
            : item,
        ),
      )
      if (activeHistoryIdRef.current === historyId) {
        setMessages(nextMessages)
      }
    } finally {
      if (loadingHistoryIdRef.current === historyId) {
        abortControllerRef.current = null
        setLoadingConversation(null)
        setIsLoading(false)
      }
    }
  }

  function handleStopThinking() {
    const historyId = loadingHistoryIdRef.current
    if (!historyId) return
    abortControllerRef.current?.abort()
  }

  function handleHistoryClick(item: HistoryItem) {
    setActiveConversation(item.id)
    if (item.messages?.length) {
      setMessages(item.messages)
      setInput("")
      textareaRef.current?.focus()
      return
    }

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

  function handleHistoryDelete(id: string) {
    saveHistory((previous) => previous.filter((item) => item.id !== id))
    if (activeHistoryIdRef.current === id) {
      setMessages([])
      setInput("")
      setActiveConversation(null)
    }
  }

  function handlePromptClick(prompt: string) {
    setInput(prompt)
    textareaRef.current?.focus()
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      if (isLoading) {
        handleStopThinking()
        return
      }
      void handleSubmit(e)
    }
  }

  const activeHistoryItem = activeHistoryId ? history.find((item) => item.id === activeHistoryId) : null
  const showActiveLoading = activeHistoryItem?.status === "pending" && activeHistoryId === loadingHistoryId

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
          <HistoryList history={history} activeId={activeHistoryId} onSelect={handleHistoryClick} onDelete={handleHistoryDelete} />
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
              onDelete={handleHistoryDelete}
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
                    <div className="text-sm whitespace-pre-wrap">{renderMessageContent(message.content)}</div>
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
                  <ThinkingPanel
                    activeStepIndex={thinkingStepIndex}
                    elapsedSeconds={thinkingElapsedSeconds}
                  />
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
              <Button
                type="submit"
                size="icon"
                disabled={!input.trim() && !isLoading}
                className="flex-shrink-0 h-10 w-10 rounded-lg"
                aria-label={isLoading ? "Stop thinking" : "Send message"}
                title={isLoading ? "Stop" : "Send"}
              >
                {isLoading ? <X className="h-4 w-4" /> : <Send className="h-4 w-4" />}
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

function normalizeHistoryItem(item: HistoryItem): HistoryItem {
  return {
    ...item,
    messages: item.messages?.map((message) => ({
      ...message,
      content: renderMessageContent(message.content),
    })),
    summary: item.summary ? renderMessageContent(item.summary) : item.summary,
  }
}

function HistoryList({
  history,
  activeId,
  onSelect,
  onDelete,
}: {
  history: HistoryItem[]
  activeId: string | null
  onSelect: (item: HistoryItem) => void
  onDelete: (id: string) => void
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
              <div
                key={item.id}
                className={cn(
                  "group flex items-start gap-2 rounded-lg border bg-background p-2 text-sm text-sidebar-foreground hover:bg-sidebar-accent transition-colors",
                  activeId === item.id ? "border-primary bg-sidebar-accent" : "border-sidebar-border bg-background",
                )}
              >
                <button type="button" onClick={() => onSelect(item)} className="min-w-0 flex-1 text-left">
                  <span className="block line-clamp-3">{item.request}</span>
                  <span className="mt-2 flex items-center justify-between gap-2 text-xs text-muted-foreground">
                    <span>{formatHistoryDate(item.createdAt)}</span>
                    <HistoryStatusBadge status={status} />
                  </span>
                </button>
                <button
                  type="button"
                  aria-label="Delete history"
                  title="Delete history"
                  onClick={() => onDelete(item.id)}
                  className="rounded-md p-1.5 text-muted-foreground opacity-100 transition-colors hover:bg-destructive/10 hover:text-destructive md:opacity-0 md:group-hover:opacity-100"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function ThinkingPanel({
  activeStepIndex,
  elapsedSeconds,
}: {
  activeStepIndex: number
  elapsedSeconds: number
}) {
  const currentStep = thinkingSteps[activeStepIndex]?.text || thinkingSteps[0].text

  return (
    <div className="max-w-[82%] rounded-lg border border-border bg-card px-4 py-3 text-sm">
      <div className="flex items-center gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <span className="thinking-dot h-3 w-3 flex-shrink-0 rounded-full border border-primary/25 bg-primary/35" />
          <div className="flex items-center gap-2 font-medium">
            <span>Thinking</span>
            <span className="text-xs font-normal text-muted-foreground">{formatElapsedTime(elapsedSeconds)}</span>
          </div>
        </div>
      </div>
      <div className="mt-3 max-w-full overflow-hidden pl-6 text-xs text-muted-foreground/55">
        <span className="thinking-flow inline-block whitespace-nowrap">{currentStep}</span>
      </div>
      <style jsx>{`
        .thinking-dot {
          animation: thinking-breathe 2.2s ease-in-out infinite;
        }

        .thinking-flow {
          animation: thinking-flow 2.8s ease-in-out infinite;
          color: currentColor;
        }

        @keyframes thinking-breathe {
          0%,
          100% {
            opacity: 0.35;
            transform: scale(0.9);
          }
          50% {
            opacity: 0.85;
            transform: scale(1.08);
          }
        }

        @keyframes thinking-flow {
          0% {
            opacity: 0.35;
            transform: translateX(0);
          }
          50% {
            opacity: 0.78;
            transform: translateX(4px);
          }
          100% {
            opacity: 0.35;
            transform: translateX(0);
          }
        }
      `}</style>
    </div>
  )
}

function HistoryStatusBadge({ status }: { status: "pending" | "ready" | "error" | "cancelled" }) {
  if (status === "pending") {
    return (
      <span className="inline-flex items-center gap-1 rounded bg-muted px-1.5 py-0.5">
        <Loader2 className="h-3 w-3 animate-spin" />
        Generating
      </span>
    )
  }

  if (status === "error") {
    return <span className="rounded bg-destructive/10 px-1.5 py-0.5 text-destructive">Incomplete</span>
  }

  if (status === "cancelled") {
    return <span className="rounded bg-muted px-1.5 py-0.5">Stopped</span>
  }

  return <span className="rounded bg-primary/10 px-1.5 py-0.5 text-primary">Ready</span>
}

function getHistoryStatus(item: HistoryItem): "pending" | "ready" | "error" | "cancelled" {
  if (item.status) return item.status
  if (item.summary || item.reportUrl) return "ready"
  return "pending"
}

function getThinkingStepIndex(elapsedSeconds: number): number {
  let activeIndex = 0
  for (let index = 0; index < thinkingSteps.length; index += 1) {
    if (elapsedSeconds >= thinkingSteps[index].afterSeconds) {
      activeIndex = index
    }
  }
  return activeIndex
}

function formatElapsedTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60
  return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`
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

function buildPlannerRequest(messages: Message[]): string {
  const userMessages = messages
    .filter((message) => message.role === "user")
    .map((message) => renderMessageContent(message.content).trim())
    .filter(Boolean)

  if (userMessages.length <= 1) return userMessages[0] || ""

  const [firstMessage, ...followUps] = userMessages

  return [
    firstMessage,
    ...followUps.map((message) => `Additional user update: ${message}`),
  ].join("\n")
}

function renderMessageContent(content: unknown): string {
  if (typeof content === "string") return content
  if (content && typeof content === "object" && "text" in content) {
    const text = (content as { text?: unknown }).text
    if (typeof text === "string") return text
  }
  if (content == null) return ""
  try {
    return JSON.stringify(content)
  } catch {
    return String(content)
  }
}

function validateRequest(request: string): string | null {
  const dateError = validateTravelDates(request)
  if (dateError) return dateError

  const budgetMatch = request.match(/(?:\b(?:budget|cost)\b|预算|费用)\D{0,12}(\d+(?:\.\d+)?)/i)
  if (budgetMatch) {
    const budget = Number(budgetMatch[1])
    if (!Number.isFinite(budget) || budget <= 0) {
      return "I can't create a plan with an invalid budget. Please enter a budget greater than 0."
    }
  }

  const peopleMatch = request.match(/(\d+)\s*(?:people|person|travelers|travellers|人)\b/i)
  if (peopleMatch) {
    const people = Number(peopleMatch[1])
    if (!Number.isInteger(people) || people <= 0 || people > 50) {
      return "I can't create a plan with that traveler count. Please enter a number of travelers between 1 and 50."
    }
  }

  const daysMatch = request.match(/(\d+)\s*(?:day|days|天)\b/i)
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
    "To continue, tell me your departure city, destination city or country, travel dates or trip length, number of travelers, total budget and currency, and any hotel, food, transport, attraction, or pace preferences.",
    "",
    "For example: I want to travel from Shanghai to Paris from 2026-06-10 to 2026-06-15 for 2 people with a budget of 15000 CNY. I prefer clean central hotels, local food, and a relaxed pace.",
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
  const budget = formatBudget(trip.budget)
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

function formatBudget(budget?: NonNullable<PlanResponse["trip_summary"]>["budget"]): string {
  if (!budget?.total) return "not specified"

  const total = `${budget.total} ${budget.currency || "CNY"}`
  if (budget.source_total && budget.source_currency && budget.source_currency !== budget.currency) {
    return `${budget.source_total} ${budget.source_currency} (approx. ${total} for planning)`
  }

  return total
}
