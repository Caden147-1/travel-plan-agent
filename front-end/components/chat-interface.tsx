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
  request_id?: string
  success: boolean
  stage?: string
  error?: string
  guidance?: string
  warnings?: string[]
  report_url?: string
  minimum_budget_required?: number
  recoverable_budget_request_id?: string
  recoverable_estimated_total?: number
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
  refinement_state?: RefinementState
  upgrade_status?: UpgradeStatus[]
  planning_baseline?: PlanningBaseline
}

interface GuidanceResponse {
  request_id?: string
  success: boolean
  ready?: boolean
  guidance?: string
  error?: string
}

interface HistoryItem {
  id: string
  request: string
  requestId?: string
  messages?: Message[]
  summary?: string
  reportUrl?: string
  createdAt: string
  updatedAt?: string
  status?: "pending" | "ready" | "error" | "cancelled"
  error?: string
  refinementState?: RefinementState
  pendingRefinement?: PendingRefinement
  planningBaselineRequest?: string
  planningBudgetTotal?: number
  planningSelectedTotal?: number
  recoverableBudgetRequestId?: string
  recoverableEstimatedTotal?: number
  recoverableMinimumBudgetRequired?: number
  lastFailureStage?: string
}

type RefinementCategory = "food" | "attractions" | "flight" | "hotel"

interface CategoryCeiling {
  reason?: string
  spend?: number
  budget?: number
  ratio?: number
}

interface RefinementState {
  categoryCeilings?: Partial<Record<RefinementCategory, CategoryCeiling>>
  categorySpend?: Record<string, { spend?: number; budget?: number; ratio?: number; budget_key?: string }>
  lastTargetCategories?: RefinementCategory[]
  maxActivityRadiusKm?: number
  allowTotalBudgetIncrease?: boolean
  totalBudget?: number
  selectedTotal?: number
}

interface PendingRefinement {
  targetCategories: RefinementCategory[]
  followupMessage: string
  needBudgetDecision: boolean
  needDistanceDecision: boolean
  reason?: "standard" | "maxed_budget"
  allowTotalBudgetIncrease?: boolean
  newTotalBudget?: number
  maxActivityRadiusKm?: number
}

interface UpgradeStatus {
  category: RefinementCategory
  status: "upgraded" | "maxed" | string
  message: string
  spend?: number
  budget?: number
  ratio?: number
}

interface RefinementContext {
  mode: "refine"
  previous_request: string
  followup_message: string
  target_categories: RefinementCategory[]
  allow_total_budget_increase: boolean
  new_total_budget?: number
  max_activity_radius_km?: number
  previous_refinement_state?: RefinementState
  previous_total_budget?: number
  previous_selected_total?: number
}

interface PlanningBaseline {
  request?: string
  total_budget?: number
  selected_total?: number
}

type DraftMap = Record<string, string>
type PendingSinceMap = Record<string, number>

const apiBase = process.env.NEXT_PUBLIC_TRAVEL_API_BASE || "http://127.0.0.1:8000"
const historyStorageKey = "travelai-plan-history"
const draftStorageKey = "travelai-plan-drafts"
const unsavedDraftKey = "__new__"

const suggestedPrompts = [
  "I want to travel from Beijing to Chengdu from 6.19 to 6.21 with a budget of 6000 CNY. I love local food.",
  "Plan a 5 day Tokyo trip from Singapore for 2 people with a 200000 CNY budget.",
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
  const [loadingByHistoryId, setLoadingByHistoryId] = useState<Record<string, boolean>>({})
  const [drafts, setDrafts] = useState<DraftMap>({})
  const [pendingSinceByHistoryId, setPendingSinceByHistoryId] = useState<PendingSinceMap>({})
  const [thinkingStepIndex, setThinkingStepIndex] = useState(0)
  const [thinkingElapsedSeconds, setThinkingElapsedSeconds] = useState(0)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const activeHistoryIdRef = useRef<string | null>(null)
  const abortControllersRef = useRef<Record<string, AbortController>>({})

  const currentDraftKey = activeHistoryId ?? unsavedDraftKey
  const activeHistoryItem = activeHistoryId ? history.find((item) => item.id === activeHistoryId) : null
  const activeConversationId = activeHistoryId ?? null
  const isActiveLoading = Boolean(activeConversationId && loadingByHistoryId[activeConversationId])
  const showActiveLoading = Boolean(activeHistoryItem?.status === "pending" && activeConversationId && loadingByHistoryId[activeConversationId])
  const maxedCategories = getMaxedCategories(activeHistoryItem?.refinementState)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, showActiveLoading, thinkingStepIndex])

  useEffect(() => {
    if (!showActiveLoading || !activeConversationId) {
      setThinkingStepIndex(0)
      setThinkingElapsedSeconds(0)
      return
    }

    const startedAt = pendingSinceByHistoryId[activeConversationId] || Date.now()
    const updateThinkingState = () => {
      const elapsedSeconds = Math.floor((Date.now() - startedAt) / 1000)
      setThinkingElapsedSeconds(elapsedSeconds)
      setThinkingStepIndex(getThinkingStepIndex(elapsedSeconds))
    }

    updateThinkingState()
    const interval = window.setInterval(updateThinkingState, 500)
    return () => window.clearInterval(interval)
  }, [activeConversationId, pendingSinceByHistoryId, showActiveLoading])

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

  useEffect(() => {
    try {
      const savedDrafts = window.localStorage.getItem(draftStorageKey)
      if (!savedDrafts) return
      const parsedDrafts = JSON.parse(savedDrafts) as DraftMap
      if (parsedDrafts && typeof parsedDrafts === "object") {
        setDrafts(parsedDrafts)
      }
    } catch {
      setDrafts({})
    }
  }, [])

  useEffect(() => {
    setInput(drafts[currentDraftKey] || "")
  }, [currentDraftKey])

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

  function updateDraft(key: string, value: string) {
    setDrafts((previous) => {
      const nextDrafts = { ...previous, [key]: value }
      window.localStorage.setItem(draftStorageKey, JSON.stringify(nextDrafts))
      return nextDrafts
    })
  }

  function clearDraft(key: string) {
    setDrafts((previous) => {
      if (!(key in previous)) return previous
      const nextDrafts = { ...previous }
      delete nextDrafts[key]
      window.localStorage.setItem(draftStorageKey, JSON.stringify(nextDrafts))
      return nextDrafts
    })
  }

  function setConversationLoading(historyId: string, isLoading: boolean) {
    setLoadingByHistoryId((previous) => {
      const next = { ...previous }
      if (isLoading) next[historyId] = true
      else delete next[historyId]
      return next
    })
    setPendingSinceByHistoryId((previous) => {
      const next = { ...previous }
      if (isLoading) next[historyId] = previous[historyId] || Date.now()
      else delete next[historyId]
      return next
    })
  }

  async function handleSubmit(e: FormEvent | KeyboardEvent) {
    e.preventDefault()
    const request = input.trim()
    if (isActiveLoading && activeConversationId && !request) {
      handleStopThinking()
      return
    }
    if (!request) return

    const previousDraftKey = currentDraftKey
    const historyId = activeHistoryIdRef.current || crypto.randomUUID()
    const existingHistory = history.find((item) => item.id === historyId)
    const previousMessages = existingHistory?.messages?.length ? existingHistory.messages : []
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: request,
    }
    const submittedMessages = [...previousMessages, userMessage]
    const publicTransportConfirmation = existingHistory
      ? isPublicIntercityTransportConfirmation(request, existingHistory)
      : false
    let plannerRequest = publicTransportConfirmation
      ? buildPublicIntercityTransportRequest(existingHistory!)
      : buildPlannerRequest(submittedMessages)
    const budgetRecovery = existingHistory ? buildFastBudgetRecovery(request, existingHistory) : null
    if (budgetRecovery) {
      const budgetAboveRecoveredMinimum =
        budgetRecovery.newTotalBudget !== undefined
        && existingHistory?.recoverableMinimumBudgetRequired !== undefined
        && budgetRecovery.newTotalBudget > existingHistory.recoverableMinimumBudgetRequired
      const budgetOverride = budgetRecovery.newTotalBudget
        ? `Use ${budgetRecovery.newTotalBudget} CNY as the new complete total budget. This replaces any earlier total budget.`
        : "Apply the user's updated budget constraints below."
      const onlyBudgetIncrease =
        budgetRecovery.actions.length === 1 && budgetRecovery.actions.includes("increase_budget")
      plannerRequest = [
        getPlanningBaselineRequest(existingHistory!),
        "",
        budgetOverride,
        onlyBudgetIncrease || budgetAboveRecoveredMinimum
          ? "Rerun the full planning pipeline with the new budget. Do not reuse the previous constrained recovery plan."
          : "Apply only the requested budget recovery changes. Do not upgrade unrelated hotels, restaurants, attractions, flights, or local transport just because the total budget changed.",
        `The user changed the budget constraints after the previous budget result: ${request}`,
      ].join("\n")
    }
    const budgetFailureRevision = !budgetRecovery && existingHistory
      ? buildBudgetFailureRevisionRequest(request, existingHistory)
      : null
    if (budgetFailureRevision) {
      plannerRequest = budgetFailureRevision
    }
    let refinementContext: RefinementContext | null = null
    let nextPendingRefinement: PendingRefinement | undefined
    let directAssistantContent = ""

    if (budgetRecovery) {
      nextPendingRefinement = undefined
    } else if (existingHistory?.pendingRefinement) {
      const resolved = resolvePendingRefinement(existingHistory.pendingRefinement, request, existingHistory)
      if (resolved.rejection) {
        directAssistantContent = resolved.rejection
        nextPendingRefinement = undefined
      } else if (resolved.question) {
        directAssistantContent = resolved.question
        nextPendingRefinement = resolved.pendingRefinement
      } else if (resolved.context) {
        refinementContext = resolved.context
        plannerRequest = getPlanningBaselineRequest(existingHistory)
      }
    } else {
      const refinementStart = buildRefinementStart(request, existingHistory)
      if (refinementStart.rejection) {
        directAssistantContent = refinementStart.rejection
      } else if (refinementStart.question) {
        directAssistantContent = refinementStart.question
        nextPendingRefinement = refinementStart.pendingRefinement
      } else if (refinementStart.context) {
        refinementContext = refinementStart.context
        plannerRequest = existingHistory ? getPlanningBaselineRequest(existingHistory) : plannerRequest
      }
    }

    const requestValidationError = refinementContext || budgetRecovery || budgetFailureRevision ? "" : validateRequest(plannerRequest)

    setActiveConversation(historyId)
    clearDraft(previousDraftKey)
    setInput("")
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
        status: directAssistantContent ? "ready" : "pending",
        error: undefined,
        pendingRefinement: nextPendingRefinement,
      }
      return [nextItem, ...previous.filter((item) => item.id !== historyId)]
    })

    if (directAssistantContent) {
      const nextMessages: Message[] = [
        ...submittedMessages,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: directAssistantContent,
        },
      ]
      saveHistory((previous) =>
        previous.map((item) =>
          item.id === historyId
            ? {
                ...item,
                requestId: item.requestId || `local-${historyId.slice(0, 8)}`,
                messages: nextMessages,
                summary: directAssistantContent,
                status: "ready",
                pendingRefinement: nextPendingRefinement,
                updatedAt: new Date().toISOString(),
              }
            : item,
        ),
      )
      if (activeHistoryIdRef.current === historyId) setMessages(nextMessages)
      return
    }

    setConversationLoading(historyId, true)
    setThinkingStepIndex(0)
    setThinkingElapsedSeconds(0)
    const abortController = new AbortController()
    abortControllersRef.current[historyId] = abortController

    try {
      if (requestValidationError) {
        const guidance = buildFallbackGuidance(plannerRequest, requestValidationError)
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
                  requestId: item.requestId || `local-${historyId.slice(0, 8)}`,
                  messages: nextMessages,
                  summary: guidance,
                  status: "error",
                  error: requestValidationError,
                  updatedAt: new Date().toISOString(),
                }
              : item,
          ),
        )
        if (activeHistoryIdRef.current === historyId) setMessages(nextMessages)
        return
      }

      if (!refinementContext && !budgetRecovery && !budgetFailureRevision) {
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
          const guidance = guidanceData.guidance || buildFallbackGuidance(plannerRequest)
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
                    requestId: guidanceData.request_id || item.requestId,
                    messages: nextMessages,
                    summary: guidance,
                    status: "error",
                    error: guidanceData.error || "The request needs more details.",
                    updatedAt: new Date().toISOString(),
                  }
                : item,
            ),
          )
          if (activeHistoryIdRef.current === historyId) setMessages(nextMessages)
          return
        }
      }

      const response = await fetch(`${apiBase}/api/plan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: plannerRequest,
          ...(refinementContext ? { refinement_context: refinementContext } : {}),
          ...(budgetRecovery
            ? {
                budget_recovery_actions: budgetRecovery.actions,
                ...(budgetRecovery.newTotalBudget ? { new_total_budget: budgetRecovery.newTotalBudget } : {}),
                ...(budgetRecovery.requestId ? { recover_budget_request_id: budgetRecovery.requestId } : {}),
              }
            : {}),
        }),
        signal: abortController.signal,
      })
      const data = (await response.json()) as PlanResponse

      if (!response.ok || !data.success) {
        const guidance = data.guidance || buildFallbackGuidance(plannerRequest, data.error)
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
                  requestId: data.request_id || item.requestId,
                  messages: nextMessages,
                  summary: guidance,
                  status: "error",
                  error: data.error || "Planning failed.",
                  recoverableBudgetRequestId: data.recoverable_budget_request_id || item.recoverableBudgetRequestId,
                  recoverableEstimatedTotal: data.recoverable_estimated_total || item.recoverableEstimatedTotal,
                  recoverableMinimumBudgetRequired: data.minimum_budget_required || item.recoverableMinimumBudgetRequired,
                  lastFailureStage: data.stage || item.lastFailureStage,
                  planningBaselineRequest: data.planning_baseline?.request || plannerRequest || item.planningBaselineRequest,
                  planningBudgetTotal: data.planning_baseline?.total_budget || data.trip_summary?.budget?.total || item.planningBudgetTotal,
                  planningSelectedTotal: data.planning_baseline?.selected_total || data.itinerary_summary?.estimated_total_cost || item.planningSelectedTotal,
                  updatedAt: new Date().toISOString(),
                }
              : item,
          ),
        )
        if (activeHistoryIdRef.current === historyId) setMessages(nextMessages)
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
                requestId: data.request_id || item.requestId,
                messages: nextMessages,
                summary,
                reportUrl: data.report_url,
                refinementState: data.refinement_state || item.refinementState,
                planningBaselineRequest: data.planning_baseline?.request || plannerRequest,
                planningBudgetTotal: data.planning_baseline?.total_budget || data.trip_summary?.budget?.total || item.planningBudgetTotal,
                planningSelectedTotal: data.planning_baseline?.selected_total || data.itinerary_summary?.estimated_total_cost || item.planningSelectedTotal,
                recoverableBudgetRequestId: undefined,
                recoverableEstimatedTotal: undefined,
                recoverableMinimumBudgetRequired: undefined,
                lastFailureStage: undefined,
                pendingRefinement: undefined,
                status: "ready",
                updatedAt: new Date().toISOString(),
              }
            : item,
        ),
      )
      if (activeHistoryIdRef.current === historyId) setMessages(nextMessages)
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
                  requestId: item.requestId || `local-${historyId.slice(0, 8)}`,
                  messages: nextMessages,
                  summary: cancelledSummary,
                  status: "cancelled",
                  error: undefined,
                  updatedAt: new Date().toISOString(),
                }
              : item,
          ),
        )
        if (activeHistoryIdRef.current === historyId) setMessages(nextMessages)
        return
      }

      const rawMessage = error instanceof Error ? error.message : "Unknown error."
      const message =
        rawMessage === "Load failed" || rawMessage === "Failed to fetch"
          ? `Cannot reach the TravelAI API at ${apiBase}. Please start the backend with python3.13 main.py and keep that terminal running.`
          : rawMessage
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
                requestId: item.requestId || `local-${historyId.slice(0, 8)}`,
                messages: nextMessages,
                summary: errorSummary,
                status: "error",
                error: message,
                updatedAt: new Date().toISOString(),
              }
            : item,
        ),
      )
      if (activeHistoryIdRef.current === historyId) setMessages(nextMessages)
    } finally {
      delete abortControllersRef.current[historyId]
      setConversationLoading(historyId, false)
    }
  }

  function handleStopThinking() {
    if (!activeConversationId) return
    abortControllersRef.current[activeConversationId]?.abort()
  }

  function handleHistoryClick(item: HistoryItem) {
    setActiveConversation(item.id)
    if (item.messages?.length) {
      setMessages(item.messages)
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
    textareaRef.current?.focus()
  }

  function handleHistoryDelete(id: string) {
    saveHistory((previous) => previous.filter((item) => item.id !== id))
    clearDraft(id)
    if (abortControllersRef.current[id]) {
      abortControllersRef.current[id].abort()
      delete abortControllersRef.current[id]
    }
    setConversationLoading(id, false)
    if (activeHistoryIdRef.current === id) {
      setMessages([])
      setInput("")
      setActiveConversation(null)
    }
  }

  function handlePromptClick(prompt: string) {
    setInput(prompt)
    updateDraft(currentDraftKey, prompt)
    textareaRef.current?.focus()
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      if (isActiveLoading && !input.trim()) {
        handleStopThinking()
        return
      }
      void handleSubmit(e)
    }
  }

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
              setActiveConversation(null)
              textareaRef.current?.focus()
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
            {maxedCategories.length > 0 && (
              <div className="mb-2 flex flex-wrap gap-2">
                {maxedCategories.map((category) => (
                  <span key={category} className="rounded-md border border-border bg-secondary px-2 py-1 text-xs text-muted-foreground">
                    {categoryLabel(category)} maxed under current budget
                  </span>
                ))}
              </div>
            )}
            <div className="relative flex items-end gap-2 bg-input rounded-lg border border-border p-2">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => {
                  const nextValue = e.target.value
                  setInput(nextValue)
                  updateDraft(currentDraftKey, nextValue)
                }}
                onKeyDown={handleKeyDown}
                placeholder="Describe your trip..."
                className="flex-1 bg-transparent resize-none outline-none text-sm px-2 py-2 max-h-[180px] min-h-[40px]"
                rows={1}
              />
              <Button
                type="submit"
                size="icon"
                disabled={!input.trim() && !isActiveLoading}
                className="flex-shrink-0 h-10 w-10 rounded-lg"
                aria-label={isActiveLoading && !input.trim() ? "Stop thinking" : "Send message"}
                title={isActiveLoading && !input.trim() ? "Stop" : "Send"}
              >
                {isActiveLoading && !input.trim() ? <X className="h-4 w-4" /> : <Send className="h-4 w-4" />}
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
    refinementState: item.refinementState,
    pendingRefinement: item.pendingRefinement,
    planningBaselineRequest: item.planningBaselineRequest,
    planningBudgetTotal: item.planningBudgetTotal,
    planningSelectedTotal: item.planningSelectedTotal,
    recoverableBudgetRequestId: item.recoverableBudgetRequestId,
    recoverableEstimatedTotal: item.recoverableEstimatedTotal,
    recoverableMinimumBudgetRequired: item.recoverableMinimumBudgetRequired,
    lastFailureStage: item.lastFailureStage,
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

function categoryLabel(category: RefinementCategory): string {
  const labels: Record<RefinementCategory, string> = {
    food: "Food",
    attractions: "Attractions",
    flight: "Flight",
    hotel: "Hotel",
  }
  return labels[category]
}

function getMaxedCategories(state?: RefinementState): RefinementCategory[] {
  const ceilings = state?.categoryCeilings || {}
  return (Object.keys(ceilings) as RefinementCategory[]).filter((category) => Boolean(ceilings[category]))
}

function detectRefinementCategories(text: string): RefinementCategory[] {
  const value = text.toLowerCase()
  const categories: RefinementCategory[] = []
  const add = (category: RefinementCategory) => {
    if (!categories.includes(category)) categories.push(category)
  }
  if (/\b(food|restaurant|restaurants|meal|meals|dining)\b|餐|饭|美食/.test(value)) add("food")
  if (/\b(attraction|attractions|poi|pois|sightseeing|museum|museums|activity|activities)\b|景点|门票|游玩/.test(value)) add("attractions")
  if (/\b(flight|flights|ticket|tickets|cabin|business|first class|airline)\b|机票|航班|商务舱|头等舱/.test(value)) add("flight")
  if (/\b(hotel|hotels|stay|lodging|room|rooms)\b|酒店|住宿/.test(value)) add("hotel")
  return categories.length ? categories : ["food", "attractions", "flight", "hotel"]
}

function isRefinementIntent(text: string): boolean {
  const value = text.toLowerCase()
  return /not satisfied|unsatisfied|spend more|more money|upgrade|better|higher|premium|luxury|improve|不满意|加钱|多花|升级|更好|高级/.test(value)
}

function parseBudgetDecision(text: string): { decided: boolean; allow: boolean; amount?: number } {
  const value = text.toLowerCase()
  const amount = parseLargestNumber(value)
  if (/不增加|不提高|不加|不要|不用|不可以|不同意|no\b|within|current budget|same budget|原预算|总预算内/.test(value)) {
    return { decided: true, allow: false }
  }
  if (amount !== undefined && amount >= 1000) {
    const looksLikeDelta = /\badd\b|increase by|raise by|加\s*\d|增加\s*\d|多\s*\d|再加/.test(value)
    const looksLikeCompleteBudget = /\bto\s*\d|new total|complete total|total budget|budget\s*(?:to|is|=|:)?\s*\d|到\s*\d|至\s*\d|总预算|完整预算|新预算/.test(value)
    if (looksLikeDelta && !looksLikeCompleteBudget) {
      return { decided: true, allow: true }
    }
    return { decided: true, allow: true, amount }
  }
  if (/yes\b|ok\b|agree|allow|increase|raise|more budget|new budget|同意|可以|增加|提高|加预算|新预算/.test(value)) {
    return { decided: true, allow: true }
  }
  return { decided: false, allow: false }
}

function parseLargestNumber(text: string): number | undefined {
  const matches = text.match(/\d+(?:,\d{3})*(?:\.\d+)?/g)
  if (!matches) return undefined
  const values = matches.map((item) => Number(item.replace(/,/g, ""))).filter((item) => Number.isFinite(item))
  if (!values.length) return undefined
  return Math.max(...values)
}

function parseRadiusDecision(text: string): { decided: boolean; radius?: number } {
  const value = text.toLowerCase()
  if (/15\s*km|15公里|十五公里/.test(value)) return { decided: true, radius: 15 }
  if (/10\s*km|10公里|十公里/.test(value)) return { decided: true, radius: 10 }
  if (/5\s*km|5公里|五公里|nearby|附近|不远|不扩大/.test(value)) return { decided: true, radius: 5 }
  if (/yes\b|ok\b|accept|farther|further|更远|远一点|可以/.test(value)) return { decided: true, radius: 15 }
  if (/no\b|不要|不可以|不能接受/.test(value)) return { decided: true, radius: 5 }
  return { decided: false }
}

function buildRefinementQuestion(pending: PendingRefinement): string {
  const categories = pending.targetCategories.map(categoryLabel).join(", ")
  const parts = [`I can refine ${categories} based on the previous report.`]
  if (pending.needBudgetDecision) {
    if (pending.reason === "maxed_budget") {
      parts.push("This category is already at the highest spec I can fit under the current budget. To upgrade it further, please provide a new complete total budget. If you do not want to increase the budget, say no and I will keep the current plan.")
    } else {
      parts.push("Can I increase the total budget? If yes, please give the new complete total budget. If no, say to keep the current budget.")
    }
  }
  if (pending.needDistanceDecision) {
    parts.push("For food/attractions, can you accept higher-spec places farther from the hotel? Please choose 5km, 10km, or 15km.")
  }
  return parts.join("\n")
}

function buildRefinementContext(
  historyItem: HistoryItem,
  followupMessage: string,
  targetCategories: RefinementCategory[],
  budgetDecision: { allow: boolean; amount?: number },
  radiusKm?: number,
): RefinementContext {
  const previousBudget = historyItem.planningBudgetTotal || historyItem.refinementState?.totalBudget
  const previousSelectedTotal = historyItem.planningSelectedTotal || historyItem.refinementState?.selectedTotal
  return {
    mode: "refine",
    previous_request: getPlanningBaselineRequest(historyItem),
    followup_message: followupMessage,
    target_categories: targetCategories,
    allow_total_budget_increase: budgetDecision.allow,
    ...(budgetDecision.allow && budgetDecision.amount ? { new_total_budget: budgetDecision.amount } : {}),
    ...(radiusKm ? { max_activity_radius_km: radiusKm } : {}),
    previous_refinement_state: historyItem.refinementState,
    ...(previousBudget ? { previous_total_budget: previousBudget } : {}),
    ...(previousSelectedTotal ? { previous_selected_total: previousSelectedTotal } : {}),
  }
}

function getPlanningBaselineRequest(historyItem: HistoryItem): string {
  return historyItem.planningBaselineRequest || historyItem.request
}

function isBudgetFailureContext(historyItem: HistoryItem): boolean {
  const previousText = [
    historyItem.error,
    historyItem.summary,
    ...(historyItem.messages || []).map((message) => message.content),
  ]
    .filter(Boolean)
    .join("\n")
    .toLowerCase()
  return (
    historyItem.lastFailureStage === "budget_feasibility"
    || /(budget_feasibility|over budget|above your.*budget|budget is.*too low|core expenses|minimum.*budget|预算.*不够|超.*预算|最低.*预算)/.test(previousText)
  )
}

function isBudgetFailureStructuralRevision(text: string): boolean {
  const value = text.toLowerCase()
  return (
    /\b(shorten|shorter|fewer days?|reduce.*days?|cut.*days?)\b|\b\d+\s*days?\s+instead of\s+\d+\s*days?\b|缩短|少.*天|改成.*天/.test(value)
    || /\b(change|shift|move|try|use)\b.{0,24}\b(date|dates|departure|departing|start|leave|leaving)\b|different dates?|new dates?|改日期|换日期|调整日期|换时间|改时间/.test(value)
  )
}

function parseTargetDurationDays(text: string): number | null {
  const value = text.toLowerCase()
  const patterns = [
    /\b(?:shorten|shorter|reduce|cut|change|make|set).*?\b(?:to|as)\s*(\d{1,2})\s*days?\b/,
    /\b(\d{1,2})\s*days?\s+instead of\s+\d{1,2}\s*days?\b/,
    /改成\s*(\d{1,2})\s*天/,
    /缩短.*?(\d{1,2})\s*天/,
  ]
  for (const pattern of patterns) {
    const match = value.match(pattern)
    if (!match) continue
    const days = Number(match[1])
    if (Number.isInteger(days) && days >= 1 && days <= 60) return days
  }
  return null
}

function addDaysToIsoDate(isoDate: string, daysToAdd: number): string | null {
  const match = isoDate.match(/^(\d{4})-(\d{2})-(\d{2})$/)
  if (!match) return null
  const date = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3])))
  if (Number.isNaN(date.getTime())) return null
  date.setUTCDate(date.getUTCDate() + daysToAdd)
  return [
    date.getUTCFullYear(),
    String(date.getUTCMonth() + 1).padStart(2, "0"),
    String(date.getUTCDate()).padStart(2, "0"),
  ].join("-")
}

function applyDurationOverrideToBaseline(baseline: string, targetDays: number): {
  request: string
  startDate?: string
  endDate?: string
} {
  const matches = [...baseline.matchAll(/\b(20\d{2}-\d{2}-\d{2})\b/g)]
  if (matches.length < 2) return { request: baseline }

  const startDate = matches[0][1]
  const oldEndMatch = matches[1]
  const oldEndDate = oldEndMatch[1]
  const oldEndIndex = oldEndMatch.index
  const newEndDate = addDaysToIsoDate(startDate, targetDays - 1)
  if (!newEndDate || oldEndIndex === undefined || newEndDate === oldEndDate) {
    return { request: baseline, startDate, endDate: newEndDate || oldEndDate }
  }

  return {
    request: `${baseline.slice(0, oldEndIndex)}${newEndDate}${baseline.slice(oldEndIndex + oldEndDate.length)}`,
    startDate,
    endDate: newEndDate,
  }
}

function buildBudgetFailureRevisionRequest(request: string, historyItem: HistoryItem): string | null {
  const baseline = getPlanningBaselineRequest(historyItem)
  if (!baseline || !isBudgetFailureContext(historyItem) || !isBudgetFailureStructuralRevision(request)) return null
  const targetDays = parseTargetDurationDays(request)
  const durationOverride = targetDays ? applyDurationOverrideToBaseline(baseline, targetDays) : { request: baseline }
  const overrideLines = targetDays
    ? [
        `Hard override: the trip must be exactly ${targetDays} days total.`,
        durationOverride.startDate && durationOverride.endDate
          ? `Use ${durationOverride.startDate} as the start date and ${durationOverride.endDate} as the end date. Ignore any previous end date that conflicts.`
          : "Replace any earlier trip length or end date that conflicts with this duration.",
      ]
    : []

  return [
    durationOverride.request,
    "",
    ...overrideLines,
    ...(overrideLines.length ? [""] : []),
    "The previous attempt failed budget feasibility. Apply only this latest structural revision to the same trip, replacing any conflicting earlier details:",
    request,
    "",
    "Ignore earlier budget-failure option replies that are not repeated in this latest revision.",
  ].join("\n")
}

function isAffirmativeReply(text: string): boolean {
  const value = text.trim().toLowerCase()
  return /^(yes|y|ok|okay|sure|agree|accepted?|confirm|confirmed|可以|同意|确认|好的|好|行|是|对)$/.test(value)
}

function isPublicIntercityTransportConfirmation(request: string, historyItem: HistoryItem): boolean {
  if (!isAffirmativeReply(request)) return false
  const lastAssistant = [...(historyItem.messages || [])].reverse().find((message) => message.role === "assistant")
  const content = renderMessageContent(lastAssistant?.content || "").toLowerCase()
  return (
    /remove car rentals?|remove car rental|instead of rental cars|instead of renting cars|use public intercity transport/.test(content)
    || /trains? or buses?|trains? or flights/.test(content)
  )
}

function buildPublicIntercityTransportRequest(historyItem: HistoryItem): string {
  return [
    getPlanningBaselineRequest(historyItem),
    "",
    "The user confirmed: remove rental cars and use public intercity transport between cities. Do not use rental cars. Use trains, buses, or flights between cities as needed to stay within budget.",
  ].join("\n")
}

type FastBudgetRecoveryAction =
  | "increase_budget"
  | "lower_hotels"
  | "lower_food"
  | "remove_paid_attractions"
  | "use_public_transport"

function buildFastBudgetRecovery(
  request: string,
  historyItem: HistoryItem,
): { requestId?: string; actions: FastBudgetRecoveryAction[]; newTotalBudget?: number } | null {
  const previousText = [
    historyItem.error,
    historyItem.summary,
    ...(historyItem.messages || []).map((message) => message.content),
  ]
    .filter(Boolean)
    .join("\n")
    .toLowerCase()
  const looksLikeBudgetFailure =
    /(budget_feasibility|over budget|above your.*budget|budget is.*too low|core expenses|minimum.*budget|预算.*不够|超.*预算|最低.*预算)/.test(
      previousText,
    )
  const canRetryBudgetPrecheck =
    (historyItem.lastFailureStage === "budget_feasibility" || looksLikeBudgetFailure)
    && Boolean(getPlanningBaselineRequest(historyItem))
  if (!historyItem.recoverableBudgetRequestId && !canRetryBudgetPrecheck) return null
  const value = request.toLowerCase()
  const actions: FastBudgetRecoveryAction[] = []
  const add = (action: FastBudgetRecoveryAction) => {
    if (!actions.includes(action)) actions.push(action)
  }

  const amount = parseLargestNumber(value)
  if (
    amount !== undefined
    && amount >= 1000
    && /(budget|rmb|cny|yuan|increase|raise|to\s*\d|new total|预算|提高|增加|加到|改到|换成|人民币|元)/.test(value)
  ) {
    add("increase_budget")
  }
  if (/(budget hotel|cheaper hotel|lower.*hotel|worse.*hotel|less nice.*hotel|不要那么好的酒店|酒店.*便宜|酒店.*降级|普通酒店|经济酒店|便宜酒店)/.test(value)) {
    add("lower_hotels")
  }
  if (/(cheaper food|cheaper restaurant|lower.*food|lower.*meal|budget meal|local meal|不要那么好的(食物|餐厅|饭)|餐厅.*便宜|食物.*便宜|吃.*便宜|普通餐|当地小吃|便宜餐)/.test(value)) {
    add("lower_food")
  }
  if (/(remove paid|free attraction|no paid attraction|skip paid|不要付费|去掉付费|免费景点|不要门票|去掉门票|不买门票)/.test(value)) {
    add("remove_paid_attractions")
  }
  if (/(public transport|public transit|train|bus|coach|no rental|remove rental|without rental|公交|公共交通|火车|大巴|巴士|不要租车|不用租车|去掉租车)/.test(value)) {
    add("use_public_transport")
  }

  if (!actions.length) return null
  const budgetAboveRecoveredMinimum =
    amount !== undefined
    && historyItem.recoverableMinimumBudgetRequired !== undefined
    && amount > historyItem.recoverableMinimumBudgetRequired
  const shouldUseRecoverablePlan = Boolean(historyItem.recoverableBudgetRequestId)
    && !(actions.length === 1 && actions.includes("increase_budget"))
    && !budgetAboveRecoveredMinimum

  return {
    ...(shouldUseRecoverablePlan ? { requestId: historyItem.recoverableBudgetRequestId } : {}),
    actions,
    ...(amount !== undefined && actions.includes("increase_budget") ? { newTotalBudget: amount } : {}),
  }
}

function buildRefinementStart(request: string, historyItem?: HistoryItem | null): {
  context?: RefinementContext
  question?: string
  pendingRefinement?: PendingRefinement
  rejection?: string
} {
  if (!historyItem?.reportUrl || !isRefinementIntent(request)) return {}
  const targetCategories = detectRefinementCategories(request)
  const maxed = targetCategories.filter((category) => historyItem.refinementState?.categoryCeilings?.[category])
  if (maxed.length > 0) {
    const pendingRefinement: PendingRefinement = {
      targetCategories: maxed,
      followupMessage: request,
      needBudgetDecision: true,
      needDistanceDecision: false,
      reason: "maxed_budget",
    }
    return {
      question: buildRefinementQuestion(pendingRefinement),
      pendingRefinement,
    }
  }

  const budgetDecision = parseBudgetDecision(request)
  const needsDistance = targetCategories.some((category) => category === "food" || category === "attractions")
  const radiusDecision = needsDistance ? parseRadiusDecision(request) : { decided: true, radius: undefined }
  const needBudgetDecision = !budgetDecision.decided
  const needDistanceDecision = needsDistance && !radiusDecision.decided
  if (needBudgetDecision || needDistanceDecision) {
    const pendingRefinement: PendingRefinement = {
      targetCategories,
      followupMessage: request,
      needBudgetDecision,
      needDistanceDecision,
      reason: "standard",
      ...(budgetDecision.decided ? { allowTotalBudgetIncrease: budgetDecision.allow, newTotalBudget: budgetDecision.amount } : {}),
      ...(radiusDecision.decided ? { maxActivityRadiusKm: radiusDecision.radius } : {}),
    }
    return { question: buildRefinementQuestion(pendingRefinement), pendingRefinement }
  }

  return {
    context: buildRefinementContext(historyItem, request, targetCategories, budgetDecision, radiusDecision.radius),
  }
}

function resolvePendingRefinement(
  pending: PendingRefinement,
  request: string,
  historyItem: HistoryItem,
): { context?: RefinementContext; question?: string; pendingRefinement?: PendingRefinement; rejection?: string } {
  const budgetDecision = pending.needBudgetDecision ? parseBudgetDecision(request) : { decided: true, allow: false }
  const radiusDecision = pending.needDistanceDecision ? parseRadiusDecision(request) : { decided: true, radius: undefined }
  const stillPending: PendingRefinement = {
    ...pending,
    needBudgetDecision: pending.needBudgetDecision && (!budgetDecision.decided || (budgetDecision.allow && !budgetDecision.amount)),
    needDistanceDecision: pending.needDistanceDecision && !radiusDecision.decided,
    ...(budgetDecision.decided ? { allowTotalBudgetIncrease: budgetDecision.allow, newTotalBudget: budgetDecision.amount } : {}),
    ...(radiusDecision.decided ? { maxActivityRadiusKm: radiusDecision.radius } : {}),
  }
  if (pending.reason === "maxed_budget" && budgetDecision.decided && !budgetDecision.allow) {
    return {
      rejection: `${pending.targetCategories.map(categoryLabel).join(", ")} will stay unchanged because the current plan is already at the highest spec under the existing budget.`,
    }
  }
  if (stillPending.needBudgetDecision || stillPending.needDistanceDecision) {
    return { question: buildRefinementQuestion(stillPending), pendingRefinement: stillPending }
  }
  return {
    context: buildRefinementContext(
      historyItem,
      pending.followupMessage,
      pending.targetCategories,
      {
        allow: stillPending.allowTotalBudgetIncrease || false,
        amount: stillPending.newTotalBudget,
      },
      stillPending.maxActivityRadiusKm,
    ),
  }
}

function buildPlannerRequest(messages: Message[]): string {
  const userMessages = messages
    .filter((message) => message.role === "user")
    .map((message) => renderMessageContent(message.content).trim())
    .filter(Boolean)

  if (userMessages.length <= 1) return userMessages[0] || ""

  return [
    "The user is continuing the same travel planning request. Combine these user messages into one complete request.",
    "Keep all valid details from earlier messages unless a later message clearly changes them.",
    "If a later message only supplies one missing field, merge it into the same trip instead of treating it as a new request.",
    "",
    ...userMessages.map((message, index) => `${index + 1}. ${message}`),
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

function inferSuggestionLead(request: string): string {
  const text = request.toLowerCase()
  const hints: string[] = []

  if (/(budget|budget-friendly|affordable|cheap|预算|省钱)/i.test(text)) {
    hints.push("I can help keep it budget-friendly by balancing transport, hotel area, and food choices.")
  }
  if (/(clean hotel|clean hotels|clean|central hotel|central hotels|central)/i.test(text)) {
    hints.push("I can prioritize clean, well-rated hotels in convenient central areas.")
  }
  if (/(local restaurant|local restaurants|local food|restaurant|restaurants|food|美食|当地)/i.test(text)) {
    hints.push("I can also lean toward local restaurants instead of tourist-focused spots.")
  }

  if (hints.length === 0) {
    return "I can help turn this into a practical travel request and suggest good tradeoffs as we narrow it down."
  }

  return hints.slice(0, 2).join(" ")
}

function buildFallbackGuidance(request = "", error?: string): string {
  const exampleRange = getExampleDateRange()
  return [
    inferSuggestionLead(request),
    "",
    error ? `What happened: ${error}` : null,
    "To continue, tell me your departure city, destination city or country, travel dates or trip length, number of travelers, total budget and currency, and any hotel, food, transport, attraction, or pace preferences.",
    "",
    `For example: I want to travel from Shanghai to Paris from ${exampleRange.start} to ${exampleRange.end} for 2 people with a budget of 15000 CNY. I prefer clean central hotels, local food, and a relaxed pace.`,
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

function getExampleDateRange(): { start: string; end: string } {
  const today = startOfDay(new Date())
  const start = new Date(today)
  start.setDate(start.getDate() + 14)

  const end = new Date(start)
  end.setDate(end.getDate() + 4)

  return {
    start: formatIsoDate(start),
    end: formatIsoDate(end),
  }
}

function formatSummary(data: PlanResponse): string {
  const trip = data.trip_summary || {}
  const destinations = trip.destination?.length ? trip.destination.join(", ") : "destination"
  const budget = formatBudget(trip.budget)
  const upgrades = data.upgrade_status?.length
    ? `\n\nUpgrade status:\n${data.upgrade_status.map((item) => `- ${item.message}`).join("\n")}`
    : ""
  const warnings = data.warnings?.length ? `\n\nWarnings:\n${data.warnings.slice(0, 3).map((item) => `- ${item}`).join("\n")}` : ""

  return [
    "Plan generated.",
    "",
    `Route: ${trip.origin || "Origin"} -> ${destinations}`,
    `Days: ${trip.duration_days || data.itinerary_summary?.total_days || "unknown"}`,
    `Budget: ${budget}`,
    data.report_url ? "The HTML report is ready." : "The pipeline finished, but no report URL was returned.",
    upgrades,
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
