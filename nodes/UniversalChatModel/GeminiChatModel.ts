import { AsyncLocalStorage } from 'node:async_hooks';
import { ChatGoogle } from '@langchain/google/node';
import { HumanMessage, type BaseMessage } from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk, ChatResult } from '@langchain/core/outputs';

export interface GeminiModalityTokenCount {
  modality?: string;
  tokenCount?: number;
}

export interface GeminiUsageMetadata {
  promptTokenCount?: number;
  cachedContentTokenCount?: number;
  candidatesTokenCount?: number;
  toolUsePromptTokenCount?: number;
  thoughtsTokenCount?: number;
  totalTokenCount?: number;
  promptTokensDetails?: GeminiModalityTokenCount[];
  cacheTokensDetails?: GeminiModalityTokenCount[];
  candidatesTokensDetails?: GeminiModalityTokenCount[];
  toolUsePromptTokensDetails?: GeminiModalityTokenCount[];
  [key: string]: unknown;
}

export interface GeminiResponseMetadata {
  usageMetadata?: GeminiUsageMetadata;
  modelVersion?: string;
  responseId?: string;
  modelStatus?: unknown;
  promptFeedback?: unknown;
  thoughts?: unknown[];
  [key: string]: unknown;
}

interface CaptureContext {
  response?: GeminiResponseMetadata;
  finishReasons: string[];
  finishMessages: string[];
  hasFunctionCall: boolean;
  hasNonThoughtOutput: boolean;
}

const responseCapture = new AsyncLocalStorage<CaptureContext>();

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === 'object'
    ? (value as Record<string, unknown>)
    : undefined;
}

function thoughtSummaryText(value: unknown): string | undefined {
  if (typeof value === 'string') return value;

  const record = asRecord(value);
  if (!record) return undefined;
  if (typeof record.text === 'string') return record.text;

  const textContent = asRecord(record.text);
  return typeof textContent?.text === 'string' ? textContent.text : undefined;
}

function normalizeRawThoughtPart(part: unknown): unknown[] {
  const record = asRecord(part);
  if (!record) return [];

  if (
    record.thought === true ||
    record.type === 'thinking' ||
    record.type === 'reasoning'
  ) {
    return [structuredClone(record)];
  }

  // Newer Gemini response schemas may represent a thought as a nested object
  // with a signature and one or more summary content blocks.
  const nestedThought =
    asRecord(record.thought) ??
    asRecord(record.thoughtSummary) ??
    asRecord(record.thought_summary);
  if (!nestedThought) return [];

  const signature =
    typeof nestedThought.signature === 'string'
      ? nestedThought.signature
      : typeof record.thoughtSignature === 'string'
        ? record.thoughtSignature
        : undefined;
  const summary = Array.isArray(nestedThought.summary)
    ? nestedThought.summary
    : [];

  return summary.flatMap((summaryPart) => {
    const text = thoughtSummaryText(summaryPart);
    return text
      ? [
          {
            text,
            thought: true,
            ...(signature ? { thoughtSignature: signature } : {}),
          },
        ]
      : [];
  });
}

function extractRawThoughts(payload: unknown): unknown[] {
  const source = asRecord(payload);
  if (!source) return [];

  const thoughts: unknown[] = [];
  const candidates = Array.isArray(source.candidates) ? source.candidates : [];
  for (const candidate of candidates) {
    const content = asRecord(asRecord(candidate)?.content);
    const parts = Array.isArray(content?.parts) ? content.parts : [];
    for (const part of parts) thoughts.push(...normalizeRawThoughtPart(part));
  }

  // Also accept the Interactions-style thought step so the capture remains
  // compatible if the Google transport changes response representation.
  const steps = Array.isArray(source.steps) ? source.steps : [];
  for (const stepValue of steps) {
    const step = asRecord(stepValue);
    if (step?.type !== 'thought') continue;

    const signature =
      typeof step.signature === 'string' ? step.signature : undefined;
    const summary = Array.isArray(step.summary) ? step.summary : [];
    for (const summaryPart of summary) {
      const text = thoughtSummaryText(summaryPart);
      if (text) {
        thoughts.push({
          text,
          thought: true,
          ...(signature ? { thoughtSignature: signature } : {}),
        });
      }
    }
  }

  return mergeUniqueThoughts([], thoughts);
}

function capturePayload(payload: unknown): void {
  if (!payload || typeof payload !== 'object') return;

  const source = payload as GeminiResponseMetadata;
  const context = responseCapture.getStore();
  if (!context) return;

  const previous = context.response ?? {};
  const capturedThoughts = extractRawThoughts(payload);
  const candidates = Array.isArray(source.candidates) ? source.candidates : [];
  for (const candidateValue of candidates) {
    const candidate = asRecord(candidateValue);
    if (typeof candidate?.finishReason === 'string') {
      context.finishReasons.push(candidate.finishReason);
    }
    if (typeof candidate?.finishMessage === 'string') {
      context.finishMessages.push(candidate.finishMessage);
    }

    const content = asRecord(candidate?.content);
    const parts = Array.isArray(content?.parts) ? content.parts : [];
    for (const partValue of parts) {
      const part = asRecord(partValue);
      if (!part) continue;
      if (asRecord(part.functionCall) || asRecord(part.function_call)) {
        context.hasFunctionCall = true;
        context.hasNonThoughtOutput = true;
        continue;
      }
      if (isThoughtBlock(part)) continue;
      if (typeof part.text === 'string') {
        if (part.text.trim().length > 0) context.hasNonThoughtOutput = true;
        continue;
      }
      if (
        Object.keys(part).some(
          (key) => !['thoughtSignature', 'thought_signature'].includes(key),
        )
      ) {
        context.hasNonThoughtOutput = true;
      }
    }
  }

  context.response = {
    ...previous,
    ...(source.modelVersion !== undefined ? { modelVersion: source.modelVersion } : {}),
    ...(source.responseId !== undefined ? { responseId: source.responseId } : {}),
    ...(source.modelStatus !== undefined ? { modelStatus: source.modelStatus } : {}),
    ...(source.promptFeedback !== undefined ? { promptFeedback: source.promptFeedback } : {}),
    ...(source.usageMetadata !== undefined
      ? { usageMetadata: structuredClone(source.usageMetadata) }
      : {}),
    ...(capturedThoughts.length > 0
      ? {
          thoughts: mergeUniqueThoughts(
            previous.thoughts,
            capturedThoughts,
          ),
        }
      : {}),
  };
}

function createCaptureContext(): CaptureContext {
  return {
    finishReasons: [],
    finishMessages: [],
    hasFunctionCall: false,
    hasNonThoughtOutput: false,
  };
}

function consumeSseText(context: CaptureContext, state: { buffer: string }, text: string): void {
  state.buffer += text;

  while (true) {
    const separator = state.buffer.search(/\r?\n\r?\n/);
    if (separator < 0) break;

    const event = state.buffer.slice(0, separator);
    const separatorMatch = state.buffer.slice(separator).match(/^\r?\n\r?\n/);
    state.buffer = state.buffer.slice(separator + (separatorMatch?.[0].length ?? 2));

    const data = event
      .split(/\r?\n/)
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart())
      .join('\n');

    if (!data || data === '[DONE]') continue;

    try {
      responseCapture.run(context, () => capturePayload(JSON.parse(data)));
    } catch {
      // Ignore incomplete/non-JSON SSE events. The LangChain parser handles API errors.
    }
  }
}

class CapturingGeminiApiClient {
  constructor(private readonly apiKey: string) {}

  hasApiKey(): boolean {
    return this.apiKey.length > 0;
  }

  async getProjectId(): Promise<string> {
    return '';
  }

  async fetch(request: Request): Promise<Response> {
    request.headers.set('x-goog-api-key', this.apiKey);
    const context = responseCapture.getStore();
    const response = await fetch(request);

    if (!context || !response.ok) return response;

    const contentType = response.headers.get('content-type') ?? '';
    if (!contentType.includes('text/event-stream') || !response.body) {
      try {
        capturePayload(await response.clone().json());
      } catch {
        // Leave parsing and error reporting to LangChain.
      }
      return response;
    }

    const decoder = new TextDecoder();
    const state = { buffer: '' };
    const body = response.body.pipeThrough(
      new TransformStream<Uint8Array, Uint8Array>({
        transform(chunk, controller) {
          consumeSseText(context, state, decoder.decode(chunk, { stream: true }));
          controller.enqueue(chunk);
        },
        flush() {
          consumeSseText(context, state, decoder.decode());
          if (state.buffer.trim()) {
            consumeSseText(context, state, '\n\n');
          }
        },
      }),
    );

    return new Response(body, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  }
}

function isThoughtBlock(block: unknown): boolean {
  if (!block || typeof block !== 'object') return false;
  const value = block as Record<string, unknown>;
  return (
    value.thought === true ||
    value.type === 'thinking' ||
    value.type === 'reasoning' ||
    asRecord(value.thought) !== undefined ||
    asRecord(value.thoughtSummary) !== undefined ||
    asRecord(value.thought_summary) !== undefined
  );
}

function extractThoughts(content: unknown): unknown[] {
  return Array.isArray(content)
    ? content.flatMap((block) => normalizeRawThoughtPart(block))
    : [];
}

function removeThoughtsFromContent(content: unknown): unknown {
  return Array.isArray(content) ? content.filter((block) => !isThoughtBlock(block)) : content;
}

function hideThoughts(message: any): void {
  message.content = removeThoughtsFromContent(message.content);

  if (message.response_metadata && typeof message.response_metadata === 'object') {
    const { thoughts: _thoughts, ...responseMetadata } = message.response_metadata;
    message.response_metadata = responseMetadata;
  }
  if (message.additional_kwargs && typeof message.additional_kwargs === 'object') {
    const { thoughts: _thoughts, ...additionalKwargs } = message.additional_kwargs;
    message.additional_kwargs = additionalKwargs;
  }
}

function responseMetadataForExposure(
  response: GeminiResponseMetadata,
  includeThoughts: boolean,
): GeminiResponseMetadata {
  if (includeThoughts) return response;
  const { thoughts: _thoughts, ...metadata } = response;
  return metadata;
}

function contentToText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';

  return content
    .map((block) => {
      if (typeof block === 'string') return block;
      if (!block || typeof block !== 'object') return '';
      const text = (block as Record<string, unknown>).text;
      return typeof text === 'string' ? text : '';
    })
    .join('');
}

function hasToolCalls(message: any): boolean {
  return (
    (Array.isArray(message?.tool_calls) && message.tool_calls.length > 0) ||
    (Array.isArray(message?.invalid_tool_calls) &&
      message.invalid_tool_calls.length > 0) ||
    (Array.isArray(message?.additional_kwargs?.tool_calls) &&
      message.additional_kwargs.tool_calls.length > 0)
  );
}

function hasMeaningfulContent(content: unknown): boolean {
  if (typeof content === 'string') return content.trim().length > 0;
  if (!Array.isArray(content)) return content !== undefined && content !== null;

  return content.some((block) => {
    if (typeof block === 'string') return block.trim().length > 0;
    const value = asRecord(block);
    if (!value || isThoughtBlock(value)) return false;
    if (typeof value.text === 'string') return value.text.trim().length > 0;
    if (
      asRecord(value.functionCall) ||
      asRecord(value.function_call) ||
      value.type === 'tool_call'
    ) {
      return true;
    }
    return Object.keys(value).some(
      (key) => !['thoughtSignature', 'thought_signature'].includes(key),
    );
  });
}

function generationHasMeaningfulOutput(generation: any): boolean {
  if (typeof generation?.text === 'string' && generation.text.trim().length > 0) {
    return true;
  }
  return (
    hasToolCalls(generation?.message) ||
    hasMeaningfulContent(generation?.message?.content)
  );
}

function resultHasMeaningfulOutput(
  result: ChatResult,
  context: CaptureContext,
): boolean {
  return (
    context.hasFunctionCall ||
    context.hasNonThoughtOutput ||
    result.generations.some(generationHasMeaningfulOutput)
  );
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values.filter((value) => value.length > 0))];
}

function responseFinishReasons(
  result: ChatResult,
  context: CaptureContext,
): string[] {
  return uniqueStrings([
    ...context.finishReasons,
    ...result.generations.flatMap((generation) => {
      const reason = (generation as any)?.generationInfo?.finishReason;
      return typeof reason === 'string' ? [reason] : [];
    }),
  ]);
}

function responseFinishMessages(
  result: ChatResult,
  context: CaptureContext,
): string[] {
  return uniqueStrings([
    ...context.finishMessages,
    ...result.generations.flatMap((generation) => {
      const message = (generation as any)?.generationInfo?.finishMessage;
      return typeof message === 'string' ? [message] : [];
    }),
  ]);
}

function isRecoverableEmptyResponse(
  result: ChatResult,
  context: CaptureContext,
): boolean {
  if (resultHasMeaningfulOutput(result, context)) return false;
  const finishReasons = responseFinishReasons(result, context);
  return (
    finishReasons.length === 0 ||
    finishReasons.every((reason) => reason.toUpperCase() === 'STOP')
  );
}

function emptyResponseError(
  result: ChatResult,
  context: CaptureContext,
  recoveryAttempted: boolean,
): Error {
  const finishReasons = responseFinishReasons(result, context);
  const finishMessages = responseFinishMessages(result, context);
  const blocked = finishReasons.some((reason) =>
    [
      'SAFETY',
      'RECITATION',
      'BLOCKLIST',
      'PROHIBITED_CONTENT',
      'SPII',
      'IMAGE_SAFETY',
    ].includes(reason.toUpperCase()),
  );
  const suffix = [
    finishReasons.length > 0
      ? `Finish reason: ${finishReasons.join(', ')}.`
      : undefined,
    finishMessages.length > 0
      ? `Provider detail: ${finishMessages.join(' | ')}`
      : undefined,
  ]
    .filter(Boolean)
    .join(' ');
  const error = new Error(
    blocked
      ? `Gemini returned no usable content because the response was blocked. ${suffix}`.trim()
      : `Gemini returned an empty final response with no text or function call${
          recoveryAttempted ? ' after one automatic recovery attempt' : ''
        }. ${suffix}`.trim(),
  );
  error.name = blocked ? 'PromptBlockedError' : 'NoCandidatesError';
  Object.assign(error, {
    finishReasons,
    finishMessages,
    responseId: context.response?.responseId,
    usageMetadata: context.response?.usageMetadata,
    emptyResponseRecoveryAttempted: recoveryAttempted,
  });
  return error;
}

const usageCountFields = [
  'promptTokenCount',
  'cachedContentTokenCount',
  'candidatesTokenCount',
  'toolUsePromptTokenCount',
  'thoughtsTokenCount',
  'totalTokenCount',
] as const;

const usageDetailFields = [
  'promptTokensDetails',
  'cacheTokensDetails',
  'candidatesTokensDetails',
  'toolUsePromptTokensDetails',
] as const;

function mergeModalityDetails(
  values: Array<GeminiModalityTokenCount[] | undefined>,
): GeminiModalityTokenCount[] | undefined {
  const totals = new Map<string, GeminiModalityTokenCount>();
  let unnamed = 0;

  for (const details of values) {
    for (const detail of details ?? []) {
      const key = detail.modality ?? `__unnamed_${unnamed++}`;
      const existing = totals.get(key);
      totals.set(key, {
        ...existing,
        ...structuredClone(detail),
        tokenCount: (existing?.tokenCount ?? 0) + (detail.tokenCount ?? 0),
      });
    }
  }

  return totals.size > 0 ? [...totals.values()] : undefined;
}

function mergeUsageMetadata(
  responses: Array<GeminiResponseMetadata | undefined>,
): GeminiUsageMetadata | undefined {
  const usages = responses
    .map((response) => response?.usageMetadata)
    .filter((usage): usage is GeminiUsageMetadata => usage !== undefined);
  if (usages.length === 0) return undefined;

  const merged: GeminiUsageMetadata = { ...structuredClone(usages.at(-1)!) };
  for (const field of usageCountFields) {
    const values = usages
      .map((usage) => usage[field])
      .filter((value): value is number => typeof value === 'number');
    if (values.length > 0) merged[field] = values.reduce((sum, value) => sum + value, 0);
  }
  for (const field of usageDetailFields) {
    const details = mergeModalityDetails(usages.map((usage) => usage[field]));
    if (details) merged[field] = details;
    else delete merged[field];
  }
  return merged;
}

function mergeResponseMetadata(
  contexts: CaptureContext[],
): GeminiResponseMetadata | undefined {
  const responses = contexts
    .map((context) => context.response)
    .filter((response): response is GeminiResponseMetadata => response !== undefined);
  if (responses.length === 0) return undefined;

  const responseIds = uniqueStrings(
    responses.flatMap((response) =>
      typeof response.responseId === 'string' ? [response.responseId] : [],
    ),
  );
  const thoughts = mergeUniqueThoughts(
    [],
    responses.flatMap((response) =>
      Array.isArray(response.thoughts) ? response.thoughts : [],
    ),
  );
  const usageMetadata = mergeUsageMetadata(responses);
  const merged: GeminiResponseMetadata = {
    ...structuredClone(responses.at(-1)!),
    ...(usageMetadata ? { usageMetadata } : {}),
    ...(thoughts.length > 0 ? { thoughts } : {}),
  };

  if (contexts.length > 1) {
    merged.emptyResponseRecovery = {
      attempted: true,
      providerRequests: contexts.length,
      emptyResponses: contexts.length - 1,
      ...(responseIds.length > 0 ? { responseIds } : {}),
    };
  }
  return merged;
}

const EMPTY_RESPONSE_RECOVERY_PROMPT =
  'The previous model turn ended with STOP but contained no text and no function call. Continue from the complete conversation and tool results above. Function calling is disabled for this recovery turn because the previous turn did not request a tool. Return a non-empty final answer to the user. Do not repeat or describe function calls whose results are already present.';

function mergeUniqueThoughts(existing: unknown, incoming: unknown[]): unknown[] {
  const thoughts = [...(Array.isArray(existing) ? existing : []), ...incoming];
  const seen = new Set<string>();

  return thoughts.filter((thought) => {
    const record = asRecord(thought);
    const text = thoughtSummaryText(thought);
    const signature =
      typeof record?.thoughtSignature === 'string'
        ? record.thoughtSignature
        : typeof record?.signature === 'string'
          ? record.signature
          : '';
    const key = text
      ? JSON.stringify({ text, signature })
      : JSON.stringify(thought);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function preserveThoughtsAndHideFromContent(
  message: any,
  capturedThoughts: unknown[] = [],
): void {
  const thoughts = mergeUniqueThoughts(
    extractThoughts(message.content),
    capturedThoughts,
  );
  const existingThoughts = message.response_metadata?.thoughts;

  if (thoughts.length > 0) {
    message.response_metadata = {
      ...message.response_metadata,
      thoughts: mergeUniqueThoughts(existingThoughts, thoughts),
    };
  }

  message.content = removeThoughtsFromContent(message.content);
}

function augmentMessage(
  message: any,
  response: GeminiResponseMetadata,
  includeThoughts: boolean,
): void {
  const rawUsage = response.usageMetadata;
  if (includeThoughts) {
    preserveThoughtsAndHideFromContent(message, response.thoughts);
  } else {
    hideThoughts(message);
  }

  if (rawUsage) {
    const existingUsage = message.usage_metadata ?? {};
    message.usage_metadata = {
      ...existingUsage,
      input_tokens: rawUsage.promptTokenCount ?? existingUsage.input_tokens ?? 0,
      output_tokens:
        rawUsage.candidatesTokenCount !== undefined || rawUsage.thoughtsTokenCount !== undefined
          ? (rawUsage.candidatesTokenCount ?? 0) + (rawUsage.thoughtsTokenCount ?? 0)
          : existingUsage.output_tokens ?? 0,
      total_tokens: rawUsage.totalTokenCount ?? existingUsage.total_tokens ?? 0,
      input_token_details: {
        ...existingUsage.input_token_details,
        cache_read:
          rawUsage.cachedContentTokenCount ??
          existingUsage.input_token_details?.cache_read ??
          0,
        tool_use:
          rawUsage.toolUsePromptTokenCount ??
          existingUsage.input_token_details?.tool_use ??
          0,
      },
      output_token_details: {
        ...existingUsage.output_token_details,
        text:
          rawUsage.candidatesTokenCount ??
          existingUsage.output_token_details?.text ??
          0,
        reasoning:
          rawUsage.thoughtsTokenCount ??
          existingUsage.output_token_details?.reasoning ??
          0,
      },
    };
  }

  const gemini = {
    ...responseMetadataForExposure(response, includeThoughts),
    ...(rawUsage
      ? {
          tokenUsage: {
            input: rawUsage.promptTokenCount,
            cache: rawUsage.cachedContentTokenCount,
            output: rawUsage.candidatesTokenCount,
            tool: rawUsage.toolUsePromptTokenCount,
            thoughts: rawUsage.thoughtsTokenCount,
            total: rawUsage.totalTokenCount,
          },
        }
      : {}),
  };

  message.response_metadata = {
    ...message.response_metadata,
    gemini,
  };
  message.additional_kwargs = {
    ...message.additional_kwargs,
    gemini,
  };

}

function augmentChatResult(
  result: ChatResult,
  response: GeminiResponseMetadata | undefined,
  includeThoughts: boolean,
): void {
  if (!response) return;

  for (const generation of result.generations) {
    if (generation.message) {
      augmentMessage(generation.message, response, includeThoughts);
      generation.text = contentToText(generation.message.content);
    }
  }

  result.llmOutput = {
    ...result.llmOutput,
    gemini: {
      ...responseMetadataForExposure(response, includeThoughts),
      ...(response.usageMetadata
        ? {
            tokenUsage: {
              input: response.usageMetadata.promptTokenCount,
              cache: response.usageMetadata.cachedContentTokenCount,
              output: response.usageMetadata.candidatesTokenCount,
              tool: response.usageMetadata.toolUsePromptTokenCount,
              thoughts: response.usageMetadata.thoughtsTokenCount,
              total: response.usageMetadata.totalTokenCount,
            },
          }
        : {}),
    },
  };
}

export function formatGeminiUsage(metadata: GeminiUsageMetadata): string {
  const fields: Array<[string, number | undefined]> = [
    ['Input', metadata.promptTokenCount],
    ['Cache', metadata.cachedContentTokenCount],
    ['Output', metadata.candidatesTokenCount],
    ['Tool', metadata.toolUsePromptTokenCount],
    ['Thoughts', metadata.thoughtsTokenCount],
    ['Total', metadata.totalTokenCount],
  ];

  return `Token Usage — ${fields
    .filter(([, value]) => value !== undefined)
    .map(([label, value]) => `${label}: ${value}`)
    .join(' | ')}`;
}

export class GeminiChatModel extends ChatGoogle {
  private readonly onUsage?: (metadata: GeminiUsageMetadata) => void;
  private readonly explicitThinkingConfig?: Record<string, unknown>;
  private readonly explicitResponseMimeType?: string;
  private readonly includeThoughts: boolean;
  private readonly recoverEmptyResponses: boolean;

  constructor(fields: Record<string, unknown>, onUsage?: (metadata: GeminiUsageMetadata) => void) {
    const apiKey = String(fields.apiKey ?? '');
    super({
      ...fields,
      model: String(fields.model ?? ''),
      apiKey,
      apiClient: new CapturingGeminiApiClient(apiKey) as any,
    } as any);
    this.onUsage = onUsage;
    this.explicitThinkingConfig =
      fields.thinkingConfig && typeof fields.thinkingConfig === 'object'
        ? (fields.thinkingConfig as Record<string, unknown>)
        : undefined;
    this.includeThoughts = this.explicitThinkingConfig?.includeThoughts === true;
    this.recoverEmptyResponses = fields.recoverEmptyResponses !== false;
    this.explicitResponseMimeType =
      typeof fields.responseMimeType === 'string' ? fields.responseMimeType : undefined;
  }

  override invocationParams(options: this['ParsedCallOptions']): any {
    const params = super.invocationParams(options) as any;

    if (this.explicitThinkingConfig && Object.keys(this.explicitThinkingConfig).length > 0) {
      params.generationConfig.thinkingConfig = this.explicitThinkingConfig;
    }
    if (this.explicitResponseMimeType) {
      params.generationConfig.responseMimeType = this.explicitResponseMimeType;
    }

    return params;
  }

  override async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    const runAttempt = async (
      attemptMessages: BaseMessage[],
      attemptOptions: this['ParsedCallOptions'] = options,
    ) => {
      const context = createCaptureContext();
      const result = await responseCapture.run(context, () =>
        super._generate(attemptMessages, attemptOptions, runManager),
      );
      return { result, context };
    };

    const first = await runAttempt(messages);
    let finalAttempt = first;
    const contexts = [first.context];

    if (
      this.recoverEmptyResponses &&
      isRecoverableEmptyResponse(first.result, first.context)
    ) {
      const recovery = await runAttempt(
        [
          ...messages,
          new HumanMessage(EMPTY_RESPONSE_RECOVERY_PROMPT),
        ],
        {
          ...options,
          tool_choice: 'none',
        } as this['ParsedCallOptions'],
      );
      contexts.push(recovery.context);
      finalAttempt = recovery;
    }

    if (!resultHasMeaningfulOutput(finalAttempt.result, finalAttempt.context)) {
      throw emptyResponseError(
        finalAttempt.result,
        finalAttempt.context,
        contexts.length > 1,
      );
    }

    const response = mergeResponseMetadata(contexts);
    augmentChatResult(finalAttempt.result, response, this.includeThoughts);
    if (response?.usageMetadata) this.onUsage?.(response.usageMetadata);
    return finalAttempt.result;
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    const createIterator = (
      attemptMessages: BaseMessage[],
      context: CaptureContext,
      attemptOptions: this['ParsedCallOptions'] = options,
    ) =>
      responseCapture.run(context, () =>
        super._streamResponseChunks(attemptMessages, attemptOptions, runManager),
      );
    const streamAttempt = async function* (
      model: GeminiChatModel,
      attemptMessages: BaseMessage[],
      context: CaptureContext,
      attemptOptions: GeminiChatModel['ParsedCallOptions'] = options,
    ): AsyncGenerator<ChatGenerationChunk> {
      const iterator = createIterator(attemptMessages, context, attemptOptions);

      let pending: ChatGenerationChunk | undefined;
      while (true) {
        const next = await responseCapture.run(context, () => iterator.next());
        if (next.done) break;
        if (pending) {
          // Intermediate chunks may contain thought summaries, but the final API
          // usage must only be attached once or LangChain will sum it repeatedly.
          if (model.includeThoughts) {
            preserveThoughtsAndHideFromContent(pending.message);
          } else {
            hideThoughts(pending.message);
          }
          pending.text = contentToText(pending.message.content);
          yield pending;
        }
        pending = next.value;
      }

      if (pending) {
        if (context.response) {
          augmentMessage(pending.message, context.response, model.includeThoughts);
        } else if (!model.includeThoughts) {
          hideThoughts(pending.message);
        }
        pending.text = contentToText(pending.message.content);
        yield pending;
      }
    };

    const firstContext = createCaptureContext();
    const firstChunks: ChatGenerationChunk[] = [];
    let firstHasMeaningfulOutput = false;

    for await (const chunk of streamAttempt(this, messages, firstContext)) {
      if (!firstHasMeaningfulOutput) {
        firstChunks.push(chunk);
        firstHasMeaningfulOutput =
          firstContext.hasFunctionCall ||
          firstContext.hasNonThoughtOutput ||
          generationHasMeaningfulOutput(chunk);
        if (firstHasMeaningfulOutput) {
          for (const bufferedChunk of firstChunks) yield bufferedChunk;
          firstChunks.length = 0;
        }
      } else {
        yield chunk;
      }
    }

    if (firstHasMeaningfulOutput) {
      if (firstContext.response?.usageMetadata) {
        this.onUsage?.(firstContext.response.usageMetadata);
      }
      return;
    }

    const firstResult: ChatResult = { generations: firstChunks };
    if (
      !this.recoverEmptyResponses ||
      !isRecoverableEmptyResponse(firstResult, firstContext)
    ) {
      throw emptyResponseError(firstResult, firstContext, false);
    }

    // Nothing meaningful from the first stream was emitted, so a continuation
    // can be made without duplicating text or tool calls.
    const recoveryContext = createCaptureContext();
    const recoveryChunks: ChatGenerationChunk[] = [];
    for await (const chunk of streamAttempt(
      this,
      [
        ...messages,
        new HumanMessage(EMPTY_RESPONSE_RECOVERY_PROMPT),
      ],
      recoveryContext,
      {
        ...options,
        tool_choice: 'none',
      } as this['ParsedCallOptions'],
    )) {
      recoveryChunks.push(chunk);
    }

    const recoveryResult: ChatResult = { generations: recoveryChunks };
    if (!resultHasMeaningfulOutput(recoveryResult, recoveryContext)) {
      throw emptyResponseError(recoveryResult, recoveryContext, true);
    }

    const response = mergeResponseMetadata([firstContext, recoveryContext]);
    const lastChunk = recoveryChunks.at(-1);
    if (lastChunk && response) {
      // The blank first stream was intentionally discarded. Attach aggregate
      // usage to the final recovery chunk so n8n reports every billed token
      // exactly once.
      augmentMessage(lastChunk.message, response, this.includeThoughts);
      lastChunk.text = contentToText(lastChunk.message.content);
    }
    if (response?.usageMetadata) this.onUsage?.(response.usageMetadata);
    for (const chunk of recoveryChunks) yield chunk;
  }
}
