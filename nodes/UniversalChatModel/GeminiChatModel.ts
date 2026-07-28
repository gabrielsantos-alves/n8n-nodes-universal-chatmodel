import { AsyncLocalStorage } from 'node:async_hooks';
import { ChatGoogle } from '@langchain/google/node';
import type { BaseMessage } from '@langchain/core/messages';
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
    const context: CaptureContext = {};
    return responseCapture.run(context, async () => {
      const result = await super._generate(messages, options, runManager);
      augmentChatResult(result, context.response, this.includeThoughts);
      if (context.response?.usageMetadata) this.onUsage?.(context.response.usageMetadata);
      return result;
    });
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    const context: CaptureContext = {};
    const iterator = responseCapture.run(context, () =>
      super._streamResponseChunks(messages, options, runManager),
    );

    let pending: ChatGenerationChunk | undefined;
    while (true) {
      const next = await responseCapture.run(context, () => iterator.next());
      if (next.done) break;
      if (pending) {
        // Intermediate chunks may contain thought summaries, but the final API
        // usage must only be attached once or LangChain will sum it repeatedly.
        if (this.includeThoughts) {
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
        augmentMessage(pending.message, context.response, this.includeThoughts);
      } else if (!this.includeThoughts) {
        hideThoughts(pending.message);
      }
      pending.text = contentToText(pending.message.content);
      if (context.response?.usageMetadata) this.onUsage?.(context.response.usageMetadata);
      yield pending;
    }
  }
}
