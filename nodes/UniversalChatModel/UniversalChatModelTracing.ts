import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import type { BaseMessage } from '@langchain/core/messages';
import type { LLMResult } from '@langchain/core/outputs';
import {
  NodeConnectionTypes,
  NodeOperationError,
  type IDataObject,
  type ISupplyDataFunctions,
} from 'n8n-workflow';

import type { GeminiResponseMetadata, GeminiUsageMetadata } from './GeminiChatModel';
import { recordAgentModelMetadata } from './AgentOutputBridge';
import {
  type ModelProvider,
  formatNormalizedModelError,
  normalizeModelError,
} from './ModelError';
import { parseStructuredOutput } from './StructuredOutput';

interface RunDetails {
  index: number;
  prompts: string[];
}

export interface ModelUsageReportEvent {
  provider: ModelProvider;
  prompts: string[];
  outputText: string;
  tokenUsage: UniversalTokenUsage;
  usageMetadata?: IDataObject;
  gemini?: IDataObject;
}

export type ModelUsageReporter = (
  event: ModelUsageReportEvent,
) => Promise<void>;

export interface UniversalTokenUsage extends IDataObject {
  inputTokens: number;
  inputUncachedTokens: number;
  outputTokens: number;
  cachedTokens: number;
  toolUsePromptTokens: number;
  thoughtsTokens: number;
  totalTokens: number;
  inputTokenDetails?: IDataObject;
  outputTokenDetails?: IDataObject;
  promptTokensDetails?: IDataObject[];
  cacheTokensDetails?: IDataObject[];
  candidatesTokensDetails?: IDataObject[];
  toolUsePromptTokensDetails?: IDataObject[];
}

function asRecord(value: unknown): Record<string, any> | undefined {
  return value !== null && typeof value === 'object'
    ? (value as Record<string, any>)
    : undefined;
}

function messageType(message: unknown): string {
  const record = asRecord(message);
  if (!record) return '';
  if (typeof record.type === 'string') return record.type.toLowerCase();
  if (typeof record.getType === 'function') {
    try {
      const type = record.getType();
      if (typeof type === 'string') return type.toLowerCase();
    } catch {
      // Fall through to legacy type detection.
    }
  }
  if (typeof record._getType === 'function') {
    try {
      const type = record._getType();
      if (typeof type === 'string') return type.toLowerCase();
    } catch {
      // Ignore malformed third-party message implementations.
    }
  }
  const constructorName = record.constructor?.name;
  return typeof constructorName === 'string'
    ? constructorName.replace(/Message(?:Chunk)?$/i, '').toLowerCase()
    : '';
}

function messageContentText(message: unknown): string {
  const record = asRecord(message);
  if (!record) return '';
  if (typeof record.content === 'string') return record.content;
  if (!Array.isArray(record.content)) return '';

  return record.content
    .map((block: unknown) => {
      if (typeof block === 'string') return block;
      const value = asRecord(block);
      if (!value) return '';
      if (typeof value.text === 'string') return value.text;
      const nestedText = asRecord(value.text);
      return typeof nestedText?.text === 'string' ? nestedText.text : '';
    })
    .filter((text: string) => text.length > 0)
    .join('\n');
}

function latestHumanInput(messages: BaseMessage[][]): string[] {
  const flattened = messages.flat();
  for (let index = flattened.length - 1; index >= 0; index -= 1) {
    const message = flattened[index];
    if (!['human', 'user'].includes(messageType(message))) continue;
    const text = messageContentText(message).trim();
    if (text.length > 0) return [text];
  }
  return [];
}

function serializeInputMessages(messages: BaseMessage[][]): IDataObject[] {
  return messages.flat().map((message) => ({
    role: messageType(message),
    content: messageContentText(message),
  }));
}

function findMessages(output: LLMResult): Record<string, any>[] {
  const messages: Record<string, any>[] = [];

  for (const generationList of output.generations ?? []) {
    for (const generation of generationList ?? []) {
      const message = asRecord((generation as any).message);
      if (message) messages.push(message);
    }
  }

  return messages;
}

function findGeminiMetadata(output: LLMResult): GeminiResponseMetadata | undefined {
  const llmOutput = asRecord(output.llmOutput);
  const directGemini = asRecord(llmOutput?.gemini);
  if (directGemini) return directGemini as GeminiResponseMetadata;

  for (const message of findMessages(output)) {
    const responseMetadata = asRecord(message.response_metadata);
    const gemini = asRecord(responseMetadata?.gemini);
    if (gemini) return gemini as GeminiResponseMetadata;

    const additionalKwargs = asRecord(message.additional_kwargs);
    const additionalGemini = asRecord(additionalKwargs?.gemini);
    if (additionalGemini) return additionalGemini as GeminiResponseMetadata;
  }

  const usageMetadata = asRecord(llmOutput?.usageMetadata);
  return usageMetadata
    ? { usageMetadata: usageMetadata as GeminiUsageMetadata }
    : undefined;
}

function findNormalizedUsage(output: LLMResult): Record<string, any> | undefined {
  for (const message of findMessages(output)) {
    const usage = asRecord(message.usage_metadata);
    if (usage) return usage;
  }

  return asRecord(asRecord(output.llmOutput)?.tokenUsage);
}

export function normalizeTokenUsage(output: LLMResult): UniversalTokenUsage {
  const geminiUsage = findGeminiMetadata(output)?.usageMetadata;
  const normalized = findNormalizedUsage(output);
  const normalizedInputDetails = asRecord(normalized?.input_token_details);
  const normalizedOutputDetails = asRecord(normalized?.output_token_details);

  const inputTokens =
    geminiUsage?.promptTokenCount ??
    normalized?.input_tokens ??
    normalized?.promptTokens ??
    0;
  const cachedTokens =
    geminiUsage?.cachedContentTokenCount ??
    normalizedInputDetails?.cache_read ??
    0;
  const thoughtsTokens =
    geminiUsage?.thoughtsTokenCount ??
    normalizedOutputDetails?.reasoning ??
    0;
  const outputTokens =
    geminiUsage?.candidatesTokenCount ??
    normalizedOutputDetails?.text ??
    normalized?.completionTokens ??
    Math.max((normalized?.output_tokens ?? 0) - thoughtsTokens, 0);
  const toolUsePromptTokens =
    geminiUsage?.toolUsePromptTokenCount ??
    normalizedInputDetails?.tool_use ??
    0;
  const totalTokens =
    geminiUsage?.totalTokenCount ??
    normalized?.total_tokens ??
    normalized?.totalTokens ??
    inputTokens + outputTokens + thoughtsTokens;

  return {
    inputTokens,
    inputUncachedTokens: Math.max(inputTokens - cachedTokens, 0),
    outputTokens,
    cachedTokens,
    toolUsePromptTokens,
    thoughtsTokens,
    totalTokens,
    ...(normalizedInputDetails
      ? { inputTokenDetails: structuredClone(normalizedInputDetails) as IDataObject }
      : {}),
    ...(normalizedOutputDetails
      ? { outputTokenDetails: structuredClone(normalizedOutputDetails) as IDataObject }
      : {}),
    ...(geminiUsage?.promptTokensDetails
      ? { promptTokensDetails: structuredClone(geminiUsage.promptTokensDetails) as IDataObject[] }
      : {}),
    ...(geminiUsage?.cacheTokensDetails
      ? { cacheTokensDetails: structuredClone(geminiUsage.cacheTokensDetails) as IDataObject[] }
      : {}),
    ...(geminiUsage?.candidatesTokensDetails
      ? {
          candidatesTokensDetails: structuredClone(
            geminiUsage.candidatesTokensDetails,
          ) as IDataObject[],
        }
      : {}),
    ...(geminiUsage?.toolUsePromptTokensDetails
      ? {
          toolUsePromptTokensDetails: structuredClone(
            geminiUsage.toolUsePromptTokensDetails,
          ) as IDataObject[],
        }
      : {}),
  };
}

function extractThoughtsFromContent(content: unknown): unknown[] {
  if (!Array.isArray(content)) return [];

  return content.filter((block) => {
    const value = asRecord(block);
    return (
      value?.thought === true ||
      value?.type === 'thinking' ||
      value?.type === 'reasoning'
    );
  });
}

function parseToolArguments(value: unknown): unknown {
  if (typeof value !== 'string') return value ?? {};

  try {
    return JSON.parse(value);
  } catch {
    // Invalid/incomplete tool calls can legitimately contain arguments that
    // are not valid JSON yet. Keep the original value observable instead of
    // discarding it or changing the message delivered to the AI consumer.
    return value;
  }
}

function normalizeObservableToolCall(value: unknown): IDataObject | undefined {
  const record = asRecord(value);
  if (!record) return undefined;

  const geminiCall = asRecord(record.functionCall) ?? asRecord(record.function_call);
  const openAiCall = asRecord(record.function);
  const call = geminiCall ?? openAiCall ?? record;
  const name =
    typeof call.name === 'string'
      ? call.name
      : typeof record.name === 'string'
        ? record.name
        : undefined;
  if (!name) return undefined;

  const rawArgs =
    call.args !== undefined
      ? call.args
      : call.arguments !== undefined
        ? call.arguments
        : record.args !== undefined
          ? record.args
          : record.arguments;
  const id =
    typeof call.id === 'string'
      ? call.id
      : typeof record.id === 'string'
        ? record.id
        : typeof record.tool_call_id === 'string'
          ? record.tool_call_id
          : undefined;
  return {
    type: 'functionCall',
    functionCall: {
      name,
      args: structuredClone(parseToolArguments(rawArgs)) as any,
      ...(id ? { id } : {}),
    },
  };
}

/**
 * Returns a provider-neutral, Gemini-compatible view of every tool call in a
 * generation. This is observability data only: the underlying AIMessage is
 * never changed, so n8n's agent continues to execute message.tool_calls.
 */
export function extractObservableToolCalls(
  generation: any,
  fallbackCalls: unknown[] = [],
): IDataObject[] {
  const capturedProviderCalls = fallbackCalls
    .map(normalizeObservableToolCall)
    .filter((call): call is IDataObject => call !== undefined);
  if (capturedProviderCalls.length > 0) {
    // The provider payload is authoritative. LangChain can replace a Gemini
    // function-call ID with its own run ID while adapting the same call, which
    // would otherwise make one invocation appear twice in observability data.
    return capturedProviderCalls;
  }

  const message = asRecord(generation?.message);
  const generationRecord = asRecord(generation);
  const additionalKwargs = asRecord(message?.additional_kwargs);
  const responseMetadata = asRecord(message?.response_metadata);
  const generationInfo = asRecord(generationRecord?.generationInfo);
  const content = Array.isArray(message?.content) ? message.content : [];
  const contentBlocks = Array.isArray(message?.contentBlocks)
    ? message.contentBlocks
    : [];
  const candidates = [
    ...(Array.isArray(generationRecord?.tool_call_chunks)
      ? generationRecord.tool_call_chunks
      : []),
    ...(Array.isArray(message?.tool_call_chunks) ? message.tool_call_chunks : []),
    ...(Array.isArray(generationInfo?.tool_call_chunks)
      ? generationInfo.tool_call_chunks
      : []),
    ...(Array.isArray(generationRecord?.tool_calls)
      ? generationRecord.tool_calls
      : []),
    ...(Array.isArray(message?.tool_calls) ? message.tool_calls : []),
    ...(Array.isArray(message?.invalid_tool_calls) ? message.invalid_tool_calls : []),
    ...(Array.isArray(additionalKwargs?.tool_calls) ? additionalKwargs.tool_calls : []),
    ...(additionalKwargs?.function_call ? [additionalKwargs.function_call] : []),
    ...(Array.isArray(responseMetadata?.tool_calls) ? responseMetadata.tool_calls : []),
    ...(Array.isArray(generationInfo?.tool_calls) ? generationInfo.tool_calls : []),
    ...[...content, ...contentBlocks].filter((block) => {
      const record = asRecord(block);
      return Boolean(
        record &&
          (record.type === 'tool_call' ||
            asRecord(record.functionCall) ||
            asRecord(record.function_call) ||
            asRecord(record.function)),
      );
    }),
  ];

  const calls: IDataObject[] = [];
  const seen = new Map<string, number>();
  for (const candidate of candidates) {
    const call = normalizeObservableToolCall(candidate);
    if (!call) continue;
    const functionCall = asRecord(call.functionCall)!;
    const key =
      typeof functionCall.id === 'string' && functionCall.id.length > 0
        ? `id:${functionCall.id}`
        : JSON.stringify({
            name: functionCall.name,
            args: functionCall.args,
          });
    const existingIndex = seen.get(key);
    if (existingIndex !== undefined) {
      calls[existingIndex] = call;
      continue;
    }
    seen.set(key, calls.length);
    calls.push(call);
  }

  return calls;
}

export function observableGenerationText(
  generation: any,
  fallbackCalls: unknown[] = [],
): string {
  const rawText = generation?.text ?? '';
  const text =
    typeof rawText === 'string'
      ? rawText
      : rawText === undefined || rawText === null
        ? ''
        : JSON.stringify(rawText);
  const toolCalls = extractObservableToolCalls(generation, fallbackCalls);
  if (toolCalls.length === 0) return text;

  const serializedCalls = JSON.stringify(toolCalls);
  return text.trim().length > 0
    ? `${text}\n${serializedCalls}`
    : serializedCalls;
}

export function extractThoughtsFromResult(output: LLMResult): unknown[] {
  const thoughts: unknown[] = [];

  for (const message of findMessages(output)) {
    thoughts.push(...extractThoughtsFromContent(message.content));

    const responseThoughts = asRecord(message.response_metadata)?.thoughts;
    if (Array.isArray(responseThoughts)) thoughts.push(...responseThoughts);

    const additionalThoughts = asRecord(message.additional_kwargs)?.thoughts;
    if (Array.isArray(additionalThoughts)) thoughts.push(...additionalThoughts);
  }

  const seen = new Set<string>();
  return thoughts.filter((thought) => {
    const record = asRecord(thought);
    const text =
      typeof record?.text === 'string'
        ? record.text
        : typeof record?.thinking === 'string'
          ? record.thinking
          : typeof record?.reasoning === 'string'
            ? record.reasoning
            : undefined;
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

function serializeGeneration(
  generation: any,
  fallbackCalls: unknown[] = [],
): IDataObject {
  const rawText = generation?.text ?? '';
  const toolCalls = extractObservableToolCalls(generation, fallbackCalls);
  const structuredOutput =
    parseStructuredOutput(rawText) ??
    parseStructuredOutput(generation?.message?.content);
  const text = observableGenerationText(generation, fallbackCalls);

  return {
    text,
    ...(toolCalls.length > 0
      ? { toolCalls: structuredClone(toolCalls) as IDataObject[] }
      : {}),
    ...(structuredOutput
      ? { structuredOutput: structuredClone(structuredOutput) }
      : {}),
    ...(generation?.generationInfo !== undefined
      ? { generationInfo: structuredClone(generation.generationInfo) as IDataObject }
      : {}),
  };
}

function compactGeminiMetadata(
  gemini: GeminiResponseMetadata | undefined,
): IDataObject | undefined {
  if (!gemini) return undefined;

  const {
    usageMetadata: _usageMetadata,
    tokenUsage: _tokenUsage,
    thoughts: _thoughts,
    ...metadata
  } = gemini as GeminiResponseMetadata & { tokenUsage?: unknown };

  return Object.keys(metadata).length > 0
    ? (structuredClone(metadata) as IDataObject)
    : undefined;
}

/**
 * Mirrors the lifecycle used by n8n's native chat-model nodes. Calling
 * addInputData/addOutputData makes the AI subnode show its execution status
 * and exposes inspectable run data in the editor.
 */
export class UniversalChatModelTracing extends BaseCallbackHandler {
  name = 'UniversalChatModelTracing';
  awaitHandlers = true;
  raiseError: boolean;

  private readonly runs = new Map<string, RunDetails>();

  constructor(
    private readonly executionFunctions: ISupplyDataFunctions,
    private readonly provider: ModelProvider,
    private readonly includeThoughts = false,
    private readonly includeTokenUsageInAgentOutput = false,
    private readonly includeIntermediateStepsInOutput = false,
    private readonly usageReporter?: ModelUsageReporter,
    failOnReporterError = false,
  ) {
    super();
    // LangChain normally logs callback failures and lets the model call
    // continue. Enable propagation only when the user explicitly selected the
    // strict Usage Reporter policy.
    this.raiseError = failOnReporterError;
  }

  async handleLLMStart(
    _llm: unknown,
    prompts: string[],
    runId: string,
  ): Promise<void> {
    const { index } = this.executionFunctions.addInputData(
      NodeConnectionTypes.AiLanguageModel,
      [[{ json: { messages: prompts } }]],
    );

    this.runs.set(runId, { index, prompts: structuredClone(prompts) });
  }

  async handleChatModelStart(
    _llm: unknown,
    messages: BaseMessage[][],
    runId: string,
  ): Promise<void> {
    const userInputs = latestHumanInput(messages);
    const { index } = this.executionFunctions.addInputData(
      NodeConnectionTypes.AiLanguageModel,
      [[{ json: { messages: serializeInputMessages(messages) } }]],
    );

    // Usage Reporter receives only the current human input. System prompts,
    // previous AI tool calls and ToolMessages remain visible in n8n's trace,
    // but are intentionally excluded from input_text.
    this.runs.set(runId, {
      index,
      prompts: structuredClone(userInputs),
    });
  }

  async handleLLMEnd(output: LLMResult, runId: string): Promise<void> {
    const runDetails = this.runs.get(runId) ?? {
      index: this.executionFunctions.getNextRunIndex(),
      prompts: [],
    };
    const gemini = findGeminiMetadata(output);
    const geminiMetadata = compactGeminiMetadata(gemini);
    const thoughts = this.includeThoughts
      ? extractThoughtsFromResult(output)
      : [];
    const tokenUsage = normalizeTokenUsage(output);
    const structuredOutputs = (output.generations ?? []).flatMap(
      (generationList) =>
        (generationList ?? []).flatMap((generation) => {
          const structuredOutput =
            parseStructuredOutput((generation as any)?.text) ??
            parseStructuredOutput((generation as any)?.message?.content);
          return structuredOutput ? [structuredOutput] : [];
        }),
    );
    const structuredOutput =
      structuredOutputs.length > 0
        ? structuredOutputs[structuredOutputs.length - 1]
        : undefined;
    const fallbackToolCalls = Array.isArray(gemini?.functionCalls)
      ? gemini.functionCalls
      : [];
    const generations = (output.generations ?? []).map((generationList) =>
      (generationList ?? []).map((generation) =>
        serializeGeneration(generation, fallbackToolCalls),
      ),
    );

    const result: IDataObject = {
      response: { generations },
      tokenUsage,
      ...(structuredOutput
        ? { structuredOutput: structuredClone(structuredOutput) }
        : {}),
      ...(structuredOutputs.length > 1
        ? {
            structuredOutputs: structuredClone(
              structuredOutputs,
            ) as IDataObject[],
          }
        : {}),
      ...(thoughts.length > 0 ? { thoughts: structuredClone(thoughts) as any[] } : {}),
      ...(gemini?.usageMetadata
        ? { usageMetadata: structuredClone(gemini.usageMetadata) as IDataObject }
        : {}),
      ...(geminiMetadata ? { gemini: geminiMetadata } : {}),
    };

    recordAgentModelMetadata({
      ...(thoughts.length > 0
        ? { thoughts: structuredClone(thoughts) }
        : {}),
      tokenUsage: structuredClone(tokenUsage),
      includeTokenUsageInAgentOutput: this.includeTokenUsageInAgentOutput,
      includeIntermediateStepsInOutput: this.includeIntermediateStepsInOutput,
      ...(structuredOutput
        ? { structuredOutput: structuredClone(structuredOutput) }
        : {}),
      ...(gemini?.usageMetadata
        ? { usageMetadata: structuredClone(gemini.usageMetadata) as IDataObject }
        : {}),
      ...(geminiMetadata
        ? { gemini: structuredClone(geminiMetadata) as IDataObject }
        : {}),
    });

    this.executionFunctions.addOutputData(
      NodeConnectionTypes.AiLanguageModel,
      runDetails.index,
      [[{ json: result }]],
    );
    this.runs.delete(runId);

    if (this.usageReporter && tokenUsage.totalTokens > 0) {
      const outputText = generations
        .flat()
        .map((generation) =>
          typeof generation.text === 'string' ? generation.text : '',
        )
        .filter((text) => text.length > 0)
        .join('\n');

      await this.usageReporter({
        provider: this.provider,
        prompts: structuredClone(runDetails.prompts),
        outputText,
        tokenUsage: structuredClone(tokenUsage),
        ...(gemini?.usageMetadata
          ? {
              usageMetadata: structuredClone(
                gemini.usageMetadata,
              ) as IDataObject,
            }
          : {}),
        ...(geminiMetadata
          ? { gemini: structuredClone(geminiMetadata) as IDataObject }
          : {}),
      });
    }
  }

  async handleLLMError(error: Error, runId: string): Promise<void> {
    const runDetails = this.runs.get(runId) ?? {
      index: this.executionFunctions.getNextRunIndex(),
      prompts: [],
    };
    const details = normalizeModelError(error, this.provider);
    const nodeError = new NodeOperationError(
      this.executionFunctions.getNode(),
      error,
      {
        functionality: 'configuration-node',
        message: formatNormalizedModelError(details),
        description: [
          details.description,
          `Category: ${details.category}`,
          details.retryable ? 'Retryable: yes' : 'Retryable: no',
          details.requestId ? `Request ID: ${details.requestId}` : undefined,
        ]
          .filter(Boolean)
          .join(' | '),
      },
    );
    (nodeError.context as Record<string, unknown>).modelError = structuredClone(
      details,
    );

    this.executionFunctions.addOutputData(
      NodeConnectionTypes.AiLanguageModel,
      runDetails.index,
      nodeError,
    );
    this.runs.delete(runId);
  }
}
