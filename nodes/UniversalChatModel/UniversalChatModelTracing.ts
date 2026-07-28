import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
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

function serializeGeneration(generation: any): IDataObject {
  const rawText = generation?.text ?? '';
  const structuredOutput =
    parseStructuredOutput(rawText) ??
    parseStructuredOutput(generation?.message?.content);
  const text =
    typeof rawText === 'string'
      ? rawText
      : rawText === undefined || rawText === null
        ? ''
        : JSON.stringify(rawText);

  return {
    text,
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
    private readonly includeTokenUsageInAgentOutput = true,
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
    const generations = (output.generations ?? []).map((generationList) =>
      (generationList ?? []).map(serializeGeneration),
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
