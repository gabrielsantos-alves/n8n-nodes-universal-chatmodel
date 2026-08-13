import {
  INodeType,
  INodeTypeDescription,
  ISupplyDataFunctions,
  ILoadOptionsFunctions,
  INodePropertyOptions,
  INodeProperties,
  IDataObject,
  NodeOperationError,
  SupplyData,
  NodeConnectionTypes,
} from 'n8n-workflow';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, parse } from 'node:path';
import { ChatOpenAI } from '@langchain/openai';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import {
  AIMessage,
  AIMessageChunk,
  SystemMessage,
  type BaseMessage,
} from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { GeminiChatModel, formatGeminiUsage } from './GeminiChatModel';
import {
  type ModelUsageReporter,
  UniversalChatModelTracing,
} from './UniversalChatModelTracing';
import {
  type ModelProvider,
  type NormalizedModelError,
  isRetryableModelError,
  normalizeModelError,
  retryAfterMsForModelError,
  toUniversalModelError,
} from './ModelError';

interface ModelExecutionSettings {
  alwaysOutputData: boolean;
  executeOnce: boolean;
  retryOnFail: boolean;
  maxTries: number;
  waitBetweenTries: number;
  onError: 'stopWorkflow' | 'continueRegularOutput' | 'continueErrorOutput';
}

interface UsageReportingOptions {
  systemMessage?: string;
  includeTokenUsageInAgentOutput?: boolean;
  includeIntermediateStepsInOutput?: boolean;
  enableUsageReporter?: boolean;
  nodeLabel?: string;
  inputTextMode?: 'label' | 'prompt';
  inputTextLabel?: string;
  includeOutputText?: boolean;
  failOnReporterError?: boolean;
  reporterMaxWaitMs?: number;
  usageReporter?: {
    settings?: UsageReportingOptions & {
      enabled?: boolean;
    };
  };
}

interface UsageReporterTool {
  invoke?: (input: IDataObject) => Promise<unknown>;
  func?: (input: IDataObject) => Promise<unknown>;
}

function asUsageReporterTool(value: unknown): UsageReporterTool | undefined {
  const candidate = Array.isArray(value) ? value[0] : value;
  if (!candidate || typeof candidate !== 'object') return undefined;

  const tool = candidate as UsageReporterTool;
  return typeof tool.invoke === 'function' || typeof tool.func === 'function'
    ? tool
    : undefined;
}

async function createUsageReporter(
  context: ISupplyDataFunctions,
  itemIndex: number,
  modelName: string,
  options: UsageReportingOptions,
): Promise<ModelUsageReporter | undefined> {
  if (options.enableUsageReporter !== true) return undefined;

  const runtimeContext = context as ISupplyDataFunctions & {
    getParentNodes?: ISupplyDataFunctions['getParentNodes'];
    getInputConnectionData?: ISupplyDataFunctions['getInputConnectionData'];
  };

  if (
    typeof runtimeContext.getParentNodes !== 'function' ||
    typeof runtimeContext.getInputConnectionData !== 'function'
  ) {
    return undefined;
  }

  const connectedReporterNodes = runtimeContext.getParentNodes(
    context.getNode().name,
    {
      connectionType: NodeConnectionTypes.AiTool,
      depth: 1,
    },
  );
  if (connectedReporterNodes.length === 0) return undefined;

  const connectedTool = await runtimeContext.getInputConnectionData(
    NodeConnectionTypes.AiTool,
    itemIndex,
  );
  const reporterTool = asUsageReporterTool(connectedTool);
  if (!reporterTool) {
    throw new NodeOperationError(
      context.getNode(),
      'The connected Usage Reporter did not provide a callable AI Tool.',
      {
        functionality: 'configuration-node',
        description:
          'Connect a Workflow Tool or another AI Tool that accepts the usage-report fields.',
      },
    );
  }

  const workflow = context.getWorkflow();
  const executionId = context.getExecutionId();
  const nodeLabel =
    typeof options.nodeLabel === 'string' && options.nodeLabel.trim()
      ? options.nodeLabel.trim()
      : context.getNode().name;
  const inputTextMode = options.inputTextMode ?? 'label';
  const inputTextLabel =
    typeof options.inputTextLabel === 'string' &&
    options.inputTextLabel.trim()
      ? options.inputTextLabel
      : 'RAG';
  const includeOutputText = options.includeOutputText !== false;
  const failOnReporterError = options.failOnReporterError === true;
  const reporterMaxWaitMs =
    typeof options.reporterMaxWaitMs === 'number' &&
    Number.isFinite(options.reporterMaxWaitMs)
      ? Math.max(0, options.reporterMaxWaitMs)
      : 1000;

  return async (event) => {
    const usage = event.tokenUsage;
    const promptText = event.prompts.join('\n');
    const payload: IDataObject = {
      model: modelName,
      input_token: usage.inputTokens,
      input_uncached_token: usage.inputUncachedTokens,
      output_token: usage.outputTokens,
      total_token: usage.totalTokens,
      cached_token: usage.cachedTokens,
      thoughts_token: usage.thoughtsTokens,
      tool_token: usage.toolUsePromptTokens,
      overhead_token:
        usage.thoughtsTokens + usage.toolUsePromptTokens,
      model_calls: 1,
      input_text:
        inputTextMode === 'prompt' ? promptText : inputTextLabel,
      output_text: includeOutputText ? event.outputText : '',
      workflow_id: workflow.id,
      workflow_name: workflow.name ?? '',
      execution_id:
        executionId && executionId !== '__UNKNOWN__' ? executionId : '',
      node: nodeLabel,
      dump: JSON.stringify({
        provider: event.provider,
        model: modelName,
        tokenUsage: usage,
        ...(event.usageMetadata
          ? { usageMetadata: event.usageMetadata }
          : {}),
        ...(event.gemini ? { gemini: event.gemini } : {}),
      }),
    };

    try {
      // ToolWorkflow exposes func(), which preserves extra telemetry fields
      // even before they are added to the workflow input schema.
      const reporterPromise = Promise.resolve().then(() =>
        typeof reporterTool.func === 'function'
          ? reporterTool.func(payload)
          : reporterTool.invoke!(payload),
      );

      // Attach a handler immediately: if the timeout wins, the already-started
      // report may still finish in the background without an unhandled rejection.
      void reporterPromise.catch(() => undefined);

      if (reporterMaxWaitMs === 0) {
        await reporterPromise;
      } else {
        let timer: ReturnType<typeof setTimeout> | undefined;
        try {
          await Promise.race([
            reporterPromise,
            new Promise<never>((_, reject) => {
              timer = setTimeout(() => {
                const timeoutError = new Error(
                  `Usage Reporter exceeded the maximum wait of ${reporterMaxWaitMs} ms`,
                );
                timeoutError.name = 'UsageReporterTimeoutError';
                reject(timeoutError);
              }, reporterMaxWaitMs);
            }),
          ]);
        } finally {
          if (timer) clearTimeout(timer);
        }
      }
    } catch (error) {
      const reporterError =
        error instanceof Error ? error : new Error(String(error));
      if (failOnReporterError) {
        throw new NodeOperationError(context.getNode(), reporterError, {
          functionality: 'configuration-node',
          message: 'Usage Reporter failed',
          description:
            'The model call succeeded, but the connected usage-reporting tool failed.',
        });
      }
      context.logger.warn(
        `Usage Reporter failed after a successful model call: ${reporterError.message}`,
      );
    }
  };
}

function sharedModelOptions(): INodeProperties[] {
  return [
    {
      displayName: 'System Message',
      name: 'systemMessage',
      type: 'string',
      typeOptions: { rows: 5 },
      default: '',
      placeholder: 'Optional instructions applied to every model call',
      description:
        'Additional system instructions appended after system messages supplied by the parent node',
    },
    {
      displayName: 'Include Token Usage in Output',
      name: 'includeTokenUsageInAgentOutput',
      type: 'boolean',
      default: false,
      description:
        'Whether tokenUsage and raw usageMetadata are exposed in compatible parent-node outputs. Usage Reporter remains independent of this setting.',
    },
    {
      displayName: 'Include Intermediate Steps in Output',
      name: 'includeIntermediateStepsInOutput',
      type: 'boolean',
      default: false,
      description:
        'Whether compatible agent outputs include a compact list of tool steps. Internal LangChain messages and Gemini signatures are always removed.',
    },
    {
      displayName: 'Usage Reporter',
      name: 'usageReporter',
      type: 'fixedCollection',
      default: {
        settings: {
          enabled: false,
          nodeLabel: '',
          inputTextMode: 'label',
          inputTextLabel: 'RAG',
          includeOutputText: true,
          reporterMaxWaitMs: 1000,
          failOnReporterError: false,
        },
      },
      description:
        'Optionally send token usage and model metadata to a connected AI Tool after every successful model call',
      options: [
        {
          displayName: 'Usage Reporter',
          name: 'settings',
          values: [
            {
              displayName: 'Enable Usage Reporter',
              name: 'enabled',
              type: 'boolean',
              default: false,
              noDataExpression: true,
              description:
                'Show an optional AI Tool connector that receives a report after every successful model call',
            },
            {
              displayName: 'Node Label',
              name: 'nodeLabel',
              type: 'string',
              default: '',
              placeholder: 'e.g. Customer Support',
              displayOptions: {
                show: {
                  enabled: [true],
                },
              },
              description:
                'Value sent in the node field. Uses this Chat Model node name when empty.',
            },
            {
              displayName: 'Input Text Mode',
              name: 'inputTextMode',
              type: 'options',
              options: [
                {
                  name: 'Fixed Label',
                  value: 'label',
                  description:
                    'Send a fixed label instead of the potentially sensitive prompt.',
                },
                {
                  name: 'Actual User Input',
                  value: 'prompt',
                  description:
                    'Send only the latest human message. System prompts, tool calls, tool results, and previous AI messages are excluded.',
                },
              ],
              default: 'label',
              displayOptions: {
                show: {
                  enabled: [true],
                },
              },
            },
            {
              displayName: 'Input Text Label',
              name: 'inputTextLabel',
              type: 'string',
              default: 'RAG',
              displayOptions: {
                show: {
                  enabled: [true],
                  inputTextMode: ['label'],
                },
              },
              description:
                'Fixed value sent in input_text when Input Text Mode is Fixed Label.',
            },
            {
              displayName: 'Include Output Text',
              name: 'includeOutputText',
              type: 'boolean',
              default: true,
              displayOptions: {
                show: {
                  enabled: [true],
                },
              },
              description:
                'Whether to send the generated response in output_text.',
            },
            {
              displayName: 'Maximum Wait',
              name: 'reporterMaxWaitMs',
              type: 'number',
              default: 1000,
              typeOptions: {
                minValue: 0,
                numberStepSize: 100,
              },
              displayOptions: {
                show: {
                  enabled: [true],
                },
              },
              description:
                'Maximum milliseconds a model call waits for the reporter. After this limit, non-strict reporting continues in the background and no longer delays the parent node. Set 0 to wait indefinitely.',
            },
            {
              displayName: 'Fail Workflow if Reporter Fails',
              name: 'failOnReporterError',
              type: 'boolean',
              default: false,
              displayOptions: {
                show: {
                  enabled: [true],
                },
              },
              description:
                'Whether a logging failure should fail the otherwise successful model call.',
            },
          ],
        },
      ],
    },
  ];
}

function resolveSharedModelOptions(
  context: ISupplyDataFunctions,
  providerOptions: UsageReportingOptions,
): UsageReportingOptions {
  const legacy = context.getNode().parameters as Record<string, any>;
  const legacyReporting =
    legacy.usageReportingOptions &&
    typeof legacy.usageReportingOptions === 'object'
      ? (legacy.usageReportingOptions as UsageReportingOptions)
      : {};
  const groupedReporting =
    providerOptions.usageReporter?.settings ?? {};

  return {
    systemMessage:
      providerOptions.systemMessage ??
      (typeof legacy.systemMessage === 'string' ? legacy.systemMessage : ''),
    includeTokenUsageInAgentOutput:
      providerOptions.includeTokenUsageInAgentOutput ??
      (typeof legacy.includeTokenUsageInAgentOutput === 'boolean'
        ? legacy.includeTokenUsageInAgentOutput
        : false),
    includeIntermediateStepsInOutput:
      providerOptions.includeIntermediateStepsInOutput ??
      (typeof legacy.includeIntermediateStepsInOutput === 'boolean'
        ? legacy.includeIntermediateStepsInOutput
        : false),
    enableUsageReporter:
      groupedReporting.enabled ??
      providerOptions.enableUsageReporter ??
      (legacy.enableUsageReporter === true),
    nodeLabel:
      groupedReporting.nodeLabel ??
      providerOptions.nodeLabel ??
      legacyReporting.nodeLabel,
    inputTextMode:
      groupedReporting.inputTextMode ??
      providerOptions.inputTextMode ??
      legacyReporting.inputTextMode,
    inputTextLabel:
      groupedReporting.inputTextLabel ??
      providerOptions.inputTextLabel ??
      legacyReporting.inputTextLabel,
    includeOutputText:
      groupedReporting.includeOutputText ??
      providerOptions.includeOutputText ??
      legacyReporting.includeOutputText,
    failOnReporterError:
      groupedReporting.failOnReporterError ??
      providerOptions.failOnReporterError ??
      legacyReporting.failOnReporterError,
    reporterMaxWaitMs:
      groupedReporting.reporterMaxWaitMs ??
      providerOptions.reporterMaxWaitMs ??
      legacyReporting.reporterMaxWaitMs,
  };
}

export function getModelExecutionSettings(
  context: ISupplyDataFunctions,
  itemIndex: number,
): ModelExecutionSettings {
  const node = context.getNode() as ReturnType<ISupplyDataFunctions['getNode']> & {
    alwaysOutputData?: boolean;
    executeOnce?: boolean;
    retryOnFail?: boolean;
    maxTries?: number;
    waitBetweenTries?: number;
    continueOnFail?: boolean;
    onError?: 'stopWorkflow' | 'continueRegularOutput' | 'continueErrorOutput';
    modelAlwaysOutputData?: boolean;
    modelExecuteOnce?: boolean;
    modelRetryOnFail?: boolean;
    modelMaxTries?: number;
    modelWaitBetweenTries?: number;
    modelOnError?: 'stopWorkflow' | 'continueRegularOutput' | 'continueErrorOutput';
  };
  const customRetryOnFail =
    node.modelRetryOnFail === true ||
    (context.getNodeParameter(
      'modelRetryOnFail',
      itemIndex,
      false,
    ) as boolean);
  const retryOnFail = node.retryOnFail === true || customRetryOnFail;
  const useNativeRetry = node.retryOnFail === true;
  const customOnError =
    node.modelOnError ??
    (context.getNodeParameter(
      'modelOnError',
      itemIndex,
      'stopWorkflow',
    ) as ModelExecutionSettings['onError']);
  const onError =
    node.onError ??
    (node.continueOnFail === true ? 'continueRegularOutput' : customOnError);

  return {
    alwaysOutputData:
      node.alwaysOutputData === true ||
      node.modelAlwaysOutputData === true ||
      (context.getNodeParameter(
        'modelAlwaysOutputData',
        itemIndex,
        false,
      ) as boolean),
    executeOnce:
      node.executeOnce === true ||
      node.modelExecuteOnce === true ||
      (context.getNodeParameter(
        'modelExecuteOnce',
        itemIndex,
        false,
      ) as boolean),
    retryOnFail,
    maxTries: retryOnFail
      ? Math.max(
          2,
          Math.min(
            5,
            Number(
              useNativeRetry
                ? (node.maxTries ?? 3)
                : (node.modelMaxTries ??
                  context.getNodeParameter('modelMaxTries', itemIndex, 3)),
            ),
          ),
        )
      : 1,
    waitBetweenTries: retryOnFail
      ? Math.max(
          0,
          Math.min(
            5000,
            Number(
              useNativeRetry
                ? (node.waitBetweenTries ?? 1000)
                : (node.modelWaitBetweenTries ??
                  context.getNodeParameter(
                    'modelWaitBetweenTries',
                    itemIndex,
                    1000,
                  )),
            ),
          ),
        )
      : 0,
    onError,
  };
}

/**
 * Current n8n editors, including 2.32.6, hide common execution settings for
 * language-model subnodes. The node supplies an equivalent set in that case.
 * The loader module is inside n8n/n8n-core, which lets us identify the running
 * host without adding a dependency on n8n itself.
 */
function shouldExposeModelExecutionProperties(): boolean {
  let parentModule = module.parent;

  while (parentModule) {
    let directory = dirname(parentModule.filename);
    const root = parse(directory).root;

    while (directory !== root) {
      const packageJsonPath = join(directory, 'package.json');
      if (existsSync(packageJsonPath)) {
        try {
          const packageJson = JSON.parse(readFileSync(packageJsonPath, 'utf8')) as {
            name?: string;
            version?: string;
          };
          if (
            (packageJson.name === 'n8n' || packageJson.name === 'n8n-core') &&
            packageJson.version
          ) {
            const match = /^(\d+)\.(\d+)/.exec(packageJson.version);
            if (!match) return true;

            const major = Number(match[1]);
            const minor = Number(match[2]);
            return major > 2 || (major === 2 && minor > 2);
          }
        } catch {
          // Continue up the module tree if this package manifest is unreadable.
        }
      }

      directory = dirname(directory);
    }

    parentModule = parentModule.parent;
  }

  // Standalone loading (tests and package inspection) uses the newer behavior.
  return true;
}

function isAbortError(error: unknown, signal?: AbortSignal): boolean {
  if (signal?.aborted) return true;
  return error instanceof Error && error.name === 'AbortError';
}

async function waitForRetry(delayMs: number, signal?: AbortSignal): Promise<void> {
  if (delayMs <= 0) return;

  await new Promise<void>((resolve, reject) => {
    const finish = () => {
      signal?.removeEventListener('abort', abort);
      resolve();
    };
    const timeout = setTimeout(finish, delayMs);
    const abort = () => {
      clearTimeout(timeout);
      signal?.removeEventListener('abort', abort);
      const error = new Error('The model request was aborted.');
      error.name = 'AbortError';
      reject(error);
    };

    if (signal?.aborted) {
      abort();
      return;
    }

    signal?.addEventListener('abort', abort, { once: true });
  });
}

async function runWithRetry<T>(
  operation: () => Promise<T>,
  settings: ModelExecutionSettings,
  provider: ModelProvider,
  signal?: AbortSignal,
): Promise<T> {
  let attempt = 1;

  while (true) {
    try {
      return await operation();
    } catch (error) {
      const normalizedError = toUniversalModelError(error, provider);
      if (
        !settings.retryOnFail ||
        attempt >= settings.maxTries ||
        isAbortError(normalizedError, signal) ||
        !isRetryableModelError(normalizedError, provider)
      ) {
        normalizeModelError(normalizedError, provider).attempts = attempt;
        throw normalizedError;
      }

      attempt += 1;
      const retryAfterMs = retryAfterMsForModelError(normalizedError, provider);
      await waitForRetry(
        Math.max(
          settings.waitBetweenTries,
          Math.min(retryAfterMs ?? 0, 60_000),
        ),
        signal,
      );
    }
  }
}

function errorDetails(
  error: unknown,
  provider: ModelProvider,
): NormalizedModelError {
  return normalizeModelError(error, provider);
}

function continuedErrorMessage(
  error: unknown,
  provider: ModelProvider,
): AIMessage {
  const normalizedError = toUniversalModelError(error, provider);
  const details = errorDetails(normalizedError, provider);
  return new AIMessage({
    content: normalizedError.message,
    additional_kwargs: {
      universalChatModelError: details,
    },
    response_metadata: {
      universalChatModelError: details,
      onError: 'continueRegularOutput',
    },
  });
}

/**
 * LangChain invokes these two protected methods inside a single traced LLM run.
 * Wrapping them keeps n8n's execution trace compact while making retries work
 * for both regular and streaming calls.
 */
export function applyModelRetry(
  model: BaseChatModel,
  settings: ModelExecutionSettings,
  provider: ModelProvider,
): BaseChatModel {
  const mutableModel = model as any;
  const originalGenerate = mutableModel._generate.bind(mutableModel);
  mutableModel._generate = async (...args: any[]) => {
    const signal = args[1]?.signal as AbortSignal | undefined;
    let result: any;
    try {
      result = await runWithRetry(
        () => originalGenerate(...args),
        settings,
        provider,
        signal,
      );
    } catch (error) {
      const normalizedError = toUniversalModelError(error, provider);
      if (
        settings.onError !== 'continueRegularOutput' ||
        isAbortError(normalizedError, signal)
      ) {
        throw normalizedError;
      }

      const message = continuedErrorMessage(normalizedError, provider);
      result = {
        generations: [
          {
            text: message.text,
            message,
          },
        ],
        llmOutput: {
          universalChatModelError: errorDetails(normalizedError, provider),
        },
      };
    }

    if (
      settings.alwaysOutputData &&
      (!Array.isArray(result?.generations) || result.generations.length === 0)
    ) {
      result.generations = [
        {
          text: '',
          message: new AIMessage(''),
        },
      ];
    }

    return result;
  };

  if (typeof mutableModel._streamResponseChunks === 'function') {
    const originalStream = mutableModel._streamResponseChunks.bind(mutableModel);
    mutableModel._streamResponseChunks = async function* (...args: any[]) {
      const signal = args[1]?.signal as AbortSignal | undefined;
      let attempt = 1;
      let emittedAnyChunk = false;

      while (true) {
        let emittedChunk = false;
        try {
          for await (const chunk of originalStream(...args)) {
            emittedChunk = true;
            emittedAnyChunk = true;
            yield chunk;
          }
          if (settings.alwaysOutputData && !emittedAnyChunk) {
            yield new ChatGenerationChunk({
              text: '',
              message: new AIMessageChunk(''),
            });
          }
          return;
        } catch (error) {
          const normalizedError = toUniversalModelError(error, provider);
          if (isAbortError(normalizedError, signal) || emittedChunk) {
            throw normalizedError;
          }

          if (
            settings.retryOnFail &&
            attempt < settings.maxTries &&
            isRetryableModelError(normalizedError, provider)
          ) {
            attempt += 1;
            const retryAfterMs = retryAfterMsForModelError(
              normalizedError,
              provider,
            );
            await waitForRetry(
              Math.max(
                settings.waitBetweenTries,
                Math.min(retryAfterMs ?? 0, 60_000),
              ),
              signal,
            );
            continue;
          }

          if (settings.onError === 'continueRegularOutput') {
            const details = errorDetails(normalizedError, provider);
            details.attempts = attempt;
            yield new ChatGenerationChunk({
              text: normalizedError.message,
              message: new AIMessageChunk({
                content: normalizedError.message,
                additional_kwargs: {
                  universalChatModelError: details,
                },
                response_metadata: {
                  universalChatModelError: details,
                  onError: 'continueRegularOutput',
                },
              }),
            });
            return;
          }

          detailsForFinalError(normalizedError, provider, attempt);
          throw normalizedError;
        }
      }
    };
  }

  return model;
}

/**
 * Adds the node-level system message after any leading system messages supplied
 * by the parent node. This keeps the extra instruction compatible with AI
 * Agent, chains, vector-store tools, and direct model invocations.
 */
export function applySystemMessage(
  model: BaseChatModel,
  systemMessage: string,
): BaseChatModel {
  const content = systemMessage.trim();
  if (!content) return model;

  const inject = (messages: BaseMessage[]): BaseMessage[] => {
    const next = [...messages];
    let systemCount = 0;
    while (
      systemCount < next.length &&
      typeof (next[systemCount] as any)?._getType === 'function' &&
      (next[systemCount] as any)._getType() === 'system'
    ) {
      systemCount += 1;
    }

    // Gemini accepts a single leading system instruction. Merge the parent
    // node's system prompt and this optional node-level addition into that
    // instruction while keeping them as separate text parts.
    const systemParts: Array<Record<string, unknown>> = [];
    for (const message of next.slice(0, systemCount)) {
      const existingContent = (message as any).content;
      if (typeof existingContent === 'string' && existingContent.length > 0) {
        systemParts.push({ type: 'text', text: existingContent });
      } else if (Array.isArray(existingContent)) {
        systemParts.push(...existingContent);
      }
    }
    systemParts.push({ type: 'text', text: content });
    next.splice(
      0,
      systemCount,
      new SystemMessage({ content: systemParts } as any),
    );
    return next;
  };

  const mutableModel = model as any;
  const originalGenerate = mutableModel._generate.bind(mutableModel);
  mutableModel._generate = (messages: BaseMessage[], ...args: any[]) =>
    originalGenerate(inject(messages), ...args);

  if (typeof mutableModel._streamResponseChunks === 'function') {
    const originalStream = mutableModel._streamResponseChunks.bind(mutableModel);
    mutableModel._streamResponseChunks = (
      messages: BaseMessage[],
      ...args: any[]
    ) => originalStream(inject(messages), ...args);
  }

  return model;
}

function detailsForFinalError(
  error: unknown,
  provider: ModelProvider,
  attempts: number,
): NormalizedModelError {
  const details = normalizeModelError(error, provider);
  details.attempts = attempts;
  return details;
}

/**
 * Wraps a LangChain model to intercept responses and log token usage metadata.
 * This creates a proxy that captures usage_metadata from AIMessage responses.
 */
export class UniversalChatModel implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'Universal Chat Model',
    name: 'universalChatModel',
    icon: 'file:universalChatModel.svg',
    group: ['transform'],
    version: 1,
    description: 'Universal AI Chat Model supporting Google Gemini (Native) and OpenAI Compatible / Local LLMs (Ollama, LM Studio, DeepSeek, OpenRouter, vLLM)',
    defaults: {
      name: 'Universal Chat Model',
    },
    inputs:
      '={{ (($parameter.provider === "gemini" && $parameter.geminiOptions && (($parameter.geminiOptions.usageReporter && $parameter.geminiOptions.usageReporter.settings && $parameter.geminiOptions.usageReporter.settings.enabled) || $parameter.geminiOptions.enableUsageReporter)) || ($parameter.provider === "openai_compatible" && $parameter.openaiOptions && (($parameter.openaiOptions.usageReporter && $parameter.openaiOptions.usageReporter.settings && $parameter.openaiOptions.usageReporter.settings.enabled) || $parameter.openaiOptions.enableUsageReporter)) || $parameter.enableUsageReporter) ? [{ type: "ai_tool", displayName: "Usage Reporter", required: false, maxConnections: 1 }] : [] }}',
    outputs: [NodeConnectionTypes.AiLanguageModel],
    credentials: [
      {
        name: 'googleGeminiApi',
        required: true,
        displayOptions: {
          show: {
            provider: ['gemini'],
          },
        },
      },
      {
        name: 'openAiCompatibleApi',
        required: false,
        displayOptions: {
          show: {
            provider: ['openai_compatible'],
          },
        },
      },
    ],
    codex: {
      categories: ['AI'],
    },
    properties: [
      ...(shouldExposeModelExecutionProperties()
        ? ([
            // n8n 2.32.6 hides common settings for model subnodes, so expose
            // equivalent native node settings for runtime and canvas status.
            {
              displayName: 'Always Output Data',
              name: 'alwaysOutputData',
              type: 'boolean',
              default: false,
              noDataExpression: true,
              isNodeSetting: true,
              description:
                'Keep an inspectable chat-model result when the provider returns an empty response',
            },
            {
              displayName: 'Execute Once',
              name: 'executeOnce',
              type: 'boolean',
              default: false,
              noDataExpression: true,
              isNodeSetting: true,
              description:
                'Resolve the model configuration from the first input item for the whole execution',
            },
            {
              displayName: 'Retry On Fail',
              name: 'retryOnFail',
              type: 'boolean',
              default: false,
              noDataExpression: true,
              isNodeSetting: true,
              description: 'Retry the model API call when it fails',
            },
            {
              displayName: 'Max Tries',
              name: 'maxTries',
              type: 'number',
              typeOptions: {
                minValue: 2,
                maxValue: 5,
                numberPrecision: 0,
              },
              default: 3,
              noDataExpression: true,
              isNodeSetting: true,
              displayOptions: {
                show: {
                  retryOnFail: [true],
                },
              },
              description: 'Total number of attempts before the model call fails',
            },
            {
              displayName: 'Wait Between Tries (ms)',
              name: 'waitBetweenTries',
              type: 'number',
              typeOptions: {
                minValue: 0,
                maxValue: 5000,
                numberPrecision: 0,
              },
              default: 1000,
              noDataExpression: true,
              isNodeSetting: true,
              displayOptions: {
                show: {
                  retryOnFail: [true],
                },
              },
              description:
                'How long to wait between model API attempts in milliseconds',
            },
            {
              displayName: 'On Error',
              name: 'onError',
              type: 'options',
              options: [
                {
                  name: 'Stop Workflow',
                  value: 'stopWorkflow',
                  description: 'Halt execution and fail the workflow',
                },
                {
                  name: 'Continue',
                  value: 'continueRegularOutput',
                  description: 'Pass the model error message through the regular AI model output',
                },
                {
                  name: 'Continue (Using Error Output)',
                  value: 'continueErrorOutput',
                  description: 'Let n8n route the failure through its error output when the host supports it',
                },
              ],
              default: 'stopWorkflow',
              noDataExpression: true,
              isNodeSetting: true,
              description: 'Action to take when the model execution fails',
            },
          ] as INodeProperties[])
        : []),

      // ─── 1. Provider Selection ───
      {
        displayName: 'Provider',
        name: 'provider',
        type: 'options',
        options: [
          {
            name: 'Google Gemini (Native)',
            value: 'gemini',
            description: 'Native Google Gemini API with advanced thinking, safety, and output schema controls',
          },
          {
            name: 'OpenAI Compatible / Local LLM / DeepSeek',
            value: 'openai_compatible',
            description: 'Custom base URL endpoints (Ollama, LM Studio, vLLM, DeepSeek, OpenRouter, LocalAI)',
          },
        ],
        default: 'gemini',
        description: 'Select the AI Model provider',
      },
      // ─── 2. GOOGLE GEMINI — Required Fields ───
      {
        displayName: 'Model Name',
        name: 'geminiModel',
        type: 'options',
        displayOptions: { show: { provider: ['gemini'] } },
        typeOptions: { loadOptionsMethod: 'getGeminiModels' },
        default: 'gemini-3.5-flash',
        description: 'Select dynamically loaded Gemini model or specify custom ID below',
      },
      {
        displayName: 'Custom Model ID',
        name: 'geminiModelCustom',
        type: 'string',
        displayOptions: { show: { provider: ['gemini'] } },
        default: '',
        placeholder: 'e.g., gemini-1.5-pro-latest',
        description: 'Optional custom Gemini model ID (overrides dropdown selection if set)',
      },

      // ─── 2a. GOOGLE GEMINI — Options (Optional) ───
      {
        displayName: 'Options',
        name: 'geminiOptions',
        type: 'collection',
        placeholder: 'Add Option',
        default: {},
        displayOptions: { show: { provider: ['gemini'] } },
        options: [
          {
            displayName: 'Temperature',
            name: 'temperature',
            type: 'number',
            typeOptions: { minValue: 0.0, maxValue: 2.0, numberPrecision: 2 },
            default: 0.2,
            description: 'Controls randomness: Lower values are more deterministic, higher values are more creative.',
          },
          {
            displayName: 'Top P',
            name: 'topP',
            type: 'number',
            typeOptions: { minValue: 0.0, maxValue: 1.0, numberPrecision: 2 },
            default: 0.95,
            description: 'Nucleus sampling threshold parameter',
          },
          {
            displayName: 'Top K',
            name: 'topK',
            type: 'number',
            default: 40,
            description: 'Top-k sampling parameter',
          },
          {
            displayName: 'Max Output Tokens',
            name: 'maxOutputTokens',
            type: 'number',
            default: 8192,
            description: 'Maximum number of tokens to generate in response',
          },
          {
            displayName: 'Response MIME Type',
            name: 'responseMimeType',
            type: 'options',
            options: [
              { name: 'Text (text/plain)', value: 'text/plain' },
              { name: 'JSON (application/json)', value: 'application/json' },
            ],
            default: 'text/plain',
            description: 'Output format requested from the Gemini model',
          },
          {
            displayName: 'Structured Output Schema (JSON)',
            name: 'responseSchema',
            type: 'string',
            typeOptions: { rows: 10 },
            default: '{\n  "type": "object",\n  "properties": {\n    "message": {\n      "type": "string"\n    }\n  },\n  "additionalProperties": false,\n  "required": ["message"]\n}',
            description: 'Optional JSON Schema for a structured response. When configured, the response MIME type is automatically set to application/json.',
          },
          {
            displayName: 'Thinking Level (Gemini 3+)',
            name: 'thinkingLevel',
            type: 'options',
            options: [
              { name: 'MINIMAL', value: 'MINIMAL' },
              { name: 'LOW', value: 'LOW' },
              { name: 'MEDIUM', value: 'MEDIUM' },
              { name: 'HIGH', value: 'HIGH' },
            ],
            default: 'MEDIUM',
            description: 'Optional reasoning depth for Gemini 3 and newer. Do not combine with Thinking Budget.',
          },
          {
            displayName: 'Thinking Budget (Gemini 2.5)',
            name: 'thinkingBudget',
            type: 'number',
            typeOptions: { minValue: -1, numberPrecision: 0 },
            default: -1,
            description: 'Optional reasoning budget for Gemini 2.5. -1 = dynamic and 0 = disabled where supported. Do not combine with Thinking Level.',
          },
          {
            displayName: 'Include Thoughts',
            name: 'includeThoughts',
            type: 'boolean',
            default: false,
            description: 'Return available thought summaries in model and consumer metadata. Gemini may take longer to generate these summaries. Thoughts remain hidden unless this option is enabled.',
          },
          {
            displayName: 'Model Request Timeout',
            name: 'requestTimeoutMs',
            type: 'number',
            default: 60000,
            typeOptions: {
              minValue: 0,
              maxValue: 900000,
              numberStepSize: 1000,
            },
            description: 'Maximum milliseconds for each Gemini provider request. A timeout is retryable when Retry On Fail is enabled. Set 0 to disable the limit.',
          },
          {
            displayName: 'Recover Empty Final Responses',
            name: 'recoverEmptyResponses',
            type: 'boolean',
            default: true,
            description: 'Automatically makes one safe continuation request when Gemini returns STOP with no text and no function call. Valid tool calls are never retried.',
          },
          {
            displayName: 'Safety Settings',
            name: 'safetySettings',
            type: 'fixedCollection',
            typeOptions: { multipleValues: true },
            placeholder: 'Add Safety Threshold Rule',
            default: {},
            options: [
              {
                name: 'values',
                displayName: 'Safety Rules',
                values: [
                  {
                    displayName: 'Category',
                    name: 'category',
                    type: 'options',
                    options: [
                      { name: 'Hate Speech', value: 'HARM_CATEGORY_HATE_SPEECH' },
                      { name: 'Harassment', value: 'HARM_CATEGORY_HARASSMENT' },
                      { name: 'Sexually Explicit', value: 'HARM_CATEGORY_SEXUALLY_EXPLICIT' },
                      { name: 'Dangerous Content', value: 'HARM_CATEGORY_DANGEROUS_CONTENT' },
                    ],
                    default: 'HARM_CATEGORY_HATE_SPEECH',
                  },
                  {
                    displayName: 'Threshold',
                    name: 'threshold',
                    type: 'options',
                    options: [
                      { name: 'Block None', value: 'BLOCK_NONE' },
                      { name: 'Block Low and Above', value: 'BLOCK_LOW_AND_ABOVE' },
                      { name: 'Block Medium and Above', value: 'BLOCK_MEDIUM_AND_ABOVE' },
                      { name: 'Block Only High', value: 'BLOCK_ONLY_HIGH' },
                    ],
                    default: 'BLOCK_MEDIUM_AND_ABOVE',
                  },
                ],
              },
            ],
          },
          ...sharedModelOptions(),
        ],
      },

      // ─── 3. OPENAI / LOCAL LLM — Required Fields ───
      {
        displayName: 'Model Name / ID',
        name: 'openaiModel',
        type: 'options',
        typeOptions: {
          loadOptionsMethod: 'getOpenAiModels',
        },
        displayOptions: { show: { provider: ['openai_compatible'] } },
        default: 'llama3',
        description: 'Dynamically loaded model from target API endpoint',
      },
      {
        displayName: 'Custom Model ID (Override)',
        name: 'openaiModelCustom',
        type: 'string',
        displayOptions: { show: { provider: ['openai_compatible'] } },
        default: '',
        placeholder: 'e.g., deepseek-r1:70b, gpt-4o-2024-08-06',
        description: 'Optional custom model string override',
      },

      // ─── 3a. OPENAI — Options (Optional) ───
      {
        displayName: 'Options',
        name: 'openaiOptions',
        type: 'collection',
        placeholder: 'Add Option',
        default: {},
        displayOptions: { show: { provider: ['openai_compatible'] } },
        options: [
          {
            displayName: 'Temperature',
            name: 'temperature',
            type: 'number',
            typeOptions: { minValue: 0.0, maxValue: 2.0, numberPrecision: 2 },
            default: 0.2,
            description: 'Controls output randomness (0.0 to 2.0)',
          },
          {
            displayName: 'Reasoning Effort',
            name: 'reasoningEffort',
            type: 'options',
            options: [
              { name: 'None / Default', value: 'none' },
              { name: 'Low', value: 'low' },
              { name: 'Medium', value: 'medium' },
              { name: 'High', value: 'high' },
            ],
            default: 'none',
            description: 'For reasoning models like DeepSeek-R1 and OpenAI o1/o3-mini',
          },
          {
            displayName: 'Frequency Penalty',
            name: 'frequencyPenalty',
            type: 'number',
            typeOptions: { minValue: -2.0, maxValue: 2.0, numberPrecision: 2 },
            default: 0,
            description: 'Penalizes tokens based on frequency (-2.0 to 2.0)',
          },
          {
            displayName: 'Presence Penalty',
            name: 'presencePenalty',
            type: 'number',
            typeOptions: { minValue: -2.0, maxValue: 2.0, numberPrecision: 2 },
            default: 0,
            description: 'Penalizes tokens based on presence (-2.0 to 2.0)',
          },
          {
            displayName: 'Max Tokens',
            name: 'maxTokens',
            type: 'number',
            default: 4096,
            description: 'Maximum number of tokens to generate',
          },
          {
            displayName: 'Seed',
            name: 'seed',
            type: 'number',
            default: 0,
            description: 'Integer seed for deterministic sampling',
          },
          {
            displayName: 'JSON Mode',
            name: 'jsonMode',
            type: 'boolean',
            default: false,
            description: 'Enforce JSON mode (response_format: { type: "json_object" })',
          },
          {
            displayName: 'Custom Headers (JSON)',
            name: 'customHeaders',
            type: 'string',
            typeOptions: { rows: 3 },
            default: '',
            placeholder: '{"HTTP-Referer": "https://n8n.io", "X-Title": "n8n Workflow"}',
            description: 'Raw JSON for additional HTTP headers (e.g. OpenRouter metadata)',
          },
          ...sharedModelOptions(),
        ],
      },
    ],
  };

  methods = {
    loadOptions: {
      async getGeminiModels(this: ILoadOptionsFunctions): Promise<INodePropertyOptions[]> {
        let apiKey = '';
        try {
          const creds = await this.getCredentials('googleGeminiApi');
          if (creds && typeof creds.apiKey === 'string') apiKey = creds.apiKey;
        } catch { /* ignore */ }

        const defaultModels: INodePropertyOptions[] = [
          { name: 'Gemini 3.5 Flash', value: 'gemini-3.5-flash' },
          { name: 'Gemini 3.5 Flash Lite', value: 'gemini-3.5-flash-lite' },
          { name: 'Gemini 3.1 Pro', value: 'gemini-3.1-pro' },
          { name: 'Gemini 2.5 Flash', value: 'gemini-2.5-flash' },
        ];

        if (!apiKey) return defaultModels;

        try {
          const response = await this.helpers.httpRequest({
            method: 'GET',
            url: 'https://generativelanguage.googleapis.com/v1beta/models',
            headers: { 'x-goog-api-key': apiKey },
            json: true,
          });

          if (response && Array.isArray((response as any).models)) {
            const options: INodePropertyOptions[] = [];
            for (const m of (response as any).models) {
              if (m.name && typeof m.name === 'string') {
                const modelId = m.name.replace(/^models\//, '');
                const methods = m.supportedGenerationMethods;
                if (!methods || (Array.isArray(methods) && methods.includes('generateContent'))) {
                  const label = m.displayName ? `${m.displayName} (${modelId})` : modelId;
                  options.push({ name: label, value: modelId });
                }
              }
            }
            if (options.length > 0) return options;
          }
        } catch { /* fallback */ }

        return defaultModels;
      },

      async getOpenAiModels(this: ILoadOptionsFunctions): Promise<INodePropertyOptions[]> {
        let baseUrl = 'http://localhost:11434/v1';
        let apiKey = 'not-needed';

        try {
          const creds = await this.getCredentials('openAiCompatibleApi');
          if (creds && typeof creds.baseUrl === 'string' && creds.baseUrl.trim()) baseUrl = creds.baseUrl.trim();
          if (creds && typeof creds.apiKey === 'string' && creds.apiKey.trim()) apiKey = creds.apiKey.trim();
        } catch { /* ignore */ }

        baseUrl = baseUrl.replace(/\/+$/, '');

        const defaultModels: INodePropertyOptions[] = [
          { name: 'llama3', value: 'llama3' },
          { name: 'gpt-4o', value: 'gpt-4o' },
          { name: 'deepseek-r1', value: 'deepseek-r1' },
          { name: 'qwen2.5', value: 'qwen2.5' },
          { name: 'claude-3-5-sonnet', value: 'claude-3-5-sonnet' },
        ];

        const headers: Record<string, string> = {};
        if (apiKey && apiKey !== 'not-needed') headers['Authorization'] = `Bearer ${apiKey}`;

        try {
          const modelsUrl = baseUrl.endsWith('/v1') ? `${baseUrl}/models` : `${baseUrl}/v1/models`;
          const response: any = await this.helpers.httpRequest({ method: 'GET', url: modelsUrl, headers, json: true });
          const rawList = response?.data || response?.models || (Array.isArray(response) ? response : null);
          if (Array.isArray(rawList) && rawList.length > 0) {
            return rawList.map((m: any) => {
              const modelId = typeof m === 'string' ? m : (m.id || m.name || JSON.stringify(m));
              return { name: String(modelId), value: String(modelId) };
            });
          }
        } catch {
          try {
            const rootUrl = baseUrl.replace(/\/v1$/, '');
            const ollamaResp: any = await this.helpers.httpRequest({ method: 'GET', url: `${rootUrl}/api/tags`, headers, json: true });
            if (ollamaResp && Array.isArray(ollamaResp.models) && ollamaResp.models.length > 0) {
              return ollamaResp.models.map((m: any) => {
                const name = m.name || m.model || JSON.stringify(m);
                return { name: String(name), value: String(name) };
              });
            }
          } catch { /* fallback */ }
        }

        return defaultModels;
      },
    },
  };

  async supplyData(this: ISupplyDataFunctions, itemIndex: number): Promise<SupplyData> {
    const executionSettings = getModelExecutionSettings(this, itemIndex);
    const executionItemIndex = executionSettings.executeOnce ? 0 : itemIndex;
    const provider = this.getNodeParameter('provider', executionItemIndex) as string;

    if (provider === 'gemini') {
      // ─── Resolve API Key ───
      const creds = await this.getCredentials('googleGeminiApi');
      const geminiApiKey =
        creds && typeof creds.apiKey === 'string' ? creds.apiKey.trim() : '';
      if (!geminiApiKey) {
        throw new Error('Google Gemini API Key is required. Configure the Google Gemini API credential.');
      }

      // ─── Resolve Model ───
      let geminiModel = this.getNodeParameter('geminiModel', executionItemIndex, 'gemini-3.5-flash') as string;
      const customModel = this.getNodeParameter('geminiModelCustom', executionItemIndex, '') as string;
      if (customModel.trim()) geminiModel = customModel.trim();

      // ─── Options (all optional) ───
      const opts = this.getNodeParameter('geminiOptions', executionItemIndex, {}) as UsageReportingOptions & {
        temperature?: number;
        topP?: number;
        topK?: number;
        maxOutputTokens?: number;
        responseMimeType?: string;
        responseSchema?: string | Record<string, unknown>;
        thinkingLevel?: string;
        thinkingBudget?: number;
        includeThoughts?: boolean;
        requestTimeoutMs?: number;
        recoverEmptyResponses?: boolean;
        safetySettings?: { values?: Array<{ category: string; threshold: string }> };
      };

      // Preserve values from workflows created with the pre-collection UI.
      const legacy = this.getNode().parameters as Record<string, any>;
      const useLegacy = <T>(current: T | undefined, name: string): T | undefined =>
        current === undefined && Object.prototype.hasOwnProperty.call(legacy, name)
          ? (legacy[name] as T)
          : current;

      opts.temperature = useLegacy(opts.temperature, 'geminiTemperature');
      opts.topP = useLegacy(opts.topP, 'geminiTopP');
      opts.topK = useLegacy(opts.topK, 'geminiTopK');
      opts.maxOutputTokens = useLegacy(opts.maxOutputTokens, 'geminiMaxTokens');
      opts.responseMimeType = useLegacy(opts.responseMimeType, 'geminiResponseMimeType');
      opts.includeThoughts = useLegacy(opts.includeThoughts, 'includeThoughts');
      opts.safetySettings = useLegacy(opts.safetySettings, 'geminiSafetySettings');
      const sharedOptions = resolveSharedModelOptions(this, opts);

      const legacyThinkingMode = legacy.thinkingMode;
      if (
        opts.thinkingLevel === undefined &&
        opts.thinkingBudget === undefined &&
        legacyThinkingMode === 'level'
      ) {
        opts.thinkingLevel = legacy.thinkingLevel;
      } else if (
        opts.thinkingLevel === undefined &&
        opts.thinkingBudget === undefined &&
        legacyThinkingMode === 'budget'
      ) {
        opts.thinkingBudget = legacy.thinkingBudget;
      }

      // Thinking is omitted entirely unless the user adds one of these options.
      const thinkingConfig: Record<string, any> = {};

      if (opts.thinkingLevel !== undefined && opts.thinkingBudget !== undefined) {
        throw new Error('Choose either Thinking Level (Gemini 3+) or Thinking Budget (Gemini 2.5), not both.');
      }
      if (opts.thinkingLevel !== undefined) {
        thinkingConfig.thinkingLevel = opts.thinkingLevel;
      }
      if (opts.thinkingBudget !== undefined) {
        thinkingConfig.thinkingBudget = opts.thinkingBudget;
      }
      // Thinking controls model reasoning, while Include Thoughts independently
      // controls whether summaries may be exposed in n8n outputs.
      const shouldIncludeThoughts = opts.includeThoughts === true;
      if (shouldIncludeThoughts) {
        thinkingConfig.includeThoughts = true;
      }

      // Response MIME type
      let responseMimeType = opts.responseMimeType;
      let responseSchema: Record<string, unknown> | undefined;
      if (typeof opts.responseSchema === 'string' && opts.responseSchema.trim().length > 0) {
        try {
          const parsedSchema = JSON.parse(opts.responseSchema) as unknown;
          if (
            parsedSchema === null ||
            typeof parsedSchema !== 'object' ||
            Array.isArray(parsedSchema)
          ) {
            throw new Error('the schema root must be a JSON object');
          }
          responseSchema = parsedSchema as Record<string, unknown>;
        } catch (error) {
          throw new Error(`Invalid JSON in Structured Output Schema: ${(error as Error).message}`);
        }
      } else if (
        opts.responseSchema !== null &&
        typeof opts.responseSchema === 'object' &&
        !Array.isArray(opts.responseSchema)
      ) {
        // Some n8n versions can return JSON editor values as an already parsed object.
        responseSchema = opts.responseSchema;
      }
      if (responseSchema) responseMimeType = 'application/json';

      // Safety Settings
      const safetySettings: Array<{ category: string; threshold: string }> = [];
      if (opts.safetySettings && Array.isArray(opts.safetySettings.values)) {
        for (const item of opts.safetySettings.values) {
          if (item.category && item.threshold) {
            safetySettings.push({ category: item.category, threshold: item.threshold });
          }
        }
      }

      const modelInput: any = {
        apiKey: geminiApiKey,
        model: geminiModel,
        maxRetries: 0,
        recoverEmptyResponses: opts.recoverEmptyResponses !== false,
        requestTimeoutMs: opts.requestTimeoutMs ?? 60_000,
      };
      const usageReporter = await createUsageReporter(
        this,
        executionItemIndex,
        geminiModel,
        sharedOptions,
      );
      modelInput.callbacks = [
        new UniversalChatModelTracing(
          this,
          'gemini',
          shouldIncludeThoughts,
          sharedOptions.includeTokenUsageInAgentOutput === true,
          sharedOptions.includeIntermediateStepsInOutput === true,
          usageReporter,
          sharedOptions.failOnReporterError === true,
        ),
      ];

      // Only set optional params if user provided them
      if (opts.temperature !== undefined) modelInput.temperature = opts.temperature;
      if (opts.topP !== undefined) modelInput.topP = opts.topP;
      if (opts.topK !== undefined) modelInput.topK = opts.topK;
      if (opts.maxOutputTokens !== undefined) modelInput.maxOutputTokens = opts.maxOutputTokens;
      if (responseMimeType !== undefined) modelInput.responseMimeType = responseMimeType;
      if (responseSchema !== undefined) modelInput.responseSchema = responseSchema;
      if (Object.keys(thinkingConfig).length > 0) modelInput.thinkingConfig = thinkingConfig;
      if (safetySettings.length > 0) modelInput.safetySettings = safetySettings;

      let model: BaseChatModel = new GeminiChatModel(modelInput, (usage) => {
        this.logAiEvent('ai-tokens-usage' as any, formatGeminiUsage(usage));
      }) as BaseChatModel;
      model = applySystemMessage(model, sharedOptions.systemMessage ?? '');

      return {
        response: applyModelRetry(model, executionSettings, 'gemini'),
      };

    } else {
      // ─── OpenAI Compatible ───

      // Resolve credentials
      let baseUrl = 'http://localhost:11434/v1';
      let openaiApiKey = 'not-needed';

      try {
        const creds = await this.getCredentials('openAiCompatibleApi');
        if (creds && typeof creds.baseUrl === 'string' && creds.baseUrl.trim()) baseUrl = creds.baseUrl.trim();
        if (creds && typeof creds.apiKey === 'string' && creds.apiKey.trim()) openaiApiKey = creds.apiKey.trim();
      } catch { /* ignore */ }

      baseUrl = baseUrl.replace(/\/+$/, '');

      // Resolve model
      let openaiModel = this.getNodeParameter('openaiModel', executionItemIndex, 'llama3') as string;
      const customModel = this.getNodeParameter('openaiModelCustom', executionItemIndex, '') as string;
      if (customModel.trim()) openaiModel = customModel.trim();

      // Options (all optional)
      const opts = this.getNodeParameter('openaiOptions', executionItemIndex, {}) as UsageReportingOptions & {
        temperature?: number;
        reasoningEffort?: string;
        frequencyPenalty?: number;
        presencePenalty?: number;
        maxTokens?: number;
        seed?: number;
        jsonMode?: boolean;
        customHeaders?: string;
      };
      const sharedOptions = resolveSharedModelOptions(this, opts);

      let parsedHeaders: Record<string, string> = {};
      if (opts.customHeaders && opts.customHeaders.trim().length > 0) {
        try {
          parsedHeaders = JSON.parse(opts.customHeaders);
        } catch (error) {
          throw new Error(`Invalid JSON in Custom Headers: ${(error as Error).message}`);
        }
      }

      const modelKwargs: Record<string, any> = {};
      const reasoningEffort = opts.reasoningEffort || 'none';

      if (opts.jsonMode === true) {
        modelKwargs.response_format = { type: 'json_object' };
      }

      const modelOptions: any = {
        apiKey: openaiApiKey,
        modelName: openaiModel,
        model: openaiModel,
        callbacks: [],
        configuration: {
          baseURL: baseUrl,
          defaultHeaders: parsedHeaders,
        },
        modelKwargs,
        maxRetries: 0,
      };
      const usageReporter = await createUsageReporter(
        this,
        executionItemIndex,
        openaiModel,
        sharedOptions,
      );
      modelOptions.callbacks = [
        new UniversalChatModelTracing(
          this,
          'openai_compatible',
          false,
          sharedOptions.includeTokenUsageInAgentOutput === true,
          sharedOptions.includeIntermediateStepsInOutput === true,
          usageReporter,
          sharedOptions.failOnReporterError === true,
        ),
      ];

      // Only set optional params if user provided them
      if (opts.temperature !== undefined) modelOptions.temperature = opts.temperature;
      if (opts.maxTokens !== undefined) modelOptions.maxTokens = opts.maxTokens;
      if (opts.frequencyPenalty !== undefined) {
        modelOptions.frequencyPenalty = opts.frequencyPenalty;
      }
      if (opts.presencePenalty !== undefined) {
        modelOptions.presencePenalty = opts.presencePenalty;
      }
      if (opts.seed !== undefined && !isNaN(Number(opts.seed))) {
        modelOptions.seed = Number(opts.seed);
      }
      if (reasoningEffort !== 'none') {
        modelOptions.reasoningEffort = reasoningEffort;
      }

      let model: BaseChatModel = new ChatOpenAI(modelOptions);
      model = applySystemMessage(model, sharedOptions.systemMessage ?? '');

      return {
        response: applyModelRetry(
          model,
          executionSettings,
          'openai_compatible',
        ),
      };
    }
  }
}
