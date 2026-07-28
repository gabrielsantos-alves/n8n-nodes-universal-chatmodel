import { AsyncLocalStorage } from 'node:async_hooks';
import Module from 'node:module';

import type { IDataObject } from 'n8n-workflow';
import {
  mergeStructuredOutput,
  parseStructuredOutput,
} from './StructuredOutput';

export interface AgentModelCallMetadata {
  thoughts?: unknown[];
  tokenUsage?: IDataObject;
  includeTokenUsageInAgentOutput?: boolean;
  usageMetadata?: IDataObject;
  gemini?: IDataObject;
  structuredOutput?: IDataObject;
}

interface AgentCaptureStore {
  calls: AgentModelCallMetadata[];
}

const agentCapture = new AsyncLocalStorage<AgentCaptureStore>();
const patchedAgentSymbol = Symbol.for(
  'n8n-nodes-universal-chatmodel.agent-output-bridge',
);
const moduleLoadHookSymbol = Symbol.for(
  'n8n-nodes-universal-chatmodel.agent-module-load-hook',
);
const agentModulePattern =
  /[\\/]n8n-nodes-langchain[\\/].*[\\/]nodes[\\/]agents[\\/]Agent[\\/]V[123][\\/]AgentV[123]\.node\.(?:js|cjs)$/i;

const tokenUsageNumberFields = [
  'inputTokens',
  'inputUncachedTokens',
  'outputTokens',
  'cachedTokens',
  'toolUsePromptTokens',
  'thoughtsTokens',
  'totalTokens',
] as const;

function cloneValue<T>(value: T): T {
  try {
    return structuredClone(value);
  } catch {
    return JSON.parse(JSON.stringify(value)) as T;
  }
}

function asRecord(value: unknown): Record<string, any> | undefined {
  return value !== null && typeof value === 'object'
    ? (value as Record<string, any>)
    : undefined;
}

function mergeThoughts(calls: AgentModelCallMetadata[]): unknown[] {
  const thoughts = calls.flatMap((call) =>
    Array.isArray(call.thoughts) ? call.thoughts : [],
  );
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

function aggregateTokenUsage(calls: AgentModelCallMetadata[]): IDataObject {
  const aggregate: IDataObject = {};

  for (const field of tokenUsageNumberFields) {
    aggregate[field] = calls.reduce((sum, call) => {
      const value = Number(call.tokenUsage?.[field] ?? 0);
      return sum + (Number.isFinite(value) ? value : 0);
    }, 0);
  }

  return aggregate;
}

function buildAgentOutputMetadata(
  calls: AgentModelCallMetadata[],
): IDataObject | undefined {
  if (calls.length === 0) return undefined;

  const thoughts = mergeThoughts(calls);
  const lastCall = calls[calls.length - 1];
  const callsWithVisibleUsage = calls.filter(
    (call) => call.includeTokenUsageInAgentOutput !== false,
  );
  const lastCallWithVisibleUsage =
    callsWithVisibleUsage[callsWithVisibleUsage.length - 1];
  const modelResponses = calls.map((call, index) => ({
    call: index + 1,
    ...(call.thoughts?.length
      ? { thoughts: cloneValue(call.thoughts) }
      : {}),
    ...(call.includeTokenUsageInAgentOutput !== false && call.tokenUsage
      ? { tokenUsage: cloneValue(call.tokenUsage) }
      : {}),
    ...(call.includeTokenUsageInAgentOutput !== false && call.usageMetadata
      ? { usageMetadata: cloneValue(call.usageMetadata) }
      : {}),
    ...(call.gemini ? { gemini: cloneValue(call.gemini) } : {}),
    ...(call.structuredOutput
      ? { structuredOutput: cloneValue(call.structuredOutput) }
      : {}),
  }));

  return {
    ...(thoughts.length > 0 ? { thoughts: cloneValue(thoughts) as any[] } : {}),
    ...(callsWithVisibleUsage.length > 0
      ? { tokenUsage: aggregateTokenUsage(callsWithVisibleUsage) }
      : {}),
    ...(lastCallWithVisibleUsage?.usageMetadata
      ? { usageMetadata: cloneValue(lastCallWithVisibleUsage.usageMetadata) }
      : {}),
    ...(lastCall.gemini ? { gemini: cloneValue(lastCall.gemini) } : {}),
    modelCalls: calls.length,
    modelResponses: cloneValue(modelResponses) as any[],
  };
}

function readPreviousCalls(args: unknown[]): AgentModelCallMetadata[] {
  const response = asRecord(args[0]);
  const metadata = asRecord(response?.metadata);
  const bridge = asRecord(metadata?.universalChatModel);
  return Array.isArray(bridge?.calls)
    ? cloneValue(bridge.calls as AgentModelCallMetadata[])
    : [];
}

function attachMetadataToAgentResult(
  result: unknown,
  calls: AgentModelCallMetadata[],
): unknown {
  const resultRecord = asRecord(result);
  if (!resultRecord) return result;

  // Agent V3 pauses between tool calls. Preserve captures in its engine request
  // metadata so the next iteration can aggregate all model calls.
  if (!Array.isArray(result) && Array.isArray(resultRecord.actions)) {
    const metadata = asRecord(resultRecord.metadata) ?? {};
    resultRecord.metadata = {
      ...metadata,
      universalChatModel: {
        calls: cloneValue(calls) as any[],
      },
    };
    return result;
  }

  const outputMetadata = buildAgentOutputMetadata(calls);
  if (!Array.isArray(result)) return result;

  const outputItems = result[0];
  if (!Array.isArray(outputItems)) return result;
  const lastStructuredOutput = calls[calls.length - 1]?.structuredOutput;

  for (const item of outputItems) {
    const itemRecord = asRecord(item);
    const json = asRecord(itemRecord?.json);
    if (!json) continue;

    // Prefer the final value produced by the Agent/output parser. The captured
    // model value is a fallback for Agent versions that only return text.
    const structuredOutput =
      parseStructuredOutput(json.output) ??
      (lastStructuredOutput
        ? cloneValue(lastStructuredOutput)
        : undefined);
    if (structuredOutput) mergeStructuredOutput(json, structuredOutput);
    if (outputMetadata) Object.assign(json, cloneValue(outputMetadata));
  }

  return result;
}

function patchAgentClass(agentClass: unknown): boolean {
  if (typeof agentClass !== 'function') return false;

  const prototype = (agentClass as any).prototype;
  if (
    !prototype ||
    typeof prototype.execute !== 'function' ||
    prototype[patchedAgentSymbol] === true
  ) {
    return false;
  }

  const originalExecute = prototype.execute;
  prototype.execute = async function (...args: unknown[]) {
    const existingStore = agentCapture.getStore();
    if (existingStore) {
      const nestedResult = await originalExecute.apply(this, args);
      return attachMetadataToAgentResult(nestedResult, existingStore.calls);
    }

    const store: AgentCaptureStore = {
      calls: readPreviousCalls(args),
    };

    return agentCapture.run(store, async () => {
      const result = await originalExecute.apply(this, args);
      return attachMetadataToAgentResult(result, store.calls);
    });
  };

  Object.defineProperty(prototype, patchedAgentSymbol, {
    value: true,
    configurable: false,
    enumerable: false,
    writable: false,
  });

  return true;
}

function patchAgentModule(filename: string, moduleExports: unknown): number {
  if (!agentModulePattern.test(filename)) return 0;

  const exportsRecord = asRecord(moduleExports);
  if (!exportsRecord) return 0;

  let patched = 0;
  for (const exportName of ['AgentV1', 'AgentV2', 'AgentV3']) {
    if (patchAgentClass(exportsRecord[exportName])) patched += 1;
  }

  return patched;
}

function patchLoadedAgentClasses(): number {
  let patched = 0;

  for (const loadedModule of Object.values(require.cache)) {
    patched += patchAgentModule(
      loadedModule?.filename ?? '',
      loadedModule?.exports,
    );
  }

  return patched;
}

let patchChecksScheduled = false;

function installAgentModuleLoadHook(): void {
  const moduleApi = Module as any;
  if (moduleApi[moduleLoadHookSymbol] === true) return;

  const originalLoad = moduleApi._load;
  moduleApi._load = function (
    request: string,
    parent: NodeModule | undefined,
    isMain: boolean,
  ) {
    const loaded = originalLoad.apply(this, arguments);

    try {
      const filename = moduleApi._resolveFilename(request, parent, isMain);
      patchAgentModule(String(filename), loaded);
    } catch {
      // Resolution already succeeded inside the original loader. A diagnostic
      // resolution failure must never affect n8n's own module loading.
    }

    return loaded;
  };

  Object.defineProperty(moduleApi, moduleLoadHookSymbol, {
    value: true,
    configurable: false,
    enumerable: false,
    writable: false,
  });
}

export function installAgentOutputBridge(): number {
  installAgentModuleLoadHook();
  const patched = patchLoadedAgentClasses();

  if (!patchChecksScheduled) {
    patchChecksScheduled = true;
    for (const delay of [0, 50, 250, 1000]) {
      const timer = setTimeout(() => patchLoadedAgentClasses(), delay);
      timer.unref?.();
    }
  }

  return patched;
}

export function recordAgentModelMetadata(
  metadata: AgentModelCallMetadata,
): void {
  const store = agentCapture.getStore();
  if (!store) return;
  store.calls.push(cloneValue(metadata));
}

installAgentOutputBridge();
