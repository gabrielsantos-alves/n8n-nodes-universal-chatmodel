import { AsyncLocalStorage } from 'node:async_hooks';
import { existsSync, mkdirSync, symlinkSync } from 'node:fs';
import Module from 'node:module';
import { join, resolve } from 'node:path';

import type { IDataObject } from 'n8n-workflow';
import {
  mergeStructuredOutput,
  parseStructuredOutput,
} from './StructuredOutput';

export interface AgentModelCallMetadata {
  thoughts?: unknown[];
  tokenUsage?: IDataObject;
  includeTokenUsageInAgentOutput?: boolean;
  includeIntermediateStepsInOutput?: boolean;
  includeServerToolDetailsInOutput?: boolean;
  grounding?: IDataObject;
  codeExecutions?: IDataObject[];
  usageMetadata?: IDataObject;
  gemini?: IDataObject;
  geminiRequests?: IDataObject[];
  structuredOutput?: IDataObject;
}

interface AgentCaptureStore {
  calls: AgentModelCallMetadata[];
}

const agentCaptureKey = Symbol.for(
  'n8n-nodes-universal-chatmodel.agent-capture-store',
);
const agentCapture: AsyncLocalStorage<AgentCaptureStore> =
  (globalThis as any)[agentCaptureKey] ??
  ((globalThis as any)[agentCaptureKey] =
    new AsyncLocalStorage<AgentCaptureStore>());
const patchedAgentSymbol = Symbol.for(
  'n8n-nodes-universal-chatmodel.agent-output-bridge',
);
const patchedExecutorSymbol = Symbol.for(
  'n8n-nodes-universal-chatmodel.agent-executor-output-bridge',
);
const moduleLoadHookSymbol = Symbol.for(
  'n8n-nodes-universal-chatmodel.agent-module-load-hook',
);
const agentModulePattern =
  /[\\/]n8n-nodes-langchain[\\/].*[\\/]nodes[\\/]agents[\\/]Agent[\\/]V[123][\\/]AgentV[123]\.node\.(?:js|cjs)$/i;
const agentExecutorModulePattern =
  /[\\/]n8n-nodes-langchain[\\/].*[\\/]nodes[\\/]agents[\\/]Agent[\\/]agents[\\/].*[\\/]execute\.(?:js|cjs)$/i;

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

export function sanitizeUndefinedTextLeakage(text: string): string {
  return text
    .replace(/(:\s*)undefined(?=[A-Za-z\u00C0-\u017F0-9])/g, '$1')
    .replace(/(^|\s)undefined(?=[A-Za-z\u00C0-\u017F0-9])/g, '$1');
}

/**
 * n8n's Agent can expose LangChain's complete AIMessage under
 * intermediateSteps[].action.messageLog. That message is required while the
 * tool loop is running, but it contains transport-only fields (including
 * Gemini thought signatures) that must not leak into the final workflow item.
 *
 * This only edits the already-finished public result. The engine request and
 * the AIMessage used by the next Gemini call keep their original metadata.
 */
function sanitizePublicIntermediateSteps(
  json: Record<string, any>,
  includeIntermediateSteps: boolean,
): void {
  if (!Array.isArray(json.intermediateSteps)) return;

  if (!includeIntermediateSteps) {
    delete json.intermediateSteps;
    return;
  }

  json.intermediateSteps = json.intermediateSteps.map((step: unknown) => {
    const stepRecord = asRecord(step);
    if (!stepRecord) return step;

    const sanitizedStep = { ...stepRecord };
    delete sanitizedStep.messageLog;
    delete sanitizedStep.message_log;

    const action = asRecord(stepRecord.action);
    if (action) {
      const sanitizedAction = { ...action };
      delete sanitizedAction.messageLog;
      delete sanitizedAction.message_log;
      sanitizedStep.action = sanitizedAction;
    }

    return sanitizedStep;
  });
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

function mergeGrounding(calls: AgentModelCallMetadata[]): IDataObject | undefined {
  const allQueries: string[] = [];
  const allSources: IDataObject[] = [];
  const seenUrls = new Set<string>();
  let lastSearchEntryPoint: unknown;
  const allSupports: unknown[] = [];

  for (const call of calls) {
    const grounding = call.grounding;
    if (!grounding || typeof grounding !== 'object') continue;

    if (Array.isArray(grounding.queries)) {
      for (const q of grounding.queries) {
        if (typeof q === 'string' && q.trim()) allQueries.push(q.trim());
      }
    }
    if (Array.isArray(grounding.sources)) {
      for (const src of grounding.sources) {
        const srcRec = asRecord(src);
        if (!srcRec) continue;
        const key =
          typeof srcRec.url === 'string'
            ? srcRec.url
            : typeof srcRec.title === 'string'
              ? srcRec.title
              : '';
        if (key && seenUrls.has(key)) continue;
        if (key) seenUrls.add(key);
        allSources.push(cloneValue(srcRec));
      }
    }
    if (Array.isArray(grounding.supports)) {
      allSupports.push(...grounding.supports);
    }
    if (grounding.searchEntryPoint) {
      lastSearchEntryPoint = grounding.searchEntryPoint;
    }
  }

  const uniqueQueries = [...new Set(allQueries)];
  if (
    uniqueQueries.length === 0 &&
    allSources.length === 0 &&
    allSupports.length === 0 &&
    !lastSearchEntryPoint
  ) {
    return undefined;
  }

  return {
    ...(uniqueQueries.length > 0 ? { queries: uniqueQueries } : {}),
    ...(allSources.length > 0 ? { sources: allSources } : {}),
    ...(allSupports.length > 0 ? { supports: cloneValue(allSupports) } : {}),
    ...(lastSearchEntryPoint ? { searchEntryPoint: cloneValue(lastSearchEntryPoint) } : {}),
  };
}

function mergeCodeExecutions(calls: AgentModelCallMetadata[]): IDataObject[] | undefined {
  const executions: IDataObject[] = [];
  for (const call of calls) {
    if (Array.isArray(call.codeExecutions)) {
      for (const exec of call.codeExecutions) {
        const execRec = asRecord(exec);
        if (execRec) executions.push(cloneValue(execRec));
      }
    }
  }
  return executions.length > 0 ? executions : undefined;
}

function buildAgentOutputMetadata(
  calls: AgentModelCallMetadata[],
): IDataObject | undefined {
  if (calls.length === 0) return undefined;

  const thoughts = mergeThoughts(calls);
  const lastCall = calls[calls.length - 1];
  const exposeServerToolDetails =
    lastCall?.includeServerToolDetailsInOutput !== false;

  const grounding = exposeServerToolDetails ? mergeGrounding(calls) : undefined;
  const codeExecutions = exposeServerToolDetails
    ? mergeCodeExecutions(calls)
    : undefined;

  const callsWithVisibleUsage = calls.filter(
    (call) => call.includeTokenUsageInAgentOutput === true,
  );
  const lastCallWithVisibleUsage =
    callsWithVisibleUsage[callsWithVisibleUsage.length - 1];
  const geminiRequests = calls.flatMap((call, callIndex) =>
    call.includeTokenUsageInAgentOutput === true && Array.isArray(call.geminiRequests)
      ? call.geminiRequests.map((request, requestIndex) => ({
          call: callIndex + 1,
          requestNumber: requestIndex + 1,
          ...cloneValue(request),
        }))
      : [],
  );

  // Capturing metadata is always enabled internally so Usage Reporter and
  // tracing remain complete. Public Agent output, however, must stay clean
  // unless the corresponding visibility option was explicitly enabled.
  const exposeUsage = callsWithVisibleUsage.length > 0;
  const hasServerToolDetails = Boolean(
    grounding || (codeExecutions && codeExecutions.length > 0),
  );
  if (thoughts.length === 0 && !exposeUsage && !hasServerToolDetails) return undefined;

  const modelResponses = exposeUsage
    ? calls.map((call, index) => ({
        call: index + 1,
        ...(call.thoughts?.length
          ? { thoughts: cloneValue(call.thoughts) }
          : {}),
        ...(exposeServerToolDetails && call.grounding
          ? { grounding: cloneValue(call.grounding) }
          : {}),
        ...(exposeServerToolDetails && call.codeExecutions?.length
          ? { codeExecutions: cloneValue(call.codeExecutions) }
          : {}),
        ...(call.includeTokenUsageInAgentOutput === true && call.tokenUsage
          ? { tokenUsage: cloneValue(call.tokenUsage) }
          : {}),
        ...(call.includeTokenUsageInAgentOutput === true && call.usageMetadata
          ? { usageMetadata: cloneValue(call.usageMetadata) }
          : {}),
        ...(call.includeTokenUsageInAgentOutput === true && call.gemini
          ? { gemini: cloneValue(call.gemini) }
          : {}),
        ...(call.structuredOutput
          ? { structuredOutput: cloneValue(call.structuredOutput) }
          : {}),
      }))
    : [];

  return {
    ...(thoughts.length > 0 ? { thoughts: cloneValue(thoughts) as any[] } : {}),
    ...(grounding ? { grounding: cloneValue(grounding) } : {}),
    ...(codeExecutions && codeExecutions.length > 0
      ? { codeExecutions: cloneValue(codeExecutions) }
      : {}),
    ...(callsWithVisibleUsage.length > 0
      ? { tokenUsage: aggregateTokenUsage(callsWithVisibleUsage) }
      : {}),
    ...(lastCallWithVisibleUsage?.usageMetadata
      ? { usageMetadata: cloneValue(lastCallWithVisibleUsage.usageMetadata) }
      : {}),
    ...(exposeUsage && lastCall.gemini
      ? { gemini: cloneValue(lastCall.gemini) }
      : {}),
    ...(geminiRequests.length > 0
      ? { geminiRequests: cloneValue(geminiRequests) as any[] }
      : {}),
    ...(exposeUsage
      ? {
          modelCalls: calls.length,
          modelResponses: cloneValue(modelResponses) as any[],
        }
      : {}),
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

    // Do not affect Agents backed by another Chat Model. A non-empty capture
    // proves that this execution actually passed through the Universal model.
    if (calls.length > 0) {
      sanitizePublicIntermediateSteps(
        json,
        calls[calls.length - 1]?.includeIntermediateStepsInOutput === true,
      );
      if (typeof json.output === 'string') {
        json.output = sanitizeUndefinedTextLeakage(json.output);
      }
    }

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

async function captureAgentExecution(
  originalExecute: (...args: unknown[]) => Promise<unknown>,
  receiver: unknown,
  args: unknown[],
): Promise<unknown> {
  const existingStore = agentCapture.getStore();
  if (existingStore) {
    const nestedResult = await originalExecute.apply(receiver, args);
    return attachMetadataToAgentResult(nestedResult, existingStore.calls);
  }

  const store: AgentCaptureStore = { calls: readPreviousCalls(args) };
  return agentCapture.run(store, async () => {
    const result = await originalExecute.apply(receiver, args);
    return attachMetadataToAgentResult(result, store.calls);
  });
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
  prototype.execute = function (...args: unknown[]) {
    return captureAgentExecution(originalExecute, this, args);
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
  const exportsRecord = asRecord(moduleExports);
  if (!exportsRecord) return 0;

  let patched = 0;
  if (agentModulePattern.test(filename)) {
    for (const exportName of ['AgentV1', 'AgentV2', 'AgentV3']) {
      if (patchAgentClass(exportsRecord[exportName])) patched += 1;
    }
  }

  // Versioned Agent methods delegate to exported executor functions. This
  // also catches executions when n8n retained the original class method.
  if (agentExecutorModulePattern.test(filename)) {
    for (const [name, original] of Object.entries(exportsRecord)) {
      if (
        !/AgentExecute$/.test(name) ||
        typeof original !== 'function' ||
        (original as any)[patchedExecutorSymbol] === true
      ) continue;

      const wrapped = function (this: unknown, ...args: unknown[]) {
        return captureAgentExecution(original as (...args: unknown[]) => Promise<unknown>, this, args);
      };
      Object.defineProperty(wrapped, patchedExecutorSymbol, { value: true });
      try {
        exportsRecord[name] = wrapped;
        patched += 1;
      } catch {
        // A read-only module namespace still has the class-level bridge.
      }
    }
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
const resolveFallbackSymbol = Symbol.for(
  'n8n-nodes-universal-chatmodel.resolve-fallback-hook',
);

function extractPackageName(request: string): string | undefined {
  if (
    !request ||
    request.startsWith('.') ||
    request.startsWith('/') ||
    request.startsWith('node:') ||
    /^[a-zA-Z]:[\\/]/.test(request)
  ) {
    return undefined;
  }
  const parts = request.split('/');
  if (request.startsWith('@')) {
    return parts.length >= 2 ? `${parts[0]}/${parts[1]}` : undefined;
  }
  return parts[0];
}

function getPackageRootDir(): string {
  const distCandidate = resolve(__dirname, '..', '..', '..');
  if (existsSync(join(distCandidate, 'package.json'))) {
    return distCandidate;
  }
  return resolve(__dirname, '..', '..');
}

function ensureNestedNodeModulesFallbacks(): void {
  try {
    const pkgRoot = getPackageRootDir();
    const localNodeModules = join(pkgRoot, 'node_modules');
    const sharedNodeModules = resolve(pkgRoot, '..');
    const zodSource = existsSync(join(localNodeModules, 'zod', 'package.json'))
      ? join(localNodeModules, 'zod')
      : existsSync(join(sharedNodeModules, 'zod', 'package.json'))
        ? join(sharedNodeModules, 'zod')
        : undefined;

    if (!zodSource) return;

    for (const subPkg of ['core', 'anthropic', 'openai', 'google-genai']) {
      const langchainSubDir = join(localNodeModules, '@langchain', subPkg);
      if (!existsSync(langchainSubDir)) continue;
      const nestedZodDir = join(langchainSubDir, 'node_modules', 'zod');
      if (existsSync(join(nestedZodDir, 'package.json'))) continue;
      try {
        mkdirSync(join(langchainSubDir, 'node_modules'), { recursive: true });
        symlinkSync(zodSource, nestedZodDir, 'junction');
      } catch {
        // Ignore read-only filesystem or permission errors; _resolveFilename hook still covers it
      }
    }
  } catch {
    // Best-effort filesystem link
  }
}

function installStalePackageCacheFallback(): void {
  const moduleApi = Module as any;
  if (moduleApi[resolveFallbackSymbol] === true) return;

  const originalResolveFilename = moduleApi._resolveFilename;
  moduleApi._resolveFilename = function (
    request: string,
    parent: NodeModule | undefined,
    isMain: boolean,
    options?: any,
  ) {
    try {
      return originalResolveFilename.apply(this, arguments);
    } catch (err: any) {
      const pkgName =
        typeof request === 'string' ? extractPackageName(request) : undefined;
      if (pkgName && parent && Array.isArray(parent.paths)) {
        const pathCache = moduleApi._pathCache;
        if (pathCache && typeof pathCache === 'object') {
          for (const key of Object.keys(pathCache)) {
            if (key.includes(pkgName)) delete pathCache[key];
          }
        }

        const localNm = join(getPackageRootDir(), 'node_modules');
        const candidatePaths = [
          localNm,
          ...parent.paths.filter((p: string) => p !== localNm),
        ];
        const validPaths = candidatePaths.filter((p: string) =>
          existsSync(join(p, pkgName, 'package.json')),
        );

        for (const validPath of validPaths) {
          try {
            const singlePathParent = Object.assign(
              Object.create(Object.getPrototypeOf(parent)),
              parent,
              { paths: [validPath] },
            );
            return originalResolveFilename.call(
              this,
              request,
              singlePathParent,
              isMain,
              options,
            );
          } catch {
            // Try next valid node_modules path that actually contains the required subpath
          }
        }
      }
      throw err;
    }
  };

  Object.defineProperty(moduleApi, resolveFallbackSymbol, {
    value: true,
    configurable: false,
    enumerable: false,
    writable: false,
  });
}

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
  ensureNestedNodeModulesFallbacks();
  installStalePackageCacheFallback();
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
