export type ModelProvider = 'gemini' | 'openai_compatible';

export type ModelErrorCategory =
  | 'authentication'
  | 'permission'
  | 'invalid_request'
  | 'model_not_found'
  | 'rate_limit'
  | 'quota'
  | 'timeout'
  | 'cancelled'
  | 'network'
  | 'dns'
  | 'connection'
  | 'tls'
  | 'invalid_url'
  | 'tool_schema'
  | 'tool_protocol'
  | 'tool_execution'
  | 'structured_output'
  | 'safety'
  | 'server'
  | 'configuration'
  | 'empty_response'
  | 'unknown';

export interface NormalizedModelError {
  provider: ModelProvider;
  category: ModelErrorCategory;
  name: string;
  message: string;
  description: string;
  retryable: boolean;
  statusCode?: number;
  status?: string;
  code?: string;
  retryAfterMs?: number;
  attempts?: number;
  requestId?: string;
  endpoint?: string;
  apiDetails?: unknown[];
  cause?: {
    name: string;
    message: string;
    code?: string;
  };
}

type UnknownRecord = Record<string, unknown>;

const RETRYABLE_HTTP_STATUSES = new Set([408, 429, 500, 502, 503, 504]);
const RETRYABLE_TRANSPORT_CODES = new Set([
  'EAI_AGAIN',
  'ECONNREFUSED',
  'ECONNRESET',
  'ENETDOWN',
  'ENETUNREACH',
  'ENOTFOUND',
  'EPIPE',
  'ETIMEDOUT',
  'UND_ERR_CONNECT_TIMEOUT',
  'UND_ERR_HEADERS_TIMEOUT',
  'UND_ERR_SOCKET',
]);
const SENSITIVE_KEY = /authorization|api[-_]?key|access[-_]?token|refresh[-_]?token|password|secret|credential/i;
const SECRET_IN_TEXT = [
  /\bBearer\s+[A-Za-z0-9._~+/-]+=*/gi,
  /\bAIza[0-9A-Za-z_-]{20,}\b/g,
  /([?&](?:key|api_key|access_token)=)[^&#\s]+/gi,
];

function asRecord(value: unknown): UnknownRecord | undefined {
  return value !== null && typeof value === 'object'
    ? (value as UnknownRecord)
    : undefined;
}

function stringValue(value: unknown): string | undefined {
  if (typeof value === 'string' && value.trim()) return value.trim();
  if (typeof value === 'number') return String(value);
  return undefined;
}

function numberValue(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim() && Number.isFinite(Number(value))) {
    return Number(value);
  }
  return undefined;
}

export function redactModelErrorText(value: string): string {
  let output = value;
  for (const pattern of SECRET_IN_TEXT) {
    output = output.replace(pattern, (match, prefix?: string) =>
      prefix ? `${prefix}[REDACTED]` : '[REDACTED]',
    );
  }
  return output.length > 8_000 ? `${output.slice(0, 8_000)}…` : output;
}

function sanitizeEndpoint(value: string | undefined): string | undefined {
  if (!value) return undefined;
  try {
    const url = new URL(value);
    for (const name of [...url.searchParams.keys()]) {
      if (SENSITIVE_KEY.test(name) || /^(key|token)$/i.test(name)) {
        url.searchParams.set(name, '[REDACTED]');
      }
    }
    return url.toString();
  } catch {
    return redactModelErrorText(value);
  }
}

function sanitizeValue(
  value: unknown,
  depth = 0,
  seen = new WeakSet<object>(),
): unknown {
  if (depth > 5) return '[TRUNCATED]';
  if (typeof value === 'string') return redactModelErrorText(value);
  if (
    value === null ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  ) {
    return value;
  }
  if (Array.isArray(value)) {
    return value.slice(0, 30).map((entry) => sanitizeValue(entry, depth + 1, seen));
  }
  const record = asRecord(value);
  if (!record) return String(value);
  if (seen.has(record)) return '[CIRCULAR]';
  seen.add(record);

  const result: UnknownRecord = {};
  for (const [key, entry] of Object.entries(record).slice(0, 50)) {
    result[key] = SENSITIVE_KEY.test(key)
      ? '[REDACTED]'
      : sanitizeValue(entry, depth + 1, seen);
  }
  return result;
}

function headerRecord(value: unknown): Record<string, string> {
  if (value instanceof Headers) {
    return Object.fromEntries(value.entries());
  }
  const record = asRecord(value);
  if (!record) return {};
  return Object.fromEntries(
    Object.entries(record)
      .filter(([, entry]) => typeof entry === 'string' || typeof entry === 'number')
      .map(([key, entry]) => [key.toLowerCase(), String(entry)]),
  );
}

function readRetryAfterMs(headers: Record<string, string>): number | undefined {
  const milliseconds = numberValue(headers['retry-after-ms']);
  if (milliseconds !== undefined && milliseconds >= 0) return milliseconds;

  const retryAfter = headers['retry-after'];
  if (!retryAfter) return undefined;
  const seconds = numberValue(retryAfter);
  if (seconds !== undefined && seconds >= 0) return seconds * 1_000;

  const date = Date.parse(retryAfter);
  return Number.isFinite(date) ? Math.max(date - Date.now(), 0) : undefined;
}

function findNestedError(record: UnknownRecord): UnknownRecord | undefined {
  const data = asRecord(record.data) ?? asRecord(asRecord(record.response)?.data);
  return (
    asRecord(data?.error) ??
    asRecord(record.error) ??
    asRecord(asRecord(record.body)?.error)
  );
}

function findCause(record: UnknownRecord): UnknownRecord | undefined {
  return asRecord(record.cause) ?? asRecord(asRecord(record.error)?.cause);
}

function statusFrom(
  record: UnknownRecord,
  nestedError: UnknownRecord | undefined,
): number | undefined {
  const response = asRecord(record.response);
  return (
    numberValue(record.statusCode) ??
    numberValue(record.status) ??
    numberValue(response?.status) ??
    numberValue(nestedError?.code)
  );
}

function apiStatusFrom(
  record: UnknownRecord,
  nestedError: UnknownRecord | undefined,
): string | undefined {
  const data = asRecord(record.data) ?? asRecord(asRecord(record.response)?.data);
  return (
    stringValue(nestedError?.status) ??
    stringValue(data?.status) ??
    (typeof record.status === 'string' && !/^\d+$/.test(record.status)
      ? record.status
      : undefined)
  );
}

function codeFrom(
  record: UnknownRecord,
  nestedError: UnknownRecord | undefined,
  cause: UnknownRecord | undefined,
): string | undefined {
  const candidates = [
    nestedError?.code,
    record.code,
    cause?.code,
    asRecord(record.error)?.code,
  ];
  for (const candidate of candidates) {
    const value = stringValue(candidate);
    if (value && !/^\d+$/.test(value)) return value;
  }
  return undefined;
}

function messageFrom(
  error: unknown,
  record: UnknownRecord,
  nestedError: UnknownRecord | undefined,
): string {
  const data = asRecord(record.data) ?? asRecord(asRecord(record.response)?.data);
  const message =
    stringValue(nestedError?.message) ??
    stringValue(data?.message) ??
    stringValue(record.message) ??
    (typeof error === 'string' ? error : undefined) ??
    'The model request failed.';
  return redactModelErrorText(message);
}

function includesAny(message: string, patterns: RegExp[]): boolean {
  return patterns.some((pattern) => pattern.test(message));
}

function classify(
  name: string,
  message: string,
  statusCode: number | undefined,
  status: string | undefined,
  code: string | undefined,
): ModelErrorCategory {
  const combined = `${name} ${message} ${status ?? ''} ${code ?? ''}`;

  if (
    name === 'AbortError' ||
    statusCode === 499 ||
    /\b(cancelled|canceled)\b/i.test(combined)
  ) {
    return 'cancelled';
  }
  if (
    statusCode === 408 ||
    statusCode === 504 ||
    /timeout|timed out|deadline.?exceeded|ECONNABORTED|ETIMEDOUT|UND_ERR_(CONNECT|HEADERS)_TIMEOUT/i.test(
      combined,
    )
  ) {
    return 'timeout';
  }
  if (/ERR_INVALID_URL|invalid url|failed to parse url|unsupported protocol/i.test(combined)) {
    return 'invalid_url';
  }
  if (/ENOTFOUND|EAI_AGAIN|getaddrinfo|name resolution|dns/i.test(combined)) {
    return 'dns';
  }
  if (/CERT_|certificate|self.?signed|unable to verify|TLS|SSL/i.test(combined)) {
    return 'tls';
  }
  if (/ECONNREFUSED|connection refused/i.test(combined)) {
    return 'connection';
  }
  if (
    /fetch failed|ECONNRESET|ENETDOWN|ENETUNREACH|EPIPE|UND_ERR_SOCKET|network/i.test(
      combined,
    )
  ) {
    return 'network';
  }
  if (
    name === 'PromptBlockedError' ||
    /prompt (?:was )?blocked|safety filter|finish.?reason.?safety/i.test(combined)
  ) {
    return 'safety';
  }
  if (
    name === 'ToolCallNotFoundError' ||
    /thought.?signature|tool.?call.?id|function.?response.*(?:missing|mismatch)|invalid.?tool.?results/i.test(
      combined,
    )
  ) {
    return 'tool_protocol';
  }
  if (
    name === 'InvalidToolError' ||
    includesAny(combined, [
      /properties: should be non-empty for OBJECT type/i,
      /function declaration/i,
      /tool schema/i,
      /invalid tool/i,
      /function.*(?:schema|parameters).*(?:invalid|unsupported)/i,
    ])
  ) {
    return 'tool_schema';
  }
  if (
    /tool execution|tool failed|error (?:executing|running) tool|ToolExecutionError/i.test(
      combined,
    )
  ) {
    return 'tool_execution';
  }
  if (
    name === 'MalformedOutputError' ||
    /structured output|response(?:Json)?Schema|response schema|malformed json|cannot parse structured|invalid JSON in Structured/i.test(
      combined,
    )
  ) {
    return 'structured_output';
  }
  if (name === 'NoCandidatesError' || /no candidates returned/i.test(combined)) {
    return 'empty_response';
  }
  if (
    statusCode === 401 ||
    /UNAUTHENTICATED|API_KEY_INVALID|invalid api key|authentication/i.test(combined)
  ) {
    return 'authentication';
  }
  if (statusCode === 403 || /PERMISSION_DENIED|forbidden|permission/i.test(combined)) {
    return 'permission';
  }
  if (statusCode === 404 || /MODEL_NOT_FOUND|model .*not found/i.test(combined)) {
    return 'model_not_found';
  }
  if (
    statusCode === 429 &&
    /quota|billing|spend|resource_exhausted|tokens per (?:day|minute)|requests per (?:day|minute)/i.test(
      combined,
    )
  ) {
    return 'quota';
  }
  if (statusCode === 429 || /rate.?limit|too many requests/i.test(combined)) {
    return 'rate_limit';
  }
  if (statusCode !== undefined && statusCode >= 500) return 'server';
  if (
    statusCode === 400 ||
    statusCode === 405 ||
    statusCode === 406 ||
    statusCode === 409 ||
    statusCode === 422 ||
    /INVALID_ARGUMENT|FAILED_PRECONDITION|bad request/i.test(combined)
  ) {
    return 'invalid_request';
  }
  if (name === 'ConfigurationError' || /configuration/i.test(combined)) {
    return 'configuration';
  }
  return 'unknown';
}

function hintFor(
  category: ModelErrorCategory,
  status: string | undefined,
): string {
  switch (category) {
    case 'authentication':
      return 'Check the API credential, whether the key is active, and whether it belongs to the intended project.';
    case 'permission':
      return 'Check API-key permissions, project access, billing, model access, and regional availability.';
    case 'invalid_request':
      return status === 'FAILED_PRECONDITION'
        ? 'Check billing and regional availability, then verify that the selected API version supports every configured feature.'
        : 'Check the selected model, API version, generation options, message contents, and request schema.';
    case 'model_not_found':
      return 'Check the model ID and confirm that the model exists and supports the endpoint/API version being used.';
    case 'rate_limit':
      return 'Reduce concurrency or request size and retry with backoff. Check RPM and TPM limits for the selected model.';
    case 'quota':
      return 'Check the project quota, billing tier, daily limits, token limits, and spend limits before retrying.';
    case 'timeout':
      return 'Reduce prompt/tool context, check network and proxy timeouts, or increase the upstream timeout.';
    case 'cancelled':
      return 'The caller cancelled the request. Cancellation is not retried or converted into regular output.';
    case 'dns':
      return 'Check the endpoint hostname, DNS resolution, proxy configuration, and network connectivity.';
    case 'connection':
      return 'The endpoint refused the connection. Check that the server is running, the port is correct, and the n8n host can reach it.';
    case 'tls':
      return 'Check the HTTPS certificate chain, hostname, corporate proxy, and local CA configuration.';
    case 'invalid_url':
      return 'Use an absolute HTTP or HTTPS Base URL, including the protocol and correct port/path.';
    case 'network':
      return 'Check internet access, proxy/firewall rules, endpoint availability, and connection stability.';
    case 'tool_schema':
      return 'Check every tool name, description, and parameter schema. Gemini object schemas must contain valid properties and use supported JSON Schema features.';
    case 'tool_protocol':
      return 'Preserve tool-call IDs and Gemini thought signatures exactly across the AI message, tool result, and next model call.';
    case 'tool_execution':
      return 'Inspect the tool node/MCP server error. The chat model can report tool-protocol failures, but the AI Agent executes external tools.';
    case 'structured_output':
      return 'Check that the schema is valid JSON, supported by Gemini, and compatible with the selected model and any enabled tools.';
    case 'safety':
      return 'Inspect the prompt feedback and safety ratings, then adjust the request or safety configuration when appropriate.';
    case 'server':
      return 'This is usually transient. Retry with backoff, reduce context if the problem persists, and check provider status.';
    case 'configuration':
      return 'Check the node configuration and remove incompatible or unsupported options.';
    case 'empty_response':
      return 'Inspect prompt feedback and finish reasons; the request may have been blocked or produced no valid candidate.';
    default:
      return 'Inspect the original message, endpoint logs, and the AI Agent/tool execution data for the underlying cause.';
  }
}

function defaultRetryable(
  category: ModelErrorCategory,
  statusCode: number | undefined,
  code: string | undefined,
): boolean {
  if (category === 'cancelled') return false;
  if (statusCode !== undefined) return RETRYABLE_HTTP_STATUSES.has(statusCode);
  if (code && RETRYABLE_TRANSPORT_CODES.has(code.toUpperCase())) return true;
  return category === 'network' || category === 'dns' || category === 'connection';
}

function providerLabel(provider: ModelProvider): string {
  return provider === 'gemini' ? 'Gemini' : 'OpenAI-compatible';
}

export function formatNormalizedModelError(details: NormalizedModelError): string {
  const identifiers = [
    details.statusCode !== undefined ? `HTTP ${details.statusCode}` : undefined,
    details.status,
    details.code && details.code !== details.status ? details.code : undefined,
  ].filter(Boolean);
  return `${providerLabel(details.provider)} request failed${
    identifiers.length > 0 ? ` (${identifiers.join(' · ')})` : ''
  }: ${details.message}`;
}

export function normalizeModelError(
  error: unknown,
  provider: ModelProvider,
): NormalizedModelError {
  const existing = asRecord(error)?.normalizedModelError;
  if (existing && typeof existing === 'object') {
    return existing as unknown as NormalizedModelError;
  }

  const record = asRecord(error) ?? {};
  const nestedError = findNestedError(record);
  const cause = findCause(record);
  const statusCode = statusFrom(record, nestedError);
  const status = apiStatusFrom(record, nestedError);
  const code = codeFrom(record, nestedError, cause);
  const name =
    stringValue(record.name) ??
    (typeof error === 'string' ? 'Error' : 'UnknownError');
  const message = messageFrom(error, record, nestedError);
  const category = classify(name, message, statusCode, status, code);
  const headers = {
    ...headerRecord(asRecord(record.response)?.headers),
    ...headerRecord(record.headers),
  };
  const requestId =
    headers['x-request-id'] ??
    headers['x-goog-request-id'] ??
    headers['x-correlation-id'];
  const data = asRecord(record.data) ?? asRecord(asRecord(record.response)?.data);
  const apiDetails = Array.isArray(nestedError?.details)
    ? (sanitizeValue(nestedError.details) as unknown[])
    : Array.isArray(data?.details)
      ? (sanitizeValue(data.details) as unknown[])
      : undefined;

  let retryable = defaultRetryable(category, statusCode, code);
  if (typeof record.isRetryable === 'function') {
    try {
      retryable = Boolean((record.isRetryable as () => boolean).call(error));
    } catch {
      // Keep the classification-based value if a third-party helper fails.
    }
  }

  return {
    provider,
    category,
    name,
    message,
    description: hintFor(category, status),
    retryable,
    ...(statusCode !== undefined ? { statusCode } : {}),
    ...(status ? { status } : {}),
    ...(code ? { code } : {}),
    ...(readRetryAfterMs(headers) !== undefined
      ? { retryAfterMs: readRetryAfterMs(headers) }
      : {}),
    ...(requestId ? { requestId: redactModelErrorText(requestId) } : {}),
    ...(sanitizeEndpoint(stringValue(record.url)) !== undefined
      ? { endpoint: sanitizeEndpoint(stringValue(record.url)) }
      : {}),
    ...(apiDetails && apiDetails.length > 0 ? { apiDetails } : {}),
    ...(cause
      ? {
          cause: {
            name: stringValue(cause.name) ?? 'Error',
            message: redactModelErrorText(
              stringValue(cause.message) ?? 'Transport request failed.',
            ),
            ...(stringValue(cause.code)
              ? { code: stringValue(cause.code) }
              : {}),
          },
        }
      : {}),
  };
}

export class UniversalModelError extends Error {
  readonly normalizedModelError: NormalizedModelError;
  readonly description: string;
  readonly retryable: boolean;
  readonly statusCode?: number;
  readonly status?: number;
  readonly code?: string;
  readonly retryAfterMs?: number;

  constructor(details: NormalizedModelError, cause?: unknown) {
    super(formatNormalizedModelError(details), { cause });
    this.name = 'UniversalModelError';
    this.normalizedModelError = details;
    this.description = details.description;
    this.retryable = details.retryable;
    this.statusCode = details.statusCode;
    this.status = details.statusCode;
    this.code = details.code ?? details.status;
    this.retryAfterMs = details.retryAfterMs;
  }
}

export function toUniversalModelError(
  error: unknown,
  provider: ModelProvider,
): Error {
  if (error instanceof UniversalModelError) return error;
  const details = normalizeModelError(error, provider);
  if (details.category === 'cancelled' && error instanceof Error) return error;
  return new UniversalModelError(details, error);
}

export function isRetryableModelError(
  error: unknown,
  provider: ModelProvider,
): boolean {
  return normalizeModelError(error, provider).retryable;
}

export function retryAfterMsForModelError(
  error: unknown,
  provider: ModelProvider,
): number | undefined {
  return normalizeModelError(error, provider).retryAfterMs;
}
