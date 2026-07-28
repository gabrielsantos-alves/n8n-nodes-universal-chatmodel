import type { IDataObject } from 'n8n-workflow';

const UNSAFE_KEYS = new Set(['__proto__', 'prototype', 'constructor']);

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function textFromContentBlocks(value: unknown[]): string {
  return value
    .map((block) => {
      if (typeof block === 'string') return block;
      const record = asRecord(block);
      return typeof record?.text === 'string' ? record.text : '';
    })
    .join('');
}

function stripJsonFence(value: string): string {
  const trimmed = value.trim();
  const match = /^```(?:json)?\s*([\s\S]*?)\s*```$/i.exec(trimmed);
  return match?.[1]?.trim() ?? trimmed;
}

/**
 * Converts a model JSON response into an n8n object without changing the
 * LangChain message content. LangChain and the AI Agent still require `text`
 * to remain a string, while n8n output items can expose the parsed fields.
 */
export function parseStructuredOutput(
  value: unknown,
): IDataObject | undefined {
  if (Array.isArray(value)) {
    return parseStructuredOutput(textFromContentBlocks(value));
  }

  const direct = asRecord(value);
  if (direct) return structuredClone(direct) as IDataObject;
  if (typeof value !== 'string' || !value.trim()) return undefined;

  let parsed: unknown = stripJsonFence(value);
  for (let depth = 0; depth < 2 && typeof parsed === 'string'; depth += 1) {
    try {
      parsed = JSON.parse(stripJsonFence(parsed));
    } catch {
      return undefined;
    }
  }

  const record = asRecord(parsed);
  return record ? (structuredClone(record) as IDataObject) : undefined;
}

/**
 * Promotes structured fields to an n8n item while preventing prototype
 * pollution. Callers decide ordering so telemetry fields can remain
 * authoritative if a user schema reuses one of their names.
 */
export function mergeStructuredOutput(
  target: Record<string, unknown>,
  structuredOutput: IDataObject,
): void {
  for (const [key, value] of Object.entries(structuredOutput)) {
    if (UNSAFE_KEYS.has(key)) continue;
    target[key] = structuredClone(value);
  }
}
