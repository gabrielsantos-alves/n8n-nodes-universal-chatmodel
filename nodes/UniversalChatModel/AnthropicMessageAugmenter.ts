export function parsePayload(input: unknown): Record<string, any> | undefined {
  if (!input) return undefined;
  if (typeof input === 'object') return input as Record<string, any>;
  if (typeof input === 'string') {
    try {
      const parsed = JSON.parse(input);
      if (parsed && typeof parsed === 'object') return parsed;
    } catch {
      // String is not valid JSON
    }
  }
  return undefined;
}

export function augmentAnthropicMessage(message: any): void {
  if (!message || typeof message !== 'object') return;

  const queries = new Set<string>();
  const sources: Array<{ title?: string; url?: string }> = [];
  const seenUrls = new Set<string>();
  const codeExecutions: Array<{
    language?: string;
    code?: string;
    outcome?: string;
    output?: string;
  }> = [];

  const addSource = (url?: string, title?: string) => {
    if (!url && !title) return;
    const key = url ?? title ?? '';
    if (seenUrls.has(key)) return;
    seenUrls.add(key);
    sources.push({ ...(title ? { title } : {}), ...(url ? { url } : {}) });
  };

  // Inspect content blocks
  if (Array.isArray(message.content)) {
    for (const block of message.content) {
      if (!block || typeof block !== 'object') continue;

      if ((block as any).type === 'text' && (block as any).text === undefined) {
        (block as any).text = '';
      }

      if (Array.isArray((block as any).citations)) {
        for (const citation of (block as any).citations) {
          if (!citation || typeof citation !== 'object') continue;
          addSource(citation.url ?? citation.uri, citation.title ?? citation.document_title);
        }
      }

      const type = (block as any).type;
      const name = (block as any).name;
      const input = (block as any).input;

      if (
        (type === 'tool_use' || type === 'server_tool_use') &&
        typeof name === 'string'
      ) {
        const parsed = parsePayload(input);
        if (name.includes('web_search')) {
          const q = typeof parsed?.query === 'string'
            ? parsed.query
            : typeof input === 'string' && !input.startsWith('{')
              ? input
              : undefined;
          if (q?.trim()) queries.add(q.trim());
          if (Array.isArray(parsed?.queries)) {
            for (const item of parsed.queries) {
              if (typeof item === 'string' && item.trim()) queries.add(item.trim());
            }
          }
        } else if (name.includes('web_fetch')) {
          const u = typeof parsed?.url === 'string' ? parsed.url.trim() : undefined;
          const t = typeof parsed?.title === 'string' ? parsed.title.trim() : undefined;
          if (u || t) addSource(u, t);
        } else if (name.includes('code_execution')) {
          const code = typeof parsed?.code === 'string'
            ? parsed.code
            : typeof input === 'string' && !input.startsWith('{')
              ? input
              : undefined;
          if (code) {
            codeExecutions.push({
              language: parsed?.language ?? 'python',
              code,
            });
          }
        }
      }

      if (type === 'web_search_tool_result') {
        let items = (block as any).content;
        if (typeof items === 'string') {
          try {
            items = JSON.parse(items);
          } catch {}
        }
        if (Array.isArray(items)) {
          for (const item of items) {
            if (item && typeof item === 'object') {
              addSource(item.url ?? item.uri, item.title ?? item.name);
            }
          }
        }
      }
    }
  }

  // Inspect tool_calls
  const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const tc of toolCalls) {
    if (!tc || typeof tc !== 'object') continue;
    const name = tc.name;
    const args = parsePayload(tc.args) ?? tc.args;
    if (typeof name === 'string' && args && typeof args === 'object') {
      if (name.includes('web_search')) {
        if (typeof args.query === 'string' && args.query.trim()) {
          queries.add(args.query.trim());
        }
        if (Array.isArray(args.queries)) {
          for (const item of args.queries) {
            if (typeof item === 'string' && item.trim()) queries.add(item.trim());
          }
        }
      } else if (name.includes('web_fetch') && typeof args.url === 'string') {
        addSource(args.url.trim(), args.title);
      } else if (name.includes('code_execution') && typeof args.code === 'string') {
        codeExecutions.push({
          language: args.language ?? 'python',
          code: args.code,
        });
      }
    }
  }

  // Inspect response_metadata.citations
  const resMeta = message.response_metadata;
  if (resMeta && typeof resMeta === 'object' && Array.isArray(resMeta.citations)) {
    for (const citation of resMeta.citations) {
      if (!citation || typeof citation !== 'object') continue;
      addSource(citation.url ?? citation.uri, citation.title ?? citation.document_title);
    }
  }

  const addKwargs = message.additional_kwargs;
  if (addKwargs && typeof addKwargs === 'object' && Array.isArray(addKwargs.citations)) {
    for (const citation of addKwargs.citations) {
      if (!citation || typeof citation !== 'object') continue;
      addSource(citation.url ?? citation.uri, citation.title ?? citation.document_title);
    }
  }

  if (queries.size > 0 || sources.length > 0) {
    const existingGrounding = message.response_metadata?.grounding ?? {};
    const existingQueries = Array.isArray(existingGrounding.queries) ? existingGrounding.queries : [];
    const mergedQueries = [...new Set([...existingQueries, ...queries])];

    const existingSources = Array.isArray(existingGrounding.sources) ? existingGrounding.sources : [];
    const mergedSources = [...existingSources];
    for (const src of sources) {
      const key = src.url ?? src.title ?? '';
      if (!mergedSources.some((s: any) => (s.url ?? s.title ?? '') === key)) {
        mergedSources.push(src);
      }
    }

    message.response_metadata = {
      ...(message.response_metadata ?? {}),
      grounding: {
        ...(mergedQueries.length > 0 ? { queries: mergedQueries } : {}),
        ...(mergedSources.length > 0 ? { sources: mergedSources } : {}),
      },
    };
  }

  if (codeExecutions.length > 0) {
    const existingCodeExecutions = Array.isArray(message.response_metadata?.codeExecutions)
      ? message.response_metadata.codeExecutions
      : [];
    message.response_metadata = {
      ...(message.response_metadata ?? {}),
      codeExecutions: [...existingCodeExecutions, ...codeExecutions],
    };
  }
}
