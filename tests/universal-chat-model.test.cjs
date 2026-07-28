const test = require('node:test');
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const { join } = require('node:path');

const {
  GeminiChatModel,
  formatGeminiUsage,
} = require('../dist/nodes/UniversalChatModel/GeminiChatModel.js');
const {
  applyModelRetry,
  getModelExecutionSettings,
  UniversalChatModel,
} = require('../dist/nodes/UniversalChatModel/UniversalChatModel.node.js');
const {
  normalizeModelError,
} = require('../dist/nodes/UniversalChatModel/ModelError.js');
const {
  parseStructuredOutput,
} = require('../dist/nodes/UniversalChatModel/StructuredOutput.js');
const {
  installAgentOutputBridge,
  recordAgentModelMetadata,
} = require('../dist/nodes/UniversalChatModel/AgentOutputBridge.js');
const {
  HumanMessage: N8n225HumanMessage,
  ToolMessage: N8n225ToolMessage,
  isAIMessage: isN8n225AIMessage,
} = require('langchain-core-n8n-225/messages');
const {
  DynamicStructuredTool: N8n225DynamicStructuredTool,
} = require('langchain-core-n8n-225/tools');
const {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  SystemMessage,
} = require('@langchain/core/messages');
const {
  ChatGenerationChunk,
} = require('@langchain/core/outputs');
const { z } = require('zod');

const usageMetadata = {
  promptTokenCount: 100,
  cachedContentTokenCount: 60,
  candidatesTokenCount: 20,
  toolUsePromptTokenCount: 7,
  thoughtsTokenCount: 13,
  totalTokenCount: 140,
  promptTokensDetails: [{ modality: 'TEXT', tokenCount: 100 }],
  cacheTokensDetails: [{ modality: 'TEXT', tokenCount: 60 }],
  candidatesTokensDetails: [{ modality: 'TEXT', tokenCount: 20 }],
  toolUsePromptTokensDetails: [{ modality: 'TEXT', tokenCount: 7 }],
};

function makeGeminiResponse() {
  return {
    candidates: [
      {
        content: {
          role: 'model',
          parts: [
            { text: 'Resumo do raciocínio', thought: true },
            { text: 'Resposta final', thoughtSignature: 'signature' },
          ],
        },
        finishReason: 'STOP',
        index: 0,
      },
    ],
    usageMetadata,
    modelVersion: 'gemini-test-001',
    responseId: 'response-test',
  };
}

test('Gemini generation controls are optional collection entries', () => {
  const description = new UniversalChatModel().description;
  const options = description.properties.find((property) => property.name === 'geminiOptions');

  assert.equal(options.type, 'collection');
  assert.deepEqual(options.default, {});
  assert.deepEqual(
    options.options.map((option) => option.name),
    [
      'temperature',
      'topP',
      'topK',
      'maxOutputTokens',
      'responseMimeType',
      'responseSchema',
      'thinkingLevel',
      'thinkingBudget',
      'includeThoughts',
      'safetySettings',
      'systemMessage',
      'includeTokenUsageInAgentOutput',
      'usageReporter',
    ],
  );
  assert.equal(
    description.properties.some((property) =>
      /^gemini(Temperature|TopP|TopK|MaxTokens|ResponseMimeType)$/.test(property.name),
    ),
    false,
  );
  const responseSchema = options.options.find((option) => option.name === 'responseSchema');
  assert.deepEqual(JSON.parse(responseSchema.default), {
    type: 'object',
    properties: {
      message: { type: 'string' },
    },
    additionalProperties: false,
    required: ['message'],
  });
  assert.equal(
    description.properties.some((property) => property.name === 'geminiApiKey'),
    false,
  );
  const credentials = Object.fromEntries(
    description.credentials.map((credential) => [credential.name, credential]),
  );
  assert.equal(credentials.googleGeminiApi.required, true);
  assert.equal(credentials.openAiCompatibleApi.required, false);
});

test('Usage Reporter is an optional AI Tool input with configurable privacy controls', () => {
  const description = new UniversalChatModel().description;
  assert.equal(
    description.inputs,
    '={{ (($parameter.provider === "gemini" && $parameter.geminiOptions && (($parameter.geminiOptions.usageReporter && $parameter.geminiOptions.usageReporter.settings && $parameter.geminiOptions.usageReporter.settings.enabled) || $parameter.geminiOptions.enableUsageReporter)) || ($parameter.provider === "openai_compatible" && $parameter.openaiOptions && (($parameter.openaiOptions.usageReporter && $parameter.openaiOptions.usageReporter.settings && $parameter.openaiOptions.usageReporter.settings.enabled) || $parameter.openaiOptions.enableUsageReporter)) || $parameter.enableUsageReporter) ? [{ type: "ai_tool", displayName: "Usage Reporter", required: false, maxConnections: 1 }] : [] }}',
  );

  const geminiOptions = description.properties.find(
    (property) => property.name === 'geminiOptions',
  );
  const openaiOptions = description.properties.find(
    (property) => property.name === 'openaiOptions',
  );
  const reportingNames = [
    'systemMessage',
    'includeTokenUsageInAgentOutput',
    'usageReporter',
  ];

  for (const options of [geminiOptions, openaiOptions]) {
    const nested = Object.fromEntries(
      options.options.map((option) => [option.name, option]),
    );
    assert.deepEqual(
      options.options
        .map((option) => option.name)
        .filter((name) => reportingNames.includes(name)),
      reportingNames,
    );
    assert.equal(nested.systemMessage.default, '');
    assert.equal(
      nested.includeTokenUsageInAgentOutput.displayName,
      'Include Token Usage in Output',
    );
    assert.equal(nested.includeTokenUsageInAgentOutput.default, true);
    assert.equal(nested.usageReporter.type, 'fixedCollection');
    const reporterValues = Object.fromEntries(
      nested.usageReporter.options[0].values.map((option) => [
        option.name,
        option,
      ]),
    );
    assert.deepEqual(Object.keys(reporterValues), [
      'enabled',
      'nodeLabel',
      'inputTextMode',
      'inputTextLabel',
      'includeOutputText',
      'failOnReporterError',
    ]);
    assert.equal(reporterValues.enabled.default, false);
    assert.deepEqual(reporterValues.nodeLabel.displayOptions.show, {
      enabled: [true],
    });
    assert.equal(reporterValues.inputTextMode.default, 'label');
    assert.equal(reporterValues.inputTextLabel.default, 'RAG');
  }

  assert.deepEqual(
    description.properties
      .map((property) => property.name)
      .filter((name) =>
        [
          'systemMessage',
          'includeTokenUsageInAgentOutput',
          'enableUsageReporter',
          'usageReportingOptions',
        ].includes(name),
      ),
    [],
  );
});

test('model execution controls are available in the Settings tab', () => {
  const description = new UniversalChatModel().description;
  const settings = Object.fromEntries(
    description.properties
      .filter((property) => property.isNodeSetting)
      .map((property) => [property.name, property]),
  );

  assert.deepEqual(
    Object.keys(settings),
    [
      'alwaysOutputData',
      'executeOnce',
      'retryOnFail',
      'maxTries',
      'waitBetweenTries',
      'onError',
    ],
  );
  assert.equal(settings.alwaysOutputData.displayName, 'Always Output Data');
  assert.equal(settings.executeOnce.displayName, 'Execute Once');
  assert.equal(settings.retryOnFail.displayName, 'Retry On Fail');
  assert.equal(settings.maxTries.default, 3);
  assert.equal(settings.maxTries.typeOptions.minValue, 2);
  assert.equal(settings.maxTries.typeOptions.maxValue, 5);
  assert.deepEqual(settings.maxTries.displayOptions.show, {
    retryOnFail: [true],
  });
  assert.equal(settings.waitBetweenTries.default, 1000);
  assert.equal(settings.waitBetweenTries.typeOptions.minValue, 0);
  assert.equal(settings.waitBetweenTries.typeOptions.maxValue, 5000);
  assert.deepEqual(settings.waitBetweenTries.displayOptions.show, {
    retryOnFail: [true],
  });
  assert.equal(settings.onError.displayName, 'On Error');
  assert.equal(settings.onError.default, 'stopWorkflow');
  assert.deepEqual(
    settings.onError.options.map((option) => ({
      name: option.name,
      value: option.value,
    })),
    [
      { name: 'Stop Workflow', value: 'stopWorkflow' },
      { name: 'Continue', value: 'continueRegularOutput' },
      {
        name: 'Continue (Using Error Output)',
        value: 'continueErrorOutput',
      },
    ],
  );
});

test('n8n 2.2.5 uses its native Settings without duplicate model controls', () => {
  const fixture = join(
    __dirname,
    'fixtures',
    'n8n-core-2.2.3',
    'load-node.cjs',
  );
  const output = execFileSync(process.execPath, [fixture], {
    encoding: 'utf8',
  });

  assert.deepEqual(JSON.parse(output), []);
});

test('native node fields resolve all execution settings used by the runtime', () => {
  const parameters = {};
  const context = {
    getNodeParameter(_name, _itemIndex, fallback) {
      return fallback;
    },
    getNode() {
      return {
        parameters,
        alwaysOutputData: true,
        executeOnce: true,
        retryOnFail: true,
        maxTries: 5,
        waitBetweenTries: 250,
        onError: 'continueRegularOutput',
      };
    },
  };

  assert.deepEqual(getModelExecutionSettings(context, 0), {
    alwaysOutputData: true,
    executeOnce: true,
    retryOnFail: true,
    maxTries: 5,
    waitBetweenTries: 250,
    onError: 'continueRegularOutput',
  });
});

test('Retry On Fail retries Gemini calls using Max Tries', async () => {
  const originalFetch = global.fetch;
  let calls = 0;
  global.fetch = async () => {
    calls += 1;
    if (calls < 3) {
      return new Response(
        JSON.stringify({ error: { code: 503, message: 'temporarily unavailable' } }),
        {
          status: 503,
          headers: { 'content-type': 'application/json' },
        },
      );
    }

    return new Response(JSON.stringify(makeGeminiResponse()), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };

  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    geminiOptions: {},
    modelRetryOnFail: true,
    modelMaxTries: 3,
    modelWaitBetweenTries: 0,
  };
  const outputs = [];
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { name: 'Universal Chat Model', parameters };
    },
    addInputData() {
      return { index: 0 };
    },
    addOutputData(connectionType, index, data) {
      outputs.push({ connectionType, index, data });
    },
    getNextRunIndex() {
      return 0;
    },
    logAiEvent() {},
  };

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    const response = await supplied.response.invoke('teste de retry');

    assert.equal(calls, 3);
    assert.equal(response.text, 'Resposta final');
    assert.equal(outputs.length, 1);
  } finally {
    global.fetch = originalFetch;
  }
});

test('native n8n 2.2.5 retry settings drive model API retries', async () => {
  const originalFetch = global.fetch;
  let calls = 0;
  global.fetch = async () => {
    calls += 1;
    if (calls === 1) {
      return new Response(
        JSON.stringify({ error: { code: 503, message: 'temporarily unavailable' } }),
        {
          status: 503,
          headers: { 'content-type': 'application/json' },
        },
      );
    }

    return new Response(JSON.stringify(makeGeminiResponse()), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };

  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    geminiOptions: {},
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return {
        name: 'Universal Chat Model',
        parameters,
        retryOnFail: true,
        maxTries: 2,
        waitBetweenTries: 0,
      };
    },
    addInputData() {
      return { index: 0 };
    },
    addOutputData() {},
    getNextRunIndex() {
      return 0;
    },
    logAiEvent() {},
  };

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    const response = await supplied.response.invoke('retry nativo');

    assert.equal(calls, 2);
    assert.equal(response.text, 'Resposta final');
  } finally {
    global.fetch = originalFetch;
  }
});

test('Gemini error mapper classifies API, transport, tool, schema, and safety failures', () => {
  const cases = [
    [
      {
        name: 'RequestError',
        statusCode: 401,
        data: {
          error: {
            code: 401,
            status: 'UNAUTHENTICATED',
            message: 'API key not valid',
          },
        },
      },
      'authentication',
      false,
    ],
    [
      {
        name: 'RequestError',
        statusCode: 403,
        data: {
          error: {
            code: 403,
            status: 'PERMISSION_DENIED',
            message: 'Permission denied',
          },
        },
      },
      'permission',
      false,
    ],
    [
      {
        name: 'RequestError',
        statusCode: 404,
        data: {
          error: {
            code: 404,
            status: 'NOT_FOUND',
            message: 'Model gemini-missing was not found',
          },
        },
      },
      'model_not_found',
      false,
    ],
    [
      {
        name: 'RequestError',
        statusCode: 429,
        headers: { 'retry-after': '2' },
        data: {
          error: {
            code: 429,
            status: 'RESOURCE_EXHAUSTED',
            message: 'Project quota exceeded for requests per minute',
          },
        },
      },
      'quota',
      true,
    ],
    [
      {
        name: 'RequestError',
        statusCode: 503,
        data: {
          error: {
            code: 503,
            status: 'UNAVAILABLE',
            message: 'The service is temporarily unavailable',
          },
        },
      },
      'server',
      true,
    ],
    [
      {
        name: 'RequestError',
        statusCode: 504,
        data: {
          error: {
            code: 504,
            status: 'DEADLINE_EXCEEDED',
            message: 'Deadline exceeded',
          },
        },
      },
      'timeout',
      true,
    ],
    [
      {
        name: 'RequestError',
        statusCode: 400,
        data: {
          error: {
            code: 400,
            status: 'INVALID_ARGUMENT',
            message:
              'Function call search in content block 2 is missing a thought_signature',
          },
        },
      },
      'tool_protocol',
      false,
    ],
    [
      {
        name: 'InvalidToolError',
        message:
          'properties: should be non-empty for OBJECT type in function declaration',
      },
      'tool_schema',
      false,
    ],
    [
      {
        name: 'MalformedOutputError',
        message: 'Cannot parse structured output as JSON',
      },
      'structured_output',
      false,
    ],
    [
      {
        name: 'PromptBlockedError',
        message: 'Prompt was blocked: SAFETY',
      },
      'safety',
      false,
    ],
    [
      {
        name: 'TypeError',
        message: 'fetch failed',
        cause: {
          name: 'Error',
          code: 'ENOTFOUND',
          message: 'getaddrinfo ENOTFOUND invalid.example',
        },
      },
      'dns',
      true,
    ],
    [
      {
        name: 'TypeError',
        code: 'ERR_INVALID_URL',
        message: 'Invalid URL',
      },
      'invalid_url',
      false,
    ],
  ];

  for (const [error, category, retryable] of cases) {
    const details = normalizeModelError(error, 'gemini');
    assert.equal(details.category, category, JSON.stringify(error));
    assert.equal(details.retryable, retryable, JSON.stringify(error));
  }

  const quota = normalizeModelError(cases[3][0], 'gemini');
  assert.equal(quota.retryAfterMs, 2_000);
});

test('Gemini error mapper preserves safe API details and redacts credentials', () => {
  const details = normalizeModelError(
    {
      name: 'RequestError',
      statusCode: 400,
      url: 'https://generativelanguage.googleapis.com/v1beta/models/test?key=AIza012345678901234567890123456789',
      headers: {
        'x-goog-request-id': 'request-123',
        authorization: 'Bearer should-not-appear',
      },
      data: {
        error: {
          code: 400,
          status: 'INVALID_ARGUMENT',
          message:
            'Bad request using Bearer secret-token and key=AIza012345678901234567890123456789',
          details: [
            {
              reason: 'INVALID_SCHEMA',
              apiKey: 'secret-key',
            },
          ],
        },
      },
    },
    'gemini',
  );

  assert.equal(details.statusCode, 400);
  assert.equal(details.status, 'INVALID_ARGUMENT');
  assert.equal(details.requestId, 'request-123');
  assert.match(details.endpoint, /%5BREDACTED%5D|\[REDACTED\]/);
  assert.doesNotMatch(JSON.stringify(details), /secret-token|secret-key|AIza/);
  assert.equal(details.apiDetails[0].reason, 'INVALID_SCHEMA');
  assert.equal(details.apiDetails[0].apiKey, '[REDACTED]');
});

test('Retry On Fail does not retry a non-transient Gemini request error', async () => {
  const originalFetch = global.fetch;
  let calls = 0;
  global.fetch = async () => {
    calls += 1;
    return new Response(
      JSON.stringify({
        error: {
          code: 400,
          status: 'INVALID_ARGUMENT',
          message: 'Invalid generation config',
        },
      }),
      {
        status: 400,
        headers: {
          'content-type': 'application/json',
          'x-goog-request-id': 'request-no-retry',
        },
      },
    );
  };

  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    geminiOptions: {},
    modelRetryOnFail: true,
    modelMaxTries: 5,
    modelWaitBetweenTries: 0,
  };
  const outputs = [];
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { name: 'Universal Chat Model', parameters };
    },
    addInputData() {
      return { index: 0 };
    },
    addOutputData(connectionType, index, data) {
      outputs.push({ connectionType, index, data });
    },
    getNextRunIndex() {
      return 0;
    },
    logAiEvent() {},
  };

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    await assert.rejects(
      () => supplied.response.invoke('invalid request'),
      (error) => {
        assert.equal(error.normalizedModelError.category, 'invalid_request');
        assert.equal(error.normalizedModelError.statusCode, 400);
        assert.equal(error.normalizedModelError.status, 'INVALID_ARGUMENT');
        assert.equal(error.normalizedModelError.retryable, false);
        assert.equal(error.normalizedModelError.attempts, 1);
        return true;
      },
    );
    assert.equal(calls, 1);
    assert.equal(
      outputs[0].data.context.modelError.category,
      'invalid_request',
    );
    assert.equal(
      outputs[0].data.context.modelError.requestId,
      'request-no-retry',
    );
  } finally {
    global.fetch = originalFetch;
  }
});

test('Retry On Fail also retries OpenAI-compatible model calls', async () => {
  const originalFetch = global.fetch;
  let calls = 0;
  global.fetch = async () => {
    calls += 1;
    if (calls === 1) {
      return new Response(
        JSON.stringify({ error: { message: 'temporarily unavailable' } }),
        {
          status: 503,
          headers: { 'content-type': 'application/json' },
        },
      );
    }

    return new Response(
      JSON.stringify({
        id: 'chatcmpl-test',
        object: 'chat.completion',
        created: 1,
        model: 'test-model',
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: 'Resposta OpenAI' },
            finish_reason: 'stop',
            logprobs: null,
          },
        ],
        usage: {
          prompt_tokens: 4,
          completion_tokens: 2,
          total_tokens: 6,
        },
      }),
      {
        status: 200,
        headers: { 'content-type': 'application/json' },
      },
    );
  };

  const parameters = {
    provider: 'openai_compatible',
    openaiModel: 'test-model',
    openaiOptions: {},
    modelRetryOnFail: true,
    modelMaxTries: 2,
    modelWaitBetweenTries: 0,
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return {
        baseUrl: 'https://example.test/v1',
        apiKey: 'test',
      };
    },
    getNode() {
      return { name: 'Universal Chat Model', parameters };
    },
    addInputData() {
      return { index: 0 };
    },
    addOutputData() {},
    getNextRunIndex() {
      return 0;
    },
    logAiEvent() {},
  };

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    const response = await supplied.response.invoke('teste OpenAI');

    assert.equal(calls, 2);
    assert.equal(response.text, 'Resposta OpenAI');
  } finally {
    global.fetch = originalFetch;
  }
});

test('streaming does not retry when only Always Output Data is enabled', async () => {
  const originalFetch = global.fetch;
  let calls = 0;
  global.fetch = async () => {
    calls += 1;
    return new Response(
      JSON.stringify({ error: { code: 503, message: 'temporarily unavailable' } }),
      {
        status: 503,
        headers: { 'content-type': 'application/json' },
      },
    );
  };

  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    geminiOptions: {},
    modelAlwaysOutputData: true,
    modelRetryOnFail: false,
    modelMaxTries: 3,
    modelWaitBetweenTries: 0,
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { name: 'Universal Chat Model', parameters };
    },
    addInputData() {
      return { index: 0 };
    },
    addOutputData() {},
    getNextRunIndex() {
      return 0;
    },
    logAiEvent() {},
  };

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    await assert.rejects(async () => {
      const stream = await supplied.response.stream('falha sem retry');
      for await (const _chunk of stream) {
        // Drain the stream so transport errors surface.
      }
    });
    assert.equal(calls, 1);
  } finally {
    global.fetch = originalFetch;
  }
});

function executionSettings(overrides = {}) {
  return {
    alwaysOutputData: false,
    executeOnce: false,
    retryOnFail: false,
    maxTries: 1,
    waitBetweenTries: 0,
    onError: 'stopWorkflow',
    ...overrides,
  };
}

test('Always Output Data creates one blank generation when the model returns none', async () => {
  const model = {
    async _generate() {
      return { generations: [] };
    },
  };
  const wrapped = applyModelRetry(
    model,
    executionSettings({ alwaysOutputData: true }),
    'gemini',
  );

  const result = await wrapped._generate([], {});
  assert.equal(result.generations.length, 1);
  assert.equal(result.generations[0].text, '');
  assert.equal(result.generations[0].message.text, '');
});

test('Always Output Data does not hide a model failure', async () => {
  let calls = 0;
  const model = {
    async _generate() {
      calls += 1;
      throw {
        name: 'RequestError',
        statusCode: 400,
        data: {
          error: {
            code: 400,
            status: 'INVALID_ARGUMENT',
            message: 'Bad request',
          },
        },
      };
    },
  };
  const wrapped = applyModelRetry(
    model,
    executionSettings({ alwaysOutputData: true }),
    'gemini',
  );

  await assert.rejects(
    () => wrapped._generate([], {}),
    (error) => {
      assert.equal(error.normalizedModelError.category, 'invalid_request');
      return true;
    },
  );
  assert.equal(calls, 1);
});

test('On Error modes stop, continue regularly, or preserve the error path', async () => {
  const makeModel = () => ({
    async _generate() {
      throw {
        name: 'RequestError',
        statusCode: 503,
        data: {
          error: {
            code: 503,
            status: 'UNAVAILABLE',
            message: 'Provider unavailable',
          },
        },
      };
    },
  });

  const stopped = applyModelRetry(
    makeModel(),
    executionSettings({ onError: 'stopWorkflow' }),
    'gemini',
  );
  await assert.rejects(
    () => stopped._generate([], {}),
    (error) => {
      assert.equal(error.normalizedModelError.category, 'server');
      return true;
    },
  );

  const continued = applyModelRetry(
    makeModel(),
    executionSettings({ onError: 'continueRegularOutput' }),
    'gemini',
  );
  const regularResult = await continued._generate([], {});
  const regularMessage = regularResult.generations[0].message;
  assert.match(regularMessage.text, /HTTP 503.*UNAVAILABLE/i);
  assert.equal(
    regularMessage.response_metadata.universalChatModelError.category,
    'server',
  );
  assert.equal(
    regularMessage.response_metadata.universalChatModelError.retryable,
    true,
  );

  const errorOutput = applyModelRetry(
    makeModel(),
    executionSettings({ onError: 'continueErrorOutput' }),
    'gemini',
  );
  await assert.rejects(
    () => errorOutput._generate([], {}),
    (error) => {
      assert.equal(error.normalizedModelError.category, 'server');
      return true;
    },
  );
});

test('Retry On Fail retries transient transport errors but not configuration errors', async () => {
  let transientCalls = 0;
  const transientModel = {
    async _generate() {
      transientCalls += 1;
      if (transientCalls < 3) {
        const error = new TypeError('fetch failed');
        error.cause = Object.assign(new Error('connection reset'), {
          code: 'ECONNRESET',
        });
        throw error;
      }
      return {
        generations: [{ text: 'ok', message: new AIMessage('ok') }],
      };
    },
  };
  const retried = applyModelRetry(
    transientModel,
    executionSettings({
      retryOnFail: true,
      maxTries: 3,
    }),
    'gemini',
  );
  const successful = await retried._generate([], {});
  assert.equal(successful.generations[0].text, 'ok');
  assert.equal(transientCalls, 3);

  let configurationCalls = 0;
  const configurationModel = {
    async _generate() {
      configurationCalls += 1;
      throw {
        name: 'ConfigurationError',
        message: 'Invalid model configuration',
      };
    },
  };
  const notRetried = applyModelRetry(
    configurationModel,
    executionSettings({
      retryOnFail: true,
      maxTries: 5,
    }),
    'gemini',
  );
  await assert.rejects(() => notRetried._generate([], {}));
  assert.equal(configurationCalls, 1);
});

test('Abort errors are never retried or converted into regular output', async () => {
  let calls = 0;
  const model = {
    async _generate() {
      calls += 1;
      const error = new Error('The operation was aborted');
      error.name = 'AbortError';
      throw error;
    },
  };
  const wrapped = applyModelRetry(
    model,
    executionSettings({
      retryOnFail: true,
      maxTries: 5,
      onError: 'continueRegularOutput',
    }),
    'gemini',
  );

  await assert.rejects(
    () => wrapped._generate([], {}),
    (error) => {
      assert.equal(error.name, 'AbortError');
      return true;
    },
  );
  assert.equal(calls, 1);
});

test('streaming retries before the first chunk and never duplicates a partial response', async () => {
  let attemptsBeforeChunk = 0;
  const retryableStreamModel = {
    async _generate() {
      return { generations: [] };
    },
    async *_streamResponseChunks() {
      attemptsBeforeChunk += 1;
      if (attemptsBeforeChunk === 1) {
        throw {
          name: 'RequestError',
          statusCode: 503,
          message: 'Unavailable',
        };
      }
      yield new ChatGenerationChunk({
        text: 'ok',
        message: new AIMessageChunk('ok'),
      });
    },
  };
  const retriedStream = applyModelRetry(
    retryableStreamModel,
    executionSettings({ retryOnFail: true, maxTries: 2 }),
    'gemini',
  );
  const retryChunks = [];
  for await (const chunk of retriedStream._streamResponseChunks([], {})) {
    retryChunks.push(chunk.text);
  }
  assert.deepEqual(retryChunks, ['ok']);
  assert.equal(attemptsBeforeChunk, 2);

  let partialAttempts = 0;
  const partialStreamModel = {
    async _generate() {
      return { generations: [] };
    },
    async *_streamResponseChunks() {
      partialAttempts += 1;
      yield new ChatGenerationChunk({
        text: 'partial',
        message: new AIMessageChunk('partial'),
      });
      throw {
        name: 'RequestError',
        statusCode: 503,
        message: 'Unavailable after partial output',
      };
    },
  };
  const partialStream = applyModelRetry(
    partialStreamModel,
    executionSettings({ retryOnFail: true, maxTries: 3 }),
    'gemini',
  );
  const partialChunks = [];
  await assert.rejects(async () => {
    for await (const chunk of partialStream._streamResponseChunks([], {})) {
      partialChunks.push(chunk.text);
    }
  });
  assert.deepEqual(partialChunks, ['partial']);
  assert.equal(partialAttempts, 1);
});

test('On Error Continue returns a model error message for custom and native n8n settings', async () => {
  const originalFetch = global.fetch;
  let calls = 0;
  global.fetch = async () => {
    calls += 1;
    return new Response(
      JSON.stringify({ error: { code: 503, message: 'temporarily unavailable' } }),
      {
        status: 503,
        headers: { 'content-type': 'application/json' },
      },
    );
  };

  const makeContext = (useNativeSetting) => {
    const parameters = {
      provider: 'gemini',
      geminiModel: 'gemini-2.5-flash',
      geminiOptions: {},
      modelOnError: useNativeSetting
        ? 'stopWorkflow'
        : 'continueRegularOutput',
    };
    return {
      getNodeParameter(name, _itemIndex, fallback) {
        return parameters[name] ?? fallback;
      },
      async getCredentials() {
        return { apiKey: 'test' };
      },
      getNode() {
        return {
          name: 'Universal Chat Model',
          parameters,
          ...(useNativeSetting
            ? { onError: 'continueRegularOutput' }
            : {}),
        };
      },
      addInputData() {
        return { index: 0 };
      },
      addOutputData() {},
      getNextRunIndex() {
        return 0;
      },
      logAiEvent() {},
    };
  };

  try {
    const custom = await new UniversalChatModel().supplyData.call(
      makeContext(false),
      0,
    );
    const customResponse = await custom.response.invoke('continue custom');
    assert.match(customResponse.text, /temporarily unavailable/i);
    assert.equal(
      customResponse.response_metadata.onError,
      'continueRegularOutput',
    );
    assert.deepEqual(
      {
        category:
          customResponse.response_metadata.universalChatModelError.category,
        statusCode:
          customResponse.response_metadata.universalChatModelError.statusCode,
        status: customResponse.response_metadata.universalChatModelError.status,
        retryable:
          customResponse.response_metadata.universalChatModelError.retryable,
        attempts:
          customResponse.response_metadata.universalChatModelError.attempts,
      },
      {
        category: 'server',
        statusCode: 503,
        status: undefined,
        retryable: true,
        attempts: 1,
      },
    );

    const native = await new UniversalChatModel().supplyData.call(
      makeContext(true),
      0,
    );
    const nativeResponse = await native.response.invoke('continue nativo');
    assert.match(nativeResponse.text, /temporarily unavailable/i);
    assert.equal(
      nativeResponse.response_metadata.onError,
      'continueRegularOutput',
    );
    assert.equal(calls, 2);
  } finally {
    global.fetch = originalFetch;
  }
});

test('Execute Once resolves provider parameters from the first input item', async () => {
  const requestedIndexes = [];
  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    geminiOptions: {},
  };
  const context = {
    getNodeParameter(name, itemIndex, fallback) {
      requestedIndexes.push({ name, itemIndex });
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { parameters, executeOnce: true };
    },
    logAiEvent() {},
  };

  await new UniversalChatModel().supplyData.call(context, 4);

  assert.equal(
    requestedIndexes
      .filter(({ name }) => ['provider', 'geminiModel', 'geminiOptions'].includes(name))
      .every(({ itemIndex }) => itemIndex === 0),
    true,
  );
});

test('legacy top-level model execution settings remain readable', async () => {
  const requestedIndexes = [];
  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    geminiOptions: {},
  };
  const context = {
    getNodeParameter(name, itemIndex, fallback) {
      requestedIndexes.push({ name, itemIndex });
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return {
        parameters,
        modelExecuteOnce: true,
        modelRetryOnFail: true,
        modelMaxTries: 4,
        modelWaitBetweenTries: 25,
      };
    },
    logAiEvent() {},
  };

  await new UniversalChatModel().supplyData.call(context, 3);

  assert.equal(
    requestedIndexes
      .filter(({ name }) =>
        ['provider', 'geminiModel', 'geminiOptions'].includes(name),
      )
      .every(({ itemIndex }) => itemIndex === 0),
    true,
  );
});

test('Existing workflows keep their legacy Gemini parameter values', async () => {
  const node = new UniversalChatModel();
  const legacyParameters = {
    provider: 'gemini',
    geminiTemperature: 0.4,
    geminiTopP: 0.7,
    geminiTopK: 12,
    geminiMaxTokens: 2048,
    geminiResponseMimeType: 'application/json',
    thinkingMode: 'budget',
    thinkingBudget: 512,
    includeThoughts: true,
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      if (name === 'provider') return 'gemini';
      if (name === 'geminiModel') return 'gemini-2.5-flash';
      if (name === 'geminiOptions') return {};
      return fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { parameters: legacyParameters };
    },
    logAiEvent() {},
  };

  const supplied = await node.supplyData.call(context, 0);
  const config = supplied.response.invocationParams({}).generationConfig;

  assert.equal(config.temperature, 0.4);
  assert.equal(config.topP, 0.7);
  assert.equal(config.topK, 12);
  assert.equal(config.maxOutputTokens, 2048);
  assert.equal(config.responseMimeType, 'application/json');
  assert.deepEqual(config.thinkingConfig, {
    thinkingBudget: 512,
    includeThoughts: true,
  });
});

test('Gemini request omits optional generation controls unless configured', () => {
  const defaultModel = new GeminiChatModel({
    apiKey: 'test',
    model: 'gemini-2.5-flash',
  });
  const defaultConfig = defaultModel.invocationParams({}).generationConfig;

  assert.equal(defaultConfig.temperature, undefined);
  assert.equal(defaultConfig.topP, undefined);
  assert.equal(defaultConfig.topK, undefined);
  assert.equal(defaultConfig.maxOutputTokens, undefined);
  assert.equal(defaultConfig.responseMimeType, undefined);
  assert.equal(defaultConfig.responseJsonSchema, undefined);
  assert.equal(defaultConfig.thinkingConfig, undefined);

  const configuredModel = new GeminiChatModel({
    apiKey: 'test',
    model: 'gemini-3-flash-preview',
    temperature: 0,
    topP: 0.8,
    topK: 20,
    maxOutputTokens: 1024,
    responseMimeType: 'application/json',
    thinkingConfig: { thinkingLevel: 'LOW', includeThoughts: true },
  });
  const configured = configuredModel.invocationParams({}).generationConfig;

  assert.equal(configured.temperature, 0);
  assert.equal(configured.topP, 0.8);
  assert.equal(configured.topK, 20);
  assert.equal(configured.maxOutputTokens, 1024);
  assert.equal(configured.responseMimeType, 'application/json');
  assert.deepEqual(configured.thinkingConfig, {
    thinkingLevel: 'LOW',
    includeThoughts: true,
  });
});

test('node System Message is appended after the parent system prompt', async () => {
  const originalFetch = global.fetch;
  let requestBody;
  global.fetch = async (request) => {
    requestBody = JSON.parse(await request.clone().text());
    return new Response(JSON.stringify(makeGeminiResponse()), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };

  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    geminiOptions: {
      systemMessage: 'Instrução adicional do Chat Model',
    },
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { name: 'Universal Chat Model', parameters };
    },
    addInputData() {
      return { index: 0 };
    },
    addOutputData() {},
    getNextRunIndex() {
      return 0;
    },
    logAiEvent() {},
  };

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    await supplied.response.invoke([
      new SystemMessage('System prompt do AI Agent'),
      new HumanMessage('Olá'),
    ]);

    assert.deepEqual(
      requestBody.systemInstruction.parts.map((part) => part.text),
      [
        'System prompt do AI Agent',
        'Instrução adicional do Chat Model',
      ],
    );
    assert.equal(requestBody.contents[0].parts[0].text, 'Olá');
  } finally {
    global.fetch = originalFetch;
  }
});

test('Gemini only requests thought summaries when Include Thoughts is enabled', async () => {
  const makeContext = (geminiOptions) => {
    const parameters = {
      provider: 'gemini',
      geminiModel: 'gemini-3.5-flash-lite',
      geminiOptions,
    };
    return {
      getNodeParameter(name, _itemIndex, fallback) {
        return parameters[name] ?? fallback;
      },
      async getCredentials() {
        return { apiKey: 'test' };
      },
      getNode() {
        return { parameters };
      },
      logAiEvent() {},
    };
  };

  const hiddenByDefault = await new UniversalChatModel().supplyData.call(
    makeContext({ thinkingLevel: 'MEDIUM' }),
    0,
  );
  assert.deepEqual(hiddenByDefault.response.invocationParams({}).generationConfig.thinkingConfig, {
    thinkingLevel: 'MEDIUM',
  });

  const disabled = await new UniversalChatModel().supplyData.call(
    makeContext({ thinkingLevel: 'MEDIUM', includeThoughts: false }),
    0,
  );
  assert.deepEqual(disabled.response.invocationParams({}).generationConfig.thinkingConfig, {
    thinkingLevel: 'MEDIUM',
  });

  const enabled = await new UniversalChatModel().supplyData.call(
    makeContext({ thinkingLevel: 'MEDIUM', includeThoughts: true }),
    0,
  );
  assert.deepEqual(enabled.response.invocationParams({}).generationConfig.thinkingConfig, {
    thinkingLevel: 'MEDIUM',
    includeThoughts: true,
  });
});

test('Gemini Structured Output Schema is parsed and forces JSON responses', async () => {
  const schema = {
    type: 'object',
    properties: {
      answer: { type: 'string' },
      confidence: { type: 'number' },
    },
    required: ['answer'],
  };
  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    geminiOptions: {
      responseMimeType: 'text/plain',
      responseSchema: JSON.stringify(schema),
    },
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { parameters };
    },
    logAiEvent() {},
  };

  const supplied = await new UniversalChatModel().supplyData.call(context, 0);
  const config = supplied.response.invocationParams({}).generationConfig;

  assert.equal(config.responseMimeType, 'application/json');
  assert.deepEqual(config.responseJsonSchema, schema);
});

test('Gemini Structured Output Schema rejects invalid JSON before making a request', async () => {
  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    geminiOptions: {
      responseSchema: '{"type":"object"',
    },
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { parameters };
    },
    logAiEvent() {},
  };

  await assert.rejects(
    () => new UniversalChatModel().supplyData.call(context, 0),
    /Invalid JSON in Structured Output Schema/,
  );
});

test('structured output parser accepts objects, JSON strings, and fenced JSON', () => {
  const expected = {
    output: 'Olá',
    abertura_de_chamado: false,
    cpf: '',
    descricao: '',
  };

  assert.deepEqual(parseStructuredOutput(expected), expected);
  assert.deepEqual(
    parseStructuredOutput(JSON.stringify(expected)),
    expected,
  );
  assert.deepEqual(
    parseStructuredOutput(`\`\`\`json\n${JSON.stringify(expected)}\n\`\`\``),
    expected,
  );
  assert.deepEqual(
    parseStructuredOutput(JSON.stringify(JSON.stringify(expected))),
    expected,
  );
  assert.equal(parseStructuredOutput('resposta normal'), undefined);
  assert.equal(parseStructuredOutput('[1,2,3]'), undefined);
});

test('Gemini response preserves thoughts and complete raw usage metadata', async () => {
  const originalFetch = global.fetch;
  global.fetch = async () =>
    new Response(JSON.stringify(makeGeminiResponse()), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  try {
    const model = new GeminiChatModel({
      apiKey: 'test',
      model: 'gemini-2.5-flash',
      thinkingConfig: { thinkingBudget: 256, includeThoughts: true },
    });
    const message = await model.invoke('teste');

    assert.equal(Array.isArray(message.content), true);
    assert.equal(message.text, 'Resposta final');
    assert.equal(message.content.length, 1);
    assert.equal(message.content[0].text, 'Resposta final');
    assert.equal(message.content[0].thought, undefined);
    assert.equal(message.response_metadata.thoughts[0].thought, true);
    assert.equal(message.response_metadata.thoughts[0].text, 'Resumo do raciocínio');
    assert.deepEqual(message.response_metadata.gemini.usageMetadata, usageMetadata);
    assert.deepEqual(message.additional_kwargs.gemini.usageMetadata, usageMetadata);
    assert.equal(message.additional_kwargs.thoughts, undefined);
    assert.equal(message.usage_metadata.input_tokens, 100);
    assert.equal(message.usage_metadata.output_tokens, 33);
    assert.equal(message.usage_metadata.input_token_details.cache_read, 60);
    assert.equal(message.usage_metadata.input_token_details.tool_use, 7);
    assert.equal(message.usage_metadata.output_token_details.text, 20);
    assert.equal(message.usage_metadata.output_token_details.reasoning, 13);
  } finally {
    global.fetch = originalFetch;
  }
});

test('Gemini hides thought summaries unless Include Thoughts is enabled', async () => {
  const originalFetch = global.fetch;
  global.fetch = async () =>
    new Response(JSON.stringify(makeGeminiResponse()), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  try {
    const model = new GeminiChatModel({
      apiKey: 'test',
      model: 'gemini-2.5-flash',
      thinkingConfig: { thinkingBudget: 256 },
    });
    const message = await model.invoke('teste');

    assert.equal(message.text, 'Resposta final');
    assert.equal(message.content.length, 1);
    assert.equal(message.response_metadata.thoughts, undefined);
    assert.equal(message.response_metadata.gemini.thoughts, undefined);
    assert.equal(message.additional_kwargs.thoughts, undefined);
    assert.equal(message.additional_kwargs.gemini.thoughts, undefined);
    assert.equal(message.usage_metadata.output_token_details.reasoning, 13);
  } finally {
    global.fetch = originalFetch;
  }
});

test('concurrent Gemini requests keep raw thoughts and usage metadata isolated', async () => {
  const originalFetch = global.fetch;
  global.fetch = async (request) => {
    const body = JSON.parse(await request.clone().text());
    const prompt = body.contents[0].parts[0].text;
    const isA = prompt === 'request-A';
    await new Promise((resolve) => setTimeout(resolve, isA ? 20 : 5));
    return new Response(
      JSON.stringify({
        candidates: [
          {
            content: {
              role: 'model',
              parts: [
                { text: `Thought ${isA ? 'A' : 'B'}`, thought: true },
                { text: `Response ${isA ? 'A' : 'B'}` },
              ],
            },
            finishReason: 'STOP',
            index: 0,
          },
        ],
        usageMetadata: {
          promptTokenCount: isA ? 10 : 100,
          candidatesTokenCount: isA ? 20 : 200,
          thoughtsTokenCount: isA ? 5 : 50,
          totalTokenCount: isA ? 35 : 350,
        },
        modelVersion: `model-${isA ? 'A' : 'B'}`,
        responseId: `response-${isA ? 'A' : 'B'}`,
      }),
      {
        status: 200,
        headers: { 'content-type': 'application/json' },
      },
    );
  };

  try {
    const model = new GeminiChatModel({
      apiKey: 'test',
      model: 'gemini-3.5-flash-lite',
      thinkingConfig: {
        thinkingLevel: 'MEDIUM',
        includeThoughts: true,
      },
    });
    const [responseA, responseB] = await Promise.all([
      model.invoke('request-A'),
      model.invoke('request-B'),
    ]);

    assert.equal(responseA.text, 'Response A');
    assert.equal(responseA.response_metadata.thoughts[0].text, 'Thought A');
    assert.equal(responseA.response_metadata.gemini.modelVersion, 'model-A');
    assert.equal(responseA.usage_metadata.input_tokens, 10);
    assert.equal(responseA.usage_metadata.output_token_details.reasoning, 5);

    assert.equal(responseB.text, 'Response B');
    assert.equal(responseB.response_metadata.thoughts[0].text, 'Thought B');
    assert.equal(responseB.response_metadata.gemini.modelVersion, 'model-B');
    assert.equal(responseB.usage_metadata.input_tokens, 100);
    assert.equal(responseB.usage_metadata.output_token_details.reasoning, 50);
  } finally {
    global.fetch = originalFetch;
  }
});

test('Gemini model interoperates with LangChain 1.1 messages from n8n 2.2.5', async () => {
  const originalFetch = global.fetch;
  global.fetch = async () =>
    new Response(JSON.stringify(makeGeminiResponse()), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  try {
    const model = new GeminiChatModel({
      apiKey: 'test',
      model: 'gemini-2.5-flash',
      thinkingConfig: { thinkingBudget: 256, includeThoughts: true },
    });
    const n8n225Message = new N8n225HumanMessage('mensagem criada pelo LangChain do n8n 2.2.5');
    const response = await model.invoke([n8n225Message]);

    assert.equal(isN8n225AIMessage(response), true);
    assert.equal(response.text, 'Resposta final');
    assert.equal(response.content[0].thought, undefined);
    assert.equal(response.response_metadata.thoughts[0].thought, true);
    assert.deepEqual(response.response_metadata.gemini.usageMetadata, usageMetadata);
  } finally {
    global.fetch = originalFetch;
  }
});

test('Gemini binds structured tools created by the LangChain version in n8n 2.2.5', async () => {
  const originalFetch = global.fetch;
  let requestBody;
  global.fetch = async (request) => {
    requestBody = JSON.parse(await request.clone().text());
    return new Response(
      JSON.stringify({
        candidates: [
          {
            content: {
              role: 'model',
              parts: [
                {
                  functionCall: {
                    id: 'call-n8n-tool',
                    name: 'lookup_order',
                    args: { orderId: '123' },
                  },
                  thoughtSignature: 'signature-n8n-tool',
                },
              ],
            },
            finishReason: 'STOP',
            index: 0,
          },
        ],
        usageMetadata,
        modelVersion: 'gemini-3.5-flash-lite',
        responseId: 'response-n8n-tool',
      }),
      {
        status: 200,
        headers: { 'content-type': 'application/json' },
      },
    );
  };

  try {
    const tool = new N8n225DynamicStructuredTool({
      name: 'lookup_order',
      description: 'Look up an order by ID',
      schema: z.object({
        orderId: z.string().describe('Order ID'),
      }),
      func: async ({ orderId }) => ({ orderId, status: 'ok' }),
    });
    const model = new GeminiChatModel({
      apiKey: 'test',
      model: 'gemini-3.5-flash-lite',
      responseSchema: {
        type: 'object',
        properties: {
          message: { type: 'string' },
        },
        required: ['message'],
      },
    }).bindTools([tool]);
    const response = await model.invoke([
      new N8n225HumanMessage('Consulte o pedido 123'),
    ]);

    assert.equal(response.tool_calls[0].name, 'lookup_order');
    const declaration =
      requestBody.tools[0].functionDeclarations.find(
        (candidate) => candidate.name === 'lookup_order',
      );
    assert.equal(declaration.description, 'Look up an order by ID');
    assert.equal(declaration.parameters.type, 'object');
    assert.equal(declaration.parameters.properties.orderId.type, 'string');
    assert.deepEqual(declaration.parameters.required, ['orderId']);
    assert.equal(
      requestBody.generationConfig.responseMimeType,
      'application/json',
    );
    assert.deepEqual(requestBody.generationConfig.responseJsonSchema, {
      type: 'object',
      properties: {
        message: { type: 'string' },
      },
      required: ['message'],
    });
  } finally {
    global.fetch = originalFetch;
  }
});

test('Gemini agent tool loop preserves call IDs and thought signatures across sequential calls', async () => {
  const originalFetch = global.fetch;
  const requestBodies = [];
  const responses = [
    {
      candidates: [
        {
          content: {
            role: 'model',
            parts: [
              {
                functionCall: {
                  id: 'call-search-1',
                  name: 'search_catalog',
                  args: { query: 'turbina' },
                },
                thoughtSignature: 'signature-search-1',
              },
            ],
          },
          finishReason: 'STOP',
          index: 0,
        },
      ],
      usageMetadata,
      modelVersion: 'gemini-3.5-flash-lite',
      responseId: 'response-tool-1',
    },
    {
      candidates: [
        {
          content: {
            role: 'model',
            parts: [
              {
                functionCall: {
                  id: 'call-stock-2',
                  name: 'check_stock',
                  args: { sku: 'TURB-001' },
                },
                thoughtSignature: 'signature-stock-2',
              },
            ],
          },
          finishReason: 'STOP',
          index: 0,
        },
      ],
      usageMetadata,
      modelVersion: 'gemini-3.5-flash-lite',
      responseId: 'response-tool-2',
    },
    {
      candidates: [
        {
          content: {
            role: 'model',
            parts: [{ text: 'Produto localizado e disponível.' }],
          },
          finishReason: 'STOP',
          index: 0,
        },
      ],
      usageMetadata,
      modelVersion: 'gemini-3.5-flash-lite',
      responseId: 'response-final',
    },
  ];

  global.fetch = async (request) => {
    requestBodies.push(JSON.parse(await request.clone().text()));
    const response = responses[requestBodies.length - 1];
    return new Response(JSON.stringify(response), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };

  try {
    const model = new GeminiChatModel({
      apiKey: 'test',
      model: 'gemini-3.5-flash-lite',
      thinkingConfig: {
        thinkingLevel: 'MEDIUM',
        includeThoughts: true,
      },
    });
    const agentModel = model.bindTools([
      {
        functionDeclarations: [
          {
            name: 'search_catalog',
            description: 'Search the product catalog',
            parameters: {
              type: 'object',
              properties: { query: { type: 'string' } },
              required: ['query'],
            },
          },
          {
            name: 'check_stock',
            description: 'Check product stock',
            parameters: {
              type: 'object',
              properties: { sku: { type: 'string' } },
              required: ['sku'],
            },
          },
        ],
      },
    ]);
    const human = new N8n225HumanMessage('Encontre uma turbina em estoque');

    const first = await agentModel.invoke([human]);
    assert.equal(first.tool_calls[0].id, 'call-search-1');
    assert.equal(first.tool_calls[0].name, 'search_catalog');
    assert.equal(first.tool_calls[0].thoughtSignature, 'signature-search-1');

    const searchResult = new N8n225ToolMessage({
      content: JSON.stringify({ sku: 'TURB-001' }),
      tool_call_id: 'call-search-1',
      name: 'search_catalog',
    });
    const second = await agentModel.invoke([human, first, searchResult]);
    assert.equal(second.tool_calls[0].id, 'call-stock-2');
    assert.equal(second.tool_calls[0].name, 'check_stock');
    assert.equal(second.tool_calls[0].thoughtSignature, 'signature-stock-2');

    const stockResult = new N8n225ToolMessage({
      content: JSON.stringify({ available: true }),
      tool_call_id: 'call-stock-2',
      name: 'check_stock',
    });
    const final = await agentModel.invoke([
      human,
      first,
      searchResult,
      second,
      stockResult,
    ]);

    assert.equal(final.text, 'Produto localizado e disponível.');
    assert.equal(requestBodies.length, 3);

    const secondRequest = requestBodies[1];
    const firstFunctionCall = secondRequest.contents
      .find((content) => content.role === 'model')
      .parts.find((part) => part.functionCall?.id === 'call-search-1');
    assert.equal(firstFunctionCall.thoughtSignature, 'signature-search-1');
    const firstFunctionResponse = secondRequest.contents
      .find((content) =>
        content.parts.some(
          (part) => part.functionResponse?.id === 'call-search-1',
        ),
      )
      .parts.find((part) => part.functionResponse);
    assert.equal(firstFunctionResponse.functionResponse.name, 'search_catalog');

    const finalRequest = requestBodies[2];
    const modelCalls = finalRequest.contents
      .filter((content) => content.role === 'model')
      .flatMap((content) => content.parts)
      .filter((part) => part.functionCall);
    assert.deepEqual(
      modelCalls.map((part) => ({
        id: part.functionCall.id,
        signature: part.thoughtSignature,
      })),
      [
        { id: 'call-search-1', signature: 'signature-search-1' },
        { id: 'call-stock-2', signature: 'signature-stock-2' },
      ],
    );
    const functionResponses = finalRequest.contents
      .filter((content) => content.role === 'user')
      .flatMap((content) => content.parts)
      .filter((part) => part.functionResponse);
    assert.deepEqual(
      functionResponses.map((part) => ({
        id: part.functionResponse.id,
        name: part.functionResponse.name,
      })),
      [
        { id: 'call-search-1', name: 'search_catalog' },
        { id: 'call-stock-2', name: 'check_stock' },
      ],
    );
  } finally {
    global.fetch = originalFetch;
  }
});

test('Gemini agent tool loop preserves parallel calls and matches every tool response', async () => {
  const originalFetch = global.fetch;
  const requestBodies = [];
  const responses = [
    {
      candidates: [
        {
          content: {
            role: 'model',
            parts: [
              {
                functionCall: {
                  id: 'call-price',
                  name: 'get_price',
                  args: { sku: 'TURB-001' },
                },
                thoughtSignature: 'signature-parallel',
              },
              {
                functionCall: {
                  id: 'call-delivery',
                  name: 'get_delivery',
                  args: { sku: 'TURB-001' },
                },
              },
            ],
          },
          finishReason: 'STOP',
          index: 0,
        },
      ],
      usageMetadata,
      modelVersion: 'gemini-3.5-flash-lite',
      responseId: 'response-parallel',
    },
    {
      candidates: [
        {
          content: {
            role: 'model',
            parts: [{ text: 'Preço e prazo consultados.' }],
          },
          finishReason: 'STOP',
          index: 0,
        },
      ],
      usageMetadata,
      modelVersion: 'gemini-3.5-flash-lite',
      responseId: 'response-parallel-final',
    },
  ];

  global.fetch = async (request) => {
    requestBodies.push(JSON.parse(await request.clone().text()));
    return new Response(JSON.stringify(responses[requestBodies.length - 1]), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };

  try {
    const model = new GeminiChatModel({
      apiKey: 'test',
      model: 'gemini-3.5-flash-lite',
    }).bindTools([
      {
        functionDeclarations: [
          {
            name: 'get_price',
            description: 'Get a product price',
            parameters: {
              type: 'object',
              properties: { sku: { type: 'string' } },
              required: ['sku'],
            },
          },
          {
            name: 'get_delivery',
            description: 'Get a delivery estimate',
            parameters: {
              type: 'object',
              properties: { sku: { type: 'string' } },
              required: ['sku'],
            },
          },
        ],
      },
    ]);
    const human = new N8n225HumanMessage('Consulte preço e entrega');
    const calls = await model.invoke([human]);

    assert.deepEqual(
      calls.tool_calls.map((call) => ({
        id: call.id,
        name: call.name,
        signature: call.thoughtSignature,
      })),
      [
        {
          id: 'call-price',
          name: 'get_price',
          signature: 'signature-parallel',
        },
        {
          id: 'call-delivery',
          name: 'get_delivery',
          signature: undefined,
        },
      ],
    );

    const price = new N8n225ToolMessage({
      content: JSON.stringify({ price: 1000 }),
      tool_call_id: 'call-price',
      name: 'get_price',
    });
    const delivery = new N8n225ToolMessage({
      content: JSON.stringify({ days: 3 }),
      tool_call_id: 'call-delivery',
      name: 'get_delivery',
    });
    const final = await model.invoke([human, calls, price, delivery]);

    assert.equal(final.text, 'Preço e prazo consultados.');
    const secondRequest = requestBodies[1];
    const modelParts = secondRequest.contents
      .filter((content) => content.role === 'model')
      .flatMap((content) => content.parts);
    assert.equal(
      modelParts.find((part) => part.functionCall?.id === 'call-price')
        .thoughtSignature,
      'signature-parallel',
    );
    assert.deepEqual(
      secondRequest.contents
        .filter((content) => content.role === 'user')
        .flatMap((content) => content.parts)
        .filter((part) => part.functionResponse)
        .map((part) => ({
          id: part.functionResponse.id,
          name: part.functionResponse.name,
        })),
      [
        { id: 'call-price', name: 'get_price' },
        { id: 'call-delivery', name: 'get_delivery' },
      ],
    );
  } finally {
    global.fetch = originalFetch;
  }
});

test('Gemini streaming interoperates with n8n 2.2.5 and keeps final usage metadata', async () => {
  const originalFetch = global.fetch;
  const event = `data: ${JSON.stringify(makeGeminiResponse())}\n\n`;
  global.fetch = async () =>
    new Response(event, {
      status: 200,
      headers: { 'content-type': 'text/event-stream' },
    });

  try {
    const model = new GeminiChatModel({
      apiKey: 'test',
      model: 'gemini-2.5-flash',
      thinkingConfig: { thinkingBudget: 256, includeThoughts: true },
    });
    const chunks = [];
    const stream = await model.stream([
      new N8n225HumanMessage('stream criado pelo LangChain do n8n 2.2.5'),
    ]);
    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    const chunkWithUsage = chunks.find((chunk) => chunk.response_metadata?.gemini);
    assert.ok(chunkWithUsage);
    assert.equal(isN8n225AIMessage(chunkWithUsage), true);
    assert.equal(
      chunks.some((chunk) =>
        Array.isArray(chunk.content)
          ? chunk.content.some((block) => block?.thought === true)
          : false,
      ),
      false,
    );
    assert.equal(chunkWithUsage.response_metadata.thoughts[0].thought, true);
    assert.deepEqual(chunkWithUsage.response_metadata.gemini.usageMetadata, usageMetadata);
    assert.equal(chunkWithUsage.usage_metadata.input_token_details.tool_use, 7);
    assert.equal(chunkWithUsage.usage_metadata.output_token_details.reasoning, 13);
  } finally {
    global.fetch = originalFetch;
  }
});

test('Gemini token log includes every API token category', () => {
  assert.equal(
    formatGeminiUsage(usageMetadata),
    'Token Usage — Input: 100 | Cache: 60 | Output: 20 | Tool: 7 | Thoughts: 13 | Total: 140',
  );
});

test('n8n tracing marks the chat-model subnode as executed and exposes thoughts and usage', async () => {
  const originalFetch = global.fetch;
  global.fetch = async () =>
    new Response(JSON.stringify(makeGeminiResponse()), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  const inputs = [];
  const outputs = [];
  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    geminiOptions: {
      thinkingBudget: 256,
      includeThoughts: true,
    },
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { name: 'Universal Chat Model', parameters };
    },
    addInputData(connectionType, data) {
      inputs.push({ connectionType, data });
      return { index: inputs.length - 1 };
    },
    addOutputData(connectionType, index, data) {
      outputs.push({ connectionType, index, data });
    },
    getNextRunIndex() {
      return inputs.length;
    },
    logAiEvent() {},
  };

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    await supplied.response.invoke('teste de tracing');

    assert.equal(inputs.length, 1);
    assert.equal(outputs.length, 1);
    assert.equal(outputs[0].index, 0);

    const trace = outputs[0].data[0][0].json;
    assert.equal(trace.thoughts.length, 1);
    assert.equal(trace.thoughts[0].text, 'Resumo do raciocínio');
    assert.equal(trace.thoughts[0].thought, true);
    assert.equal(trace.tokenUsage.inputTokens, 100);
    assert.equal(trace.tokenUsage.inputUncachedTokens, 40);
    assert.equal(trace.tokenUsage.outputTokens, 20);
    assert.equal(trace.tokenUsage.cachedTokens, 60);
    assert.equal(trace.tokenUsage.toolUsePromptTokens, 7);
    assert.equal(trace.tokenUsage.thoughtsTokens, 13);
    assert.equal(trace.tokenUsage.totalTokens, 140);
    assert.deepEqual(trace.usageMetadata, usageMetadata);
    assert.equal(trace.gemini.modelVersion, 'gemini-test-001');
    assert.equal(trace.gemini.usageMetadata, undefined);
    assert.equal(trace.response.generations[0][0].text, 'Resposta final');
    assert.equal(
      JSON.stringify(trace).split('Resumo do raciocínio').length - 1,
      1,
    );
  } finally {
    global.fetch = originalFetch;
  }
});

test('n8n tracing hides thoughts unless Include Thoughts is enabled', async () => {
  const originalFetch = global.fetch;
  global.fetch = async () =>
    new Response(JSON.stringify(makeGeminiResponse()), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  const outputs = [];
  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    geminiOptions: {
      thinkingBudget: 256,
    },
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { name: 'Universal Chat Model', parameters };
    },
    addInputData() {
      return { index: 0 };
    },
    addOutputData(connectionType, index, data) {
      outputs.push({ connectionType, index, data });
    },
    getNextRunIndex() {
      return 0;
    },
    logAiEvent() {},
  };

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    await supplied.response.invoke('teste sem thoughts');

    const trace = outputs[0].data[0][0].json;
    assert.equal(trace.thoughts, undefined);
    assert.equal(trace.response.generations[0][0].text, 'Resposta final');
    assert.equal(trace.tokenUsage.thoughtsTokens, 13);
    assert.equal(JSON.stringify(trace).includes('Resumo do raciocÃ­nio'), false);
  } finally {
    global.fetch = originalFetch;
  }
});

test('Usage Reporter receives exact per-call tokens and workflow metadata', async () => {
  const originalFetch = global.fetch;
  global.fetch = async () =>
    new Response(JSON.stringify(makeGeminiResponse()), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  const reports = [];
  const warnings = [];
  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    geminiOptions: {
      thinkingBudget: 256,
      usageReporter: {
        settings: {
          enabled: true,
          inputTextMode: 'label',
          inputTextLabel: 'RAG',
          includeOutputText: true,
          failOnReporterError: false,
        },
      },
    },
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { name: 'Modelo Resumidor RAG', parameters };
    },
    getParentNodes(_nodeName, options) {
      assert.equal(options.connectionType, 'ai_tool');
      return [{ name: 'MonitorarTokens11' }];
    },
    async getInputConnectionData(connectionType) {
      assert.equal(connectionType, 'ai_tool');
      return {
        async func(payload) {
          reports.push(structuredClone(payload));
          return 'ok';
        },
        async invoke() {
          throw new Error('invoke should not be used when func is available');
        },
      };
    },
    getWorkflow() {
      return { id: 'workflow-rag', name: 'Produtos Digitais' };
    },
    getExecutionId() {
      return 'execution-123';
    },
    addInputData() {
      return { index: 0 };
    },
    addOutputData() {},
    getNextRunIndex() {
      return 0;
    },
    logAiEvent() {},
    logger: {
      warn(message) {
        warnings.push(message);
      },
    },
  };

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    const message = await supplied.response.invoke('resuma os documentos');

    assert.equal(message.text, 'Resposta final');
    assert.equal(reports.length, 1);
    assert.equal(warnings.length, 0);

    const report = reports[0];
    assert.equal(report.model, 'gemini-2.5-flash');
    assert.equal(report.input_token, 100);
    assert.equal(report.input_uncached_token, 40);
    assert.equal(report.output_token, 20);
    assert.equal(report.cached_token, 60);
    assert.equal(report.thoughts_token, 13);
    assert.equal(report.tool_token, 7);
    assert.equal(report.overhead_token, 20);
    assert.equal(report.total_token, 140);
    assert.equal(report.model_calls, 1);
    assert.equal(report.input_text, 'RAG');
    assert.equal(report.output_text, 'Resposta final');
    assert.equal(report.workflow_id, 'workflow-rag');
    assert.equal(report.workflow_name, 'Produtos Digitais');
    assert.equal(report.execution_id, 'execution-123');
    assert.equal(report.node, 'Modelo Resumidor RAG');

    const dump = JSON.parse(report.dump);
    assert.equal(dump.provider, 'gemini');
    assert.equal(dump.model, 'gemini-2.5-flash');
    assert.equal(dump.tokenUsage.inputTokens, 100);
    assert.equal(dump.usageMetadata.toolUsePromptTokenCount, 7);
    assert.equal(dump.gemini.modelVersion, 'gemini-test-001');
  } finally {
    global.fetch = originalFetch;
  }
});

test('Usage Reporter failures are non-blocking by default', async () => {
  const originalFetch = global.fetch;
  global.fetch = async () =>
    new Response(JSON.stringify(makeGeminiResponse()), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  const warnings = [];
  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    enableUsageReporter: true,
    geminiOptions: {},
    usageReportingOptions: {
      inputTextMode: 'prompt',
      includeOutputText: false,
    },
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { name: 'Modelo RAG', parameters };
    },
    getParentNodes() {
      return [{ name: 'Reporter com falha' }];
    },
    async getInputConnectionData() {
      return {
        async invoke(payload) {
          assert.match(payload.input_text, /resuma/);
          assert.equal(payload.output_text, '');
          throw new Error('subworkflow indisponível');
        },
      };
    },
    getWorkflow() {
      return { id: 'workflow-rag', name: 'RAG' };
    },
    getExecutionId() {
      return 'execution-456';
    },
    addInputData() {
      return { index: 0 };
    },
    addOutputData() {},
    getNextRunIndex() {
      return 0;
    },
    logAiEvent() {},
    logger: {
      warn(message) {
        warnings.push(message);
      },
    },
  };

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    const message = await supplied.response.invoke('resuma este RAG');

    assert.equal(message.text, 'Resposta final');
    assert.equal(warnings.length, 1);
    assert.match(warnings[0], /subworkflow indisponível/);
  } finally {
    global.fetch = originalFetch;
  }
});

test('concurrent Usage Reporter events keep prompts, outputs, and tokens isolated', async () => {
  const originalFetch = global.fetch;
  global.fetch = async (request) => {
    const body = JSON.parse(await request.clone().text());
    const prompt = body.contents[0].parts[0].text;
    const isA = prompt === 'request-A';
    await new Promise((resolve) => setTimeout(resolve, isA ? 15 : 2));

    return new Response(
      JSON.stringify({
        candidates: [
          {
            content: {
              role: 'model',
              parts: [{ text: `Response ${isA ? 'A' : 'B'}` }],
            },
            finishReason: 'STOP',
            index: 0,
          },
        ],
        usageMetadata: {
          promptTokenCount: isA ? 10 : 100,
          candidatesTokenCount: isA ? 20 : 200,
          thoughtsTokenCount: isA ? 5 : 50,
          totalTokenCount: isA ? 35 : 350,
        },
        modelVersion: `model-${isA ? 'A' : 'B'}`,
        responseId: `response-${isA ? 'A' : 'B'}`,
      }),
      {
        status: 200,
        headers: { 'content-type': 'application/json' },
      },
    );
  };

  const reports = [];
  let runIndex = 0;
  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-3.5-flash-lite',
    enableUsageReporter: true,
    geminiOptions: {
      thinkingLevel: 'MEDIUM',
    },
    usageReportingOptions: {
      inputTextMode: 'prompt',
    },
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { name: 'Modelo Concorrente', parameters };
    },
    getParentNodes() {
      return [{ name: 'MonitorarTokens11' }];
    },
    async getInputConnectionData() {
      return {
        async invoke(payload) {
          reports.push(structuredClone(payload));
        },
      };
    },
    getWorkflow() {
      return { id: 'workflow-concurrent', name: 'Concorrência' };
    },
    getExecutionId() {
      return 'execution-concurrent';
    },
    addInputData() {
      return { index: runIndex++ };
    },
    addOutputData() {},
    getNextRunIndex() {
      return runIndex;
    },
    logAiEvent() {},
    logger: {
      warn() {},
    },
  };

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    await Promise.all([
      supplied.response.invoke('request-A'),
      supplied.response.invoke('request-B'),
    ]);

    assert.equal(reports.length, 2);
    const reportA = reports.find((report) => report.output_text === 'Response A');
    const reportB = reports.find((report) => report.output_text === 'Response B');
    assert.match(reportA.input_text, /request-A/);
    assert.equal(reportA.input_token, 10);
    assert.equal(reportA.output_token, 20);
    assert.equal(reportA.thoughts_token, 5);
    assert.equal(reportA.total_token, 35);
    assert.match(reportB.input_text, /request-B/);
    assert.equal(reportB.input_token, 100);
    assert.equal(reportB.output_token, 200);
    assert.equal(reportB.thoughts_token, 50);
    assert.equal(reportB.total_token, 350);
  } finally {
    global.fetch = originalFetch;
  }
});

test('n8n tracing exposes parsed Structured Output separately from generation text', async () => {
  const originalFetch = global.fetch;
  const structured = {
    output: 'Olá! Como posso ajudar?',
    abertura_de_chamado: false,
    cpf: '',
    descricao: '',
  };
  const response = makeGeminiResponse();
  response.candidates[0].content.parts[1].text = JSON.stringify(structured);
  global.fetch = async () =>
    new Response(JSON.stringify(response), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  const outputs = [];
  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    geminiOptions: {
      responseSchema: JSON.stringify({
        type: 'object',
        properties: {
          output: { type: 'string' },
          abertura_de_chamado: { type: 'boolean' },
          cpf: { type: 'string' },
          descricao: { type: 'string' },
        },
        required: ['output', 'abertura_de_chamado', 'cpf', 'descricao'],
      }),
    },
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { name: 'Universal Chat Model', parameters };
    },
    addInputData() {
      return { index: 0 };
    },
    addOutputData(connectionType, index, data) {
      outputs.push({ connectionType, index, data });
    },
    getNextRunIndex() {
      return 0;
    },
    logAiEvent() {},
  };

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    const modelResponse = await supplied.response.invoke('olá');
    assert.equal(modelResponse.text, JSON.stringify(structured));

    const trace = outputs[0].data[0][0].json;
    assert.equal(
      trace.response.generations[0][0].text,
      JSON.stringify(structured),
    );
    assert.deepEqual(
      trace.response.generations[0][0].structuredOutput,
      structured,
    );
    assert.deepEqual(trace.structuredOutput, structured);
  } finally {
    global.fetch = originalFetch;
  }
});

test('streaming sends one complete Usage Reporter event with final token metadata', async () => {
  const originalFetch = global.fetch;
  const thoughtChunk = {
    candidates: [
      {
        content: {
          role: 'model',
          parts: [{ text: 'Resumo do raciocínio', thought: true }],
        },
        index: 0,
      },
    ],
    modelVersion: 'gemini-test-001',
    responseId: 'response-test',
  };
  const finalChunk = {
    candidates: [
      {
        content: {
          role: 'model',
          parts: [{ text: 'Resposta final', thoughtSignature: 'signature' }],
        },
        finishReason: 'STOP',
        index: 0,
      },
    ],
    usageMetadata,
    modelVersion: 'gemini-test-001',
    responseId: 'response-test',
  };
  const event =
    `data: ${JSON.stringify(thoughtChunk)}\n\n` +
    `data: ${JSON.stringify(finalChunk)}\n\n`;
  global.fetch = async () =>
    new Response(event, {
      status: 200,
      headers: { 'content-type': 'text/event-stream' },
    });

  const outputs = [];
  const reports = [];
  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    enableUsageReporter: true,
    geminiOptions: {
      thinkingBudget: 256,
      includeThoughts: true,
    },
    usageReportingOptions: {
      nodeLabel: 'RAG Streaming',
    },
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { name: 'Universal Chat Model', parameters };
    },
    getParentNodes() {
      return [{ name: 'MonitorarTokens11' }];
    },
    async getInputConnectionData() {
      return {
        async invoke(payload) {
          reports.push(structuredClone(payload));
        },
      };
    },
    getWorkflow() {
      return { id: 'workflow-stream', name: 'RAG Streaming' };
    },
    getExecutionId() {
      return 'execution-stream';
    },
    addInputData() {
      return { index: 0 };
    },
    addOutputData(connectionType, index, data) {
      outputs.push({ connectionType, index, data });
    },
    getNextRunIndex() {
      return 0;
    },
    logAiEvent() {},
    logger: {
      warn() {},
    },
  };

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    const stream = await supplied.response.stream('teste de tracing com streaming');
    const visibleChunks = [];
    for await (const chunk of stream) {
      visibleChunks.push(chunk.text);
      assert.equal(
        Array.isArray(chunk.content)
          ? chunk.content.some((block) => block?.thought === true)
          : false,
        false,
      );
    }

    assert.equal(visibleChunks.join(''), 'Resposta final');
    assert.equal(visibleChunks.join('').includes('Resumo do raciocínio'), false);
    assert.equal(outputs.length, 1);
    const trace = outputs[0].data[0][0].json;
    assert.equal(trace.thoughts[0].text, 'Resumo do raciocínio');
    assert.equal(trace.tokenUsage.inputTokens, 100);
    assert.equal(trace.tokenUsage.outputTokens, 20);
    assert.equal(trace.tokenUsage.cachedTokens, 60);
    assert.equal(trace.tokenUsage.toolUsePromptTokens, 7);
    assert.equal(trace.tokenUsage.thoughtsTokens, 13);
    assert.deepEqual(trace.usageMetadata, usageMetadata);
    assert.equal(trace.response.generations[0][0].text, 'Resposta final');
    assert.equal(reports.length, 1);
    assert.equal(reports[0].output_text, 'Resposta final');
    assert.equal(reports[0].input_token, 100);
    assert.equal(reports[0].thoughts_token, 13);
    assert.equal(reports[0].tool_token, 7);
    assert.equal(reports[0].node, 'RAG Streaming');
    assert.equal(
      JSON.stringify(trace).split('Resumo do raciocínio').length - 1,
      1,
    );
  } finally {
    global.fetch = originalFetch;
  }
});

test('raw Gemini capture preserves nested thought summaries dropped by the model adapter', async () => {
  const originalFetch = global.fetch;
  const nestedSummaryResponse = {
    candidates: [
      {
        content: {
          role: 'model',
          parts: [
            {
              thought: {
                signature: 'nested-signature',
                summary: [
                  {
                    text: {
                      text: 'Resumo no formato aninhado',
                    },
                  },
                ],
              },
            },
            { text: 'Resposta sem pensamento misturado' },
          ],
        },
        finishReason: 'STOP',
        index: 0,
      },
    ],
    usageMetadata,
    modelVersion: 'gemini-3.5-flash-lite',
    responseId: 'nested-response',
  };
  global.fetch = async () =>
    new Response(JSON.stringify(nestedSummaryResponse), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  try {
    const model = new GeminiChatModel({
      apiKey: 'test',
      model: 'gemini-3.5-flash-lite',
      thinkingConfig: {
        thinkingLevel: 'MEDIUM',
        includeThoughts: true,
      },
    });
    const response = await model.invoke('teste');

    assert.equal(response.text, 'Resposta sem pensamento misturado');
    assert.equal(
      response.response_metadata.thoughts[0].text,
      'Resumo no formato aninhado',
    );
    assert.equal(response.response_metadata.thoughts[0].thought, true);
    assert.equal(
      response.response_metadata.thoughts[0].thoughtSignature,
      'nested-signature',
    );
    assert.equal(response.response_metadata.gemini.thoughts.length, 1);
  } finally {
    global.fetch = originalFetch;
  }
});

test('AI Agent output receives separate thoughts and aggregated model metadata', async () => {
  const originalFetch = global.fetch;
  global.fetch = async () =>
    new Response(JSON.stringify(makeGeminiResponse()), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    geminiOptions: {
      thinkingBudget: 256,
      includeThoughts: true,
    },
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { name: 'Universal Chat Model', parameters };
    },
    addInputData() {
      return { index: 0 };
    },
    addOutputData() {},
    getNextRunIndex() {
      return 0;
    },
    logAiEvent() {},
  };

  const fixture = join(
    __dirname,
    'fixtures',
    'n8n-nodes-langchain',
    'dist',
    'nodes',
    'agents',
    'Agent',
    'V2',
    'AgentV2.node.cjs',
  );
  const { AgentV2 } = require(fixture);
  installAgentOutputBridge();

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    const agent = new AgentV2();
    agent.run = async () => {
      await supplied.response.invoke('primeira chamada');
      await supplied.response.invoke('segunda chamada');
      return [[{ json: { output: 'Somente a resposta final' } }]];
    };

    const result = await agent.execute();
    const output = result[0][0].json;

    assert.equal(output.output, 'Somente a resposta final');
    assert.equal(output.output.includes('Resumo do raciocínio'), false);
    assert.equal(output.thoughts.length, 1);
    assert.equal(output.thoughts[0].text, 'Resumo do raciocínio');
    assert.equal(output.modelCalls, 2);
    assert.equal(output.modelResponses.length, 2);
    assert.equal(output.tokenUsage.inputTokens, 200);
    assert.equal(output.tokenUsage.inputUncachedTokens, 80);
    assert.equal(output.tokenUsage.outputTokens, 40);
    assert.equal(output.tokenUsage.cachedTokens, 120);
    assert.equal(output.tokenUsage.toolUsePromptTokens, 14);
    assert.equal(output.tokenUsage.thoughtsTokens, 26);
    assert.equal(output.tokenUsage.totalTokens, 280);
    assert.deepEqual(output.usageMetadata, usageMetadata);
    assert.equal(output.gemini.modelVersion, 'gemini-test-001');
    assert.equal(output.gemini.usageMetadata, undefined);
  } finally {
    global.fetch = originalFetch;
  }
});

test('AI Agent can hide token usage while Usage Reporter still receives it', async () => {
  const originalFetch = global.fetch;
  global.fetch = async () =>
    new Response(JSON.stringify(makeGeminiResponse()), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  const reports = [];
  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    geminiOptions: {
      includeTokenUsageInAgentOutput: false,
      usageReporter: {
        settings: {
          enabled: true,
        },
      },
    },
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { name: 'Modelo de Produção', parameters };
    },
    getParentNodes() {
      return [{ name: 'Monitor de Tokens' }];
    },
    async getInputConnectionData() {
      return {
        async invoke(payload) {
          reports.push(structuredClone(payload));
        },
      };
    },
    getWorkflow() {
      return { id: 'workflow-agent', name: 'Agente de Produção' };
    },
    getExecutionId() {
      return 'execution-agent';
    },
    addInputData() {
      return { index: 0 };
    },
    addOutputData() {},
    getNextRunIndex() {
      return 0;
    },
    logAiEvent() {},
    logger: {
      warn() {},
    },
  };

  const fixture = join(
    __dirname,
    'fixtures',
    'n8n-nodes-langchain',
    'dist',
    'nodes',
    'agents',
    'Agent',
    'V2',
    'AgentV2.node.cjs',
  );
  const { AgentV2 } = require(fixture);
  installAgentOutputBridge();

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    const agent = new AgentV2();
    agent.run = async () => {
      await supplied.response.invoke('monitore esta chamada');
      return [[{ json: { output: 'Resposta final limpa' } }]];
    };

    const result = await agent.execute();
    const output = result[0][0].json;

    assert.equal(output.output, 'Resposta final limpa');
    assert.equal(output.tokenUsage, undefined);
    assert.equal(output.usageMetadata, undefined);
    assert.equal(output.modelResponses[0].tokenUsage, undefined);
    assert.equal(output.modelResponses[0].usageMetadata, undefined);
    assert.equal(reports.length, 1);
    assert.equal(reports[0].input_token, 100);
    assert.equal(reports[0].output_token, 20);
    assert.equal(reports[0].cached_token, 60);
    assert.equal(reports[0].thoughts_token, 13);
    assert.equal(reports[0].tool_token, 7);
    assert.equal(reports[0].total_token, 140);
  } finally {
    global.fetch = originalFetch;
  }
});

test('AI Agent loaded after the community node still receives metadata', async () => {
  const fixture = join(
    __dirname,
    'fixtures',
    'n8n-nodes-langchain',
    'dist',
    'nodes',
    'agents',
    'Agent',
    'V1',
    'AgentV1.node.cjs',
  );

  assert.equal(require.cache[require.resolve(fixture)], undefined);
  installAgentOutputBridge();
  const { AgentV1 } = require(fixture);

  const agent = new AgentV1();
  agent.run = async () => {
    recordAgentModelMetadata({
      thoughts: [{ text: 'Pensamento carregado tardiamente' }],
      tokenUsage: {
        inputTokens: 12,
        inputUncachedTokens: 7,
        outputTokens: 5,
        cachedTokens: 5,
        toolUsePromptTokens: 2,
        thoughtsTokens: 3,
        totalTokens: 22,
      },
    });

    return [[{ json: { output: 'Resposta final tardia' } }]];
  };

  const result = await agent.execute();
  const output = result[0][0].json;

  assert.equal(output.output, 'Resposta final tardia');
  assert.equal(output.thoughts[0].text, 'Pensamento carregado tardiamente');
  assert.equal(output.tokenUsage.inputTokens, 12);
  assert.equal(output.tokenUsage.cachedTokens, 5);
  assert.equal(output.modelCalls, 1);
});

test('AI Agent promotes Structured Output JSON into separate output fields', async () => {
  const fixture = join(
    __dirname,
    'fixtures',
    'n8n-nodes-langchain',
    'dist',
    'nodes',
    'agents',
    'Agent',
    'V1',
    'AgentV1.node.cjs',
  );
  const { AgentV1 } = require(fixture);
  installAgentOutputBridge();

  const structured = {
    output: 'Olá! Tudo bem?',
    abertura_de_chamado: false,
    cpf: '',
    descricao: '',
  };
  const agent = new AgentV1();
  agent.run = async () => {
    recordAgentModelMetadata({
      structuredOutput: structured,
      tokenUsage: {
        inputTokens: 10,
        inputUncachedTokens: 10,
        outputTokens: 5,
        cachedTokens: 0,
        toolUsePromptTokens: 0,
        thoughtsTokens: 2,
        totalTokens: 17,
      },
    });
    return [[{ json: { output: JSON.stringify(structured) } }]];
  };

  const result = await agent.execute();
  const output = result[0][0].json;

  assert.equal(output.output, 'Olá! Tudo bem?');
  assert.equal(output.abertura_de_chamado, false);
  assert.equal(output.cpf, '');
  assert.equal(output.descricao, '');
  assert.equal(output.tokenUsage.totalTokens, 17);
  assert.deepEqual(output.modelResponses[0].structuredOutput, structured);
});

test('AI Agent keeps its output contract when structured JSON has no output field', async () => {
  const fixture = join(
    __dirname,
    'fixtures',
    'n8n-nodes-langchain',
    'dist',
    'nodes',
    'agents',
    'Agent',
    'V1',
    'AgentV1.node.cjs',
  );
  const { AgentV1 } = require(fixture);
  installAgentOutputBridge();

  const structured = {
    message: 'Olá!',
    category: 'greeting',
  };
  const serialized = JSON.stringify(structured);
  const agent = new AgentV1();
  agent.run = async () => {
    recordAgentModelMetadata({ structuredOutput: structured });
    return [[{ json: { output: serialized } }]];
  };

  const result = await agent.execute();
  const output = result[0][0].json;

  assert.equal(output.output, serialized);
  assert.equal(output.message, 'Olá!');
  assert.equal(output.category, 'greeting');
});

test('parallel AI Agent executions keep thoughts and token usage isolated', async () => {
  const fixture = join(
    __dirname,
    'fixtures',
    'n8n-nodes-langchain',
    'dist',
    'nodes',
    'agents',
    'Agent',
    'V2',
    'AgentV2.node.cjs',
  );
  const { AgentV2 } = require(fixture);
  installAgentOutputBridge();

  const makeAgent = (label, delay, tokens) => {
    const agent = new AgentV2();
    agent.run = async () => {
      recordAgentModelMetadata({
        thoughts: [{ text: `Pensamento ${label} 1` }],
        tokenUsage: {
          inputTokens: tokens,
          totalTokens: tokens,
        },
      });
      await new Promise((resolve) => setTimeout(resolve, delay));
      recordAgentModelMetadata({
        thoughts: [{ text: `Pensamento ${label} 2` }],
        tokenUsage: {
          outputTokens: tokens,
          totalTokens: tokens,
        },
      });
      return [[{ json: { output: `Resposta ${label}` } }]];
    };
    return agent;
  };

  const [resultA, resultB] = await Promise.all([
    makeAgent('A', 20, 10).execute(),
    makeAgent('B', 5, 100).execute(),
  ]);
  const outputA = resultA[0][0].json;
  const outputB = resultB[0][0].json;

  assert.deepEqual(
    outputA.thoughts.map((thought) => thought.text),
    ['Pensamento A 1', 'Pensamento A 2'],
  );
  assert.deepEqual(
    outputB.thoughts.map((thought) => thought.text),
    ['Pensamento B 1', 'Pensamento B 2'],
  );
  assert.equal(outputA.tokenUsage.inputTokens, 10);
  assert.equal(outputA.tokenUsage.outputTokens, 10);
  assert.equal(outputA.tokenUsage.totalTokens, 20);
  assert.equal(outputB.tokenUsage.inputTokens, 100);
  assert.equal(outputB.tokenUsage.outputTokens, 100);
  assert.equal(outputB.tokenUsage.totalTokens, 200);
});

test('AI Agent V3 preserves metadata across tool-call iterations', async () => {
  const fixture = join(
    __dirname,
    'fixtures',
    'n8n-nodes-langchain',
    'dist',
    'nodes',
    'agents',
    'Agent',
    'V3',
    'AgentV3.node.cjs',
  );
  const { AgentV3 } = require(fixture);
  installAgentOutputBridge();

  const agent = new AgentV3();
  agent.run = async (response) => {
    recordAgentModelMetadata({
      tokenUsage: {
        inputTokens: 10,
        inputUncachedTokens: 8,
        outputTokens: 4,
        cachedTokens: 2,
        toolUsePromptTokens: 1,
        thoughtsTokens: 3,
        totalTokens: 17,
      },
    });

    if (!response) {
      return {
        actions: [{ nodeName: 'Tool' }],
        metadata: { iterationCount: 1 },
      };
    }

    return [[{ json: { output: 'Resposta depois da ferramenta' } }]];
  };

  const request = await agent.execute();
  assert.equal(request.metadata.universalChatModel.calls.length, 1);

  const result = await agent.execute({ metadata: request.metadata });
  const output = result[0][0].json;

  assert.equal(output.output, 'Resposta depois da ferramenta');
  assert.equal(output.modelCalls, 2);
  assert.equal(output.tokenUsage.inputTokens, 20);
  assert.equal(output.tokenUsage.cachedTokens, 4);
  assert.equal(output.tokenUsage.outputTokens, 8);
  assert.equal(output.tokenUsage.toolUsePromptTokens, 2);
  assert.equal(output.tokenUsage.thoughtsTokens, 6);
  assert.equal(output.tokenUsage.totalTokens, 34);
});

test('Usage Reporter can fail the workflow without repeating a successful model call', async () => {
  const originalFetch = global.fetch;
  let apiCalls = 0;
  let reporterCalls = 0;
  global.fetch = async () => {
    apiCalls += 1;
    return new Response(JSON.stringify(makeGeminiResponse()), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };

  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    modelRetryOnFail: true,
    modelMaxTries: 3,
    modelWaitBetweenTries: 0,
    geminiOptions: {
      usageReporter: {
        settings: {
          enabled: true,
          failOnReporterError: true,
        },
      },
    },
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { name: 'Production Model', parameters };
    },
    getParentNodes() {
      return [{ name: 'Failing Reporter' }];
    },
    async getInputConnectionData() {
      return {
        async func() {
          reporterCalls += 1;
          throw new Error('telemetry database unavailable');
        },
      };
    },
    getWorkflow() {
      return { id: 'workflow-fail-reporter', name: 'Production' };
    },
    getExecutionId() {
      return 'execution-fail-reporter';
    },
    addInputData() {
      return { index: 0 };
    },
    addOutputData() {},
    getNextRunIndex() {
      return 0;
    },
    logAiEvent() {},
    logger: {
      warn() {},
    },
  };

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    await assert.rejects(
      () => supplied.response.invoke('do not retry this completed call'),
      /Usage Reporter failed|telemetry database unavailable/,
    );
    assert.equal(apiCalls, 1);
    assert.equal(reporterCalls, 1);
  } finally {
    global.fetch = originalFetch;
  }
});

test('Usage Reporter remains optional when enabled without a connected tool', async () => {
  const originalFetch = global.fetch;
  let connectionReads = 0;
  global.fetch = async () =>
    new Response(JSON.stringify(makeGeminiResponse()), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    geminiOptions: {
      usageReporter: {
        settings: {
          enabled: true,
        },
      },
    },
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { name: 'Optional Reporter Model', parameters };
    },
    getParentNodes() {
      return [];
    },
    async getInputConnectionData() {
      connectionReads += 1;
      throw new Error('must not read a missing optional connection');
    },
    getWorkflow() {
      return { id: 'workflow-no-reporter', name: 'No Reporter' };
    },
    getExecutionId() {
      return 'execution-no-reporter';
    },
    addInputData() {
      return { index: 0 };
    },
    addOutputData() {},
    getNextRunIndex() {
      return 0;
    },
    logAiEvent() {},
  };

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    const message = await supplied.response.invoke('reporter is optional');
    assert.equal(message.text, 'Resposta final');
    assert.equal(connectionReads, 0);
  } finally {
    global.fetch = originalFetch;
  }
});

test('grouped Usage Reporter privacy settings can send the prompt and omit output text', async () => {
  const originalFetch = global.fetch;
  const reports = [];
  global.fetch = async () =>
    new Response(JSON.stringify(makeGeminiResponse()), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    geminiOptions: {
      usageReporter: {
        settings: {
          enabled: true,
          nodeLabel: 'Customer Support',
          inputTextMode: 'prompt',
          includeOutputText: false,
        },
      },
    },
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { name: 'Fallback Node Name', parameters };
    },
    getParentNodes() {
      return [{ name: 'Reporter' }];
    },
    async getInputConnectionData() {
      return {
        async func(payload) {
          reports.push(structuredClone(payload));
        },
      };
    },
    getWorkflow() {
      return { id: 'workflow-privacy', name: 'Privacy Test' };
    },
    getExecutionId() {
      return '__UNKNOWN__';
    },
    addInputData() {
      return { index: 0 };
    },
    addOutputData() {},
    getNextRunIndex() {
      return 0;
    },
    logAiEvent() {},
    logger: {
      warn() {},
    },
  };

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    await supplied.response.invoke('customer request 123');

    assert.equal(reports.length, 1);
    assert.match(reports[0].input_text, /customer request 123/);
    assert.equal(reports[0].output_text, '');
    assert.equal(reports[0].node, 'Customer Support');
    assert.equal(reports[0].execution_id, '');
  } finally {
    global.fetch = originalFetch;
  }
});

test('OpenAI-compatible calls send normalized usage to Usage Reporter', async () => {
  const originalFetch = global.fetch;
  const reports = [];
  global.fetch = async () =>
    new Response(
      JSON.stringify({
        id: 'chatcmpl-usage-report',
        object: 'chat.completion',
        created: 1,
        model: 'compatible-model',
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: 'Compatible response' },
            finish_reason: 'stop',
            logprobs: null,
          },
        ],
        usage: {
          prompt_tokens: 11,
          completion_tokens: 7,
          total_tokens: 18,
          prompt_tokens_details: {
            cached_tokens: 3,
          },
          completion_tokens_details: {
            reasoning_tokens: 2,
          },
        },
      }),
      {
        status: 200,
        headers: { 'content-type': 'application/json' },
      },
    );

  const parameters = {
    provider: 'openai_compatible',
    openaiModel: 'compatible-model',
    openaiOptions: {
      usageReporter: {
        settings: {
          enabled: true,
          inputTextLabel: 'AGENT',
        },
      },
    },
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return {
        apiKey: 'test',
        baseUrl: 'https://compatible.example/v1',
      };
    },
    getNode() {
      return { name: 'Compatible Model', parameters };
    },
    getParentNodes() {
      return [{ name: 'Reporter' }];
    },
    async getInputConnectionData() {
      return {
        async invoke(payload) {
          reports.push(structuredClone(payload));
        },
      };
    },
    getWorkflow() {
      return { id: 'workflow-openai', name: 'OpenAI Compatible' };
    },
    getExecutionId() {
      return 'execution-openai';
    },
    addInputData() {
      return { index: 0 };
    },
    addOutputData() {},
    getNextRunIndex() {
      return 0;
    },
    logAiEvent() {},
    logger: {
      warn() {},
    },
  };

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    const message = await supplied.response.invoke('compatible usage');

    assert.equal(message.text, 'Compatible response');
    assert.equal(reports.length, 1);
    assert.equal(reports[0].model, 'compatible-model');
    assert.equal(reports[0].input_token, 11);
    assert.equal(reports[0].cached_token, 3);
    assert.equal(reports[0].input_uncached_token, 8);
    assert.equal(reports[0].output_token, 5);
    assert.equal(reports[0].thoughts_token, 2);
    assert.equal(reports[0].total_token, 18);
    assert.equal(reports[0].input_text, 'AGENT');
  } finally {
    global.fetch = originalFetch;
  }
});

test('AI Agent keeps thought text hidden when Include Thoughts is disabled', async () => {
  const originalFetch = global.fetch;
  global.fetch = async () =>
    new Response(JSON.stringify(makeGeminiResponse()), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  const parameters = {
    provider: 'gemini',
    geminiModel: 'gemini-2.5-flash',
    geminiOptions: {
      thinkingBudget: 256,
    },
  };
  const context = {
    getNodeParameter(name, _itemIndex, fallback) {
      return parameters[name] ?? fallback;
    },
    async getCredentials() {
      return { apiKey: 'test' };
    },
    getNode() {
      return { name: 'Hidden Thoughts Model', parameters };
    },
    addInputData() {
      return { index: 0 };
    },
    addOutputData() {},
    getNextRunIndex() {
      return 0;
    },
    logAiEvent() {},
  };

  const fixture = join(
    __dirname,
    'fixtures',
    'n8n-nodes-langchain',
    'dist',
    'nodes',
    'agents',
    'Agent',
    'V2',
    'AgentV2.node.cjs',
  );
  const { AgentV2 } = require(fixture);
  installAgentOutputBridge();

  try {
    const supplied = await new UniversalChatModel().supplyData.call(context, 0);
    const agent = new AgentV2();
    agent.run = async () => {
      await supplied.response.invoke('keep thought text private');
      return [[{ json: { output: 'Visible answer only' } }]];
    };

    const result = await agent.execute();
    const output = result[0][0].json;
    assert.equal(output.output, 'Visible answer only');
    assert.equal(output.thoughts, undefined);
    assert.equal(
      JSON.stringify(output).includes('Resumo do racioc'),
      false,
    );
    assert.equal(output.tokenUsage.thoughtsTokens, 13);
  } finally {
    global.fetch = originalFetch;
  }
});
