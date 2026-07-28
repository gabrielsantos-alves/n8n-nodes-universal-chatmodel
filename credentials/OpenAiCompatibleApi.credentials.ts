import { ICredentialType, INodeProperties } from 'n8n-workflow';

export class OpenAiCompatibleApi implements ICredentialType {
  name = 'openAiCompatibleApi';
  displayName = 'OpenAI Compatible / Local LLM API';
  documentationUrl = 'https://docs.n8n.io/';
  properties: INodeProperties[] = [
    {
      displayName: 'Base URL',
      name: 'baseUrl',
      type: 'string',
      default: 'http://localhost:11434/v1',
      required: true,
      placeholder: 'http://localhost:11434/v1',
      description: 'Endpoint base URL (Ollama, LM Studio, DeepSeek, OpenRouter, vLLM, LocalAI)',
    },
    {
      displayName: 'API Key',
      name: 'apiKey',
      type: 'string',
      typeOptions: {
        password: true,
      },
      default: 'not-needed',
      description: 'API key for OpenAI / OpenRouter / DeepSeek or local proxy (defaults to "not-needed")',
    },
  ];
}
