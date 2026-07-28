import { ICredentialType, INodeProperties } from 'n8n-workflow';

export class GoogleGeminiApi implements ICredentialType {
  name = 'googleGeminiApi';
  displayName = 'Google Gemini API';
  documentationUrl = 'https://ai.google.dev/';
  properties: INodeProperties[] = [
    {
      displayName: 'API Key',
      name: 'apiKey',
      type: 'string',
      typeOptions: {
        password: true,
      },
      default: '',
      required: true,
      description: 'Google AI Studio API key',
    },
  ];
}
