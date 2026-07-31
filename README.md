# n8n Universal Chat Model

Community node para conectar o **AI Agent** do n8n ao Google Gemini nativo ou
a APIs compatíveis com OpenAI, incluindo Ollama, LM Studio, DeepSeek,
OpenRouter, LocalAI e vLLM.

O node foi projetado para fluxos de agente com múltiplas ferramentas,
function calling sequencial ou paralelo, streaming, thoughts disponibilizados
pelo provedor, Structured Output e telemetria detalhada de tokens.

## Compatibilidade

- n8n alvo e validado: **2.32.6**
- Peer runtime: `n8n-workflow >= 2.32.1 < 3.0.0`
- Node.js: `>= 22.22 < 25`
- AI Agent V1, V2 e V3
- Pacote AI do n8n 2.32.6: `@n8n/n8n-nodes-langchain` 2.32.4
- Interoperabilidade validada com o `@langchain/core` 1.2.0 usado pelo pacote AI
