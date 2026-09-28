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

## Token usage na saída do AI Agent

Ative **Include Token Usage in Output** em **Options** do Universal Chat Model
conectado ao AI Agent. O resultado do Agent inclui `tokenUsage` no nível superior
do JSON, além de `usageMetadata` quando o Gemini a fornece. A saída do subnó
Chat Model mostra a contagem independentemente dessa opção.

Para investigar de onde vem o consumo no Gemini, ative **Include Gemini Request
Details in Output** em **Options**. O AI Agent adiciona `geminiRequests`, com uma
entrada por chamada à API. Cada entrada contém `call` (chamada do modelo pelo
Agent), `requestNumber` (requisição ao Gemini dentro dessa chamada), `model`,
`request` (corpo JSON enviado), `response` ou `responseChunks` no streaming, e
`usageMetadata` dessa requisição. A opção também mostra `tokenUsage` na saída
do Agent. Ela vem desligada por padrão porque o conteúdo completo pode incluir
instruções de sistema, histórico, esquemas e resultados de ferramentas, além
de aumentar o tamanho dos dados salvos na execução. Resumos de pensamento na
resposta só aparecem com **Include Thoughts** ativo.

O Gemini informa a contagem total de entrada e saída e alguns detalhes, como
modalidade e cache, mas não atribui um número exato de tokens a cada trecho do
prompt. O campo `request` permite inspecionar o conteúdo que entrou em cada
chamada e identificar histórico ou resultados de ferramentas que aumentaram a
entrada; não se deve interpretar seu tamanho em caracteres como tokens exatos.
