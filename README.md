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

## Recursos

### Google Gemini nativo

- Temperature, Top P, Top K e Max Output Tokens opcionais.
- Thinking Level ou Thinking Budget opcionais.
- Captura de thought summaries quando a API as disponibiliza.
- Preservação de `thoughtSignature` em tool loops.
- Structured Output Schema opcional.
- Safety Settings.
- Metadados brutos relevantes da Gemini API.

### OpenAI compatible

- Base URL configurável.
- Suporte a APIs locais e remotas compatíveis com Chat Completions.
- Temperature, penalties, Max Tokens, Seed, JSON Mode e Reasoning Effort.
- Headers customizados.

### Agentes e ferramentas

- Tool calls sequenciais e paralelas.
- Preservação dos IDs das chamadas.
- Recuperação segura de uma resposta terminal vazia sem repetir tool calls.
- Compatibilidade com MCP Client Tool e structured tools.
- Thoughts, tokens e metadados separados do texto final do AI Agent.
- Exposição de token usage no AI Agent controlada por uma opção independente.
- System Message adicional combinado com as instruções do node pai.
- Isolamento de metadados entre execuções concorrentes.

### Recuperação de resposta vazia

Uma resposta de tool calling normalmente possui `text: ""`: a informação útil
está em `tool_calls`, e o AI Agent deve executar a ferramenta. O node não
considera esse caso uma falha e nunca repete a chamada.

Se Gemini terminar com `STOP` sem texto, sem conteúdo e sem function call, a
opção **Recover Empty Final Responses** realiza uma única continuação segura.
O histórico completo e os resultados das tools permanecem presentes. Se a
continuação também vier vazia, o node devolve um erro detalhado em vez de
produzir silenciosamente `output: ""`.

Os tokens das duas requisições ao provedor são somados. O metadata
`gemini.emptyResponseRecovery` informa quantas requisições foram feitas e os
response IDs envolvidos.

### Uso de tokens

Quando o provedor informa os dados, o output inclui:

```json
{
  "tokenUsage": {
    "inputTokens": 0,
    "inputUncachedTokens": 0,
    "outputTokens": 0,
    "cachedTokens": 0,
    "toolUsePromptTokens": 0,
    "thoughtsTokens": 0,
    "totalTokens": 0
  }
}
```

Os nomes e valores exatos disponíveis dependem da resposta do provedor.
Use **Include Token Usage in Output** para mostrar ou ocultar esses campos em
outputs de nodes pais compatíveis. Essa opção não interfere na captura interna
nem no envio ao Usage Reporter. A configuração fica recolhida em **Options** e
só aparece no formulário quando for adicionada.

### Usage Reporter

Ative **Enable Usage Reporter** para exibir a entrada opcional **Usage
Reporter** do tipo AI Tool. Desativada, a entrada desaparece e o Chat Model
permanece com o formato circular padrão. Quando conectada, essa tool é
executada uma vez após cada chamada bem-sucedida da LLM. Ela não é apresentada
ao modelo e não pode ser escolhida pelo agente. A ativação e todas as
configurações do reporter ficam agrupadas em **Options → Usage Reporter**. As
subopções de label, privacidade, texto de output e tratamento de erro só
aparecem quando **Enable Usage Reporter** está ativado.

Isso permite conectar diretamente um Workflow Tool de telemetria e remover
wrappers ou monkey patches sobre `invoke` e `generate`:

```text
Workflow Tool de consumo -> Usage Reporter do Universal Chat Model
Universal Chat Model -> AI Language Model do Agent, Chain ou Vector Store Tool
```

O payload enviado mantém compatibilidade com workflows de consumo existentes e
inclui campos adicionais:

```json
{
  "model": "gemini-3.5-flash",
  "input_token": 0,
  "input_uncached_token": 0,
  "output_token": 0,
  "cached_token": 0,
  "thoughts_token": 0,
  "tool_token": 0,
  "overhead_token": 0,
  "total_token": 0,
  "model_calls": 1,
  "input_text": "RAG",
  "output_text": "",
  "workflow_id": "",
  "workflow_name": "",
  "execution_id": "",
  "node": "",
  "dump": "{}"
}
```

Por padrão, `input_text` recebe o rótulo `RAG` para não enviar prompts
potencialmente sensíveis. A opção **Input Text Mode** permite enviar o prompt
real. Falhas do reporter geram aviso e não interrompem a resposta da LLM, a
menos que **Fail Workflow if Reporter Fails** seja ativado.

### System Message

O campo opcional **System Message** aplica instruções adicionais a todas as
chamadas do modelo. Quando o AI Agent, Chain ou outro node pai já envia um
system prompt, o texto configurado no Chat Model é acrescentado depois dessas
instruções, preservando ambos. O campo fica dentro de **Options**, então não é
exibido quando você prefere definir o system prompt apenas no AI Agent.

### Tratamento de erros

O node classifica erros de:

- autenticação e permissão;
- request inválido e modelo inexistente;
- rate limit e quota;
- timeout, DNS, conexão, rede e TLS;
- Structured Output;
- tool schema, tool protocol e thought signatures;
- safety e respostas sem candidates;
- falhas transitórias do provedor.

Em `On Error: Continue`, os detalhes são disponibilizados em
`universalChatModelError`. Chaves, tokens e query parameters sensíveis são
removidos das mensagens.

O retry só ocorre para erros transitórios, como `408`, `429`, `5xx` e certas
falhas de transporte. Erros de configuração, autenticação, schema e
cancelamentos não são repetidos.

## Structured Output Schema

Ao adicionar a opção **Structured Output Schema (JSON)**, o node configura a
resposta como `application/json`. O valor inicial é:

```json
{
  "type": "object",
  "properties": {
    "message": {
      "type": "string"
    }
  },
  "additionalProperties": false,
  "required": ["message"]
}
```

O Gemini suporta apenas um subconjunto de JSON Schema. Consulte a
[documentação oficial](https://ai.google.dev/gemini-api/docs/structured-output)
antes de usar schemas muito grandes ou profundamente aninhados.

No tracing do Chat Model, o JSON permanece em `generation.text` para manter o
contrato do LangChain e também aparece parseado em `structuredOutput`. No
output final do AI Agent, as propriedades são promovidas para campos reais:

```json
{
  "output": "Olá! Como posso ajudar?",
  "abertura_de_chamado": false,
  "cpf": "",
  "descricao": ""
}
```

## Instalação pelo n8n

Como o pacote é publicado no npm, abra **Settings → Community Nodes** e instale:

```text
n8n-nodes-universal-chatmodel
```

Para instalações manuais, use o usuário que executa o n8n:

```bash
cd ~/.n8n/nodes
npm install n8n-nodes-universal-chatmodel@1.0.26
```

Depois, reinicie o n8n.

## Desenvolvimento

```bash
npm ci
npm test
npm run verify:package
```

## Publicação no npm pelo GitHub Codespaces

Depois de enviar os arquivos atualizados ao GitHub, abra um Codespace na raiz
do repositório e execute:

```bash
npm ci
npm test
npm publish --dry-run
npm whoami
npm publish
```

O `prepack` executa novamente os testes e valida os arquivos publicáveis antes
do envio. Se sua conta npm exigir autenticação de dois fatores, o comando de
publicação solicitará o código correspondente.

Confirme a versão publicada com:

```bash
npm view n8n-nodes-universal-chatmodel version
```

O workflow `.github/workflows/ci.yml`:

1. instala dependências com `npm ci`;
2. executa build e todos os testes;
3. verifica a estrutura publicável;
4. executa `npm publish --dry-run`, sem publicar ou exigir token npm.

## Segurança

- Nunca faça commit de API keys, tokens do n8n, `.env`, banco SQLite ou pasta
  `.n8n`.
- Revise os logs e execution data antes de compartilhá-los.
- Atualize dependências somente depois de validar novamente no n8n 2.32.6.
- Tool execution acontece no AI Agent/MCP server. O Chat Model trata erros de
  API e protocolo, mas não substitui logs e observabilidade das ferramentas.

## Licença

[MIT](LICENSE)
