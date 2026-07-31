# Changelog

Todas as mudanças relevantes deste projeto serão registradas aqui.

## 1.0.26 — 2026-07-31

- Atualiza o alvo de produção para **n8n 2.32.6** e Node.js 22.22–24.x.
- Alinha `n8n-workflow` a 2.32.1, versão usada pelo n8n 2.32.6.
- Valida interoperabilidade com `@n8n/n8n-nodes-langchain` 2.32.4 e o
  `@langchain/core` 1.2.0 usado por esse pacote.
- Atualiza `@langchain/core` para 1.2.4 e `@langchain/openai` para 1.5.5.
- Mantém compatibilidade com o AI Agent V3, tools estruturadas, tool loops,
  streaming, thoughts, token usage e recuperação de resposta final vazia.
- Adiciona metadados do repositório GitHub e CI para validar build, testes e
  `npm publish --dry-run` antes da publicação manual pelo Codespaces.
- Amplia a suíte para 64 testes, incluindo uma verificação explícita do conjunto
  de versões usado pelo n8n 2.32.6.

## 1.0.25 — 2026-07-30

- Força `functionCallingConfig: NONE` somente na recuperação de uma resposta
  terminal vazia, evitando um segundo `STOP` vazio depois de loops longos com
  ferramentas e impedindo a repetição de chamadas com efeitos colaterais.
- Mantém `AUTO` nas chamadas normais do agente e preserva todas as function
  calls válidas antes da recuperação.
- Remove **Base URL (Override)** e **API Key (Override)** do formulário OpenAI
  Compatible; ambos passam a vir exclusivamente da credencial.
- Ignora com segurança os overrides antigos que ainda existam no JSON de
  workflows salvos.
- Valida o comportamento em 63 testes, incluindo loop multi-tool, recuperação,
  streaming, Retry On Fail e credenciais OpenAI Compatible.

## 1.0.24 — 2026-07-30

- Detecta a diferença entre uma geração textual vazia e uma chamada estruturada
  de ferramenta, cujo `text` vazio é esperado.
- Recupera automaticamente uma resposta terminal `STOP` sem texto e sem
  function call com uma única solicitação de continuação.
- Nunca repete uma geração que contenha tool calls, evitando execução duplicada
  de ferramentas com efeitos colaterais.
- Soma os tokens cobrados da tentativa vazia e da recuperação em `tokenUsage`,
  `usageMetadata`, tracing e Usage Reporter.
- Aplica a mesma proteção a chamadas normais e streaming.
- Converte uma segunda resposta vazia ou bloqueada em erro detalhado, em vez de
  encerrar o AI Agent silenciosamente com `output: ""`.
- Adiciona a opção **Recover Empty Final Responses**, ativada por padrão.
- Valida um loop com três tools, resposta terminal vazia, recuperação, safety,
  contagem agregada e streaming em 63 testes.

## 1.0.23 — 2026-07-28

- Integra `Always Output Data`, `Execute Once`, `Retry On Fail`, `Max Tries`,
  `Wait Between Tries` e `On Error` com os campos nativos do node no n8n.
- Faz os indicadores visuais de Always Output Data, Execute Once, Retry On Fail
  e Continue On Error aparecerem no canvas quando ativados.
- Corrige a leitura das Settings em editores novos, nos quais model subnodes não
  recebem automaticamente todas as configurações comuns.
- Mantém compatibilidade com os antigos campos `model*` salvos pelas versões
  anteriores do community node.
- Confirma que o n8n 2.2.5 continua usando suas Settings nativas, sem campos
  duplicados no formulário.
- Amplia a suíte para 57 testes.

## 1.0.22 — 2026-07-28

- Corrige `Fail Workflow if Reporter Fails`: falhas do Usage Reporter agora
  interrompem a execução quando a opção estrita está ativa.
- Garante que uma falha de telemetria após uma resposta bem-sucedida não repita
  a chamada da LLM nem gere cobrança duplicada.
- Amplia a suíte para 55 testes, incluindo reporter opcional sem conexão,
  privacidade do prompt/output, OpenAI-compatible e thoughts ocultos no Agent.
- Valida os manifests reais do n8n 2.2.5, n8n-core 2.2.3 e
  `@n8n/n8n-nodes-langchain` 2.2.3.
- Executa o pipeline do GitLab em Node.js 20.19, 22 e 24.
- Adiciona inspeção do `.tgz` para impedir fontes, testes, `node_modules` ou
  arquivos potencialmente sensíveis no artifact.

## 1.0.21 — 2026-07-28

- Renomeia o campo visível para `Include Token Usage in Output`.
- Remove da interface a associação exclusiva com o AI Agent, já que o Chat
  Model também pode ser conectado a outros nodes compatíveis.
- Mantém o identificador interno anterior para não quebrar workflows salvos.
- Mantém 50 testes aprovados.

## 1.0.20 — 2026-07-28

- Agrupa toda a configuração do reporter em uma única opção `Usage Reporter`.
- Mantém `Enable Usage Reporter` como chave principal dentro do grupo.
- Mostra Node Label, Input Text Mode/Label, Include Output Text e política de
  erro apenas como subopções quando a chave estiver ativada.
- Preserva a leitura dos formatos salvos nas versões 1.0.17, 1.0.18 e 1.0.19.
- Mantém 50 testes aprovados, incluindo compatibilidade com n8n 2.2.5.

## 1.0.19 — 2026-07-28

- Move `System Message`, `Include Token Usage in AI Agent Output` e
  `Enable Usage Reporter` para dentro de `Options`.
- Mantém essas configurações totalmente opcionais nos provedores Gemini e
  OpenAI-compatible.
- Recolhe também Node Label, privacidade do prompt, output text e política de
  erro do Usage Reporter dentro de `Options`.
- Mantém compatibilidade com workflows salvos nas versões 1.0.17 e 1.0.18.
- Valida o novo layout e o comportamento com 50 testes.

## 1.0.18 — 2026-07-28

- Adiciona `Include Token Usage in AI Agent Output` para controlar a exposição
  de tokens no resultado do Agent sem desativar a captura interna.
- Mantém o envio de tokens ao `Usage Reporter` mesmo quando os metadados estão
  ocultos no output do AI Agent.
- Adiciona `Enable Usage Reporter`; o conector de AI Tool só aparece quando a
  opção é ativada, mantendo o Chat Model circular quando desligada.
- Move todas as opções de Usage Reporting para o final da configuração.
- Adiciona `System Message`, combinado depois das instruções de sistema
  recebidas do AI Agent ou de outro node pai.
- Amplia a suíte para 50 testes.

## 1.0.17 — 2026-07-27

- Adiciona a entrada opcional `Usage Reporter` para executar uma AI Tool de
  telemetria após cada chamada bem-sucedida da LLM.
- Envia tokens de input, input sem cache, output, cache, thoughts, tool use,
  overhead e total, além de contexto do workflow e metadados brutos.
- Permite usar um rótulo fixo como `RAG` ou enviar o prompt real.
- Mantém falhas do reporter não bloqueantes por padrão.
- Valida reporting em chamadas normais, streaming e execuções concorrentes.
- Amplia a suíte para 48 testes.

## 1.0.16 — 2026-07-27

- Faz `Include Thoughts` controlar explicitamente toda exposição dos resumos de
  pensamento.
- Mantém thoughts ocultos por padrão no Chat Model, tracing e output do AI
  Agent, mesmo quando Thinking Level ou Thinking Budget estiver configurado.
- Preserva `thoughtsTokens` nas métricas de uso sem revelar o conteúdo do
  pensamento.
- Amplia a suíte para 44 testes.

## 1.0.15 — 2026-07-27

- Converte JSON de Structured Output em campos reais no output do AI Agent.
- Mantém `generation.text` como string para compatibilidade com LangChain.
- Adiciona `structuredOutput` separado ao tracing do Chat Model.
- Preserva o contrato `output` do Chat Trigger quando o schema usa outro nome,
  como `message`.
- Impede prototype pollution ao promover campos estruturados.
- Amplia a suíte para 42 testes.

## 1.0.14 — 2026-07-27

- Prepara o projeto para versionamento e empacotamento pelo GitLab.
- Adiciona pipeline de CI para build, 38 testes e geração do pacote `.tgz`.
- Documenta instalação manual, instalação direta do GitLab e uso em Docker.
- Mantém compatibilidade testada com n8n 2.2.5 e n8n 2.x atual.
- Normaliza erros de API, rede, timeout, schema, safety e ferramentas.
- Implementa retry seletivo para erros transitórios.
- Expõe thoughts disponíveis, consumo detalhado de tokens e metadados no AI Agent.
- Adiciona Structured Output Schema opcional.

## 1.0.13

- Adiciona tratamento detalhado de erros e proteção de dados sensíveis.
- Adiciona Settings equivalentes a Retry On Fail, Always Output Data, Execute
  Once e On Error.

## 1.0.12

- Adiciona Structured Output Schema e melhorias de thoughts e token usage.
- Preserva tool call IDs e thought signatures em agentes com múltiplas tools.
