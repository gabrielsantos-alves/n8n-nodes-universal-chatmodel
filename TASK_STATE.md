# TASK_STATE: Fix n8n Node Loading & Module Collisions

## Objetivo
Resolver o erro de carregamento e atualização do pacote no n8n de produção (`Error updating package "n8n-nodes-universal-chatmodel": The specified package could not be loaded` / `Cannot find module '.../@langchain/core/node_modules/zod/v4/core/index.cjs'`).

## Causa Raiz Confirmada pela Stack Trace
1. Na v1.0.37 em produção, `zod@4.x` estava aninhado em `node_modules/@langchain/core/node_modules/zod` (porque na raiz do n8n havia `zod@3.25.67` do `token-guard`). O CJS Loader do Node.js memorizou esse `package.json` no cache interno em memória (`packageJsonCache`).
2. Na atualização para v1.0.39 (que adicionou `@langchain/anthropic`), o `npm` deduplicou `zod@4.x` movendo-o de `node_modules/@langchain/core/node_modules/zod` para `node_modules/zod`.
3. Sem reiniciar o container do n8n, ao carregar `UniversalChatModel.node.js`, os imports top-level de `@langchain/core/messages` disparavam `require('zod/v4/core')`. O `resolveExports` do Node.js lia o `package.json` obsoleto do cache da RAM e tentava acessar `.../@langchain/core/node_modules/zod/v4/core/index.cjs` (que não existia mais no disco), lançando `MODULE_NOT_FOUND` sem continuar a busca em `parent.paths`.

## Passos Concluídos
- [x] 1. Isolar `augmentAnthropicMessage` e `parsePayload` em `nodes/UniversalChatModel/AnthropicMessageAugmenter.ts`.
- [x] 2. Reexportar `augmentAnthropicMessage` em `nodes/UniversalChatModel/AnthropicChatModel.ts`.
- [x] 3. Atualizar `nodes/UniversalChatModel/UniversalChatModelTracing.ts` para importar de `AnthropicMessageAugmenter`.
- [x] 4. Implementar Lazy Loading completo em `nodes/UniversalChatModel/UniversalChatModel.node.ts` (zero imports de `@langchain/*` no top-level durante a validação de instalação/atualização do n8n).
- [x] 5. Implementar `installStalePackageCacheFallback` (`Module._resolveFilename`) e `ensureNestedNodeModulesFallbacks` (symlinks) em `nodes/UniversalChatModel/AgentOutputBridge.ts` para contornar o cache de `package.json` do Node.js após hot-updates do `npm`.
- [x] 6. Fixar `@anthropic-ai/sdk: ^0.115.0` e incrementar versão para `1.0.41` em `package.json`.
- [x] 7. Adicionar teste reproduzindo o bug do `packageJsonCache` do Node.js durante hot-update em `tests/universal-chat-model.test.cjs` e atualizar `CHANGELOG.md`.
- [x] 8. Validar suíte completa (`npm test`: 108/108 testes passando).

## Estado Atual
Pronto para publicação da versão `1.0.41`.
