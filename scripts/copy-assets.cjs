const { copyFileSync, mkdirSync } = require('node:fs');
const { resolve } = require('node:path');

const projectRoot = resolve(__dirname, '..');
const sourceDirectory = resolve(projectRoot, 'nodes', 'UniversalChatModel');
const destinationDirectory = resolve(
  projectRoot,
  'dist',
  'nodes',
  'UniversalChatModel',
);

mkdirSync(destinationDirectory, { recursive: true });

for (const file of [
  'UniversalChatModel.node.json',
  'universalChatModel.svg',
]) {
  copyFileSync(
    resolve(sourceDirectory, file),
    resolve(destinationDirectory, file),
  );
}
