const { existsSync, readFileSync } = require('node:fs');
const { resolve } = require('node:path');

const projectRoot = resolve(__dirname, '..');
const packageJson = JSON.parse(
  readFileSync(resolve(projectRoot, 'package.json'), 'utf8'),
);

const expectedFiles = [
  packageJson.main,
  packageJson.types,
  ...packageJson.n8n.nodes,
  ...packageJson.n8n.credentials,
  'dist/nodes/UniversalChatModel/UniversalChatModel.node.json',
  'dist/nodes/UniversalChatModel/universalChatModel.svg',
];

if (!packageJson.name.startsWith('n8n-nodes-')) {
  throw new Error('The package name must start with "n8n-nodes-".');
}

if (!packageJson.keywords.includes('n8n-community-node-package')) {
  throw new Error(
    'The package must include the "n8n-community-node-package" keyword.',
  );
}

const missingFiles = expectedFiles.filter(
  (file) => !existsSync(resolve(projectRoot, file)),
);

if (missingFiles.length > 0) {
  throw new Error(`Missing release files: ${missingFiles.join(', ')}`);
}

console.log(
  `Package ${packageJson.name}@${packageJson.version} is ready (${expectedFiles.length} release files verified).`,
);
