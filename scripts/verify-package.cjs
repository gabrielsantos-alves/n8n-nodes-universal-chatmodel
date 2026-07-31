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

if (packageJson.engines?.node !== '>=22.22 <25') {
  throw new Error('Node.js compatibility must remain aligned with n8n 2.32.6.');
}

if (packageJson.peerDependencies?.['n8n-workflow'] !== '>=2.32.1 <3.0.0') {
  throw new Error('n8n-workflow peer compatibility must target n8n 2.32.6.');
}

if (
  packageJson.repository?.url !==
  'git+https://github.com/gabrielsantos-alves/n8n-nodes-universal-chatmodel.git'
) {
  throw new Error('GitHub repository metadata is missing or incorrect.');
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
