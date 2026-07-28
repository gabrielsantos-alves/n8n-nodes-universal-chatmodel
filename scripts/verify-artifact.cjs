const { execFileSync } = require('node:child_process');
const {
  existsSync,
  readFileSync,
  readdirSync,
  statSync,
} = require('node:fs');
const { basename, resolve } = require('node:path');

const projectRoot = resolve(__dirname, '..');
const packageJson = JSON.parse(
  readFileSync(resolve(projectRoot, 'package.json'), 'utf8'),
);
const expectedName = `${packageJson.name}-${packageJson.version}.tgz`;
const argument = process.argv[2];
const artifact = argument
  ? resolve(projectRoot, argument)
  : resolve(projectRoot, expectedName);

if (!existsSync(artifact) || !statSync(artifact).isFile()) {
  const available = readdirSync(projectRoot)
    .filter((name) => name.endsWith('.tgz'))
    .join(', ');
  throw new Error(
    `Artifact ${basename(artifact)} was not found.` +
      (available ? ` Available: ${available}` : ''),
  );
}

const entries = execFileSync('tar', ['-tzf', artifact], {
  cwd: projectRoot,
  encoding: 'utf8',
})
  .split(/\r?\n/)
  .map((entry) => entry.trim().replace(/\\/g, '/'))
  .filter(Boolean);
const entrySet = new Set(entries);
const expectedFiles = [
  'package/package.json',
  'package/README.md',
  'package/CHANGELOG.md',
  'package/LICENSE',
  `package/${packageJson.main}`,
  `package/${packageJson.types}`,
  ...packageJson.n8n.nodes.map((file) => `package/${file}`),
  ...packageJson.n8n.credentials.map((file) => `package/${file}`),
  'package/dist/nodes/UniversalChatModel/UniversalChatModel.node.json',
  'package/dist/nodes/UniversalChatModel/universalChatModel.svg',
];
const missing = expectedFiles.filter((file) => !entrySet.has(file));

if (missing.length > 0) {
  throw new Error(`Artifact is missing: ${missing.join(', ')}`);
}

const forbidden = entries.filter(
  (entry) =>
    entry.includes('/node_modules/') ||
    entry.startsWith('package/tests/') ||
    entry.startsWith('package/nodes/') ||
    entry.startsWith('package/credentials/') ||
    entry.endsWith('/.env') ||
    entry.endsWith('/.npmrc') ||
    /(?:api[_-]?key|access[_-]?token|secret)\.(?:txt|json)$/i.test(entry),
);

if (forbidden.length > 0) {
  throw new Error(
    `Artifact contains development or sensitive files: ${forbidden.join(', ')}`,
  );
}

console.log(
  `Artifact ${basename(artifact)} is clean (${entries.length} files, ${expectedFiles.length} required files verified).`,
);
