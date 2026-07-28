const { rmSync } = require('node:fs');
const { dirname, resolve } = require('node:path');

const projectRoot = resolve(__dirname, '..');
const distDirectory = resolve(projectRoot, 'dist');

if (dirname(distDirectory) !== projectRoot) {
  throw new Error('Refusing to clean a path outside the project root.');
}

rmSync(distDirectory, { recursive: true, force: true });
