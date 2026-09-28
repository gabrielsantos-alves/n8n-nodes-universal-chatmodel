const executor = require('../agents/ToolsAgent/V2/execute.js');

class AgentV2 {
  async execute(...args) {
    return executor.toolsAgentExecute.call(this, ...args);
  }
}

module.exports = { AgentV2 };
