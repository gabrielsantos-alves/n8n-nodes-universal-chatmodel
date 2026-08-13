class AgentV2 {
  async execute(...args) {
    return this.run(...args);
  }
}

module.exports = { AgentV2 };
