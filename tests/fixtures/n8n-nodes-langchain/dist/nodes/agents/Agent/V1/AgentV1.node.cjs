class AgentV1 {
  async execute(...args) {
    return this.run(...args);
  }
}

module.exports = { AgentV1 };
