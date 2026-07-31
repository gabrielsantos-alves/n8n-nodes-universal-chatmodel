const {
  UniversalChatModel,
} = require('../../../dist/nodes/UniversalChatModel/UniversalChatModel.node.js');

const settings = new UniversalChatModel().description.properties
  .filter((property) => property.isNodeSetting)
  .map((property) => property.name);

process.stdout.write(JSON.stringify(settings));
