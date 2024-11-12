const { v4: uuidv4 } = require("uuid");

exports.getJoinUrl = (req, res) => {
  res.json({
    joinUrl: `/join/${uuidv4()}`,
  });
};

exports.getOldJoinUrl = (req, res) => {
  res.json({
    joinUrl: `/join/${req.params.meeting_id}`,
    name: req.query.name,
  });
};

exports.joinRoom = (req, res) => {
  res.json({
    roomId: req.params.rooms,
    name: req.query.name,
  });
};
