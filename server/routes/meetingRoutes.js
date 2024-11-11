const express = require("express");
const { getJoinUrl, getOldJoinUrl, joinRoom } = require("../controllers/meetingControllers");

const router = express.Router();

router.get("/join", getJoinUrl);
router.get("/joinold/:meeting_id", getOldJoinUrl);
router.get("/join/:rooms", joinRoom);

module.exports = router;
