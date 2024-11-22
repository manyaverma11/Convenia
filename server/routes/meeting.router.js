const express = require("express");
const { authenticate } = require("../middlewares/authenticate");
const {
  createMeeting,
  getMeetingsForUser,
} = require("../controllers/meeting.controllers");

const router = express.Router();

router.use(authenticate);

// Create a new meeting
router.post("/newmeet", createMeeting);

// Get all meetings for a user
router.get("/meetings", getMeetingsForUser);

module.exports = router;
