const { createTranscript } = require("../controllers/transcript.controllers");

const express = require("express");
const router = express.Router();

router.post("/transcript", createTranscript);

module.exports = router;
