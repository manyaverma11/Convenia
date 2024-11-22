const Transcript = require("../models/transcriptSchema");

const createTranscript = async (req, res) => {
  const { meetingID, url } = req.body;

  if (!meetingID || !url) {
    return res.status(400).send("Meeting ID and URL are required.");
  }

  try {
    const newTranscript = new Transcript({ meetingID, url });
    const savedTranscript = await newTranscript.save();

    res.status(200).send(savedTranscript);
  } catch (error) {
    console.error(error);
    res.status(500).send("Failed to save transcript.");
  }
};

module.exports = { createTranscript };
