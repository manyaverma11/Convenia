const mongoose = require("mongoose");

const transcriptSchema = new mongoose.Schema(
  {
    meetingID: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Meeting",
      required: true,
    },
    url: { type: String, required: true },
  },
  { timestamps: true }
);

const Transcript = mongoose.model("Transcript", transcriptSchema);
module.exports = Transcript;
