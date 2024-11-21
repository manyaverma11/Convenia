const mongoose = require("mongoose");

const transcriptSchema = new mongoose.Schema(
  {
    meetingID: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Meeting",
      required: true,
    },
    url: {
      type: String,
      required: true,
    },
    createdAt: { type: Date, default: Date.now },
  },
  { timestamps: true }
);

const Transcript = new mongoose.model("Transcript", userSchema);
module.exports = Transcript;
