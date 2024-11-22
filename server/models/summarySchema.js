const mongoose = require("mongoose");

const summarySchema = new mongoose.Schema(
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

const Summary = mongoose.model("Summary", summarySchema);
module.exports = Summary;
