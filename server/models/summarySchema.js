const mongoose = require("mongoose");

const summarySchema = new mongoose.Schema(
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
    createdAt: {
      type: Date,
      default: Date.now,
    },
  },
  { timestamps: true }
);

const Summary = new mongoose.model("Summary", userSchema);
module.exports = Summary;
