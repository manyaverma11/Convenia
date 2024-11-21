const mongoose = require("mongoose");

const meetingSchema = new mongoose.Schema(
  {
    meetingID: { type: String, required: true, unique: true },
    title: { type: String, required: true },
    host: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
    participants: [{ type: mongoose.Schema.Types.ObjectId, ref: "User" }],
    startTime: { type: Date, required: true, default: Date.now },
    endTime: { type: Date },
    transcript: { type: mongoose.Schema.Types.ObjectId, ref: "Transcript" },
    summary: { type: mongoose.Schema.Types.ObjectId, ref: "Summary" },
  },
  { timestamps: true }
);

const Meeting = new mongoose.model("Meeting", userSchema);
module.exports = Meeting;
