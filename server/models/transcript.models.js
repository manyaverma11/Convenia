import { Timestamp } from "bson";
import mongoose from "mongoose";

const transcriptSchema = new mongoose.Schema({
  meetingID: { 
    type: mongoose.Schema.Types.ObjectId, ref: "Meeting", required: true
   },
  content: {
     type: String, required: true 
  },
  createdAt: { type: Date, default: Date.now }
});

export const transcript=mongoose.model("transcript",transcriptSchema)