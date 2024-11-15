import { Timestamp } from "bson";
import mongoose from "mongoose";

const summarySchema = new mongoose.Schema({
  meetingID: {
     type: mongoose.Schema.Types.ObjectId, ref: "Meeting", required: true },
  content: { 
    type: String, required: true 
  },
  createdAt: { 
    type: Date, default: Date.now 
  }
});

export const summary=mongoose.model("summary",summarySchema)
