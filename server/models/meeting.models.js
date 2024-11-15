import { Timestamp } from "bson";
import mongoose from "mongoose";

const meetingSchema=new mongoose.Schema({
  meetingID: { type: String, required: true, unique: true },
    title: { type: String, required: true },
    host: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
    participants: [{ type: mongoose.Schema.Types.ObjectId, ref: "User" }],
    startTime: { type: Date, required: true },
    endTime: { type: Date },
    createdAt: { type: Date, default: Date.now },
    transcript: { type: mongoose.Schema.Types.ObjectId, ref: "Transcript" },
    summary: { type: mongoose.Schema.Types.ObjectId, ref: "Summary" }
},{Timestamp:true});

export const meeting=mongoose.model("meeting",meetingSchema)