// controllers/meetingController.js

import { Meeting } from "../models/meetingSchema.js";
import asyncHandler from "../utils/asyncHandler.js";
import ApiError from "../utils/apiError.js";
import apiResponse from "../utils/apiResponse.js";

// Create Meeting
export const createMeeting = asyncHandler(async (req, res, next) => {
  const { meetingID, title, host, participants, startTime, endTime } = req.body;

  const newMeeting = await Meeting.create({
    meetingID,
    title,
    host,
    participants,
    startTime,
    endTime,
  });

  apiResponse(res, 201, true, "Meeting created successfully", newMeeting);
});

// Get Meeting by ID
export const getMeetingById = asyncHandler(async (req, res, next) => {
  const meeting = await Meeting.findById(req.params.id)
    .populate("participants")
    .populate("transcript")
    .populate("summary");

  if (!meeting) {
    return next(new ApiError(404, "Meeting not found"));
  }

  apiResponse(res, 200, true, "Meeting found", meeting);
});
