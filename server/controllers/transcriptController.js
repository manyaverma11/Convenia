// controllers/transcriptController.js

import { Transcript } from "../models/transcriptSchema.js";
import asyncHandler from "../utils/asyncHandler.js";
import ApiError from "../utils/apiError.js";
import apiResponse from "../utils/apiResponse.js";

// Create Transcript
export const createTranscript = asyncHandler(async (req, res, next) => {
  const { meetingID, content } = req.body;

  const newTranscript = await Transcript.create({ meetingID, content });

  apiResponse(res, 201, true, "Transcript created successfully", newTranscript);
});
