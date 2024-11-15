// controllers/summaryController.js

import { Summary } from "../models/summarySchema.js";
import asyncHandler from "../utils/asyncHandler.js";
import ApiError from "../utils/apiError.js";
import apiResponse from "../utils/apiResponse.js";

// Create Summary
export const createSummary = asyncHandler(async (req, res, next) => {
  const { meetingID, content } = req.body;

  const newSummary = await Summary.create({ meetingID, content });

  apiResponse(res, 201, true, "Summary created successfully", newSummary);
});
