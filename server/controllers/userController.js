// controllers/userController.js

import { User } from "../models/userSchema.js";
import asyncHandler from "../utils/asyncHandler.js";
import ApiError from "../utils/apiError.js";
import apiResponse from "../utils/apiResponse.js";

// Create User
export const createUser = asyncHandler(async (req, res, next) => {
  const { name, email } = req.body;

  const existingUser = await User.findOne({ email });
  if (existingUser) {
    return next(new ApiError(400, "User already exists"));
  }

  const newUser = await User.create({ name, email });
  apiResponse(res, 201, true, "User created successfully", newUser);
});

// Find User by Email
export const findUserByEmail = asyncHandler(async (req, res, next) => {
  const { email } = req.params;
  const user = await User.findOne({ email });

  if (!user) {
    return next(new ApiError(404, "User not found"));
  }

  apiResponse(res, 200, true, "User found", user);
});
