const mongoose = require("mongoose");

const userSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      unique: true,
      reqired: true,
    },

    email: {
      type: String,
      required: true,
      unique: true,
    },

    createdAt: {
      type: Date,
      default: Date.now,
    },
    password: {
      type: String,
      required: true,
    },

    meetings: [
      {
        type: mongoose.Schema.Types.ObjectId,
        ref: "Meeting",
      },
    ],
  },
  { timestamps: true }
);

const User = new mongoose.model("User", userSchema);
module.exports = User;
