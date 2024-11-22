const Meeting = require("../models/meetingSchema");
const User = require("../models/userSchema");

// Create a new meeting
const createMeeting = async (req, res) => {
  const { meetingTitle } = req.body;
  const userId = req.user._id;

  if (!meetingTitle || meetingTitle.trim() === "") {
    return res.status(400).send("Meeting title is required!");
  }

  try {
    const meetingID = Math.random().toString(36).substr(2, 8); // Generate a random meeting ID
    const newMeeting = new Meeting({
      meetingID,
      title: meetingTitle,
      host: userId,
      participants: [userId],
    });

    const savedMeeting = await newMeeting.save();

    // Add the meeting to the user's list
    await User.findByIdAndUpdate(userId, {
      $push: { meetings: savedMeeting._id },
    });

    res.redirect(`/user/${meetingID}`); // Redirect to the meeting room
  } catch (error) {
    console.error(error);
    res.status(500).send("Failed to create meeting.");
  }
};

// Get meetings for a user
const getMeetingsForUser = async (req, res) => {
  try {
    const userId = req.user._id;
    const user = await User.findById(userId).populate("meetings");

    if (!user) {
      return res.status(404).send("User not found.");
    }

    res.render("meetings", { meetings: user.meetings });
  } catch (error) {
    console.error(error);
    res.status(500).send("Failed to fetch meetings.");
  }
};

module.exports = { createMeeting, getMeetingsForUser };
