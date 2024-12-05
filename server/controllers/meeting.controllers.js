const Meeting = require("../models/meetingSchema");
const User = require("../models/userSchema");

// Create a new meeting
const generateRoomId = async (req, res) => {
  let roomid = "";
  let uniqueId = false;
  while (!uniqueId) {
    for (let i = 0; i < 8; i++) {
      roomid += String.fromCharCode(Math.floor(Math.random() * 26) + 97);
    }
    const meeting = await Meeting.findOne({ meetingID: roomid });
    if (!meeting) {
      uniqueId = true;
    }
  }

  return roomid;
};

const createMeeting = async (req, res) => {
  const { meetingTitle } = req.body;
  console.log(meetingTitle);
  const userId = req.user._id;
  const roomId = await generateRoomId();
  if (!meetingTitle || meetingTitle.trim() === "") {
    return res.status(400).send("Meeting title is required!");
  }
  if (!roomId || roomId.trim() === "") {
    return res.status(400).send("Room ID is required!");
  }

  try {
    const newMeeting = new Meeting({
      meetingID: roomId,
      title: meetingTitle,
      host: userId,
      startTime: Date.now(),
    });

    const savedMeeting = await newMeeting.save();

    // Add the meeting to the user's list
    await User.findByIdAndUpdate(userId, {
      $push: { meetings: savedMeeting._id },
    });

    res.status(200).json({ success: true, redirectUrl: `/user/${roomId}` });
  } catch (error) {
    console.error(error);
    res.status(500).send("Failed to create meeting.");
  }
};

// Get meetings for a user
const getMeetingsForUser = async (req, res) => {
  try {
    const userId = req.user._id;
    console.log(userId);
    const user = await User.findById(userId).populate("meetings");

    if (!user) {
      return res.status(404).send("User not found.");
    }
    res.send(user);
  } catch (error) {
    console.error(error);
    res.status(500).send("Failed to fetch meetings.");
  }
};

module.exports = { createMeeting, getMeetingsForUser };
