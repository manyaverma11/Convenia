const express = require("express");
const app = express();
require("cookie-parser");
const fs = require("fs");
const path = require("path");
const https = require("https");
const AWS = require("aws-sdk");

AWS.config.update({
  region: "https://19cd51f4383fbe15726448aace900109.r2.cloudflarestorage.com/", // e.g., 'us-west-2'
  accessKeyId: "d5269b9726b4c711cfdd975ab0c29492",
  secretAccessKey:
    "a6aecf0669a4abbe82b591255646071f2c819fcc0d5a3f8344bb75b4b7d029fa",
});

const s3 = new AWS.S3();

app.get("/generate-presigned-url", (req, res) => {
  const params = {
    Bucket: "convenia",
    Key: `uploads/${Date.now()}.mp4`, // Expiry time of the URL in seconds
    ContentType: "video/mp4", // Set appropriate content type
    ACL: "public-read", // or any ACL you require
  };

  s3.getSignedUrl("putObject", params, (err, url) => {
    if (err) {
      console.log(err);
      return res.status(500).json({ error: "Error creating signed URL" });
    }
    console.log(url);
    res.json({ url, key: params.Key });
  });
});

const { v4: uuidV4 } = require("uuid");
const bodyParser = require("body-parser");
require("dotenv").config();

//const server = require('http').Server(app)
const server = https.createServer(
  {
    key: fs.readFileSync(path.join(__dirname, "cert", "key.pem")),
    cert: fs.readFileSync(path.join(__dirname, "cert", "cert.pem")),
  },
  app
);

const io = require("socket.io")(server);
const PORT = process.env.PORT || 8080;

const router = require("./routes/user.router");
require("./config/db-connection");

app.set("view engine", "ejs");
app.use(express.static("public"));

app.use(express.json());
app.use(bodyParser.urlencoded({ extended: true }));

io.on("connection", (socket) => {
  socket.on("join-room", (roomId, userId) => {
    socket.join(roomId);
    socket.to(roomId).emit("user-connected", userId);

    socket.on("send-chat", (message, username) => {
      io.to(roomId).emit("show-to-room", message, username);
    });

    socket.on("disconnect", () => {
      socket.to(roomId).emit("user-disconnected", userId);
    });
  });
});

app.use(router);
// const meetingRouter = require("./routes/meeting.router");

// // Existing setup...

// app.use(meetingRouter); // Add the meeting routes
const meetingRouter = require("./routes/meeting.router");
const transcriptRouter = require("./routes/transcript.router");

app.use(meetingRouter);
app.use(transcriptRouter);

server.listen(PORT, () => {
  console.log(`server is running at ${PORT}`);
});
