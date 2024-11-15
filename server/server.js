const express = require("express");
const app = express();

const PORT = process.env.PORT || 3030;
const server = require("http").Server(app);
const { v4: uuidv4 } = require("uuid");
const io = require("socket.io")(server);
const { ExpressPeerServer } = require("peer");
const url = require("url");
const peerServer = ExpressPeerServer(server, {
    debug: true,
});
const path = require("path");


import connectDB from './db.js';
import { createUser, findUserByEmail } from './controllers/userController.js';
import { createMeeting, getMeetingById } from './controllers/meetingController.js';
import { createTranscript } from './controllers/transcriptController.js';
import { createSummary } from './controllers/summaryController.js';
import ApiError from './utils/apiError.js';
import asyncHandler from './utils/asyncHandler.js';

// Connect to MongoDB
connectDB();

// API Routes
app.post('/create-user', asyncHandler(createUser));
app.get('/find-user/:email', asyncHandler(findUserByEmail));
app.post('/create-meeting', asyncHandler(createMeeting));
app.get('/get-meeting/:id', asyncHandler(getMeetingById));
app.post('/create-transcript', asyncHandler(createTranscript));
app.post('/create-summary', asyncHandler(createSummary));

// Error handling middleware
app.use((err, req, res, next) => {
  if (err instanceof ApiError) {
    return apiResponse(res, err.statusCode, false, err.message);
  }
  apiResponse(res, 500, false, 'Internal Server Error');
});


app.set("view engine", "ejs");
app.use("/public", express.static(path.join(__dirname, "static")));
app.use("/peerjs", peerServer);

app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "static", "index.html"));
});

app.get("/join", (req, res) => {
    res.redirect(
        url.format({
            pathname: `/join/${uuidv4()}`,
            query: req.query,
        })
    );
});

app.get("/joinold", (req, res) => {
    res.redirect(
        url.format({
            pathname: req.query.meeting_id,
            query: req.query,
        })
    );
});

app.get("/join/:rooms", (req, res) => {
    res.render("room", { roomid: req.params.rooms, Myname: req.query.name });
});

io.on("connection", (socket) => {
    socket.on("join-room", (roomId, id, myname) => {
        socket.join(roomId);
        socket.to(roomId).emit("user-connected", id, myname);



        socket.on("tellName", (myname) => {
            console.log(myname);
            socket.to(roomId).emit("AddName", myname);

        });

        socket.on("disconnect", () => {
            socket.to(roomId).emit("user-disconnected", id);
        });
    });
});

server.listen(PORT);