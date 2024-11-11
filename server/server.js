const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const { ExpressPeerServer } = require("peer");
const path = require("path");

const app = express();
const server = http.Server(app);
const io = new Server(server);
const peerServer = ExpressPeerServer(server, { debug: true });
const PORT = process.env.PORT || 3000;

// Middleware
app.use("/public", express.static(path.join(__dirname, "static")));
app.use("/peerjs", peerServer);

// Routes
const meetingRoutes = require("./routes/meetingRoutes.js");
app.use("/", meetingRoutes);

io.on("connection", (socket) => {
  socket.on("join-room", (roomid, id, myname) => {
    socket.join(roomid);
    socket.to(roomid).broadcast.emit("user-connected", id, myname);

    socket.on("disconnect", () => {
      socket.to(roomid).broadcast.emit("user-disconnected", id);
    });
  });
});

server.listen(PORT, () => console.log(`Server running on port ${PORT}`));
