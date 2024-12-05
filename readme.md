# Convenia

Convenia is a real-time video conferencing application that facilitates seamless communication with features like secure room creation, peer-to-peer connections, and dynamic meeting management. Built using modern web technologies, it enables private and efficient collaboration.

---

## Features

- **Real-Time Video Conferencing**: Supports multiple participants in video rooms.
- **Room Management**: Unique room IDs for secure and private meetings.
- **Dynamic UI Rendering**: Uses EJS templates for dynamic content rendering.
- **WebRTC Integration**: Peer-to-peer audio, video, and data streaming via the Peer.js library.
- **Real-Time Communication**: Socket.io for instant updates and chat features.
- **Secure Tunneling**: Ngrok integration for sharing secure public URLs during development or testing.

---

## Technologies Used

### Frontend:

- **HTML/CSS**: Structure and styling.
- **JavaScript**: Dynamic and interactive components.
- **EJS**: Template engine for server-side rendering.

### Backend:

- **Node.js**: Server-side runtime environment.
- **Express.js**: Framework for routing and middleware.
- **Socket.io**: Real-time communication.
- **Peer.js**: WebRTC wrapper for peer-to-peer connections.

### Database:

- **MongoDB**: NoSQL database for storing user and meeting data.

### Other Tools:

- **Ngrok**: For exposing localhost to the internet.
- **UUID**: Generating unique room IDs.

---

## Setup Instructions

### Prerequisites

- Node.js installed on your system.
- MongoDB setup (local or cloud-based).
- Ngrok installed (optional for development).

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/manyaverma11/Convenia.git
   cd Convenia
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Configure environment variables:
   Create a `.env` file in the root directory and add the following:

   ```env
   MONGODB_URI=your-mongodb-uri
   PORT=3000
   ```

4. Start the server:

   ```bash
   npm start
   ```

5. (Optional) Run Ngrok to expose your local server:
   ```bash
   ngrok http 3000
   ```

---

## Usage

1. Access the application at `http://localhost:3000` (or Ngrok URL for public access).
2. Generate a unique meeting room link and share it with participants.
3. Join the room for real-time video conferencing.

---

## File Structure

```
Convenia/
│
├── public/           # Static assets (CSS, JS, images)
├── views/            # EJS templates for frontend rendering
├── server/           # Core backend logic
├── .env              # Environment variables
├── package.json      # Node.js dependencies and scripts
└── README.md         # Project documentation
```

---

## Future Enhancements

- **AI-Powered Features**: Automated meeting transcriptions and summaries.
- **User Authentication**: Secure login/signup functionality.
- **Chat Enhancements**: Persistent chat history and media sharing.

---

## Contributing

We welcome contributions! Please fork the repository, make your changes, and submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---
