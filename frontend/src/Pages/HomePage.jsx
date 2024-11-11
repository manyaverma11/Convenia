import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./HomePage.css";

const HomePage = () => {
  const [meetingCode, setMeetingCode] = useState("");
  const [participantName, setParticipantName] = useState("");
  const navigate = useNavigate();

  const createMeeting = async () => {
    if (!participantName.trim()) {
      alert("Please enter your name.");
    } else {
      try {
        const response = await fetch(`/join`);
        const data = await response.json();
        navigate(`${data.joinUrl}?name=${encodeURIComponent(participantName)}`);
      } catch (error) {
        console.error("Error creating a new meeting:", error);
      }
    }
  };

  const joinMeeting = () => {
    if (meetingCode.trim() && participantName.trim()) {
      navigate(`/joinold/${meetingCode}?name=${encodeURIComponent(participantName)}`);
    } else {
      alert("Please enter both a valid meeting code and your name.");
    }
  };

  return (
    <div>
      <div className="logo_container">Convenia</div>
      <div className="intro_container">Where Conversations Turn into Connections</div>
      <div className="sub_intro_container">
        Seamless video calls, powerful collaborations, and instant AI insights with Convenia
      </div>
      <div className="buttons_container">
        <button className="create_meet_container" onClick={createMeeting}>
          Create Meeting
        </button>
        <input
          type="text"
          placeholder="Enter meeting code"
          value={meetingCode}
          onChange={(e) => setMeetingCode(e.target.value)}
        />
        <input
          type="text"
          placeholder="Enter your name for the meeting"
          value={participantName}
          onChange={(e) => setParticipantName(e.target.value)}
        />
        <button className="join_container" onClick={joinMeeting}>
          Join
        </button>
      </div>
    </div>
  );
};

export default HomePage;
