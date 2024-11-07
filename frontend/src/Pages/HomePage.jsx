import React, { useState , useRef, useEffect} from "react";
import './HomePage.css'

const HomePage = () => {

  const[isDropdown, setDropdown] = useState(false);
  const toggleDropdown = () => {
    setDropdown(!isDropdown);
  };

  const dropdownRef = useRef(null); // Ref for the dropdown element

  const handleClickOutside = (event) => {
    // Check if the click is outside the dropdown
    if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
      setDropdown(false);
    }
  };

  useEffect(() => {
    if (isDropdown) {
      document.addEventListener("mousedown", handleClickOutside);
    } else {
      document.removeEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isDropdown]);

  return (
    <div>
      <div className="logo_container">Convenia</div>
      <div className="intro_container">Where Conversations Turn into Connections</div>
      <div className="sub_intro_container">Seamless video calls, powerful collaborations, and instant AI insights with Convenia</div>
      <div className="buttons_container">
        <div className="dropdown" ref={dropdownRef}>
          <button className="create_meet_container" onClick={toggleDropdown}>
            Create meeting
          </button>
          {isDropdown && (
            <div className="dropdown_menu" >
              <button className="dropdown_item">Start an instant meeting</button>
              <button className="dropdown_item">Create a meeting for later</button>
            </div>
          )}
        </div>
        <input type="text" name="myInput" placeholder="Enter meeting code" />
        <button className="join_container">Join</button>
      </div>
    </div>
  );
};

export default HomePage;
