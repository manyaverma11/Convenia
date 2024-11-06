import React from "react";
import './HomePage.css'

const HomePage = () => {
  return (
    <div>
      <div className="logo">Convenia</div>
      <button className="create_meet">Create meeting</button>
      <input type="text" name="myInput" />
      <button className="join">Join</button>
    </div>
  );
};

export default HomePage;
