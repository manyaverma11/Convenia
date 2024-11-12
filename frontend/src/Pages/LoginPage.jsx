import Button from "react-bootstrap/Button";
import { useState } from "react";
import Form from "react-bootstrap/Form";
import "./LoginPage.css";

const LoginPage = () => {
  const [signUp, setSignUp] = useState(false);

  const myFunction = () => {
    var x = document.getElementById("pwd");
    if (x.type === "password") {
      x.type = "text";
    } else {
      x.type = "password";
    }
  };

  return (
    <div className="outer_container">
      <div className="convenia"> Convenia</div>
      <div className="content">
        <p className="intro1">Welcome to Convenia </p>
        <p className="intro2"> Your Hub for Seamless Communication</p>
        <p className="intro3">
          Log in to access your Convenia workspace, where real-time meetings,
          collaborative chat, and AI-powered tools bring your team closer, no
          matter where you are. With secure access, integrated features, and
          automated AI transcripts, Convenia helps you focus on what matters -
          effective communication and productivity.
        </p>
      </div>
      <Form>
        {!signUp ? (
          <div id="login">
            <div className="login-head">Login</div>
            <Form.Group className="mb-3" controlId="formBasicEmail">
              <input type="email" name="myInput" placeholder="Email" />
            </Form.Group>
            <Form.Group className="mb-3" controlId="formBasicPassword">
              <input type="password" placeholder="Password" id="pwd" />
            </Form.Group>
            {/* <input type="checkbox" onClick={myFunction} /> Show Password */}
            <div className="fpwd">Forgot Password</div>
            <button className="continue_button" type="submit">
              Continue
            </button>
            <div className="signup">
              Don't have an account yet?{" "}
              <span
                className="signup-link"
                onClick={() => setSignUp(true)}
                style={{ cursor: "pointer", color: "#00d4ff" }}
              >
                Sign up
              </span>
            </div>
          </div>
        ) : (
          <div id="signup">
            <div className="login-head">Sign Up</div>
            <Form.Group className="mb-3" controlId="formSignUpEmail">
              <input type="email" placeholder="Email" />
            </Form.Group>
            <Form.Group className="mb-3" controlId="formSignUpPassword">
              <input type="password" placeholder="Password" />
            </Form.Group>
            <Form.Group className="mb-3" controlId="formConfirmPassword">
              <input type="password" placeholder="Confirm Password" />
            </Form.Group>
            <button className="continue_button" type="submit">
              Sign Up
            </button>
            <div className="signup">
              Already have an account?{" "}
              <span className="signup-link" onClick={() => setSignUp(false)}>
                Log in
              </span>
            </div>
          </div>
        )}
      </Form>
    </div>
  );
};

export default LoginPage;
