import { useState } from 'react'
import './App.css'
import HomePage from './Pages/HomePage.jsx'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <HomePage />
    </>
  )
}

export default App
