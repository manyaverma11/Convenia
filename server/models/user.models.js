import { Timestamp } from "bson";
import mongoose from "mongoose";

const userSchema= new mongoose.Schema({
  name:{
    type:String,
    unique:true,
    reqired:true
  },
  
  email: { 
    type: String, required: true, unique: true 
  },

  createdAt: { 
    type: Date, default: Date.now 
  },

  meetings: [{ 
    type: mongoose.Schema.Types.ObjectId, ref: "Meeting" 
  }]

},{Timestamp:true});

export const user=mongoose.model("user",userSchema);