import mongoose from 'mongoose';

const otpSchema = new mongoose.Schema({
  mobile: {
    type: String,
    required: true,
    trim: true
  },
  otp: {
    type: String,
    required: true
  },
  createdAt: {
    type: Date,
    default: Date.now,
    expires: 300 // OTP expires after 5 minutes (300 seconds)
  },
  verified: {
    type: Boolean,
    default: false
  }
});

const OTP = mongoose.model('OTP', otpSchema);

export default OTP;
