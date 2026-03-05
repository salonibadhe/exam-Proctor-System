import mongoose from 'mongoose';
import bcrypt from 'bcryptjs';

const userSchema = new mongoose.Schema({
  fullName: {
    type: String,
    required: true,
    trim: true
  },
  email: {
    type: String,
    required: true,  // Now required for both teachers and students
    unique: true,
    lowercase: true,
    trim: true
  },
  mobile: {
    type: String,
    required: function () {
      return this.role === 'teacher';
    },
    unique: true,
    sparse: true,
    trim: true,
    validate: {
      validator: function (v) {
        return /^[0-9]{10}$/.test(v);
      },
      message: 'Mobile number must be 10 digits'
    }
  },
  password: {
    type: String,
    required: true,  // Now required for both teachers and students
    minlength: 6
  },
  role: {
    type: String,
    enum: ['teacher', 'student'],
    required: true
  },
  profilePhoto: {
    type: String, // Base64 encoded image string
    required: false
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

// Hash password before saving
userSchema.pre('save', async function (next) {
  if (!this.isModified('password')) {
    next();
  }
  const salt = await bcrypt.genSalt(10);
  this.password = await bcrypt.hash(this.password, salt);
});

// Method to compare password
userSchema.methods.matchPassword = async function (enteredPassword) {
  return await bcrypt.compare(enteredPassword, this.password);
};

const User = mongoose.model('User', userSchema);

export default User;
