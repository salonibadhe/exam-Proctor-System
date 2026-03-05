import express from 'express';
import { body, validationResult } from 'express-validator';
import User from '../models/User.js';
import OTP from '../models/OTP.js';
import generateToken from '../utils/generateToken.js';
import { protect } from '../middleware/auth.js';
import { generateOTP, sendOTP } from '../utils/otpService.js';

const router = express.Router();

// @route   POST /api/auth/signup
// @desc    Register a new user
// @access  Public
router.post(
  '/signup',
  [
    body('fullName').notEmpty().withMessage('Full name is required'),
    body('role').isIn(['teacher', 'student']).withMessage('Role must be either teacher or student'),
    body('email').isEmail().withMessage('Valid email is required'),
    body('password').isLength({ min: 6 }).withMessage('Password must be at least 6 characters'),
    body('mobile').if(body('role').equals('teacher')).matches(/^[0-9]{10}$/).withMessage('Valid 10-digit mobile number is required for teachers')
  ],
  async (req, res) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const { fullName, email, password, mobile, role, profilePhoto } = req.body;

      // Check if user exists by email
      const userExists = await User.findOne({ email });
      if (userExists) {
        return res.status(400).json({ message: 'User already exists with this email' });
      }

      // For teachers, also check mobile number
      if (role === 'teacher') {
        const mobileExists = await User.findOne({ mobile });
        if (mobileExists) {
          return res.status(400).json({ message: 'User already exists with this mobile number' });
        }
      }

      // Create user
      const userData = {
        fullName,
        email,
        password,
        role,
        profilePhoto
      };

      // Add mobile for teachers
      if (role === 'teacher') {
        userData.mobile = mobile;
      }

      const user = await User.create(userData);

      if (user) {
        const response = {
          _id: user._id,
          fullName: user.fullName,
          role: user.role,
          token: generateToken(user._id)
        };

        response.email = user.email;
        if (role === 'teacher') {
          response.mobile = user.mobile;
        }

        res.status(201).json(response);
      } else {
        res.status(400).json({ message: 'Invalid user data' });
      }
    } catch (error) {
      console.error(error);
      res.status(500).json({ message: 'Server error' });
    }
  }
);

// @route   POST /api/auth/login
// @desc    Authenticate user and get token
// @access  Public
router.post(
  '/login',
  [
    body('email').isEmail().withMessage('Valid email is required'),
    body('password').notEmpty().withMessage('Password is required')
  ],
  async (req, res) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const { email, password } = req.body;

      // Check for user
      const user = await User.findOne({ email });
      if (!user) {
        return res.status(401).json({ message: 'Invalid credentials' });
      }

      // Check password
      const isMatch = await user.matchPassword(password);
      if (!isMatch) {
        return res.status(401).json({ message: 'Invalid credentials' });
      }

      res.json({
        _id: user._id,
        fullName: user.fullName,
        email: user.email,
        role: user.role,
        token: generateToken(user._id)
      });
    } catch (error) {
      console.error(error);
      res.status(500).json({ message: 'Server error' });
    }
  }
);

// @route   GET /api/auth/me
// @desc    Get current logged in user
// @access  Private
router.get('/me', protect, async (req, res) => {
  try {
    const user = await User.findById(req.user._id).select('-password');
    res.json(user);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Server error' });
  }
});

// @route   POST /api/auth/send-otp
// @desc    Send OTP to teacher's mobile number
// @access  Public
router.post(
  '/send-otp',
  [
    body('mobile')
      .notEmpty()
      .withMessage('Mobile number is required')
      .matches(/^[0-9]{10}$/)
      .withMessage('Mobile number must be 10 digits')
  ],
  async (req, res) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const { mobile } = req.body;

      // Check if teacher exists with this mobile
      const teacher = await User.findOne({ mobile, role: 'teacher' });
      if (!teacher) {
        return res.status(404).json({ message: 'No teacher account found with this mobile number' });
      }

      // Generate OTP
      const otp = generateOTP();

      // Delete any existing OTPs for this mobile
      await OTP.deleteMany({ mobile });

      // Save new OTP
      await OTP.create({ mobile, otp });

      // Send OTP via SMS
      const result = await sendOTP(mobile, otp);

      res.json({
        success: true,
        message: result.message || 'OTP sent successfully',
        // For development only - remove in production
        ...(process.env.NODE_ENV === 'development' && { otp })
      });
    } catch (error) {
      console.error(error);
      res.status(500).json({ message: 'Error sending OTP' });
    }
  }
);

// @route   POST /api/auth/verify-otp
// @desc    Verify OTP and login teacher
// @access  Public
router.post(
  '/verify-otp',
  [
    body('mobile')
      .notEmpty()
      .withMessage('Mobile number is required')
      .matches(/^[0-9]{10}$/)
      .withMessage('Mobile number must be 10 digits'),
    body('otp')
      .notEmpty()
      .withMessage('OTP is required')
      .isLength({ min: 6, max: 6 })
      .withMessage('OTP must be 6 digits')
  ],
  async (req, res) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const { mobile, otp } = req.body;

      // Find OTP record
      const otpRecord = await OTP.findOne({ mobile, otp, verified: false });
      if (!otpRecord) {
        return res.status(401).json({ message: 'Invalid or expired OTP' });
      }

      // Mark OTP as verified
      otpRecord.verified = true;
      await otpRecord.save();

      // Find teacher
      const teacher = await User.findOne({ mobile, role: 'teacher' });
      if (!teacher) {
        return res.status(404).json({ message: 'Teacher account not found' });
      }

      // Return teacher data with token
      res.json({
        _id: teacher._id,
        fullName: teacher.fullName,
        mobile: teacher.mobile,
        role: teacher.role,
        token: generateToken(teacher._id)
      });
    } catch (error) {
      console.error(error);
      res.status(500).json({ message: 'Error verifying OTP' });
    }
  }
);

export default router;
