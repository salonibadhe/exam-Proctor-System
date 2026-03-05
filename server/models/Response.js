import mongoose from 'mongoose';

const testCaseResultSchema = new mongoose.Schema({
  testCaseId: mongoose.Schema.Types.ObjectId,
  passed: Boolean,
  input: String,
  expectedOutput: String,
  actualOutput: String,
  executionTime: Number, // milliseconds
  error: String
}, { _id: false });

const responseSchema = new mongoose.Schema({
  studentId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  examId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Exam',
    required: true
  },
  questionId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Question',
    required: true
  },
  questionType: {
    type: String,
    required: true,
    enum: ['mcq', 'coding']
  },
  
  // MCQ response fields
  selectedAnswer: {
    type: String,
    enum: ['A', 'B', 'C', 'D', '']
  },
  
  // Coding response fields
  submittedCode: {
    type: String
  },
  language: {
    type: String,
    enum: ['javascript', 'python', 'cpp', 'java']
  },
  testCaseResults: [testCaseResultSchema],
  testCasesPassed: {
    type: Number,
    default: 0
  },
  totalTestCases: {
    type: Number,
    default: 0
  },
  
  // Common fields
  isCorrect: {
    type: Boolean,
    required: true
  },
  pointsEarned: {
    type: Number,
    default: 0
  },
  submittedAt: {
    type: Date,
    default: Date.now
  }
});

const Response = mongoose.model('Response', responseSchema);

export default Response;
