import mongoose from 'mongoose';

const testCaseSchema = new mongoose.Schema({
  input: {
    type: String,
    required: true
  },
  expectedOutput: {
    type: String,
    required: true
  },
  isHidden: {
    type: Boolean,
    default: false
  },
  explanation: {
    type: String
  }
}, { _id: true });

const questionSchema = new mongoose.Schema({
  examId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Exam',
    required: true
  },
  questionType: {
    type: String,
    required: true,
    enum: ['mcq', 'coding'],
    default: 'mcq'
  },
  questionText: {
    type: String,
    required: true,
    trim: true
  },
  points: {
    type: Number,
    default: 1,
    min: 1
  },
  
  // MCQ-specific fields
  optionA: {
    type: String,
    trim: true
  },
  optionB: {
    type: String,
    trim: true
  },
  optionC: {
    type: String,
    trim: true
  },
  optionD: {
    type: String,
    trim: true
  },
  correctAnswer: {
    type: String,
    enum: ['A', 'B', 'C', 'D', '']
  },
  
  // Coding-specific fields
  problemStatement: {
    type: String,
    trim: true
  },
  inputFormat: {
    type: String,
    trim: true
  },
  outputFormat: {
    type: String,
    trim: true
  },
  constraints: {
    type: String,
    trim: true
  },
  testCases: [testCaseSchema],
  timeLimit: {
    type: Number,
    default: 2 // seconds
  },
  memoryLimit: {
    type: Number,
    default: 256 // MB
  },
  supportedLanguages: [{
    type: String,
    enum: ['javascript', 'python', 'cpp', 'java']
  }],
  starterCode: {
    javascript: { type: String, default: '' },
    python: { type: String, default: '' },
    cpp: { type: String, default: '' },
    java: { type: String, default: '' }
  },
  
  createdAt: {
    type: Date,
    default: Date.now
  }
});

// Validation: MCQ questions must have options and correct answer
questionSchema.pre('save', function(next) {
  if (this.questionType === 'mcq') {
    if (!this.optionA || !this.optionB || !this.optionC || !this.optionD || !this.correctAnswer) {
      return next(new Error('MCQ questions must have all options and correct answer'));
    }
  }
  
  if (this.questionType === 'coding') {
    if (!this.problemStatement || !this.testCases || this.testCases.length === 0) {
      return next(new Error('Coding questions must have problem statement and at least one test case'));
    }
    
    if (!this.supportedLanguages || this.supportedLanguages.length === 0) {
      this.supportedLanguages = ['javascript']; // Default to JavaScript
    }
  }
  
  next();
});

const Question = mongoose.model('Question', questionSchema);

export default Question;
