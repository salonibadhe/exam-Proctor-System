import mongoose from 'mongoose';

const examSchema = new mongoose.Schema({
  title: {
    type: String,
    required: true,
    trim: true
  },
  description: {
    type: String,
    trim: true
  },
  examCode: {
    type: String,
    unique: true,
    uppercase: true,
    sparse: true
  },
  examDate: {
    type: Date,
    required: true
  },
  durationMinutes: {
    type: Number,
    required: true,
    min: 1
  },
  teacherId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

// Generate unique exam code before saving
examSchema.pre('save', async function(next) {
  if (!this.examCode || this.examCode === '') {
    this.examCode = await generateExamCode();
  }
  next();
});

// Function to generate a unique 6-character exam code
async function generateExamCode() {
  const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  let code = '';
  let isUnique = false;
  
  while (!isUnique) {
    code = '';
    for (let i = 0; i < 6; i++) {
      code += characters.charAt(Math.floor(Math.random() * characters.length));
    }
    
    const existingExam = await mongoose.model('Exam').findOne({ examCode: code });
    if (!existingExam) {
      isUnique = true;
    }
  }
  
  return code;
}

const Exam = mongoose.model('Exam', examSchema);

export default Exam;
