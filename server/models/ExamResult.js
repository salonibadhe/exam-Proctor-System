import mongoose from 'mongoose';

const examResultSchema = new mongoose.Schema({
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
  score: {
    type: Number,
    required: true,
    min: 0
  },
  totalQuestions: {
    type: Number,
    required: true,
    min: 1
  },
  totalPoints: {
    type: Number,
    required: true,
    min: 1
  },
  submittedAt: {
    type: Date,
    default: Date.now
  },
  violations: [
    {
      type: { type: String }, // e.g. "tab_switch" | "copy_paste" | "keyboard_shortcut" | "no_face" | "multiple_faces"
      count: { type: Number, default: 0 }
    }
  ]
});

// Ensure a student can only submit an exam once
examResultSchema.index({ studentId: 1, examId: 1 }, { unique: true });

const ExamResult = mongoose.model('ExamResult', examResultSchema);

export default ExamResult;
