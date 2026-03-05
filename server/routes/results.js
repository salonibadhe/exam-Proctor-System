import express from 'express';
import { body, validationResult } from 'express-validator';
import ExamResult from '../models/ExamResult.js';
import Response from '../models/Response.js';
import Exam from '../models/Exam.js';
import Question from '../models/Question.js';
import { protect, authorizeRole } from '../middleware/auth.js';
import { executeCode } from '../services/codeExecutor.js';

const router = express.Router();

// @route   POST /api/results/submit
// @desc    Submit exam responses and create result
// @access  Private/Student
router.post(
  '/submit',
  [
    protect,
    authorizeRole('student'),
    body('examId').notEmpty().withMessage('Exam ID is required'),
    body('responses').isArray().withMessage('Responses must be an array')
  ],
  async (req, res) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const { examId, responses, violations = [] } = req.body;

      // Check if exam exists
      const exam = await Exam.findById(examId);
      if (!exam) {
        return res.status(404).json({ message: 'Exam not found' });
      }

      // Check if student has already submitted
      const existingResult = await ExamResult.findOne({
        studentId: req.user._id,
        examId
      });

      if (existingResult) {
        return res.status(400).json({ message: 'You have already submitted this exam' });
      }

      // Get all questions with correct answers
      const questions = await Question.find({ examId });
      const questionMap = {};
      questions.forEach(q => {
        questionMap[q._id.toString()] = q;
      });

      // Calculate score and prepare responses
      let totalScore = 0;
      let totalPoints = 0;
      const responsesToInsert = [];

      for (const r of responses) {
        const question = questionMap[r.questionId];
        if (!question) continue;

        totalPoints += question.points || 1;
        let responseData = {
          studentId: req.user._id,
          examId,
          questionId: r.questionId,
          questionType: question.questionType,
          isCorrect: false,
          pointsEarned: 0
        };

        // Handle MCQ questions
        if (question.questionType === 'mcq') {
          const isCorrect = r.selectedAnswer === question.correctAnswer;
          responseData.selectedAnswer = r.selectedAnswer || '';
          responseData.isCorrect = isCorrect;
          responseData.pointsEarned = isCorrect ? (question.points || 1) : 0;
          totalScore += responseData.pointsEarned;
        }
        // Handle Coding questions
        else if (question.questionType === 'coding') {
          if (!r.code || !r.language) {
            responseData.submittedCode = r.code || '';
            responseData.language = r.language || 'javascript';
            responseData.testCaseResults = [];
            responseData.testCasesPassed = 0;
            responseData.totalTestCases = question.testCases.length;
          } else {
            // Execute code against test cases
            const testResults = await executeCode(
              r.code,
              r.language,
              question.testCases,
              question.timeLimit
            );

            const passedCount = testResults.filter(tr => tr.passed).length;
            const totalTestCases = testResults.length;
            const percentage = totalTestCases > 0 ? passedCount / totalTestCases : 0;

            // Partial credit based on test cases passed
            const pointsEarned = Math.round(percentage * (question.points || 1));

            responseData.submittedCode = r.code;
            responseData.language = r.language;
            responseData.testCaseResults = testResults;
            responseData.testCasesPassed = passedCount;
            responseData.totalTestCases = totalTestCases;
            responseData.isCorrect = passedCount === totalTestCases;
            responseData.pointsEarned = pointsEarned;
            totalScore += pointsEarned;
          }
        }

        responsesToInsert.push(responseData);
      }

      // Insert responses
      await Response.insertMany(responsesToInsert);

      // Create exam result
      const result = await ExamResult.create({
        studentId: req.user._id,
        examId,
        score: totalScore,
        totalQuestions: questions.length,
        totalPoints: totalPoints,
        violations: violations.filter(v => v.count > 0) // only store types that occurred
      });

      res.status(201).json({
        result,
        percentage: totalPoints > 0 ? Math.round((totalScore / totalPoints) * 100) : 0
      });
    } catch (error) {
      console.error('Error submitting exam:', error);
      res.status(500).json({ message: error.message || 'Server error' });
    }
  }
);

// @route   GET /api/results/student
// @desc    Get all results for logged in student
// @access  Private/Student
router.get('/student', protect, authorizeRole('student'), async (req, res) => {
  try {
    const results = await ExamResult.find({ studentId: req.user._id })
      .populate('examId')
      .sort({ submittedAt: -1 });

    res.json(results);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Server error' });
  }
});

// @route   GET /api/results/exam/:examId
// @desc    Get all results for an exam (teacher only)
// @access  Private/Teacher
router.get('/exam/:examId', protect, authorizeRole('teacher'), async (req, res) => {
  try {
    const exam = await Exam.findById(req.params.examId);

    if (!exam) {
      return res.status(404).json({ message: 'Exam not found' });
    }

    // Verify exam belongs to teacher
    if (exam.teacherId.toString() !== req.user._id.toString()) {
      return res.status(403).json({ message: 'Not authorized to view these results' });
    }

    const results = await ExamResult.find({ examId: req.params.examId })
      .populate('studentId', 'fullName email')
      .sort({ submittedAt: -1 });

    res.json(results);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Server error' });
  }
});

// @route   GET /api/results/exam/:examId/student/:studentId
// @desc    Get detailed responses for a student on an exam (teacher only)
// @access  Private/Teacher
router.get('/exam/:examId/student/:studentId', protect, authorizeRole('teacher'), async (req, res) => {
  try {
    const exam = await Exam.findById(req.params.examId);

    if (!exam) {
      return res.status(404).json({ message: 'Exam not found' });
    }

    if (exam.teacherId.toString() !== req.user._id.toString()) {
      return res.status(403).json({ message: 'Not authorized to view these results' });
    }

    const responses = await Response.find({
      examId: req.params.examId,
      studentId: req.params.studentId
    }).populate('questionId', 'questionText questionType');

    res.json(responses);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Server error' });
  }
});

export default router;
