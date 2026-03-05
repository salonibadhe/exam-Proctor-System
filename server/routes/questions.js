import express from 'express';
import { body, validationResult } from 'express-validator';
import Question from '../models/Question.js';
import Exam from '../models/Exam.js';
import { protect, authorizeRole } from '../middleware/auth.js';
import { runCode } from '../services/codeExecutor.js';

const router = express.Router();

// @route   POST /api/questions
// @desc    Add a question to an exam (MCQ or Coding)
// @access  Private/Teacher
router.post(
  '/',
  [
    protect,
    authorizeRole('teacher'),
    body('examId').notEmpty().withMessage('Exam ID is required'),
    body('questionType').isIn(['mcq', 'coding']).withMessage('Question type must be mcq or coding'),
    body('questionText').notEmpty().withMessage('Question text is required')
  ],
  async (req, res) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const { examId, questionType, questionText, points = 1 } = req.body;

      // Verify exam exists and belongs to teacher
      const exam = await Exam.findById(examId);
      if (!exam) {
        return res.status(404).json({ message: 'Exam not found' });
      }

      if (exam.teacherId.toString() !== req.user._id.toString()) {
        return res.status(403).json({ message: 'Not authorized to add questions to this exam' });
      }

      let questionData = {
        examId,
        questionType,
        questionText,
        points
      };

      // Add type-specific fields
      if (questionType === 'mcq') {
        const { optionA, optionB, optionC, optionD, correctAnswer } = req.body;

        if (!optionA || !optionB || !optionC || !optionD || !correctAnswer) {
          return res.status(400).json({ message: 'MCQ questions require all options and correct answer' });
        }

        questionData = {
          ...questionData,
          optionA,
          optionB,
          optionC,
          optionD,
          correctAnswer
        };
      } else if (questionType === 'coding') {
        const {
          problemStatement,
          inputFormat,
          outputFormat,
          constraints,
          testCases,
          timeLimit = 2,
          memoryLimit = 256,
          supportedLanguages = ['javascript'],
          starterCode = {}
        } = req.body;

        if (!problemStatement || !testCases || testCases.length === 0) {
          return res.status(400).json({ message: 'Coding questions require problem statement and at least one test case' });
        }

        questionData = {
          ...questionData,
          problemStatement,
          inputFormat,
          outputFormat,
          constraints,
          testCases,
          timeLimit,
          memoryLimit,
          supportedLanguages,
          starterCode
        };
      }

      const question = await Question.create(questionData);

      res.status(201).json(question);
    } catch (error) {
      console.error('Error creating question:', error);
      res.status(500).json({ message: error.message || 'Server error' });
    }
  }
);

// @route   GET /api/questions/exam/:examId
// @desc    Get all questions for an exam
// @access  Private
router.get('/exam/:examId', protect, async (req, res) => {
  try {
    const exam = await Exam.findById(req.params.examId);
    if (!exam) {
      return res.status(404).json({ message: 'Exam not found' });
    }

    let questions = await Question.find({ examId: req.params.examId }).sort({ createdAt: 1 });

    // If student, hide answers/hidden test cases and SHUFFLE questions
    if (req.user.role === 'student') {
      // Fisher-Yates shuffle
      for (let i = questions.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [questions[i], questions[j]] = [questions[j], questions[i]];
      }

      questions = questions.map(q => {
        const questionObj = q.toObject();

        // Hide correct answer for MCQ
        if (questionObj.questionType === 'mcq') {
          delete questionObj.correctAnswer;
        }

        // Hide hidden test cases and expected outputs for coding questions
        if (questionObj.questionType === 'coding' && questionObj.testCases) {
          questionObj.testCases = questionObj.testCases
            .filter(tc => !tc.isHidden)
            .map(tc => ({
              _id: tc._id,
              input: tc.input,
              explanation: tc.explanation,
              // Don't include expectedOutput for students
            }));
        }

        return questionObj;
      });
    }

    res.json(questions);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Server error' });
  }
});

// @route   GET /api/questions/exam/:examId/with-answers
// @desc    Get all questions with answers for grading (teacher or for submission)
// @access  Private
router.get('/exam/:examId/with-answers', protect, async (req, res) => {
  try {
    const questions = await Question.find({ examId: req.params.examId }).sort({ createdAt: 1 });
    res.json(questions);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Server error' });
  }
});

// @route   DELETE /api/questions/:id
// @desc    Delete a question
// @access  Private/Teacher
router.delete('/:id', protect, authorizeRole('teacher'), async (req, res) => {
  try {
    const question = await Question.findById(req.params.id);

    if (!question) {
      return res.status(404).json({ message: 'Question not found' });
    }

    // Verify exam belongs to teacher
    const exam = await Exam.findById(question.examId);
    if (exam.teacherId.toString() !== req.user._id.toString()) {
      return res.status(403).json({ message: 'Not authorized to delete this question' });
    }

    await question.deleteOne();
    res.json({ message: 'Question deleted' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Server error' });
  }
});

// @route   POST /api/questions/run-code
// @desc    Run code with sample input (for testing during exam)
// @access  Private/Student
router.post('/run-code', protect, async (req, res) => {
  try {
    const { code, language, input } = req.body;

    if (!code || !language || input === undefined) {
      return res.status(400).json({ message: 'Code, language, and input are required' });
    }

    const result = await runCode(code, language, input);
    res.json(result);
  } catch (error) {
    console.error('Error running code:', error);
    res.status(500).json({
      success: false,
      output: '',
      error: error.message || 'Server error'
    });
  }
});

// @route   POST /api/questions/run-test-cases
// @desc    Run code against all visible test cases
// @access  Private/Student
router.post('/run-test-cases', protect, async (req, res) => {
  try {
    const { code, language, questionId } = req.body;

    if (!code || !language || !questionId) {
      return res.status(400).json({ message: 'Code, language, and questionId are required' });
    }

    const question = await Question.findById(questionId);
    if (!question) {
      return res.status(404).json({ message: 'Question not found' });
    }

    // Only run visible test cases for students
    const testCases = question.testCases.filter(tc => !tc.isHidden);

    const results = await runCode(code, language, null, testCases);
    res.json(results);
  } catch (error) {
    console.error('Error running test cases:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Server error'
    });
  }
});

export default router;
