import { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { examAPI, questionAPI } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { ArrowLeft, Plus, Trash2, Code2, X } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

type Exam = {
  _id: string;
  title: string;
  description?: string;
  examCode: string;
  examDate: string;
  durationMinutes: number;
  teacherId: string;
  createdAt: string;
};

type TestCase = {
  _id?: string;
  input: string;
  expectedOutput: string;
  isHidden: boolean;
  explanation?: string;
};

type Question = {
  _id: string;
  examId: string;
  questionType: 'mcq' | 'coding';
  questionText: string;
  points?: number;
  // MCQ fields
  optionA?: string;
  optionB?: string;
  optionC?: string;
  optionD?: string;
  correctAnswer?: string;
  // Coding fields
  problemStatement?: string;
  inputFormat?: string;
  outputFormat?: string;
  constraints?: string;
  testCases?: TestCase[];
  timeLimit?: number;
  supportedLanguages?: string[];
  starterCode?: {
    javascript?: string;
  };
  createdAt: string;
};

const ExamManagement = () => {
  const { examId } = useParams();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [exam, setExam] = useState<Exam | null>(null);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [questionType, setQuestionType] = useState<'mcq' | 'coding'>('mcq');
  
  // MCQ question data
  const [mcqData, setMcqData] = useState({
    questionText: "",
    optionA: "",
    optionB: "",
    optionC: "",
    optionD: "",
    correctAnswer: "A",
    points: 1,
  });

  // Coding question data
  const [codingData, setCodingData] = useState({
    questionText: "",
    problemStatement: "",
    inputFormat: "",
    outputFormat: "",
    constraints: "",
    timeLimit: 2,
    points: 10,
    starterCode: "function solution(input) {\n  // Write your code here\n  \n}",
  });

  const [testCases, setTestCases] = useState<TestCase[]>([{
    input: "",
    expectedOutput: "",
    isHidden: false,
    explanation: ""
  }]);

  useEffect(() => {
    fetchExam();
    fetchQuestions();
  }, [examId]);

  const fetchExam = async () => {
    try {
      const data = await examAPI.getById(examId!);
      setExam(data);
    } catch (error) {
      console.error('Error fetching exam:', error);
    }
  };

  const fetchQuestions = async () => {
    try {
      const data = await questionAPI.getByExamId(examId!);
      setQuestions(data);
    } catch (error) {
      console.error('Error fetching questions:', error);
    }
  };

  const addTestCase = () => {
    setTestCases([...testCases, {
      input: "",
      expectedOutput: "",
      isHidden: false,
      explanation: ""
    }]);
  };

  const removeTestCase = (index: number) => {
    setTestCases(testCases.filter((_, i) => i !== index));
  };

  const updateTestCase = (index: number, field: keyof TestCase, value: any) => {
    const updated = [...testCases];
    updated[index] = { ...updated[index], [field]: value };
    setTestCases(updated);
  };

  const handleAddQuestion = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      let questionData: any = {
        examId: examId!,
        questionType,
      };

      if (questionType === 'mcq') {
        questionData = {
          ...questionData,
          ...mcqData,
        };
      } else {
        questionData = {
          ...questionData,
          ...codingData,
          testCases: testCases.filter(tc => tc.input && tc.expectedOutput),
          supportedLanguages: ['javascript'],
          starterCode: {
            javascript: codingData.starterCode
          }
        };
      }

      await questionAPI.create(questionData);

      toast({
        title: "Question added!",
        description: `${questionType === 'mcq' ? 'MCQ' : 'Coding'} question has been added to the exam.`,
      });

      setDialogOpen(false);
      // Reset forms
      setMcqData({
        questionText: "",
        optionA: "",
        optionB: "",
        optionC: "",
        optionD: "",
        correctAnswer: "A",
        points: 1,
      });
      setCodingData({
        questionText: "",
        problemStatement: "",
        inputFormat: "",
        outputFormat: "",
        constraints: "",
        timeLimit: 2,
        points: 10,
        starterCode: "function solution(input) {\n  // Write your code here\n  \n}",
      });
      setTestCases([{
        input: "",
        expectedOutput: "",
        isHidden: false,
        explanation: ""
      }]);
      fetchQuestions();
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "Error",
        description: error.response?.data?.message || error.message,
      });
    }
  };

  const handleDeleteQuestion = async (questionId: string) => {
    try {
      await questionAPI.delete(questionId);

      toast({
        title: "Question deleted",
        description: "The question has been removed from the exam.",
      });
      fetchQuestions();
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "Error",
        description: error.response?.data?.message || error.message,
      });
    }
  };

  if (!exam) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-xl">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b bg-card shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <Button variant="ghost" onClick={() => navigate("/teacher/dashboard")}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Dashboard
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto space-y-8">
          <Card>
            <CardHeader>
              <CardTitle className="text-3xl">{exam.title}</CardTitle>
              <CardDescription>{exam.description}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex gap-4 text-sm">
                <span className="font-semibold">Exam Code:</span>
                <span className="font-mono text-primary">{exam.examCode}</span>
              </div>
              <div className="flex gap-4 text-sm">
                <span className="font-semibold">Date:</span>
                <span>{new Date(exam.examDate).toLocaleDateString()}</span>
              </div>
              <div className="flex gap-4 text-sm">
                <span className="font-semibold">Duration:</span>
                <span>{exam.durationMinutes} minutes</span>
              </div>
            </CardContent>
          </Card>

          <div className="flex justify-between items-center">
            <h2 className="text-2xl font-bold">Questions ({questions.length})</h2>
            <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
              <DialogTrigger asChild>
                <Button>
                  <Plus className="w-4 h-4 mr-2" />
                  Add Question
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
                <DialogHeader>
                  <DialogTitle>Add New Question</DialogTitle>
                  <DialogDescription>
                    Create a question for this exam (MCQ or Coding)
                  </DialogDescription>
                </DialogHeader>
                
                <Tabs value={questionType} onValueChange={(v) => setQuestionType(v as 'mcq' | 'coding')}>
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="mcq">Multiple Choice</TabsTrigger>
                    <TabsTrigger value="coding"><Code2 className="w-4 h-4 mr-2" />Coding</TabsTrigger>
                  </TabsList>
                  
                  <form onSubmit={handleAddQuestion} className="mt-4">
                    <TabsContent value="mcq" className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="mcq-question">Question Text</Label>
                        <Textarea
                          id="mcq-question"
                          value={mcqData.questionText}
                          onChange={(e) => setMcqData({ ...mcqData, questionText: e.target.value })}
                          required={questionType === 'mcq'}
                          rows={3}
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="optionA">Option A</Label>
                          <Input
                            id="optionA"
                            value={mcqData.optionA}
                            onChange={(e) => setMcqData({ ...mcqData, optionA: e.target.value })}
                            required={questionType === 'mcq'}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="optionB">Option B</Label>
                          <Input
                            id="optionB"
                            value={mcqData.optionB}
                            onChange={(e) => setMcqData({ ...mcqData, optionB: e.target.value })}
                            required={questionType === 'mcq'}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="optionC">Option C</Label>
                          <Input
                            id="optionC"
                            value={mcqData.optionC}
                            onChange={(e) => setMcqData({ ...mcqData, optionC: e.target.value })}
                            required={questionType === 'mcq'}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="optionD">Option D</Label>
                          <Input
                            id="optionD"
                            value={mcqData.optionD}
                            onChange={(e) => setMcqData({ ...mcqData, optionD: e.target.value })}
                            required={questionType === 'mcq'}
                          />
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="correct">Correct Answer</Label>
                          <select
                            id="correct"
                            value={mcqData.correctAnswer}
                            onChange={(e) => setMcqData({ ...mcqData, correctAnswer: e.target.value })}
                            className="w-full px-3 py-2 border rounded-md"
                            required={questionType === 'mcq'}
                          >
                            <option value="A">A</option>
                            <option value="B">B</option>
                            <option value="C">C</option>
                            <option value="D">D</option>
                          </select>
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="mcq-points">Points</Label>
                          <Input
                            id="mcq-points"
                            type="number"
                            min="1"
                            value={mcqData.points}
                            onChange={(e) => setMcqData({ ...mcqData, points: parseInt(e.target.value) })}
                          />
                        </div>
                      </div>
                    </TabsContent>

                    <TabsContent value="coding" className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="coding-title">Question Title</Label>
                        <Input
                          id="coding-title"
                          value={codingData.questionText}
                          onChange={(e) => setCodingData({ ...codingData, questionText: e.target.value })}
                          placeholder="e.g., Find Sum of Array"
                          required={questionType === 'coding'}
                        />
                      </div>
                      
                      <div className="space-y-2">
                        <Label htmlFor="problem">Problem Statement</Label>
                        <Textarea
                          id="problem"
                          value={codingData.problemStatement}
                          onChange={(e) => setCodingData({ ...codingData, problemStatement: e.target.value })}
                          placeholder="Describe the problem in detail..."
                          required={questionType === 'coding'}
                          rows={4}
                        />
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="input-format">Input Format</Label>
                          <Textarea
                            id="input-format"
                            value={codingData.inputFormat}
                            onChange={(e) => setCodingData({ ...codingData, inputFormat: e.target.value })}
                            placeholder="Array of integers"
                            rows={2}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="output-format">Output Format</Label>
                          <Textarea
                            id="output-format"
                            value={codingData.outputFormat}
                            onChange={(e) => setCodingData({ ...codingData, outputFormat: e.target.value })}
                            placeholder="Single integer"
                            rows={2}
                          />
                        </div>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="constraints">Constraints</Label>
                        <Textarea
                          id="constraints"
                          value={codingData.constraints}
                          onChange={(e) => setCodingData({ ...codingData, constraints: e.target.value })}
                          placeholder="1 ≤ array length ≤ 1000"
                          rows={2}
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="starter-code">Starter Code (JavaScript)</Label>
                        <Textarea
                          id="starter-code"
                          value={codingData.starterCode}
                          onChange={(e) => setCodingData({ ...codingData, starterCode: e.target.value })}
                          rows={4}
                          className="font-mono text-sm"
                        />
                      </div>

                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <Label>Test Cases</Label>
                          <Button type="button" variant="outline" size="sm" onClick={addTestCase}>
                            <Plus className="w-4 h-4 mr-1" />Add Test Case
                          </Button>
                        </div>
                        
                        {testCases.map((tc, index) => (
                          <Card key={index} className="p-4">
                            <div className="space-y-3">
                              <div className="flex justify-between items-center">
                                <span className="font-semibold text-sm">Test Case {index + 1}</span>
                                {testCases.length > 1 && (
                                  <Button
                                    type="button"
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => removeTestCase(index)}
                                  >
                                    <X className="w-4 h-4" />
                                  </Button>
                                )}
                              </div>
                              
                              <div className="grid grid-cols-2 gap-3">
                                <div className="space-y-1">
                                  <Label className="text-xs">Input</Label>
                                  <Textarea
                                    value={tc.input}
                                    onChange={(e) => updateTestCase(index, 'input', e.target.value)}
                                    placeholder='[1,2,3,4,5]'
                                    rows={2}
                                    className="font-mono text-sm"
                                  />
                                </div>
                                <div className="space-y-1">
                                  <Label className="text-xs">Expected Output</Label>
                                  <Textarea
                                    value={tc.expectedOutput}
                                    onChange={(e) => updateTestCase(index, 'expectedOutput', e.target.value)}
                                    placeholder='15'
                                    rows={2}
                                    className="font-mono text-sm"
                                  />
                                </div>
                              </div>
                              
                              <div className="space-y-1">
                                <Label className="text-xs">Explanation (Optional)</Label>
                                <Input
                                  value={tc.explanation}
                                  onChange={(e) => updateTestCase(index, 'explanation', e.target.value)}
                                  placeholder="1+2+3+4+5 = 15"
                                  className="text-sm"
                                />
                              </div>

                              <div className="flex items-center space-x-2">
                                <input
                                  type="checkbox"
                                  id={`hidden-${index}`}
                                  checked={tc.isHidden}
                                  onChange={(e) => updateTestCase(index, 'isHidden', e.target.checked)}
                                  className="rounded"
                                />
                                <Label htmlFor={`hidden-${index}`} className="text-sm cursor-pointer">
                                  Hidden test case (not shown to students)
                                </Label>
                              </div>
                            </div>
                          </Card>
                        ))}
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="time-limit">Time Limit (seconds)</Label>
                          <Input
                            id="time-limit"
                            type="number"
                            min="1"
                            max="10"
                            value={codingData.timeLimit}
                            onChange={(e) => setCodingData({ ...codingData, timeLimit: parseInt(e.target.value) })}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="coding-points">Points</Label>
                          <Input
                            id="coding-points"
                            type="number"
                            min="1"
                            value={codingData.points}
                            onChange={(e) => setCodingData({ ...codingData, points: parseInt(e.target.value) })}
                          />
                        </div>
                      </div>
                    </TabsContent>

                    <Button type="submit" className="w-full mt-4">
                      Add {questionType === 'mcq' ? 'MCQ' : 'Coding'} Question
                    </Button>
                  </form>
                </Tabs>
              </DialogContent>
            </Dialog>
          </div>

          <div className="space-y-4">
            {questions.map((question, index) => (
              <Card key={question._id}>
                <CardHeader>
                  <div className="flex justify-between items-start">
                    <div className="flex items-center gap-2">
                      <CardTitle className="text-lg">Question {index + 1}</CardTitle>
                      <span className="text-xs bg-primary/10 text-primary px-2 py-1 rounded-full">
                        {question.questionType === 'mcq' ? 'MCQ' : 'Coding'}
                      </span>
                      <span className="text-xs bg-accent/10 text-accent px-2 py-1 rounded-full">
                        {question.points || 1} pts
                      </span>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDeleteQuestion(question._id)}
                    >
                      <Trash2 className="w-4 h-4 text-destructive" />
                    </Button>
                  </div>
                  <CardDescription className="text-base text-foreground">
                    {question.questionText}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
                  {question.questionType === 'mcq' ? (
                    <>
                      <div className={`p-3 rounded-md ${question.correctAnswer === 'A' ? 'bg-accent/20 border border-accent' : 'bg-muted'}`}>
                        <span className="font-semibold mr-2">A.</span>
                        {question.optionA}
                      </div>
                      <div className={`p-3 rounded-md ${question.correctAnswer === 'B' ? 'bg-accent/20 border border-accent' : 'bg-muted'}`}>
                        <span className="font-semibold mr-2">B.</span>
                        {question.optionB}
                      </div>
                      <div className={`p-3 rounded-md ${question.correctAnswer === 'C' ? 'bg-accent/20 border border-accent' : 'bg-muted'}`}>
                        <span className="font-semibold mr-2">C.</span>
                        {question.optionC}
                      </div>
                      <div className={`p-3 rounded-md ${question.correctAnswer === 'D' ? 'bg-accent/20 border border-accent' : 'bg-muted'}`}>
                        <span className="font-semibold mr-2">D.</span>
                        {question.optionD}
                      </div>
                    </>
                  ) : (
                    <div className="space-y-3">
                      <div className="bg-muted p-4 rounded-md">
                        <p className="text-sm font-semibold mb-2">Problem Statement:</p>
                        <p className="text-sm whitespace-pre-wrap">{question.problemStatement}</p>
                      </div>
                      
                      {question.inputFormat && (
                        <div className="flex gap-4">
                          <div className="flex-1">
                            <p className="text-xs font-semibold mb-1">Input Format:</p>
                            <p className="text-xs text-muted-foreground">{question.inputFormat}</p>
                          </div>
                          <div className="flex-1">
                            <p className="text-xs font-semibold mb-1">Output Format:</p>
                            <p className="text-xs text-muted-foreground">{question.outputFormat}</p>
                          </div>
                        </div>
                      )}
                      
                      {question.constraints && (
                        <div>
                          <p className="text-xs font-semibold mb-1">Constraints:</p>
                          <p className="text-xs text-muted-foreground">{question.constraints}</p>
                        </div>
                      )}
                      
                      <div>
                        <p className="text-sm font-semibold mb-2">Test Cases:</p>
                        <div className="space-y-2">
                          {question.testCases?.map((tc, idx) => (
                            <div key={idx} className="bg-muted/50 p-2 rounded text-xs">
                              <div className="flex gap-2">
                                <span className="font-semibold">Input:</span>
                                <span className="font-mono">{tc.input}</span>
                              </div>
                              <div className="flex gap-2">
                                <span className="font-semibold">Output:</span>
                                <span className="font-mono">{tc.expectedOutput}</span>
                              </div>
                              {tc.isHidden && (
                                <span className="text-xs text-muted-foreground italic">(Hidden)</span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div className="flex gap-4 text-xs text-muted-foreground">
                        <span>⏱️ Time Limit: {question.timeLimit}s</span>
                        <span>💻 Language: JavaScript</span>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>

          {questions.length === 0 && (
            <Card>
              <CardContent className="text-center py-12">
                <h3 className="text-xl font-semibold mb-2">No questions yet</h3>
                <p className="text-muted-foreground mb-4">Add questions to this exam to get started</p>
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  );
};

export default ExamManagement;
