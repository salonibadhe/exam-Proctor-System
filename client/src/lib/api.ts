import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests if available
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Handle response errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/auth';
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authAPI = {
  signup: async (data: { fullName: string; email?: string; password?: string; mobile?: string; role: string }) => {
    const response = await api.post('/auth/signup', data);
    return response.data;
  },

  login: async (data: { email: string; password: string }) => {
    const response = await api.post('/auth/login', data);
    return response.data;
  },

  sendOTP: async (data: { mobile: string }) => {
    const response = await api.post('/auth/send-otp', data);
    return response.data;
  },

  verifyOTP: async (data: { mobile: string; otp: string }) => {
    const response = await api.post('/auth/verify-otp', data);
    return response.data;
  },

  getMe: async () => {
    const response = await api.get('/auth/me');
    return response.data;
  },

  logout: () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  },
};

// Exam API
export const examAPI = {
  create: async (data: { title: string; description?: string; examDate: string; durationMinutes: number }) => {
    const response = await api.post('/exams', data);
    return response.data;
  },

  getAll: async () => {
    const response = await api.get('/exams');
    return response.data;
  },

  getById: async (id: string) => {
    const response = await api.get(`/exams/${id}`);
    return response.data;
  },

  delete: async (id: string) => {
    const response = await api.delete(`/exams/${id}`);
    return response.data;
  },

  getByCode: async (code: string) => {
    const response = await api.get(`/exams/code/${code}`);
    return response.data;
  },
};

// Question API
export const questionAPI = {
  create: async (data: any) => {
    const response = await api.post('/questions', data);
    return response.data;
  },

  getByExamId: async (examId: string) => {
    const response = await api.get(`/questions/exam/${examId}`);
    return response.data;
  },

  getWithAnswers: async (examId: string) => {
    const response = await api.get(`/questions/exam/${examId}/with-answers`);
    return response.data;
  },

  delete: async (id: string) => {
    const response = await api.delete(`/questions/${id}`);
    return response.data;
  },

  runCode: async (data: { code: string; language: string; input: string }) => {
    const response = await api.post('/questions/run-code', data);
    return response.data;
  },

  runTestCases: async (data: { code: string; language: string; questionId: string }) => {
    const response = await api.post('/questions/run-test-cases', data);
    return response.data;
  },
};

// Result API
export const resultAPI = {
  submit: async (data: { examId: string; responses: any[]; violations?: { type: string; count: number }[] }) => {
    const response = await api.post('/results/submit', data);
    return response.data;
  },

  getStudentResults: async () => {
    const response = await api.get('/results/student');
    return response.data;
  },

  getExamResults: async (examId: string) => {
    const response = await api.get(`/results/exam/${examId}`);
    return response.data;
  },

  getStudentResponses: async (examId: string, studentId: string) => {
    const response = await api.get(`/results/exam/${examId}/student/${studentId}`);
    return response.data;
  },

  getStudentResultsByTeacher: async (studentId: string) => {
    const response = await api.get(`/results/teacher/student/${studentId}`);
    return response.data;
  },
};

export default api;
