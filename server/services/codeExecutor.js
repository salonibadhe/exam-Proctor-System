import { VM } from 'vm2';
import fs from 'fs';
import path from 'path';
import { exec } from 'child_process';
import { v4 as uuidv4 } from 'uuid';
import os from 'os';

// Helper to write code to a temp file
const writeTempFile = async (code, extension) => {
  const filename = `${uuidv4()}.${extension}`;
  const filepath = path.join(os.tmpdir(), filename);
  await fs.promises.writeFile(filepath, code);
  return filepath;
};

// Helper to delete temp file
const deleteTempFile = async (filepath) => {
  try {
    await fs.promises.unlink(filepath);
  } catch (err) {
    console.error(`Error deleting temp file ${filepath}:`, err);
  }
};

/**
 * Execute JavaScript code with test cases
 */
export const executeJavaScriptCode = async (code, testCases, timeLimit = 2) => {
  const results = [];

  for (const testCase of testCases) {
    const startTime = Date.now();

    try {
      const vm = new VM({
        timeout: timeLimit * 1000,
        sandbox: {}
      });

      let input;
      try {
        input = JSON.parse(testCase.input);
      } catch (e) {
        input = testCase.input;
      }

      const wrappedCode = `
        ${code}
        
        const functionMatch = \`${code.replace(/`/g, '\\`')}\`.match(/function\\s+(\\w+)/);
        const functionName = functionMatch ? functionMatch[1] : null;
        
        if (!functionName) {
          throw new Error('No function found in code');
        }
        
        const result = eval(functionName)(${JSON.stringify(input)});
        JSON.stringify(result);
      `;

      const output = vm.run(wrappedCode);
      const executionTime = Date.now() - startTime;

      const actualOutput = String(output).trim();
      const expectedOutput = String(testCase.expectedOutput).trim();
      const passed = actualOutput === expectedOutput;

      results.push({
        testCaseId: testCase._id,
        passed,
        input: testCase.input,
        expectedOutput: testCase.expectedOutput,
        actualOutput,
        executionTime,
        error: null
      });

    } catch (error) {
      const executionTime = Date.now() - startTime;
      results.push({
        testCaseId: testCase._id,
        passed: false,
        input: testCase.input,
        expectedOutput: testCase.expectedOutput,
        actualOutput: '',
        executionTime,
        error: error.message || 'Execution error'
      });
    }
  }

  return results;
};

/**
 * Execute Python code
 */
const executePythonCode = async (code, testCases, timeLimit = 2) => {
  const results = [];
  const filepath = await writeTempFile(code, 'py');

  for (const testCase of testCases) {
    const startTime = Date.now();
    try {
      // Prepare python script wrapper to handle input/output
      // This wrapper reads input from stdin and prints output to stdout
      // It assumes the user's code reads from stdin and prints to stdout
      const input = testCase.input;

      const result = await new Promise((resolve, reject) => {
        const process = exec(`python "${filepath}"`, { timeout: timeLimit * 1000 }, (error, stdout, stderr) => {
          if (error) {
            reject({ error, stderr });
          } else {
            resolve(stdout);
          }
        });

        if (process.stdin) {
          process.stdin.write(input);
          process.stdin.end();
        }
      });

      const executionTime = Date.now() - startTime;
      const actualOutput = String(result).trim();
      const expectedOutput = String(testCase.expectedOutput).trim();
      const passed = actualOutput === expectedOutput;

      results.push({
        testCaseId: testCase._id,
        passed,
        input: testCase.input,
        expectedOutput: testCase.expectedOutput,
        actualOutput,
        executionTime,
        error: null
      });

    } catch (err) {
      const executionTime = Date.now() - startTime;
      results.push({
        testCaseId: testCase._id,
        passed: false,
        input: testCase.input,
        expectedOutput: testCase.expectedOutput,
        actualOutput: '',
        executionTime,
        error: err.stderr || err.error?.message || 'Execution error'
      });
    }
  }

  await deleteTempFile(filepath);
  return results;
};

/**
 * Execute C++ code
 */
const executeCppCode = async (code, testCases, timeLimit = 2) => {
  const results = [];
  const sourceFile = await writeTempFile(code, 'cpp');
  // Use a fixed name for the executable based on source file to avoid collisions if possible, or random
  // On Windows .exe is needed
  const exeFile = sourceFile.replace('.cpp', '.exe');

  try {
    // Compile
    await new Promise((resolve, reject) => {
      exec(`g++ "${sourceFile}" -o "${exeFile}"`, (error, stdout, stderr) => {
        if (error) reject(stderr);
        else resolve();
      });
    });

    for (const testCase of testCases) {
      const startTime = Date.now();
      try {
        const input = testCase.input;
        const result = await new Promise((resolve, reject) => {
          const child = exec(`"${exeFile}"`, { timeout: timeLimit * 1000 }, (error, stdout, stderr) => {
            if (error) reject({ error, stderr });
            else resolve(stdout);
          });

          if (child.stdin) {
            child.stdin.write(input);
            child.stdin.end();
          }
        });

        const executionTime = Date.now() - startTime;
        const actualOutput = String(result).trim();
        const expectedOutput = String(testCase.expectedOutput).trim();
        const passed = actualOutput === expectedOutput;

        results.push({
          testCaseId: testCase._id,
          passed,
          input: testCase.input,
          expectedOutput: testCase.expectedOutput,
          actualOutput,
          executionTime,
          error: null
        });
      } catch (err) {
        const executionTime = Date.now() - startTime;
        results.push({
          testCaseId: testCase._id,
          passed: false,
          input: testCase.input,
          expectedOutput: testCase.expectedOutput,
          actualOutput: '',
          executionTime,
          error: err.stderr || err.error?.message || 'Execution Error'
        });
      }
    }
  } catch (compileError) {
    // Compilation failed, fail all test cases
    results.push(...testCases.map(tc => ({
      testCaseId: tc._id,
      passed: false,
      input: tc.input,
      expectedOutput: tc.expectedOutput,
      actualOutput: '',
      executionTime: 0,
      error: 'Compilation Error: ' + compileError
    })));
  } finally {
    await deleteTempFile(sourceFile);
    if (fs.existsSync(exeFile)) await deleteTempFile(exeFile);
  }

  return results;
};

/**
 * Execute Java code
 */
const executeJavaCode = async (code, testCases, timeLimit = 2) => {
  const results = [];

  // Extract class name
  const match = code.match(/public\s+class\s+(\w+)/);
  const className = match ? match[1] : 'Main';
  const filename = `${className}.java`;
  const tempDir = os.tmpdir();
  const filepath = path.join(tempDir, filename);

  await fs.promises.writeFile(filepath, code);

  try {
    // Compile
    await new Promise((resolve, reject) => {
      exec(`javac "${filepath}"`, (error, stdout, stderr) => {
        if (error) reject(stderr);
        else resolve();
      });
    });

    for (const testCase of testCases) {
      const startTime = Date.now();
      try {
        const input = testCase.input;
        const result = await new Promise((resolve, reject) => {
          const child = exec(`java -cp "${tempDir}" ${className}`, { timeout: timeLimit * 1000 }, (error, stdout, stderr) => {
            if (error) reject({ error, stderr });
            else resolve(stdout);
          });

          if (child.stdin) {
            child.stdin.write(input);
            child.stdin.end();
          }
        });

        const executionTime = Date.now() - startTime;
        const actualOutput = String(result).trim();
        const expectedOutput = String(testCase.expectedOutput).trim();
        const passed = actualOutput === expectedOutput;

        results.push({
          testCaseId: testCase._id,
          passed,
          input: testCase.input,
          expectedOutput: testCase.expectedOutput,
          actualOutput,
          executionTime,
          error: null
        });
      } catch (err) {
        const executionTime = Date.now() - startTime;
        results.push({
          testCaseId: testCase._id,
          passed: false,
          input: testCase.input,
          expectedOutput: testCase.expectedOutput,
          actualOutput: '',
          executionTime,
          error: err.stderr || err.error?.message || 'Execution Error'
        });
      }
    }

  } catch (compileError) {
    results.push(...testCases.map(tc => ({
      testCaseId: tc._id,
      passed: false,
      input: tc.input,
      expectedOutput: tc.expectedOutput,
      actualOutput: '',
      executionTime: 0,
      error: 'Compilation Error: ' + compileError
    })));
  } finally {
    // Cleanup .java and .class files
    await deleteTempFile(filepath);
    const classFile = path.join(tempDir, `${className}.class`);
    if (fs.existsSync(classFile)) await deleteTempFile(classFile);
  }

  return results;
};


/**
 * Execute code in supported languages
 */
export const executeCode = async (code, language, testCases, timeLimit = 2) => {
  switch (language.toLowerCase()) {
    case 'javascript':
      return await executeJavaScriptCode(code, testCases, timeLimit);
    case 'python':
      return await executePythonCode(code, testCases, timeLimit);
    case 'cpp':
      return await executeCppCode(code, testCases, timeLimit);
    case 'java':
      return await executeJavaCode(code, testCases, timeLimit);
    default:
      throw new Error(`Unsupported language: ${language}`);
  }
};

/**
 * Run code with sample test cases (for "Run Code" button)
 */
export const runCode = async (code, language, sampleInput, testCases = null) => {
  if (testCases) {
    try {
      return await executeCode(code, language, testCases);
    } catch (error) {
      return testCases.map(tc => ({
        testCaseId: tc._id,
        passed: false,
        input: tc.input,
        expectedOutput: tc.expectedOutput,
        actualOutput: '',
        executionTime: 0,
        error: error.message
      }));
    }
  }

  try {
    const testCase = {
      _id: 'sample',
      input: sampleInput,
      expectedOutput: '', // Not used during simple run
    };

    let result;
    const timeLimit = 5; // 5 seconds for run code

    switch (language.toLowerCase()) {
      case 'javascript': {
        // Re-using existing VM logic for consistency with previous implementation but adaptable
        const vm = new VM({ timeout: 5000, sandbox: {} });
        let input;
        try { input = JSON.parse(sampleInput); } catch (e) { input = sampleInput; }

        const wrappedCode = `
                ${code}
                const functionMatch = \`${code.replace(/`/g, '\\`')}\`.match(/function\\s+(\\w+)/);
                const functionName = functionMatch ? functionMatch[1] : null;
                if (!functionName) throw new Error('No function found in code');
                const result = eval(functionName)(${JSON.stringify(input)});
                JSON.stringify(result);
             `;
        const output = vm.run(wrappedCode);
        return { success: true, output, error: null };
      }
      case 'python': {
        const results = await executePythonCode(code, [testCase], timeLimit);
        const res = results[0];
        return {
          success: !res.error,
          output: res.actualOutput,
          error: res.error
        };
      }
      case 'cpp': {
        const results = await executeCppCode(code, [testCase], timeLimit);
        const res = results[0];
        return {
          success: !res.error,
          output: res.actualOutput,
          error: res.error
        };
      }
      case 'java': {
        const results = await executeJavaCode(code, [testCase], timeLimit);
        const res = results[0];
        return {
          success: !res.error,
          output: res.actualOutput,
          error: res.error
        };
      }
      default:
        throw new Error(`Unsupported language: ${language}`);
    }

  } catch (error) {
    return {
      success: false,
      output: '',
      error: error.message
    };
  }
};
