require('dotenv').config();
const express = require('express');
const cors = require('cors');
const mysql = require('mysql2/promise');
const axios = require('axios');
const OpenAI = require('openai');
const { GoogleGenerativeAI } = require("@google/generative-ai");

const app = express();
app.use(cors());
app.use(express.json());

// Database connection
const pool = mysql.createPool({
  host: process.env.DB_HOST || 'localhost',
  user: process.env.DB_USER || 'root',
  password: process.env.DB_PASSWORD || '',
  database: process.env.DB_NAME || 'todo_app',
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
});

// // OpenAI configuration
// const openai = new OpenAI({
//   apiKey: process.env.OPENAI_API_KEY
// });
// Initialize Gemini
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Routes
app.get('/todos', async (req, res) => {
  try {
    const [rows] = await pool.query('SELECT * FROM todos');
    res.json(rows);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch todos' });
  }
});

app.post('/todos', async (req, res) => {
  const { title, description } = req.body;
  if (!title) {
    return res.status(400).json({ error: 'Title is required' });
  }

  try {
    const [result] = await pool.query(
      'INSERT INTO todos (title, description, completed) VALUES (?, ?, ?)',
      [title, description || '', false]
    );
    const [newTodo] = await pool.query('SELECT * FROM todos WHERE id = ?', [result.insertId]);
    res.status(201).json(newTodo[0]);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to create todo' });
  }
});

app.delete('/todos/:id', async (req, res) => {
  const { id } = req.params;
  try {
    await pool.query('DELETE FROM todos WHERE id = ?', [id]);
    res.status(204).end();
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to delete todo' });
  }
});

app.post('/summarize', async (req, res) => {
  try {
    const [todos] = await pool.query('SELECT * FROM todos WHERE completed = false');
    if (todos.length === 0) {
      return res.status(400).json({ error: 'No pending todos to summarize' });
    }

    // Prepare prompt
    const todoList = todos.map(todo => `- ${todo.title}: ${todo.description || 'No description'}`).join('\n');
    const prompt = `Summarize these to-dos in a concise paragraph, grouping similar items:\n\n${todoList}`;

    // Call Gemini
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const summary = response.text();

    if (process.env.SLACK_WEBHOOK_URL) {
      try {
        await axios.post(process.env.SLACK_WEBHOOK_URL, {
          text: `*To-Do List Summary*\n${summary}`
        });
        return res.json({ success: true, message: 'Summary generated and sent to Slack successfully', summary });
      } catch (slackError) {
        console.error('Slack error:', slackError);
        return res.json({ success: true, message: 'Summary generated but failed to send to Slack', summary });
      }
    }

    res.json({ success: true, message: 'Summary generated successfully', summary });
  } catch (error) {
    console.error('Gemini error:', error);
    res.status(500).json({ 
      error: 'Failed to generate summary',
      details: error.message || 'Unknown Gemini API error'
    });
  }
});
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});