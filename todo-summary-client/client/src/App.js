// client/src/App.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [todos, setTodos] = useState([]);
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [message, setMessage] = useState({ text: '', type: '' });
  const [isLoading, setIsLoading] = useState(false);

  const API_URL = 'http://localhost:5000';

  useEffect(() => {
    fetchTodos();
  }, []);

  const fetchTodos = async () => {
    try {
      const response = await axios.get(`${API_URL}/todos`);
      setTodos(response.data);
    } catch (error) {
      showMessage('Failed to fetch todos', 'error');
    }
  };

  const addTodo = async (e) => {
    e.preventDefault();
    if (!title.trim()) return;

    try {
      const response = await axios.post(`${API_URL}/todos`, {
        title,
        description
      });
      setTodos([...todos, response.data]);
      setTitle('');
      setDescription('');
      showMessage('Todo added successfully', 'success');
    } catch (error) {
      showMessage('Failed to add todo', 'error');
    }
  };

  const deleteTodo = async (id) => {
    try {
      await axios.delete(`${API_URL}/todos/${id}`);
      setTodos(todos.filter(todo => todo.id !== id));
      showMessage('Todo deleted successfully', 'success');
    } catch (error) {
      showMessage('Failed to delete todo', 'error');
    }
  };

  const summarizeTodos = async () => {
    setIsLoading(true);
    try {
      const response = await axios.post(`${API_URL}/summarize`);
      if (response.data.success) {
        showMessage(response.data.message, 'success');
        if (response.data.summary) {
          alert(`Summary:\n\n${response.data.summary}`);
        }
      } else {
        showMessage(response.data.error || 'Failed to generate summary', 'error');
      }
    } catch (error) {
      showMessage(error.response?.data?.error || 'Failed to generate summary', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const showMessage = (text, type) => {
    setMessage({ text, type });
    setTimeout(() => setMessage({ text: '', type: '' }), 3000);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Todo Summary Assistant</h1>
        {message.text && (
          <div className={`message ${message.type}`}>
            {message.text}
          </div>
        )}
      </header>

      <div className="app-content">
        {/* Left Panel - Todo List */}
        <div className="todo-list-panel">
          <div className="panel-header">
            <h2>Your Tasks</h2>
            <button 
              onClick={summarizeTodos} 
              disabled={todos.length === 0 || isLoading}
              className="summary-button"
            >
              {isLoading ? (
                <span className="button-loading">
                  <span className="spinner"></span> Generating...
                </span>
              ) : (
                'Generate and Send Summary'
              )}
            </button>
          </div>

          <div className="todo-items-container">
            {todos.length === 0 ? (
              <div className="empty-state">
                <p>No tasks yet. Add one to get started!</p>
              </div>
            ) : (
              <ul className="todo-list">
                {todos.map(todo => (
                  <li key={todo.id} className="todo-item">
                    <div className="todo-content">
                      <h3>{todo.title}</h3>
                      {todo.description && <p>{todo.description}</p>}
                    </div>
                    <button 
                      onClick={() => deleteTodo(todo.id)}
                      className="delete-button"
                    >
                      ×
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* Right Panel - Add Todo Form */}
        <div className="add-todo-panel">
          <div className="panel-header">
            <h2>Add New Task</h2>
          </div>
          <form onSubmit={addTodo} className="todo-form">
            <div className="form-group">
              <label htmlFor="title">Title*</label>
              <input
                id="title"
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="What needs to be done?"
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="description">Description (optional)</label>
              <textarea
                id="description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Add details..."
                rows="3"
              />
            </div>
            <button type="submit" className="add-button">
              Add Task
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;