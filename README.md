# Todo Summary Assistant

A full-stack application for managing todos with AI-powered summaries and Slack integration.

## Features
- Create, view, and delete todos
- Generate AI summaries using Gemini/OpenAI
- Send summaries to Slack
- Responsive split-panel UI

## Tech Stack
- **Frontend**: React
- **Backend**: Node.js/Express
- **Database**: MySQL
- **AI**: Google Gemini API
- **Notifications**: Slack Webhooks

## Setup Instructions

### 1. Prerequisites
- Node.js (v16+)
- MySQL server
- XAMPP (for local MySQL)
- Gemini API key
- Slack workspace

### 2. Installation
```bash
git clone https://github.com/yourusername/todo-summary-assistant.git
cd todo-summary-assistant

## Running the Server
cd server
npm start  # Production

## Start Frontend
open in other terminal 
cd ../client
npm start

Note - run both server and client parallaly in terminal

Database Setup
Using XAMPP
Start Apache and MySQL in XAMPP

Open phpMyAdmin (http://localhost/phpmyadmin)

Create database:
CREATE DATABASE todo_app;
USE todo_app;

CREATE TABLE todos (
  id INT AUTO_INCREMENT PRIMARY KEY,
  title VARCHAR(255) NOT NULL,
  description TEXT,
  completed BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);