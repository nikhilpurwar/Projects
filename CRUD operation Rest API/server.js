const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());

// Connect to MongoDB
mongoose.connect('mongodb://127.0.0.1:27017/usersDB');

// Define User schema
const UserSchema = new mongoose.Schema({
    name: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    age: { type: Number, required: true }
});

const User = mongoose.model('User', UserSchema);

// POST /api/users - Create a new user
app.post('/api/users', async (req, res) => {
    try {
        const { name, email, age } = req.body;
        if (!name || !email || !age) return res.status(400).json({ error: 'All fields are required' });
        const user = new User({ name, email, age });
        await user.save();
        res.status(201).json(user);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// GET /api/users - Retrieve all users
app.get('/api/users', async (req, res) => {
    const users = await User.find();
    res.json(users);
});

// GET /api/users/:id - Retrieve a user by ID
app.get('/api/users/:id', async (req, res) => {
    try {
        const user = await User.findById(req.params.id);
        if (!user) return res.status(404).json({ error: 'User not found' });
        res.json(user);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// PUT /api/users/:id - Update a user by ID
app.put('/api/users/:id', async (req, res) => {
    try {
        const { name, email, age } = req.body;
        const user = await User.findByIdAndUpdate(req.params.id, { name, email, age }, { new: true });
        if (!user) return res.status(404).json({ error: 'User not found' });
        res.json(user);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// DELETE /api/users/:id - Delete a user by ID
app.delete('/api/users/:id', async (req, res) => {
    try {
        const user = await User.findByIdAndDelete(req.params.id);
        if (!user) return res.status(404).json({ error: 'User not found' });
        res.json({ message: 'User deleted successfully' });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Start the server
const PORT = 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));