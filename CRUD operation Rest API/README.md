# RESTful API for User Management

## 📌 Requirements
Ensure you have the following installed on your system:
- **Node.js** (v14 or later) – (https://nodejs.org/)
- **MongoDB** (running locally or cloud-based like MongoDB Atlas) – (https://www.mongodb.com/try/download/community)

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-repo/user-crud-api.git
cd user-crud-api
```

### 2️⃣ Install Dependencies
```sh
npm install
```

### 3️⃣ Start MongoDB Server (If running locally)
```sh
mongod
```

### 4️⃣ Run the Server
```sh
node server.js
```
✔ The server should start on `http://localhost:3000`

## 📡 API Endpoints & Commands

### **1️⃣ Create a User (POST)**
```sh
curl -X POST "http://localhost:3000/api/users" -H "Content-Type: application/json" -d '{"name": "Alice", "email": "alice@example.com", "age": 28}'
```
#### 📌 Response:
```json
{
  "_id": "123abc456def",
  "name": "Alice",
  "email": "alice@example.com",
  "age": 28
}
```

### **2️⃣ Get All Users (GET)**
```sh
curl -X GET http://localhost:3000/api/users
```

### **3️⃣ Get a Single User by ID (GET)**
```sh
curl -X GET http://localhost:3000/api/users/123abc456def
```

### **4️⃣ Update a User (PUT)**
```sh
curl -X PUT "http://localhost:3000/api/users/123abc456def" -H "Content-Type: application/json" -d '{"name": "Alice Updated", "email": "alice.updated@example.com", "age": 30}'
```

### **5️⃣ Delete a User (DELETE)**
```sh
curl -X DELETE http://localhost:3000/api/users/123abc456def
```

## 📌 Approach to Building this API
1. **Set Up Express.js Server** – Created an Express app and configured middleware for JSON parsing.
2. **Connected to MongoDB** – Used Mongoose for data modeling and connected to a local MongoDB instance.
3. **Defined User Schema & Model** – Created a Mongoose schema with fields: `name`, `email`, and `age`.
4. **Implemented CRUD Operations** – Built RESTful routes (`POST`, `GET`, `PUT`, `DELETE`) for user management.
5. **Handled Errors Properly** – Used `try-catch` blocks and meaningful error messages.
6. **Tested API** – Used `curl` testing all endpoints.

🚀 **API is now ready for integration!**

