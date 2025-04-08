<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Redirect to Index</title>
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
        }

        .redirect-btn {
            padding: 15px 30px;
            font-size: 1.2rem;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        .redirect-btn:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>

    <button class="redirect-btn" onclick="window.location.href='http://127.0.0.1:5000/'">Go to Index Page</button>

</body>
</html>
