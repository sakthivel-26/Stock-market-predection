# 📈 Stock-market-predection

Welcome to **Stock-market-predection**! This project aims to provide robust tools for predicting stock market trends using machine learning, interactive dashboards, and RESTful APIs. It leverages Python, FastAPI, Streamlit, and modern data science libraries for end-to-end stock analysis and forecasting.

---

## 📝 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## 🚀 Introduction

**Stock-market-predection** is an open-source project focused on building a comprehensive stock market prediction system. It combines a REST API (with FastAPI) for backend services and a rich, interactive web frontend (with Streamlit) to visualize predictions, trends, and technical indicators. The system is containerized with Docker for easy setup and deployment.

---

## ✨ Features

- 📊 **Stock Data Visualization:** Interactive charts using Plotly and Streamlit.
- 🤖 **ML Model Integration:** Pre-trained machine learning model for stock price prediction.
- 📦 **API Server:** FastAPI-based RESTful API for programmatic access.
- 🐳 **Dockerized:** Seamless deployment with Docker.
- 🧮 **Technical Indicators:** Built-in calculation of technical analysis features using `ta`.
- 🔗 **Real-time Data:** Fetches up-to-date stock data from Yahoo Finance.
- 🌐 **CORS Enabled:** Ready for web integration.

---

## ⚡ Installation

### Prerequisites

- [Docker](https://www.docker.com/) installed on your system.

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sakthivel-26/Stock-market-predection.git
   cd Stock-market-predection
   ```

2. **Build the Docker image:**
   ```bash
   docker build -t stock-market-predection .
   ```

3. **Run the container:**
   ```bash
   docker run -p 7860:7860 -p 8000:8000 stock-market-predection
   ```

---

## 🛠 Usage

### Web Application

- Access the interactive dashboard at [http://localhost:7860](http://localhost:7860).

### API Server

- Access the REST API at [http://localhost:8000](http://localhost:8000).

#### Example Endpoints

> _See `api_server.py` for available endpoints and usage._

### Model Training

- To retrain the model, modify and run `train_model.py`.

---

## 📂 Project Structure

```plaintext
.
├── app.py             # Streamlit app for dashboard and visualization
├── api_server.py      # FastAPI server for backend prediction API
├── train_model.py     # Script to train the ML model
├── stock_model.pkl    # Pre-trained model (generated after training)
├── requirements.txt   # Python dependencies
├── Dockerfile         # Docker configuration
└── ...
```

---

## 🤝 Contributing

Contributions are welcome! 🚀

1. Fork the repo
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request


---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

> Made with ❤️ by contributors to Stock-market-predection

---

**Note:** This repository is for educational and research purposes. Trading in the stock market carries risk. Use predictions at your own discretion.

## License
This project is licensed under the **MIT** License.

---
🔗 GitHub Repo: https://github.com/sakthivel-26/Stock-market-predection