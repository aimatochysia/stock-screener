# Stock Screener

A web-based stock screening system designed for the Jakarta Stock Exchange (JKSE). It automates daily data collection, performs technical analysis, and presents the results through an interactive frontend.

---

## 📌 Overview

The Stock Screener operates as a distributed system with automated data pipelines, multi-repository data storage, and a modern React frontend. It processes approximately 1,000 JKSE stocks daily, providing users with comprehensive financial and technical insights.

---

## 🧱 System Architecture

The architecture follows a producer-consumer pattern, where backend Python scripts collect and process stock data, storing results across multiple GitHub repositories that serve as a distributed data layer for the frontend React application.

### Key Components:

- **Data Collection (`getter.py`)**: Fetches daily OHLCV data for ~1,000 JKSE stocks using the Yahoo Finance API.
- **Data Aggregation (`infoer.py`)**: Consolidates data across repositories and enriches it with stock metadata.
- **Technical Analysis (`logiczer.py`)**: Calculates technical indicators like RSI, ATR, SMA, and identifies support/resistance levels.
- **Stock List Management (`lister.py`)**: Maintains the master ticker list from sahamidx.com.
- **Workflow Orchestration**: Automated via GitHub Actions to execute daily/weekly data processing.
- **Data Storage**: Distributed across 7 `stock-db-*` repositories and a `stock-results` repository.
- **Frontend Interface**: Built with React.js and hosted on Vercel, providing interactive stock screening and visualization.

---

## 🔄 Data Processing Pipeline

The system implements a sequential three-stage pipeline executed daily via GitHub Actions:
1. **Collection (`getter.py`)**: Retrieves raw OHLCV data from Yahoo Finance and shards it across multiple repositories.
2. **Aggregation (`infoer.py`)**: Consolidates data and enriches it with stock metadata and company information.
3. **Analysis (`logiczer.py`)**: Computes technical indicators, support/resistance levels, and generates trading signals.

---

## ⚙️ Technology Stack

### Backend:

- **Python**: For data collection and processing using the `yfinance` library.
- **GitHub Actions**: For workflow orchestration with scheduled execution.
- **Git Repositories**: For distributed data storage across multiple repositories.
- **External APIs**: Yahoo Finance for market data, sahamidx.com for stock listings.

### Frontend:

- **React.js**: For building the user interface with a component-based architecture.
- **Vercel**: For hosting the frontend application.
- **Data Fetching**: Custom `useStockData` hook for API integration.
- **UI Components**: `StockTable` with infinite scrolling, `StockChart` with technical indicators.

---

## 🚀 Getting Started

### Prerequisites:

- **Node.js**: Ensure Node.js is installed.
- **Yarn**: Install Yarn package manager.

### Installation:

1. Clone the repository:

   ```bash
   git clone https://github.com/mirellarhea/stock-screener.git
   cd stock-screener

2. Install dependencies:
   
   ```bash
   yarn install

3. Run the development server:
   ```bash
   yarn dev
The application will be available at http://localhost:3000.

## 📈 Features

### Prerequisites:

- **Stock Screening**: Filter stocks based on various financial and technical criteria.
- **Technical Indicators**: View indicators like RSI, ATR, SMA, and support/resistance levels.
- **Interactive Charts**: Visualize stock data with interactive charts.
- **Infinite Scrolling**: Browse through a large number of stocks seamlessly.

## 🔐 Configuration & Environment
Configuration settings and environment variables are managed through `.env` files. Ensure that the necessary environment variables are set before running the application.

## 🧪 Development & Testing
- **Development Server**: Use `yarn dev` to run the development server.
- **Testing**: Implement unit and integration tests to ensure the reliability of the system.
- **Linting**: Use ESLint and Prettier for code quality and formatting.

## 📂 Project Structure
The project is organized as follows:
```plaintext
stock-screener/
├── .env               # Environment variables
├── .gitignore         # Git ignore file
├── package.json       # Project metadata and dependencies
├── yarn.lock          # Yarn lock file
├── src/               # Source code
│   ├── components/    # React components
│   ├── hooks/         # Custom hooks
│   ├── pages/         # Page components
│   └── utils/         # Utility functions
└── public/            # Public assets
    └── images/        # Image assets
```
## 📄 License
This project is licensed under the MIT License.
