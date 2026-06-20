# Stock Direction Predictor

## Overview
A machine learning pipeline that predicts next-day stock price direction (up or down) using historical market data. Three classification models are trained and compared: Logistic Regression, Random Forest, and XGBoost. Models are tuned using hyperparameter optimization and proper time series methodology to prevent data leakage.

## Features

### Data Sources
- Stock Price Data via Yahoo Finance
- CBOE Volatility Index (^VIX) via Yahoo Finance
- 10-Year Treasury Yield (^TNX) via Yahoo Finance

### Engineered Features
- Daily returns and 3-day lagged returns
- 14-day Relative Strength Index (RSI)
- Volume change and intraday price range
- Time-based features (month, day, day of week)

### Models
- Logistic Regression (baseline)
- Random Forest Classifier
- XGBoost Classifier

All models tuned using GridSearchCV with TimeSeriesSplit cross-validation.

## Project Structure
```
stock-direction-predictor/

├── data/
│   ├── raw/            # Raw data pulled from Yahoo Finance
│   └── processed/      # Cleaned and feature engineered data
├── src/
│   ├── data_pull.py    # Downloads market data via yfinance
│   ├── features.py     # Feature engineering and data processing
│   ├── train.py        # Model training and hyperparameter tuning
│   └── evaluate.py     # Model evaluation and confusion matrices
├── models/             # Saved trained models
├── outputs/
│   └── figures/        # Confusion matrix plots
├── README.md
└── requirements.txt
```

## Installation
```bash
git clone https://github.com/Andrew45679/stock-direction-predictor.git
cd stock-direction-predictor
pip install -r requirements.txt
```

## Usage
Run each file in order:

```bash
# 1. Pull market data (you will be prompted to enter a ticker symbol)
python src/data_pull.py

# 2. Engineer features and process data
python src/features.py

# 3. Train and tune models
python src/train.py

# 4. Evaluate models and generate confusion matrices
python src/evaluate.py
```

## Results

### S&P 500 (^GSPC)
| Model | Accuracy | Precision (Up) |
|---|---|---|
| Logistic Regression | 43.95% | 29% |
| Random Forest | 44.76% | 0% |
| **XGBoost** | **48.79%** | **71%** |


*Confusion matrices for S&P 500 test data (2023-2024):*

![Logistic Regression](outputs/figures/Logistic%20Regression_confusion_matrix.png)

*Logistic Regression struggled to identify upward moves, predicting down the majority of the time.*


![Random Forest](outputs/figures/Random%20Forest%20Classifier_confusion_matrix.png)

*Random Forest predicted almost exclusively downward moves on S&P 500 test data.*


![XGBoost](outputs/figures/XGBoost%20Classifier_confusion_matrix.png)

*XGBoost performed best, correctly identifying upward moves 71% of the time.*

### Apple (AAPL)
| Model | Accuracy | Precision (Up) |
|---|---|---|
| Logistic Regression | 43.37% | 48% |
| **Random Forest** | **47.39%** | **85%** |
| XGBoost | 44.58% | 52% |

### Tesla (TSLA)
| Model | Accuracy | Precision (Up) |
|---|---|---|
| Logistic Regression | 45.38% | 53% |
| **Random Forest** | **47.79%** | **54%** |
| XGBoost | 46.59% | 52% |

## Key Findings

After training on data from 2015 to 2022, all three models were evaluated on 2023-2024 market data which turned out to be one of the most unusual periods in recent market history. Overall accuracy came in below 50% across the board, which isn't surprising given that no model trained on pre-2023 data could have anticipated the AI-driven bull run that defined that period. That said, there were some genuinely interesting results. XGBoost called upward moves on the S&P 500 correctly 71% of the time, and Random Forest hit 85% precision on AAPL upward predictions. TSLA gave the most balanced results across all three models, which makes sense given how volatile the stock tends to be. The biggest takeaway is that precision on upward predictions matters more than raw accuracy for this kind of problem.

## Methodology

Data was collected using the yfinance API, pulling daily OHLCV data for the target ticker alongside two market-wide indicators — the VIX volatility index and the 10-year Treasury yield. Features were engineered from the raw price data including daily returns, three days of lagged returns, a 14-day RSI, intraday price range, and volume change. Time-based features such as month, day, and day of week were also included to capture any seasonal patterns. The data was split chronologically with everything before 2023 used for training and 2023 onwards held out for testing, ensuring no future data leaked into the training process. All three models were tuned using GridSearchCV with TimeSeriesSplit cross-validation. Features were scaled using StandardScaler fitted only on training data before being applied to the test set.

## Future Work

- Retrain models on a rolling window basis every 3-6 months to adapt to changing market conditions
- Add news sentiment features using a financial news API to capture market moving events
- Build a Streamlit dashboard for live predictions on any ticker
- Expand evaluation to a wider range of tickers across different sectors

## Technologies Used
- yfinance
- pandas
- numpy
- xgboost
- scikit-learn
- matplotlib
- seaborn
- joblib
