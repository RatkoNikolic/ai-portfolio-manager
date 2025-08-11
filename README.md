 # 🧠 AI Portfolio Manager (AIPM)

A comprehensive AI-driven portfolio management system that combines multi-analyst quantitative analysis with large language model (LLM) decision-making for systematic trading signal generation and portfolio optimization.

This project is part of the **master's thesis** *"Automating Portfolio Management Using a Multi-Agent System Based on Large Language Models"* for the Master in Computational Finance program at the School of Computing at the Union University, Belgrade.

**Author:** Ratko Nikolić  
**Mentor:** Prof. Dr. Branko Urošević  
**Institution:** School of Computing, Union University, Belgrade  
**Program:** [Master in Computational Finance](https://mcf.raf.edu.rs/)  
**Academic Year:** 2024/2025 

## 🏗️ System Architecture

### Core Components

**📊 Multi-Analyst Framework**
- **Fundamentals Analyst**: TTM-based financial statement analysis, growth metrics, and company health indicators with backtesting-aware price data
- **Valuation Analyst**: P/E, P/B, PEG ratios, DCF-based valuations, and relative valuation using previous trading day prices for backtesting accuracy
- **Technical Analyst**: Moving averages, RSI, MACD, Bollinger Bands, and momentum indicators with corrected MA ratio calculations and backtesting-aware data handling  
- **Risk Analyst**: Volatility, beta, drawdown, and risk-adjusted return metrics with backtesting-aware data handling and properly annualized risk metrics
- **Sentiment Analyst**: Market sentiment analysis and behavioral indicators
- **Macro Analyst**: Minimal 7-metric macro set with explicit date alignment (gdp_growth_pct, inflation_yoy_pct, unemployment_rate_pct, fed_funds_rate_pct, yield_spread_pct, credit_spread_bp, vix_index)

**🤖 AI-Powered Decision Layer**
- **Signal Analyst**: LLM-based trading signal generation from multi-analyst outputs (supports: gpt-4o, gpt-4o-mini)
- **Portfolio Manager**: Intelligent portfolio construction, rebalancing, and management (supports: gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini, o3-mini, o4-mini, gpt-5)
- **Orchestrator**: Coordinates all components with automatic dependency management

**💾 Data Management**
- **SQLite Database**: Stores raw financial data, analysis outputs, signals, and portfolio states
- **Data Manager**: Handles all database operations and data retrieval
- **Automatic Dependency Resolution**: Creates missing analysis outputs and signals as needed

### Key Features

- 🔄 **Automatic Dependency Management**: Missing analysis outputs and signals are created automatically
- 📈 **Comprehensive Analysis**: Six specialized analysts provide multi-dimensional market insights
- 🎯 **LLM-Driven Signals**: GPT-4o powered trading signal generation with confidence scoring
- 💼 **Portfolio Management**: Create, rebalance, and manage portfolios with customizable parameters
- 🕐 **Backtesting Support**: Historical analysis across multiple market regimes (2021-2023)
- 🐛 **Debug Mode**: Detailed analysis outputs for development and validation

## 📚 Documentation

This system includes the following documentation files:

- **`README.md`** (this file): Complete setup guide, usage instructions, and parameter reference
- **`notes/SYSTEM_ARCHITECTURE.md`**: Detailed technical documentation covering:
  - System architecture and component relationships  
  - Database schema and data flow diagrams
  - Complete analyst specifications (roles, metrics, formulas, examples)
  - AI decision layer details (Signal Analyst, Portfolio Manager)
  - Real input/output examples and implementation details
- **`notes/cli_calls.txt`**: Comprehensive command-line usage examples including:
  - Database creation and population commands
  - Analysis workflows (single/batch ticker and date combinations)
  - Signal generation examples
  - Portfolio management scenarios
  - Backtesting and testing commands
- **`data/analysis.ipynb`**: Interactive Jupyter notebook for exploring and experimenting with the analyst outputs and managed portfolios generated for this thesis

## 🔧 Prerequisites

### Technical Requirements
- **Python 3.9+** (recommended: Python 3.10 or 3.11 for optimal compatibility)
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 8GB RAM (16GB recommended for large-scale analysis)
- **Storage**: 2GB free space for database and analysis outputs

### API Keys Required

- **Alpha Vantage API Key**: Financial statement data
  - Get free key: https://www.alphavantage.co/support/#api-key
  - Rate limit: 5 calls/minute (free), 75 calls/minute (premium)
- **FRED API Key**: Macroeconomic data from Federal Reserve
  - Get free key: https://fred.stlouisfed.org/docs/api/api_key.html
  - No rate limits for personal use
- **OpenAI API Key**: Signal generation and portfolio management
  - Required for: Signal Analyst and Portfolio Manager
  - Signal Analyst models: gpt-4o, gpt-4o-mini
  - Portfolio Manager models: gpt-4o (default), gpt-4o-mini, gpt-4.1, gpt-4.1-mini, o3-mini, o4-mini
  - Get key: https://platform.openai.com/api-keys

## 🚀 Setup Instructions

### Environment Setup

1. **Clone Repository**
```bash
git clone <repository-url>
cd aipm-thesis
```

2. **Create Virtual Environment**
```bash
python -m venv aipm-venv

# Windows
aipm-venv\Scripts\activate

# macOS/Linux  
source aipm-venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment Variables**
Create a `.env` file in the project root:
```env
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
FRED_API_KEY=your_fred_key_here
OPENAI_API_KEY=your_openai_key_here
DB_NAME=aipm_thesis.db
```

### Database Setup Options

#### Option 1: Use Existing Database (Recommended for Quick Start)
The repository includes `data/aipm_thesis.db` with pre-populated data for the **top 25 S&P 500 companies** covering **2021-2023** (36 months, ~900 analysis data points).

**Included Companies**: AAPL, NVDA, MSFT, AMZN, META, GOOGL, AVGO, TSLA, BRK-B, GOOG, JPM, LLY, V, XOM, COST, MA, UNH, NFLX, WMT, PG, JNJ, HD, ABBV, BAC, CRM

**Ready to use immediately** - no additional setup required!

#### Option 2: Create New Database
```bash
# Create database with default tickers (recommended)
python create_db.py

# Create database with custom tickers
python create_db.py --tickers AAPL MSFT GOOGL AMZN TSLA

# Create database with specific date range
python create_db.py --tickers AAPL MSFT --start-date 2020-01-01 --end-date 2023-12-31

# Create database with custom name
python create_db.py --db-name my_custom_db.db --tickers AAPL MSFT
```

**⚠️ Note**: Database creation takes 10-15 minutes depending on the number of tickers due to API rate limits.

## 📖 Usage Guide

### Basic Analysis Commands

#### Single Ticker Analysis
```bash
# Complete analysis (all 6 analysts)
python main.py --ticker MSFT --date 2021-09-01

# Single analyst
python main.py --ticker MSFT --date 2021-09-01 --analyst fundamentals
python main.py --ticker MSFT --date 2021-09-01 --analyst technical

# Debug mode (saves detailed outputs to test/ folder)
python main.py --ticker MSFT --date 2021-09-01 --debug
```

#### Batch Analysis
```bash
# Multiple tickers, single date
python main.py --tickers AAPL MSFT GOOGL --date 2021-09-01

# Multiple tickers, multiple dates  
python main.py --tickers AAPL MSFT --dates 2021-09-01 2021-08-01 2021-07-01
```

### Signal Generation

#### Generate Trading Signals
```bash
# Single ticker signal
python main.py --ticker MSFT --date 2021-10-01 --signal

# Batch signal generation
python main.py --tickers AAPL MSFT GOOGL --date 2021-10-01 --signal

# Custom model selection (Signal Analyst supports: gpt-4o, gpt-4o-mini)
python main.py --ticker MSFT --date 2021-10-01 --signal --model gpt-4o-mini
```

### Portfolio Management

The Portfolio Manager supports a broader range of models than the Signal Analyst.

**Supported Models:**
- `gpt-4o` (default)
- `gpt-4o-mini`
- `gpt-4.1`
- `gpt-4.1-mini` 
- `o3-mini`
- `o4-mini`
- `gpt-5` 

#### Create Portfolio
```bash
python main.py --create_portfolio \
  --portfolio_name my_portfolio \
  --portfolio_date 2021-01-01 \
  --investment 100000 \
  --num_tickers 10 \
  --tickers AAPL MSFT NVDA AMZN META GOOGL AVGO TSLA BRK-B GOOG \
  --risk normal

# With custom model
python main.py --create_portfolio \
  --portfolio_name my_portfolio \
  --portfolio_date 2021-01-01 \
  --investment 100000 \
  --num_tickers 10 \
  --tickers AAPL MSFT NVDA AMZN META GOOGL AVGO TSLA BRK-B GOOG \
  --risk normal \
  --model o3-mini
```

#### Rebalance Portfolio
```bash
python main.py --rebalance_portfolio \
  --portfolio_name my_portfolio \
  --rebalance_date 2021-04-01 \
  --num_tickers 10 \
  --tickers AAPL MSFT NVDA AMZN META GOOGL AVGO TSLA BRK-B GOOG \
  --risk normal

# With custom model
python main.py --rebalance_portfolio \
  --portfolio_name my_portfolio \
  --rebalance_date 2021-04-01 \
  --num_tickers 10 \
  --tickers AAPL MSFT NVDA AMZN META GOOGL AVGO TSLA BRK-B GOOG \
  --risk normal \
  --model gpt-4.1
```

#### Manage Portfolio (Create + Periodic Rebalancing)
```bash
python main.py --manage_portfolio \
  --portfolio_name my_portfolio \
  --start_date 2021-01-01 \
  --end_date 2021-12-31 \
  --num_tickers 10 \
  --tickers AAPL MSFT NVDA AMZN META GOOGL AVGO TSLA BRK-B GOOG \
  --risk normal \
  --investment 100000 \
  --frequency monthly

# With custom model  
python main.py --manage_portfolio \
  --portfolio_name my_portfolio \
  --start_date 2021-01-01 \
  --end_date 2021-12-31 \
  --num_tickers 10 \
  --tickers AAPL MSFT NVDA AMZN META GOOGL AVGO TSLA BRK-B GOOG \
  --risk normal \
  --investment 100000 \
  --frequency monthly \
  --model o4-mini
```

### 📚 Extensive Examples
For comprehensive usage examples, see `notes/cli_calls.txt` which contains:
- Complete analysis workflows
- Backtesting commands
- Portfolio management scenarios
- Advanced configuration options
- Database creation and population commands

## 📊 Real Example: MSFT Analysis (2021-10-01)

### Complete Analysis Output

**Multi-Analyst Results:**
```json
{
  "fundamentals": {
    "revenue_growth_ttm": 0.1981,      // 19.81% TTM growth
    "eps_ttm": 8.9434,                 // $8.94 earnings per share
    "roe_ttm": 0.4467,                 // 44.67% return on equity
    "operating_margin_ttm": 0.4214,    // 42.14% operating margin
    "current_ratio": 2.1648,           // Strong liquidity position
    "debt_to_equity": 1.207            // Moderate leverage
  },
  "valuation": {
    "price_earnings_ratio": 25.21,     // P/E using previous day price ($273.24)
    "pe_growth_ratio": 1.02,           // PEG ratio - growth fairly priced
    "price_book_ratio": 13.36,         // P/B ratio
    "ev_ebitda_ratio": 21.96,          // EV/EBITDA multiple
    "price_sales_ratio": 11.2,         // P/S ratio
    "price_fcf_ratio": 20.69           // Price/Free Cash Flow
  },
  "technical": {
    "price_momentum_12m": 0.3391,      // 33.91% 12-month momentum
    "ma_50_200_ratio": 1.1472,         // Bullish MA crossover (50/200)
    "rsi_14d": 32.19,                  // Near oversold territory
    "macd_line": -2.1242,              // Momentum weakness in pullback
    "volume_trend_30d": "neutral",     // Balanced volume pattern
    "bb_width_pct": 0.0859            // Bollinger Band width
  },
  "risk": {
    "market_beta": 1.1506,             // Moderate market correlation
    "sharpe_ratio": 1.1053,            // Strong risk-adjusted returns
    "sortino_ratio": 1.4367,           // Excellent downside protection
    "max_drawdown_pct": -0.2804,       // -28.04% max drawdown
    "volatility_30d_pct": 0.1962,      // 19.62% annualized volatility
    "tracking_error_pct": 0.093704     // 9.37% vs sector tracking error
  },
  "sentiment": {
    "sentiment_score": 1,              // Highly positive sentiment
    "explanation": "New Surface devices and security enhancements drive positive outlook"
  },
  "macro": {
    "gdp_growth_pct": 7.4,             // Strong economic expansion
    "inflation_yoy_pct": 6.23,         // Elevated inflation levels
    "unemployment_rate_pct": 4.5,      // Recovering labor market
    "fed_funds_rate_pct": 0.08,        // Accommodative monetary policy
    "vix_index": 21.15                 // Elevated market uncertainty
  }
}
```

### Generated Trading Signal

```json
{
  "ticker": "MSFT",
  "signal": "buy",
  "confidence": 0.88,
  "time_horizon": "3M",
  "explanation": "Strong fundamentals with 42.14% operating margins, reasonable valuation with PEG of 1.02, and technical indicators suggesting rebound potential from oversold levels."
}
```

### Portfolio Construction Example

```json
{
  "date": "2021-10-01",
  "portfolio_value": 100000.0,
  "allocations": [
    {"ticker": "AAPL", "allocation": 0.15, "shares": 108.16, "value": 15000},
    {"ticker": "MSFT", "allocation": 0.15, "shares": 54.90, "value": 15000},
    {"ticker": "GOOGL", "allocation": 0.15, "shares": 112.88, "value": 15000},
    {"ticker": "TSLA", "allocation": 0.10, "shares": 38.69, "value": 10000},
    {"ticker": "NVDA", "allocation": 0.05, "shares": 241.83, "value": 5000}
  ]
}
```

## 📋 Available Parameters

### Analysis Types (`--analyst`)
- `fundamentals` - TTM-based financial statement analysis, growth metrics, and profitability ratios
- `valuation` - P/E, P/B, PEG ratios, DCF-based valuations, and relative valuation
- `technical` - Moving averages, RSI, MACD, Bollinger Bands, and momentum indicators with accurate MA ratios and correction-aware interpretations
- `risk` - Volatility, beta, drawdown, and risk-adjusted return metrics with industry-standard annualized tracking error
- `sentiment` - Market sentiment analysis and behavioral indicators
- `macro` - Minimal 7-metric macro set covering growth, price, labor, policy, curve, credit, and volatility (gdp_growth_pct, inflation_yoy_pct, unemployment_rate_pct, fed_funds_rate_pct, yield_spread_pct, credit_spread_bp, vix_index)

### Models
**Portfolio Manager Models (`--model`)**
- `gpt-4o` (default) - OpenAI GPT-4 Omni
- `gpt-4o-mini` - OpenAI GPT-4 Omni Mini
- `gpt-4.1` - OpenAI GPT-4.1
- `gpt-4.1-mini` - OpenAI GPT-4.1 Mini
- `o3-mini` - OpenAI O3 Mini
- `o4-mini` - OpenAI O4 Mini
- `gpt-5` - OpenAI GPT-5

**Signal Analyst Models (`--model`)**
- `gpt-4o` (default) - OpenAI GPT-4 Omni
- `gpt-4o-mini` - OpenAI GPT-4 Omni Mini
*(Limited to models supporting structured JSON output)*

### Risk Preferences (`--risk`)
- `low` - Conservative allocation with focus on stability and capital preservation
- `normal` (default) - Balanced approach between growth and stability
- `high` - Aggressive allocation with focus on growth and higher risk tolerance

### Portfolio Management Frequency (`--frequency`)
- `monthly` - Rebalance on the 1st of each month
- `quarterly` - Rebalance on the 1st of each quarter (Jan, Apr, Jul, Oct)

### Sector Preferences (`--sectors`)
Available S&P 500 sectors for sector-focused allocation:
- `"Technology"` - Technology companies (ETF: XLK)
- `"Financial Services"` - Financial and banking services (ETF: XLF)
- `"Healthcare"` - Healthcare and pharmaceutical companies (ETF: XLV)
- `"Consumer Cyclical"` - Consumer discretionary goods and services (ETF: XLY)
- `"Consumer Defensive"` - Consumer staples and necessities (ETF: XLP)
- `"Industrials"` - Industrial and manufacturing companies (ETF: XLI)
- `"Communication Services"` - Telecommunications and media (ETF: XLC)
- `"Utilities"` - Utility companies (ETF: XLU)
- `"Energy"` - Energy and oil companies (ETF: XLE)
- `"Basic Materials"` - Materials and chemicals (ETF: XLB)
- `"Real Estate"` - Real estate investment trusts (ETF: XLRE)

**Usage Examples:**
```bash
# Single sector preference
--sectors "Technology"

# Multiple sector preferences
--sectors "Technology" "Healthcare" "Financial Services"
```

### Trading Signal Values
Generated by Signal Analyst:
- `strong_buy` - Highest conviction bullish recommendation
- `buy` - Bullish recommendation with moderate conviction
- `hold` - Neutral recommendation (no clear directional edge)
- `sell` - Bearish recommendation with moderate conviction
- `strong_sell` - Highest conviction bearish recommendation

### Signal Time Horizons
- `1M` - 1 month investment horizon
- `3M` - 3 month investment horizon (default)
- `6M` - 6 month investment horizon

### Date Formats
All dates must be in `YYYY-MM-DD` format:
- `--date 2021-09-01`
- `--dates 2021-09-01 2021-08-01 2021-07-01`
- `--portfolio_date 2021-01-01`
- `--start_date 2021-01-01 --end_date 2021-12-31`

### Ticker Symbols
Use standard stock ticker symbols:
- Single ticker: `--ticker MSFT`
- Multiple tickers: `--tickers AAPL MSFT GOOGL AMZN TSLA`

**Available in Database:** AAPL, NVDA, MSFT, AMZN, META, GOOGL, AVGO, TSLA, BRK-B, GOOG, JPM, LLY, V, XOM, COST, MA, UNH, NFLX, WMT, PG, JNJ, HD, ABBV, BAC, CRM

### Investment Amounts
- `--investment 100000` - Portfolio creation initial investment amount (integer or float)
- No minimum or maximum limits, but consider practical transaction costs

### Portfolio Size
- `--num_tickers 10` - Number of tickers to include in portfolio
- Recommended range: 5-20 tickers for optimal diversification

### Debug and Output Options
- `--debug` - Enable debug mode (saves detailed outputs to `test/` folder)
- `--output filename.json` - Save results to custom file in `test/` folder
- `--verbose` - Enable verbose logging for detailed system output

## 📊 Viewing Results

### Method 1: Debug Mode (Development & Validation)
```bash
python main.py --ticker MSFT --date 2021-09-01 --debug
```