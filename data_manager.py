#!/usr/bin/env python3
"""
Data Manager for AI Portfolio Manager
Serves analysts with relevant data from SQLite database.
"""

import pandas as pd
import sqlite3
import json
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """Manages data access for all analysts in the AI Portfolio Manager system using SQLite."""
    
    def __init__(self, db_path: str = None):
        """Initialize DataManager with SQLite connection."""
        if db_path is None:
            # Get database name from environment variable, default to 'aipm.db'
            db_name = os.getenv('DB_NAME', 'aipm.db')
            db_path = f'data/{db_name}'
        
        self.db_path = db_path
        self.connection = None
        self._connect()
        self._ensure_portfolio_table()
        logger.info(f"DataManager initialized with SQLite database: {db_path}")
    
    def _connect(self):
        """Establish connection to SQLite database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            # Enable foreign keys and set row factory for named tuples
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.row_factory = sqlite3.Row
            logger.info("SQLite connection established")
        except Exception as e:
            logger.error(f"Error connecting to SQLite database: {e}")
            raise
    
    def _execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute a query and return results."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []
    
    def _execute_write(self, query: str, params: tuple = ()): 
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error executing write query: {e}")
            return False
    
    def get_financials(self, ticker: str, analysis_date: str, num_quarters: int = 4) -> Dict:
        """
        Get financial statements for a ticker up to the analysis date.
        
        Args:
            ticker: Stock ticker symbol
            analysis_date: Analysis date in YYYY-MM-DD format
            num_quarters: Number of quarters to retrieve (default 4, use 8 for TTM growth calculations)
            
        Returns:
            Dict containing income statement, balance sheet, and cash flow data
        """
        try:
            # Get income statement data
            income_query = """
                SELECT * FROM income_statement_quarterly 
                WHERE ticker = ? AND report_date <= ?
                ORDER BY report_date DESC
                LIMIT ?
            """
            income_data = self._execute_query(income_query, (ticker, analysis_date, num_quarters))
            
            # Get balance sheet data
            balance_query = """
                SELECT * FROM balance_statement_quarterly 
                WHERE ticker = ? AND report_date <= ?
                ORDER BY report_date DESC
                LIMIT ?
            """
            balance_data = self._execute_query(balance_query, (ticker, analysis_date, num_quarters))
            
            # Get cash flow data
            cashflow_query = """
                SELECT * FROM cashflow_statement_quarterly 
                WHERE ticker = ? AND report_date <= ?
                ORDER BY report_date DESC
                LIMIT ?
            """
            cashflow_data = self._execute_query(cashflow_query, (ticker, analysis_date, num_quarters))
            
            if not income_data and not balance_data and not cashflow_data:
                logger.warning(f"No financial data found for {ticker}")
                return None
            
            # Convert to the expected format
            financials = {
                'income_statement': {
                    'quarterlyReports': [dict(row) for row in income_data]
                },
                'balance_sheet': {
                    'quarterlyReports': [dict(row) for row in balance_data]
                },
                'cash_flow': {
                    'quarterlyReports': [dict(row) for row in cashflow_data]
                }
            }
            
            logger.info(f"Retrieved financial data for {ticker} (filtered to {analysis_date})")
            return financials
            
        except Exception as e:
            logger.error(f"Error retrieving financials for {ticker}: {e}")
            return None
    
    def get_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get daily price data for a ticker within the specified date range.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        try:
            query = """
                SELECT date, open, high, low, close, volume 
                FROM stock_prices_daily 
                WHERE ticker = ? AND date BETWEEN ? AND ?
                ORDER BY date ASC
            """
            
            data = self._execute_query(query, (ticker, start_date, end_date))
            
            if not data:
                logger.warning(f"No price data found for {ticker} between {start_date} and {end_date}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame([dict(row) for row in data])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            logger.info(f"Retrieved {len(df)} price records for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving prices for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_closing_price(self, ticker: str, target_date: str) -> float:
        """
        Get the closing price for a ticker on or before the target date.
        This represents the price available to analysts on the analysis date.
        
        Args:
            ticker: Stock ticker symbol
            target_date: Target date in YYYY-MM-DD format
            
        Returns:
            Closing price as float, or None if no data available
        """
        try:
            # Get price data up to the target date
            prices = self.get_prices(ticker, '2000-01-01', target_date)
            
            if prices.empty:
                logger.warning(f"No price data found for {ticker} on or before {target_date}")
                return None
            
            # Get the most recent closing price on or before the target date
            prices = prices[prices.index <= pd.to_datetime(target_date)]
            if prices.empty:
                logger.warning(f"No price data found for {ticker} on or before {target_date}")
                return None
            
            closing_price = prices['close'].iloc[-1]
            logger.info(f"Retrieved closing price for {ticker} on {prices.index[-1].strftime('%Y-%m-%d')}: ${closing_price:.2f}")
            return float(closing_price)
            
        except Exception as e:
            logger.error(f"Error retrieving closing price for {ticker}: {e}")
            return None
    
    def get_sector_info(self, ticker: str) -> Dict:
        """
        Get sector information for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict containing sector, industry, and sector_etf
        """
        try:
            query = "SELECT * FROM sector_information WHERE ticker = ?"
            result = self._execute_query(query, (ticker,))
            
            if not result:
                logger.warning(f"No sector information found for {ticker}")
                return {
                    'sector': 'Unknown',
                    'industry': 'Unknown',
                    'sector_etf': 'SPY'
                }
            
            row = result[0]
            sector_info = {
                'sector': row['sector'] or 'Unknown',
                'industry': row['industry'] or 'Unknown',
                'sector_etf': row['sector_etf'] or 'SPY'
            }
            
            logger.info(f"Retrieved sector info for {ticker}: {sector_info['sector']} ({sector_info['sector_etf']})")
            return sector_info
            
        except Exception as e:
            logger.error(f"Error retrieving sector info for {ticker}: {e}")
            return {
                'sector': 'Unknown',
                'industry': 'Unknown',
                'sector_etf': 'SPY'
            }
    
    def get_macro_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get macroeconomic data within the specified date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with macroeconomic indicators
        """
        try:
            query = """
                SELECT * FROM macroeconomic_monthly 
                WHERE date BETWEEN ? AND ?
                ORDER BY date ASC
            """
            
            data = self._execute_query(query, (start_date, end_date))
            
            if not data:
                logger.warning(f"No macro data found between {start_date} and {end_date}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame([dict(row) for row in data])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            logger.info(f"Retrieved {len(df)} macro records")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving macro data: {e}")
            return pd.DataFrame()
    
    def get_sentiment_data(self, ticker: str, analysis_date: str) -> Dict:
        """
        Get sentiment data for a ticker for the month of the analysis date.
        For monthly sentiment data, we use the sentiment from the month of the analysis date.
        
        Args:
            ticker: Stock ticker symbol
            analysis_date: Analysis date in YYYY-MM-DD format
            
        Returns:
            Dict containing sentiment score and explanation, or None if no data found
        """
        try:
            # Convert analysis date to month start (YYYY-MM-01)
            from datetime import datetime
            analysis_dt = datetime.strptime(analysis_date, '%Y-%m-%d')
            month_start = analysis_dt.replace(day=1).strftime('%Y-%m-%d')
            
            # Get sentiment for the month of the analysis date
            query = """
                SELECT sentiment, explanation 
                FROM sentiment_monthly 
                WHERE ticker = ? AND date = ?
            """
            
            data = self._execute_query(query, (ticker, month_start))
            
            if data:
                row = data[0]
                sentiment_data = {
                    'sentiment': row['sentiment'],
                    'explanation': row['explanation']
                }
                logger.info(f"Found sentiment data for {ticker} for month {month_start}")
                return sentiment_data
            
            # If no data for that month, get the latest sentiment before the analysis date
            query_latest = """
                SELECT sentiment, explanation 
                FROM sentiment_monthly 
                WHERE ticker = ? AND date <= ?
                ORDER BY date DESC
                LIMIT 1
            """
            
            data_latest = self._execute_query(query_latest, (ticker, analysis_date))
            
            if data_latest:
                row = data_latest[0]
                sentiment_data = {
                    'sentiment': row['sentiment'],
                    'explanation': row['explanation']
                }
                logger.info(f"Using latest sentiment data for {ticker} before {analysis_date}")
                return sentiment_data
            
            logger.warning(f"No sentiment data found for {ticker} on or before {analysis_date}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving sentiment data for {ticker}: {e}")
            return None
    
    def get_sector_etf_prices(self, sector_etf: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get sector ETF prices for comparison.
        
        Args:
            sector_etf: Sector ETF symbol (e.g., 'XLK')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with sector ETF prices
        """
        try:
            df = self.get_prices(sector_etf, start_date, end_date)
            return df
        except Exception as e:
            logger.error(f"Error retrieving sector ETF prices for {sector_etf}: {e}")
            return pd.DataFrame()
    
    def get_market_data(self, ticker: str = 'SPY', start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get market benchmark data (SPY by default).
        
        Args:
            ticker: Market benchmark ticker (default: 'SPY')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with market benchmark prices
        """
        try:
            df = self.get_prices(ticker, start_date, end_date)
            return df
        except Exception as e:
            logger.error(f"Error retrieving market data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_all_data_for_analysis(self, ticker: str, analysis_date: str) -> Dict:
        """
        Get all necessary data for a complete analysis.
        
        Args:
            ticker: Stock ticker symbol
            analysis_date: Analysis date in YYYY-MM-DD format
            
        Returns:
            Dict containing all data needed for analysis
        """
        try:
            # Calculate date ranges
            analysis_dt = datetime.strptime(analysis_date, '%Y-%m-%d')
            start_date = '2020-01-01'  # Fixed start date as per requirements
            end_date = analysis_date
            
            # Get all data
            financials = self.get_financials(ticker, analysis_date)
            prices = self.get_prices(ticker, start_date, end_date)
            sector_info = self.get_sector_info(ticker)
            macro_data = self.get_macro_data(start_date, end_date)
            sector_etf_prices = self.get_sector_etf_prices(sector_info['sector_etf'], start_date, end_date)
            market_data = self.get_market_data('SPY', start_date, end_date)
            sentiment_data = self.get_sentiment_data(ticker, analysis_date)
            
            data_package = {
                'ticker': ticker,
                'analysis_date': analysis_date,
                'financials': financials,
                'prices': prices,
                'sector_info': sector_info,
                'macro_data': macro_data,
                'sector_etf_prices': sector_etf_prices,
                'market_data': market_data,
                'sentiment_data': sentiment_data
            }
            
            logger.info(f"Retrieved complete data package for {ticker}")
            return data_package
            
        except Exception as e:
            logger.error(f"Error retrieving complete data package for {ticker}: {e}")
            return None
    
    def close_connection(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def _ensure_analysis_outputs_table(self):
        """Ensure the analysis_outputs table exists."""
        try:
            # First, check if table exists and has correct schema
            cursor = self.connection.cursor()
            cursor.execute("PRAGMA table_info(analysis_outputs)")
            columns = cursor.fetchall()
            
            if not columns:
                # Table doesn't exist, create it
                create_table_query = """
                CREATE TABLE analysis_outputs (
                    ticker TEXT,
                    date DATE,
                    analysis_output TEXT,
                    created_at DATE,
                    PRIMARY KEY (ticker, date)
                )
                """
                self.connection.execute(create_table_query)
                self.connection.commit()
                logger.info("analysis_outputs table created")
            else:
                # Check if schema is correct
                column_names = [col[1] for col in columns]
                if 'analysis_output' not in column_names or 'result' in column_names:
                    # Drop and recreate with correct schema
                    self.connection.execute("DROP TABLE IF EXISTS analysis_outputs")
                    create_table_query = """
                    CREATE TABLE analysis_outputs (
                        ticker TEXT,
                        date DATE,
                        analysis_output TEXT,
                        created_at DATE,
                        PRIMARY KEY (ticker, date)
                    )
                    """
                    self.connection.execute(create_table_query)
                    self.connection.commit()
                    logger.info("analysis_outputs table recreated with correct schema")
                else:
                    logger.info("analysis_outputs table already exists with correct schema")
                    
        except Exception as e:
            logger.error(f"Error ensuring analysis_outputs table: {e}")
            raise
    
    def save_analysis_output(self, ticker: str, analysis_date: str, result: Dict) -> bool:
        """
        Save or update analysis output in the database.
        
        Args:
            ticker: Stock ticker symbol
            analysis_date: Analysis date in YYYY-MM-DD format
            result: Analysis result dictionary to be JSON stringified
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._ensure_analysis_outputs_table()
            
            # Convert result to JSON string
            import json
            result_json = json.dumps(result, indent=2)
            
            # Get current date for created_at
            from datetime import datetime
            created_at = datetime.now().strftime('%Y-%m-%d')
            
            # Use INSERT OR REPLACE to handle overwriting existing entries
            query = """
            INSERT OR REPLACE INTO analysis_outputs (ticker, date, analysis_output, created_at)
            VALUES (?, ?, ?, ?)
            """
            
            self.connection.execute(query, (ticker, analysis_date, result_json, created_at))
            self.connection.commit()
            
            logger.info(f"Analysis output saved to database for {ticker} on {analysis_date}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving analysis output for {ticker}: {e}")
            return False
    
    def get_analysis_output(self, ticker: str, analysis_date: str) -> Dict:
        """
        Retrieve analysis output from the database.
        
        Args:
            ticker: Stock ticker symbol
            analysis_date: Analysis date in YYYY-MM-DD format
            
        Returns:
            Dict: Analysis result or None if not found
        """
        try:
            self._ensure_analysis_outputs_table()
            
            query = "SELECT analysis_output FROM analysis_outputs WHERE ticker = ? AND date = ?"
            result = self._execute_query(query, (ticker, analysis_date))
            
            if result:
                import json
                return json.loads(result[0]['analysis_output'])
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving analysis output for {ticker}: {e}")
            return None
    
    def _ensure_signals_table(self):
        """Ensure the signals table exists with correct schema."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("PRAGMA table_info(signals)")
            columns = cursor.fetchall()
            
            if not columns:
                # Table doesn't exist, create it
                create_table_query = """
                CREATE TABLE signals (
                    ticker TEXT,
                    date DATE,
                    signal TEXT,
                    created_at DATE,
                    PRIMARY KEY (ticker, date)
                )
                """
                self.connection.execute(create_table_query)
                self.connection.commit()
                logger.info("signals table created")
            else:
                # Check if schema is correct
                column_names = [col[1] for col in columns]
                required_columns = ['ticker', 'date', 'signal', 'created_at']
                
                if not all(col in column_names for col in required_columns):
                    # Drop and recreate with correct schema
                    self.connection.execute("DROP TABLE IF EXISTS signals")
                    create_table_query = """
                    CREATE TABLE signals (
                        ticker TEXT,
                        date DATE,
                        signal TEXT,
                        created_at DATE,
                        PRIMARY KEY (ticker, date)
                    )
                    """
                    self.connection.execute(create_table_query)
                    self.connection.commit()
                    logger.info("signals table recreated with correct schema")
                else:
                    logger.info("signals table already exists with correct schema")
                    
        except Exception as e:
            logger.error(f"Error ensuring signals table: {e}")
            raise
    
    def save_signal(self, signal_result: Dict) -> bool:
        """
        Save signal result to the database.
        
        Args:
            signal_result: Signal result dictionary
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._ensure_signals_table()
            
            ticker = signal_result['ticker']
            analysis_date = signal_result['analysis_date']
            signal_json = json.dumps(signal_result)
            created_at = datetime.now().strftime('%Y-%m-%d')
            
            # Use INSERT OR REPLACE to handle overwriting existing entries
            query = """
            INSERT OR REPLACE INTO signals 
            (ticker, date, signal, created_at)
            VALUES (?, ?, ?, ?)
            """
            
            self.connection.execute(query, (ticker, analysis_date, signal_json, created_at))
            self.connection.commit()
            
            logger.info(f"Signal saved to database for {ticker} on {analysis_date}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            return False
    
    def get_signal(self, ticker: str, analysis_date: str) -> Dict:
        """
        Retrieve signal from the database.
        
        Args:
            ticker: Stock ticker symbol
            analysis_date: Analysis date in YYYY-MM-DD format
            
        Returns:
            Dict: Signal result or None if not found
        """
        try:
            self._ensure_signals_table()
            
            query = """
            SELECT ticker, date, signal, created_at
            FROM signals WHERE ticker = ? AND date = ?
            """
            result = self._execute_query(query, (ticker, analysis_date))
            
            if result:
                row = result[0]
                signal_result = json.loads(row['signal'])
                return signal_result
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving signal for {ticker}: {e}")
            return None 

    def _ensure_portfolio_table(self):
        query = '''
        CREATE TABLE IF NOT EXISTS portfolio (
            portfolio_name TEXT,
            date DATE,
            call TEXT,
            response TEXT,
            state TEXT,
            notes TEXT
        )'''
        self._execute_query(query)

    def save_portfolio(self, portfolio_name: str, date: str, call: str, response: str, state: str, notes: str):
        query = '''
        INSERT INTO portfolio (portfolio_name, date, call, response, state, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        '''
        self._execute_write(query, (portfolio_name, date, call, response, state, notes))

    def get_portfolio_state(self, portfolio_name: str, analysis_date: str):
        query = '''
        SELECT state FROM portfolio WHERE portfolio_name = ? AND date <= ? ORDER BY date DESC LIMIT 1
        '''
        result = self._execute_query(query, (portfolio_name, analysis_date))
        if result and result[0]['state']:
            return json.loads(result[0]['state'])
        return None

    def get_latest_signal(self, ticker: str):
        query = '''
        SELECT signal FROM signals WHERE ticker = ? ORDER BY date DESC LIMIT 1
        '''
        result = self._execute_query(query, (ticker,))
        if result and result[0]['signal']:
            return json.loads(result[0]['signal'])
        return None 

    def get_portfolio_call(self, portfolio_name: str, analysis_date: Optional[str] = None):
        """Retrieve the most recent call (input parameters) for a portfolio, optionally up to a given date."""
        if analysis_date:
            query = '''
            SELECT call FROM portfolio WHERE portfolio_name = ? AND date <= ? ORDER BY date DESC LIMIT 1
            '''
            result = self._execute_query(query, (portfolio_name, analysis_date))
        else:
            query = '''
            SELECT call FROM portfolio WHERE portfolio_name = ? ORDER BY date DESC LIMIT 1
            '''
            result = self._execute_query(query, (portfolio_name,))
        if result and result[0]['call']:
            return json.loads(result[0]['call'])
        return None 

    def get_latest_portfolio_state_date(self, portfolio_name: str) -> Optional[str]:
        """Return the latest date for a given portfolio from the portfolio table."""
        query = '''
        SELECT date FROM portfolio WHERE portfolio_name = ? ORDER BY date DESC LIMIT 1
        '''
        result = self._execute_query(query, (portfolio_name,))
        if result and result[0]['date']:
            return result[0]['date']
        return None 

    def delete_portfolio_row(self, portfolio_name: str, date: str):
        query = '''
        DELETE FROM portfolio WHERE portfolio_name = ? AND date = ?
        '''
        self._execute_write(query, (portfolio_name, date)) 