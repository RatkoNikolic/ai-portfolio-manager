#!/usr/bin/env python3
"""
Orchestrator for AI Portfolio Manager
Coordinates all analysts and produces final recommendations.
"""

import json
import sys
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import sqlite3
from dotenv import load_dotenv

# Import data manager and all analysts
from data_manager import DataManager
from analysts.fundamentals_analyst import FundamentalsAnalyst
from analysts.valuation_analyst import ValuationAnalyst
from analysts.technical_analyst import TechnicalAnalyst
from analysts.risk_analyst import RiskAnalyst
from analysts.sentiment_analyst import SentimentAnalyst
from analysts.macro_analyst import MacroAnalyst
from analysts.signal_analyst import SignalAnalyst
from analysts.portfolio_manager import PortfolioManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseNotFoundError(Exception):
    """Exception raised when the required database is not found."""
    pass

class RawDataMissingError(Exception):
    """Exception raised when raw data is missing and needs to be populated."""
    pass

class Orchestrator:
    """Coordinates all analysts and produces final recommendations."""
    
    def __init__(self):
        """Initialize the Orchestrator."""
        # First check if database exists before initializing data manager
        if not self._check_database_exists():
            db_name = os.getenv('DB_NAME', 'aipm.db')
            raise DatabaseNotFoundError(
                f"Database '{db_name}' not found. Please create and populate the database first using:\n"
                f"python create_db.py\n"
                f"See notes/cli_calls.txt for more options."
            )
        
        self.dm = DataManager()
        
        # Initialize all analysts
        self.fundamentals_analyst = FundamentalsAnalyst(self.dm)
        self.valuation_analyst = ValuationAnalyst(self.dm)
        self.technical_analyst = TechnicalAnalyst(self.dm)
        self.risk_analyst = RiskAnalyst(self.dm)
        self.sentiment_analyst = SentimentAnalyst(self.dm)
        self.macro_analyst = MacroAnalyst(self.dm)
        
        # Initialize signal analyst (optional, requires OpenAI API key)
        self.signal_analyst = None
        try:
            self.signal_analyst = SignalAnalyst()
            logger.info("Signal Analyst initialized")
        except Exception as e:
            logger.warning(f"Signal Analyst not initialized: {e}")
        
        self.portfolio_manager = PortfolioManager()
        
        # Set orchestrator reference in portfolio manager for dependency management
        self.portfolio_manager.set_orchestrator(self)
        
        logger.info("Orchestrator initialized with all analysts")
    
    def _check_database_exists(self) -> bool:
        """Check if the database exists and has the required tables."""
        try:
            # Get database name from environment variable, default to 'aipm.db'
            db_name = os.getenv('DB_NAME', 'aipm.db')
            db_path = f'data/{db_name}'
            
            # Check if database file exists
            if not os.path.exists(db_path):
                logger.error(f"Database file not found: {db_path}")
                return False
            
            # Check if database has required tables
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # List of required tables
            required_tables = [
                'stock_prices_daily',
                'income_statement_quarterly',
                'balance_statement_quarterly', 
                'cashflow_statement_quarterly',
                'sector_information',
                'macroeconomic_monthly'
            ]
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            existing_tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            missing_tables = [table for table in required_tables if table not in existing_tables]
            
            if missing_tables:
                logger.error(f"Database missing required tables: {missing_tables}")
                return False
            
            logger.info(f"Database {db_path} exists with all required tables")
            return True
            
        except Exception as e:
            logger.error(f"Error checking database: {e}")
            return False

    def _check_raw_data_available(self, ticker: str, analysis_date: str) -> bool:
        """Check if required raw data is available for analysis."""
        try:
            # Try to get essential data - if this fails, raw data is missing
            closing_price = self.dm.get_closing_price(ticker, analysis_date)
            if closing_price is None:
                return False
            
            # Check if we have some financial data (any table)
            financials = self.dm.get_financials(ticker, analysis_date)
            if not financials or not any(financials.values()):
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error checking raw data for {ticker}: {e}")
            return False

    def _ensure_analysis_output_exists(self, ticker: str, analysis_date: str) -> Dict:
        """
        Ensure analysis output exists for the given ticker and date.
        If it doesn't exist, create it by running the full analysis.
        If raw data is missing, raise RawDataMissingError.
        
        Returns:
            Dict: The analysis output
        """
        try:
            # First check if analysis output already exists
            existing_output = self.dm.get_analysis_output(ticker, analysis_date)
            if existing_output:
                logger.info(f"Analysis output already exists for {ticker} on {analysis_date}")
                return existing_output
            
            # Check if raw data is available
            if not self._check_raw_data_available(ticker, analysis_date):
                raise RawDataMissingError(
                    f"Raw data missing for {ticker} on {analysis_date}. "
                    f"Please create and populate the database with the required ticker data.\n"
                    f"See the database creation commands in notes/cli_calls.txt, for example:\n"
                    f"python create_db.py --tickers {ticker}"
                )
            
            # Run the full analysis to create the missing output
            logger.info(f"Creating missing analysis output for {ticker} on {analysis_date}")
            analysis_output = self.analyze_ticker(ticker, analysis_date)
            
            if not analysis_output or not analysis_output.get('analyses'):
                raise Exception(f"Failed to create analysis output for {ticker} on {analysis_date}")
            
            logger.info(f"Successfully created analysis output for {ticker} on {analysis_date}")
            return analysis_output
            
        except RawDataMissingError:
            # Re-raise raw data errors as-is
            raise
        except Exception as e:
            logger.error(f"Error ensuring analysis output exists for {ticker}: {e}")
            raise Exception(f"Failed to create analysis output for {ticker} on {analysis_date}: {str(e)}")

    def _ensure_signal_exists(self, ticker: str, analysis_date: str, model: str = "gpt-4o") -> Dict:
        """
        Ensure signal exists for the given ticker and date.
        If it doesn't exist, create it by running signal generation.
        If analysis output is missing, create it first.
        
        Returns:
            Dict: The signal result
        """
        try:
            # First check if signal already exists
            existing_signal = self.dm.get_signal(ticker, analysis_date)
            if existing_signal:
                logger.info(f"Signal already exists for {ticker} on {analysis_date}")
                return existing_signal
            
            # Check if signal analyst is available
            if not self.signal_analyst:
                raise Exception("Signal Analyst not available (OpenAI API key required)")
            
            # Ensure analysis output exists (this will create it if missing)
            analysis_output = self._ensure_analysis_output_exists(ticker, analysis_date)
            
            # Generate the signal
            logger.info(f"Creating missing signal for {ticker} on {analysis_date}")
            signal_result = self.generate_signal(ticker, analysis_date, model)
            
            if signal_result.get('error'):
                raise Exception(f"Failed to generate signal: {signal_result['error']}")
            
            logger.info(f"Successfully created signal for {ticker} on {analysis_date}")
            return signal_result
            
        except RawDataMissingError:
            # Re-raise raw data errors as-is
            raise
        except Exception as e:
            logger.error(f"Error ensuring signal exists for {ticker}: {e}")
            raise Exception(f"Failed to create signal for {ticker} on {analysis_date}: {str(e)}")

    def ensure_signals_exist(self, tickers: List[str], analysis_date: str, model: str = "gpt-4o") -> List[Dict]:
        """
        Ensure signals exist for all given tickers and date.
        Create any missing signals automatically.
        
        Returns:
            List[Dict]: List of signal results
        """
        signals = []
        missing_data_errors = []
        
        for ticker in tickers:
            try:
                signal = self._ensure_signal_exists(ticker, analysis_date, model)
                signals.append(signal)
            except RawDataMissingError as e:
                missing_data_errors.append(str(e))
            except Exception as e:
                logger.error(f"Failed to ensure signal exists for {ticker}: {e}")
                # Continue with other tickers, but log the error
        
        # If we have raw data errors, raise them as a batch
        if missing_data_errors:
            raise RawDataMissingError("\n\n".join(missing_data_errors))
        
        return signals
    
    def run_fundamentals_analysis(self, ticker: str, analysis_date: str) -> Dict:
        """Run fundamentals analysis."""
        try:
            logger.info(f"Running fundamentals analysis for {ticker}")
            result = self.fundamentals_analyst.analyze(ticker, analysis_date)
            return result
            
        except Exception as e:
            logger.error(f"Error in fundamentals analysis: {e}")
            return {"analyst": "fundamentals", "metrics": {}}
    
    def run_valuation_analysis(self, ticker: str, analysis_date: str) -> Dict:
        """Run valuation analysis."""
        try:
            logger.info(f"Running valuation analysis for {ticker}")
            result = self.valuation_analyst.analyze(ticker, analysis_date)
            return result
            
        except Exception as e:
            logger.error(f"Error in valuation analysis: {e}")
            return {"analyst": "valuation", "metrics": {}}
    
    def run_technical_analysis(self, ticker: str, analysis_date: str) -> Dict:
        """Run technical analysis."""
        try:
            logger.info(f"Running technical analysis for {ticker}")
            result = self.technical_analyst.analyze(ticker, analysis_date)
            return result
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {"analyst": "technical", "metrics": {}}
    
    def run_risk_analysis(self, ticker: str, analysis_date: str) -> Dict:
        """Run risk analysis."""
        try:
            logger.info(f"Running risk analysis for {ticker}")
            result = self.risk_analyst.analyze(ticker, analysis_date)
            return result
            
        except Exception as e:
            logger.error(f"Error in risk analysis: {e}")
            return {"analyst": "risk", "metrics": {}}
    
    def run_sentiment_analysis(self, ticker: str, analysis_date: str) -> Dict:
        """Run sentiment analysis."""
        try:
            logger.info(f"Running sentiment analysis for {ticker}")
            result = self.sentiment_analyst.analyze(ticker, analysis_date)
            return result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {"analyst": "sentiment", "metrics": {}}
    
    def run_macro_analysis(self, ticker: str, analysis_date: str) -> Dict:
        """Run macro analysis."""
        try:
            logger.info(f"Running macro analysis for {analysis_date}")
            result = self.macro_analyst.analyze(analysis_date)
            return result
            
        except Exception as e:
            logger.error(f"Error in macro analysis: {e}")
            return {"analyst": "macro", "metrics": {}}
    
    def run_fundamentals_analysis_debug(self, ticker: str, analysis_date: str) -> Dict:
        """Run fundamentals analysis with debug mode."""
        try:
            logger.info(f"Running fundamentals analysis for {ticker}")
            result = self.fundamentals_analyst.analyze(ticker, analysis_date, debug=True)
            return result
            
        except Exception as e:
            logger.error(f"Error in fundamentals analysis: {e}")
            return {"analyst": "fundamentals", "metrics": {}}
    
    def run_valuation_analysis_debug(self, ticker: str, analysis_date: str) -> Dict:
        """Run valuation analysis with debug mode."""
        try:
            logger.info(f"Running valuation analysis for {ticker}")
            result = self.valuation_analyst.analyze(ticker, analysis_date, debug=True)
            return result
            
        except Exception as e:
            logger.error(f"Error in valuation analysis: {e}")
            return {"analyst": "valuation", "metrics": {}}
    
    def run_technical_analysis_debug(self, ticker: str, analysis_date: str) -> Dict:
        """Run technical analysis with debug mode."""
        try:
            logger.info(f"Running technical analysis for {ticker}")
            result = self.technical_analyst.analyze(ticker, analysis_date, debug=True)
            return result
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {"analyst": "technical", "metrics": {}}
    
    def run_risk_analysis_debug(self, ticker: str, analysis_date: str) -> Dict:
        """Run risk analysis with debug mode."""
        try:
            logger.info(f"Running risk analysis for {ticker}")
            result = self.risk_analyst.analyze(ticker, analysis_date, debug=True)
            return result
            
        except Exception as e:
            logger.error(f"Error in risk analysis: {e}")
            return {"analyst": "risk", "metrics": {}}
    
    def run_sentiment_analysis_debug(self, ticker: str, analysis_date: str) -> Dict:
        """Run sentiment analysis with debug mode."""
        try:
            logger.info(f"Running sentiment analysis for {ticker}")
            result = self.sentiment_analyst.analyze(ticker, analysis_date, debug=True)
            return result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {"analyst": "sentiment", "metrics": {}}
    
    def run_macro_analysis_debug(self, ticker: str, analysis_date: str) -> Dict:
        """Run macro analysis with debug mode."""
        try:
            logger.info(f"Running macro analysis for {analysis_date}")
            result = self.macro_analyst.analyze(analysis_date, debug=True)
            return result
            
        except Exception as e:
            logger.error(f"Error in macro analysis: {e}")
            return {"analyst": "macro", "metrics": {}}
    
    def run_single_analyst(self, analyst_name, ticker, analysis_date, debug=False):
        if analyst_name == "fundamentals":
            return self.fundamentals_analyst.analyze(ticker, analysis_date, debug=debug)
        elif analyst_name == "valuation":
            return self.valuation_analyst.analyze(ticker, analysis_date, debug=debug)
        elif analyst_name == "technical":
            return self.technical_analyst.analyze(ticker, analysis_date, debug=debug)
        elif analyst_name == "risk":
            return self.risk_analyst.analyze(ticker, analysis_date, debug=debug)
        elif analyst_name == "sentiment":
            return self.sentiment_analyst.analyze(ticker, analysis_date, debug=debug)
        elif analyst_name == "macro":
            return self.macro_analyst.analyze(analysis_date, debug=debug)
        else:
            raise ValueError(f"Unknown analyst: {analyst_name}")
    
    def generate_signal(self, ticker: str, analysis_date: str, model: str = "gpt-4o") -> Dict:
        """
        Generate a trading signal for a ticker and date.
        Automatically creates missing analysis outputs if needed.
        
        Args:
            ticker: Stock ticker symbol
            analysis_date: Analysis date in YYYY-MM-DD format
            model: OpenAI model to use for signal generation
            
        Returns:
            Dict containing the signal or error information
        """
        try:
            # Check if signal analyst is available
            if not self.signal_analyst:
                return {
                    "ticker": ticker,
                    "analysis_date": analysis_date,
                    "error": "Signal Analyst not available (OpenAI API key required)",
                    "signal": None
                }
            
            # Try to use dependency management to ensure analysis output exists
            try:
                analysis_output = self._ensure_analysis_output_exists(ticker, analysis_date)
            except RawDataMissingError as e:
                return {
                    "ticker": ticker,
                    "analysis_date": analysis_date,
                    "error": str(e),
                    "signal": None
                }
            
            # Update model if different from default
            if model != self.signal_analyst.model:
                # Validate model supports structured output
                supported_models = ["gpt-4o", "gpt-4o-mini"]
                if model not in supported_models:
                    logger.warning(f"Model {model} does not support structured output. Using gpt-4o instead.")
                    model = "gpt-4o"
                self.signal_analyst.model = model
            
            # Generate signal
            signal_result = self.signal_analyst.analyze(analysis_output)
            
            if signal_result:
                # Save to database
                if self.dm.save_signal(signal_result):
                    return signal_result
                else:
                    return {
                        "ticker": ticker,
                        "analysis_date": analysis_date,
                        "error": "Failed to save signal to database",
                        "signal": None
                    }
            else:
                return {
                    "ticker": ticker,
                    "analysis_date": analysis_date,
                    "error": "Failed to generate signal",
                    "signal": None
                }
                
        except Exception as e:
            logger.error(f"Error generating signal for {ticker}: {e}")
            return {
                "ticker": ticker,
                "analysis_date": analysis_date,
                "error": str(e),
                "signal": None
            }
    
    def get_signal(self, ticker: str, analysis_date: str) -> Dict:
        """
        Retrieve signal from the database.
        
        Args:
            ticker: Stock ticker symbol
            analysis_date: Analysis date in YYYY-MM-DD format
            
        Returns:
            Dict: Signal result or None if not found
        """
        return self.dm.get_signal(ticker, analysis_date)
    
    def analyze_ticker(self, ticker: str, analysis_date: str) -> Dict:
        """Perform complete analysis for a ticker."""
        try:
            logger.info(f"Starting complete analysis for {ticker}")
            
            # Check if raw data is available first
            if not self._check_raw_data_available(ticker, analysis_date):
                raise RawDataMissingError(
                    f"Raw data missing for {ticker} on {analysis_date}. "
                    f"Please create and populate the database with the required ticker data.\n"
                    f"See the database creation commands in notes/cli_calls.txt, for example:\n"
                    f"python create_db.py --tickers {ticker}"
                )
            
            # Get sector information
            sector_info = self.dm.get_sector_info(ticker)
            sector_etf = sector_info.get('sector_etf', 'SPY')
            sector = sector_info.get('sector', 'Unknown')
            industry = sector_info.get('industry', 'Unknown')
            
            # Get closing price (price available to analysts on analysis date)
            closing_price = self.dm.get_closing_price(ticker, analysis_date)
            
            # Run all analyses sequentially
            fundamentals = self.run_fundamentals_analysis(ticker, analysis_date)
            valuation = self.run_valuation_analysis(ticker, analysis_date)
            technical = self.run_technical_analysis(ticker, analysis_date)
            risk = self.run_risk_analysis(ticker, analysis_date)
            sentiment = self.run_sentiment_analysis(ticker, analysis_date)
            macro = self.run_macro_analysis(ticker, analysis_date)
            
            # Convert boolean values to integers for JSON serialization
            def convert_bools_to_ints(obj):
                if isinstance(obj, dict):
                    return {k: convert_bools_to_ints(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_bools_to_ints(v) for v in obj]
                elif isinstance(obj, bool):
                    return int(obj)
                else:
                    return obj
            
            final_result = {
                "ticker": ticker,
                "analysis_date": analysis_date,
                "closing_price": closing_price,
                "sector": sector,
                "industry": industry,
                "sector_etf": sector_etf,
                "analyses": {
                    "fundamentals": convert_bools_to_ints(fundamentals.get('metrics', {})),
                    "valuation": convert_bools_to_ints(valuation.get('metrics', {})),
                    "technical": convert_bools_to_ints(technical.get('metrics', {})),
                    "risk": convert_bools_to_ints(risk.get('metrics', {})),
                    "sentiment": convert_bools_to_ints(sentiment.get('metrics', {})),
                    "macro": convert_bools_to_ints(macro.get('metrics', {}))
                }
            }
            
            # Save to database (main mode)
            success = self.dm.save_analysis_output(ticker, analysis_date, final_result)
            if success:
                logger.info(f"Analysis output saved to database for {ticker}")
            else:
                logger.error(f"Failed to save analysis output to database for {ticker}")
            
            logger.info(f"Completed analysis for {ticker}")
            return final_result
            
        except RawDataMissingError:
            # Re-raise raw data errors as-is for the caller to handle
            raise
        except Exception as e:
            logger.error(f"Error in complete analysis for {ticker}: {e}")
            # Get sector information even in error case
            try:
                sector_info = self.dm.get_sector_info(ticker)
                sector_etf = sector_info.get('sector_etf', 'SPY')
                sector = sector_info.get('sector', 'Unknown')
                industry = sector_info.get('industry', 'Unknown')
            except:
                sector_etf = 'SPY'
                sector = 'Unknown'
                industry = 'Unknown'
            
            error_result = {
                "ticker": ticker,
                "analysis_date": analysis_date,
                "closing_price": None,
                "sector": sector,
                "industry": industry,
                "sector_etf": sector_etf,
                "analyses": {}
            }
            
            # Save error result to database as well
            self.dm.save_analysis_output(ticker, analysis_date, error_result)
            
            return error_result
    
    def analyze_ticker_debug(self, ticker: str, analysis_date: str) -> Dict:
        """Perform complete analysis for a ticker with debug information."""
        try:
            logger.info(f"Starting complete analysis for {ticker} in debug mode")
            
            # Get sector information
            sector_info = self.dm.get_sector_info(ticker)
            sector_etf = sector_info.get('sector_etf', 'SPY')
            sector = sector_info.get('sector', 'Unknown')
            industry = sector_info.get('industry', 'Unknown')
            
            # Get closing price (price available to analysts on analysis date)
            closing_price = self.dm.get_closing_price(ticker, analysis_date)
            
            # Run all analyses sequentially with debug mode
            fundamentals = self.run_fundamentals_analysis_debug(ticker, analysis_date)
            valuation = self.run_valuation_analysis_debug(ticker, analysis_date)
            technical = self.run_technical_analysis_debug(ticker, analysis_date)
            risk = self.run_risk_analysis_debug(ticker, analysis_date)
            sentiment = self.run_sentiment_analysis_debug(ticker, analysis_date)
            macro = self.run_macro_analysis_debug(ticker, analysis_date)
            
            # Convert boolean values to integers for JSON serialization
            def convert_bools_to_ints(obj):
                if isinstance(obj, dict):
                    return {k: convert_bools_to_ints(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_bools_to_ints(v) for v in obj]
                elif isinstance(obj, bool):
                    return int(obj)
                else:
                    return obj
            
            final_result = {
                "ticker": ticker,
                "analysis_date": analysis_date,
                "closing_price": closing_price,
                "sector": sector,
                "industry": industry,
                "sector_etf": sector_etf,
                "analyses": {
                    "fundamentals": convert_bools_to_ints(fundamentals.get('metrics', {})),
                    "valuation": convert_bools_to_ints(valuation.get('metrics', {})),
                    "technical": convert_bools_to_ints(technical.get('metrics', {})),
                    "risk": convert_bools_to_ints(risk.get('metrics', {})),
                    "sentiment": convert_bools_to_ints(sentiment.get('metrics', {})),
                    "macro": convert_bools_to_ints(macro.get('metrics', {}))
                },
                "debug_data": {
                    "fundamentals": fundamentals.get('data_used', {}),
                    "valuation": valuation.get('data_used', {}),
                    "technical": technical.get('data_used', {}),
                    "risk": risk.get('data_used', {}),
                    "sentiment": sentiment.get('data_used', {}),
                    "macro": macro.get('data_used', {})
                }
            }
            
            # Save to test folder (debug mode)
            import os
            import json
            os.makedirs("test", exist_ok=True)
            date_str = analysis_date.replace("-", "")
            output_file = f"test/complete_{ticker}_{date_str}_debug.json"
            with open(output_file, "w") as f:
                json.dump(final_result, f, indent=2)
            logger.info(f"Debug result saved to {output_file}")
            
            logger.info(f"Completed analysis for {ticker} in debug mode")
            return final_result
            
        except Exception as e:
            logger.error(f"Error in complete analysis for {ticker}: {e}")
            # Get sector information even in error case
            try:
                sector_info = self.dm.get_sector_info(ticker)
                sector_etf = sector_info.get('sector_etf', 'SPY')
                sector = sector_info.get('sector', 'Unknown')
                industry = sector_info.get('industry', 'Unknown')
            except:
                sector_etf = 'SPY'
                sector = 'Unknown'
                industry = 'Unknown'
            
            error_result = {
                "ticker": ticker,
                "analysis_date": analysis_date,
                "closing_price": None,
                "sector": sector,
                "industry": industry,
                "sector_etf": sector_etf,
                "analyses": {},
                "debug_data": {}
            }
            
            # Save error result to test folder as well
            import os
            import json
            os.makedirs("test", exist_ok=True)
            date_str = analysis_date.replace("-", "")
            output_file = f"test/complete_{ticker}_{date_str}_debug.json"
            with open(output_file, "w") as f:
                json.dump(error_result, f, indent=2)
            
            return error_result
    
    def run_portfolio(self, portfolio_name, analysis_date, investment, num_tickers, tickers, risk_preference, sector_preferences, model='gpt-4o', debug=False):
        try:
            pm = PortfolioManager(model)
            pm.set_orchestrator(self)
            return pm.run(
                portfolio_name=portfolio_name,
                analysis_date=analysis_date,
                investment=investment,
                num_tickers=num_tickers,
                tickers=tickers,
                risk_preference=risk_preference,
                sector_preferences=sector_preferences,
                debug=debug
            )
        except Exception as e:
            if "Raw data missing" in str(e):
                # Handle raw data missing error specifically
                return {
                    "portfolio_name": portfolio_name,
                    "analysis_date": analysis_date,
                    "error": str(e),
                    "new_state": None,
                    "notes": f"Portfolio creation failed due to missing raw data: {str(e)}"
                }
            else:
                return {
                    "portfolio_name": portfolio_name,
                    "analysis_date": analysis_date,
                    "error": str(e),
                    "new_state": None,
                    "notes": f"Portfolio creation failed: {str(e)}"
                }
    
    def run_portfolio_rebalance(self, portfolio_name: str, analysis_date: str, num_tickers: int, tickers: List[str], risk_preference: str, sector_preferences: Optional[List[str]], model: str = "gpt-4o", debug: bool = False) -> Dict:
        """
        Rebalance a portfolio based on the previous state.
        
        Args:
            portfolio_name: Name of the portfolio to rebalance
            analysis_date: Date to rebalance the portfolio (YYYY-MM-DD)
            num_tickers: Number of tickers to include in the rebalanced portfolio
            tickers: List of tickers to consider for rebalancing
            risk_preference: Risk preference (low, normal, high)
            sector_preferences: Optional list of sector preferences
            model: OpenAI model to use for rebalancing
            debug: If True, run in debug mode
            
        Returns:
            Dict containing the rebalancing result.
        """
        try:
            logger.info(f"Starting portfolio rebalance for {portfolio_name} on {analysis_date}")
            
            # Create portfolio manager instance
            pm = PortfolioManager(model)
            pm.set_orchestrator(self)
            
            # Call PortfolioManager.rebalance_portfolio with all required parameters
            result = pm.rebalance_portfolio(
                portfolio_name=portfolio_name,
                analysis_date=analysis_date,
                num_tickers=num_tickers,
                tickers=tickers,
                risk_preference=risk_preference,
                sector_preferences=sector_preferences,
                model=model,
                debug=debug
            )
            
            logger.info(f"Portfolio rebalance completed for {portfolio_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error in portfolio rebalance for {portfolio_name}: {e}")
            if "Raw data missing" in str(e):
                # Handle raw data missing error specifically
                return {
                    "portfolio_name": portfolio_name,
                    "analysis_date": analysis_date,
                    "error": str(e),
                    "new_state": None,
                    "notes": f"Portfolio rebalance failed due to missing raw data: {str(e)}"
                }
            else:
                return {
                    "portfolio_name": portfolio_name,
                    "analysis_date": analysis_date,
                    "error": str(e),
                    "new_state": None,
                    "notes": f"Rebalance failed: {str(e)}"
                }
    
    def manage_portfolio(self, portfolio_name, start_date, end_date, num_tickers, tickers, risk_preference, investment, frequency, model='gpt-4o', sector_preferences=None, debug=False):
        """
        Manage a portfolio by creating it on start_date and rebalancing it periodically.
        Args:
            portfolio_name: Name of the portfolio to manage
            start_date: Date to start managing (YYYY-MM-DD)
            end_date: Date to stop managing (YYYY-MM-DD)
            num_tickers: Number of tickers in the portfolio
            tickers: List of tickers
            risk_preference: Risk preference
            investment: Initial investment
            frequency: 'monthly' or 'quarterly'
            model: LLM model
            sector_preferences: Optional sector preferences
            debug: Debug mode
        Returns:
            Dict with summary of creation and rebalances
        """
        try:
            pm = PortfolioManager(model)
            pm.set_orchestrator(self)
            return pm.manage_portfolio(
                portfolio_name=portfolio_name,
                start_date=start_date,
                end_date=end_date,
                num_tickers=num_tickers,
                tickers=tickers,
                risk_preference=risk_preference,
                investment=investment,
                frequency=frequency,
                sector_preferences=sector_preferences,
                debug=debug
            )
        except Exception as e:
            if "Raw data missing" in str(e):
                # Handle raw data missing error specifically
                return {
                    "portfolio_name": portfolio_name,
                    "start_date": start_date,
                    "end_date": end_date,
                    "error": str(e),
                    "initial_state": None,
                    "rebalance_results": [],
                    "final_state": None,
                    "notes": f"Portfolio management failed due to missing raw data: {str(e)}"
                }
            else:
                return {
                    "portfolio_name": portfolio_name,
                    "start_date": start_date,  
                    "end_date": end_date,
                    "error": str(e),
                    "initial_state": None,
                    "rebalance_results": [],
                    "final_state": None,
                    "notes": f"Portfolio management failed: {str(e)}"
                }
        
    def _get_rebalance_dates(self, start_date: str, end_date: str, frequency: str) -> List[str]:
        """
        Generates a list of rebalance dates based on the start date, end date, and frequency.
        Args:
            start_date: The date to start rebalancing (YYYY-MM-DD)
            end_date: The date to end rebalancing (YYYY-MM-DD)
            frequency: The frequency of rebalancing ("monthly" or "quarterly")
        Returns:
            List of rebalance dates (YYYY-MM-DD).
        """
        from datetime import datetime, timedelta
        from dateutil.relativedelta import relativedelta
        dates = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        if frequency == "monthly":
            while current <= end:
                dates.append(current.strftime("%Y-%m-%d"))
                # Move to the 1st of next month
                current = (current + relativedelta(months=1)).replace(day=1)
        elif frequency == "quarterly":
            # Find the first quarter start on or after start_date
            month = ((current.month - 1) // 3) * 3 + 1
            current = current.replace(month=month, day=1)
            while current <= end:
                dates.append(current.strftime("%Y-%m-%d"))
                # Move to the 1st of next quarter
                current = (current + relativedelta(months=3)).replace(day=1)
        else:
            raise ValueError('Frequency must be "monthly" or "quarterly"')
        return dates
    
    def close_connections(self):
        """Close all database connections."""
        self.dm.close_connection()
        logger.info("All connections closed")


 