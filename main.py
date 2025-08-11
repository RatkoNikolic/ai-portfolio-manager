#!/usr/bin/env python3
"""
Main Script for AI Portfolio Manager
Entry point for the AI portfolio management analysis system.
"""

import argparse
import json
import sys
import logging
from datetime import datetime
from typing import List, Dict
import os
import numpy as np
from dotenv import load_dotenv

# Import all components
from data_manager import DataManager
from orchestrator import Orchestrator, DatabaseNotFoundError, RawDataMissingError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIPortfolioManagerSystem:
    """Main system class for AI Portfolio Manager analysis."""
    
    def __init__(self):
        """Initialize the AI Portfolio Manager system."""
        try:
            self.orchestrator = Orchestrator()
            logger.info("AI Portfolio Manager System initialized")
        except DatabaseNotFoundError as e:
            logger.error(str(e))
            # Get database name from environment
            db_name = os.getenv('DB_NAME', 'aipm.db')
            print(f"\n{'='*80}")
            print("DATABASE NOT FOUND")
            print(f"{'='*80}")
            print(f"The database '{db_name}' was not found or is missing required tables.")
            print("Please create and populate the database first using the database creation script:")
            print("")
            print("# Create database with default tickers (recommended)")
            print("python create_db.py")
            print("")
            print("# Create database with custom tickers")
            print("python create_db.py --tickers AAPL MSFT GOOGL AMZN TSLA")
            print("")
            print("# Create database with specific date range")
            print("python create_db.py --tickers AAPL MSFT --start-date 2020-01-01 --end-date 2023-12-31")
            print("")
            print("Note: Database creation may take 10-15 minutes depending on the number of tickers.")
            print("Make sure you have valid API keys in your .env file:")
            print("- ALPHA_VANTAGE_API_KEY (for financial statements)")
            print("- FRED_API_KEY (for macroeconomic data)")
            print(f"{'='*80}")
            raise SystemExit(1)
    
    def run_single_analyst(self, analyst_name: str, ticker: str, analysis_date: str, debug: bool = False) -> Dict:
        """Run a single analyst via the orchestrator."""
        try:
            logger.info(f"Running {analyst_name} analysis for {ticker}")
            result = self.orchestrator.run_single_analyst(analyst_name, ticker, analysis_date, debug=debug)
            return result
        except Exception as e:
            logger.error(f"Error running {analyst_name} analysis: {e}")
            error_result = {
                "analyst": analyst_name,
                "error": str(e),
                "metrics": {}
            }
            return error_result
    
    def run_complete_analysis(self, ticker: str, analysis_date: str, debug: bool = False) -> Dict:
        """Run complete analysis using orchestrator."""
        try:
            # Check if analysis already exists (unless in debug mode)
            if not debug:
                existing_analysis = self.orchestrator.dm.get_analysis_output(ticker, analysis_date)
                if existing_analysis:
                    logger.info(f"Analysis already exists for {ticker} on {analysis_date}, using cached result")
                    return existing_analysis
            
            logger.info(f"Running complete analysis for {ticker}")
            if debug:
                result = self.orchestrator.analyze_ticker_debug(ticker, analysis_date)
            else:
                result = self.orchestrator.analyze_ticker(ticker, analysis_date)
            return result
        except RawDataMissingError as e:
            logger.error(f"Raw data missing for {ticker}: {e}")
            print(f"\n{'='*80}")
            print("RAW DATA MISSING")
            print(f"{'='*80}")
            print(str(e))
            print(f"{'='*80}")
            return {
                "ticker": ticker,
                "analysis_date": analysis_date,
                "error": str(e),
                "composite_score": 0,
                "recommendation": f"Raw data missing: {str(e)}",
                "analyses": {}
            }
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            return {
                "ticker": ticker,
                "analysis_date": analysis_date,
                "error": str(e),
                "composite_score": 0,
                "recommendation": f"Error in analysis: {str(e)}",
                "analyses": {}
            }
    
    def run_batch_analysis(self, tickers: List[str], analysis_date: str, 
                          analyst: str = None, debug: bool = False) -> List[Dict]:
        """Run analysis for multiple tickers."""
        try:
            results = []
            for i, ticker in enumerate(tickers, 1):
                logger.info(f"Processing {ticker} ({i}/{len(tickers)})")
                if analyst:
                    result = self.run_single_analyst(analyst, ticker, analysis_date, debug=debug)
                else:
                    result = self.run_complete_analysis(ticker, analysis_date, debug=debug)
                results.append(result)
                print(f"✓ Completed {ticker} ({i}/{len(tickers)})")
            return results
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            return []
    
    def run_batch_ticker_date_analysis(self, tickers: List[str], dates: List[str], 
                                     analyst: str = None, debug: bool = False) -> List[Dict]:
        """Run analysis for multiple tickers across multiple dates."""
        try:
            total_analyses = len(tickers) * len(dates)
            current_analysis = 0
            results = []
            
            for date in dates:
                logger.info(f"Processing date: {date}")
                for ticker in tickers:
                    current_analysis += 1
                    logger.info(f"Processing {ticker} on {date} ({current_analysis}/{total_analyses})")
                    
                    if analyst:
                        result = self.run_single_analyst(analyst, ticker, date, debug=debug)
                    else:
                        result = self.run_complete_analysis(ticker, date, debug=debug)
                    
                    # Add date information to result for identification
                    if isinstance(result, dict):
                        result['analysis_date'] = date
                        result['ticker'] = ticker
                    
                    results.append(result)
                    print(f"✓ Completed {ticker} on {date} ({current_analysis}/{total_analyses})")
            
            return results
        except Exception as e:
            logger.error(f"Error in batch ticker-date analysis: {e}")
            return []
    
    def run_single_signal(self, ticker: str, analysis_date: str, model: str = "gpt-4o", debug: bool = False) -> Dict:
        """Run signal generation for a single ticker and date."""
        try:
            logger.info(f"Generating signal for {ticker} on {analysis_date}")
            
            # Check if signal already exists
            existing_signal = self.orchestrator.get_signal(ticker, analysis_date)
            if existing_signal and not debug:
                logger.info(f"Signal already exists for {ticker} on {analysis_date}")
                return existing_signal
            
            # Generate new signal
            result = self.orchestrator.generate_signal(ticker, analysis_date, model=model)
            return result
        except RawDataMissingError as e:
            logger.error(f"Raw data missing for {ticker}: {e}")
            print(f"\n{'='*80}")
            print("RAW DATA MISSING")
            print(f"{'='*80}")
            print(str(e))
            print(f"{'='*80}")
            return {
                "ticker": ticker,
                "analysis_date": analysis_date,
                "error": str(e),
                "signal": None
            }           
        except Exception as e:
            logger.error(f"Error in single signal generation: {e}")
            return {
                "ticker": ticker,
                "analysis_date": analysis_date,
                "error": str(e),
                "signal": None
            }
    
    def run_batch_signals(self, tickers: List[str], analysis_date: str, model: str = "gpt-4o", debug: bool = False) -> List[Dict]:
        """Run signal generation for multiple tickers on a single date."""
        try:
            results = []
            for i, ticker in enumerate(tickers, 1):
                logger.info(f"Processing {ticker} ({i}/{len(tickers)})")
                result = self.run_single_signal(ticker, analysis_date, model=model, debug=debug)
                results.append(result)
                print(f"✓ Completed {ticker} ({i}/{len(tickers)})")
            return results
        except Exception as e:
            logger.error(f"Error in batch signal generation: {e}")
            return []
    
    def run_batch_ticker_date_signals(self, tickers: List[str], dates: List[str], model: str = "gpt-4o", debug: bool = False) -> List[Dict]:
        """Run signal generation for multiple tickers across multiple dates."""
        try:
            total_analyses = len(tickers) * len(dates)
            current_analysis = 0
            results = []
            
            for date in dates:
                logger.info(f"Processing date: {date}")
                for ticker in tickers:
                    current_analysis += 1
                    logger.info(f"Processing {ticker} on {date} ({current_analysis}/{total_analyses})")
                    
                    result = self.run_single_signal(ticker, date, model=model, debug=debug)
                    results.append(result)
                    print(f"✓ Completed {ticker} on {date} ({current_analysis}/{total_analyses})")
            
            return results
        except Exception as e:
            logger.error(f"Error in batch ticker-date signal generation: {e}")
            return []
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save results to JSON file in test folder."""
        try:
            # Ensure output file is saved to test folder
            if not output_file.startswith('test/'):
                output_file = f"test/{output_file}"
            
            # Ensure test directory exists
            os.makedirs("test", exist_ok=True)
            
            # Custom JSON encoder to handle boolean values and other non-serializable types
            class CustomJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, bool):
                        return int(obj)  # Convert bool to int (0 or 1)
                    if isinstance(obj, (np.integer, np.floating)):
                        return float(obj)  # Convert numpy types to Python types
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()  # Convert numpy arrays to lists
                    return super().default(obj)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, cls=CustomJSONEncoder)
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def print_summary(self, results: List[Dict]):
        """Print a summary of results."""
        try:
            print("\n" + "="*60)
            
            # Check if these are signal results or analysis results
            is_signal_results = any("signal" in result for result in results if isinstance(result, dict))
            
            if is_signal_results:
                print("SIGNAL GENERATION SUMMARY")
                print("="*60)
                
                successful_signals = [r for r in results if r.get('signal') and not r.get('error')]
                failed_signals = [r for r in results if r.get('error')]
                
                for result in successful_signals:
                    ticker = result.get("ticker", "Unknown")
                    date = result.get("analysis_date", "Unknown")
                    signal = result.get("signal", "Unknown")
                    confidence = result.get("confidence", 0)
                    time_horizon = result.get("time_horizon", "Unknown")
                    
                    print(f"\n{ticker} ({date}): {signal.upper()} (confidence: {confidence:.2f}, horizon: {time_horizon})")
                
                if failed_signals:
                    print(f"\nFailed signals: {len(failed_signals)}")
                    for result in failed_signals:
                        ticker = result.get("ticker", "Unknown")
                        date = result.get("analysis_date", "Unknown")
                        error = result.get("error", "Unknown error")
                        print(f"  {ticker} ({date}): {error}")
                
                print(f"\nTotal signals processed: {len(results)}")
                print(f"Successful: {len(successful_signals)}")
                print(f"Failed: {len(failed_signals)}")
                
            else:
                print("ANALYSIS SUMMARY")
                print("="*60)
                
                # Group results by ticker and date for better summary
                ticker_date_results = {}
                for result in results:
                    ticker = result.get("ticker", "Unknown")
                    date = result.get("analysis_date", "Unknown")
                    key = f"{ticker}_{date}"
                    
                    if key not in ticker_date_results:
                        ticker_date_results[key] = {
                            "ticker": ticker,
                            "date": date,
                            "analyst": result.get("analyst", "complete"),
                            "has_analyses": "analyses" in result,
                            "metrics_count": 0
                        }
                    
                    if "analyses" in result:
                        analyses = result["analyses"]
                        for analyst_name, metrics in analyses.items():
                            if metrics:
                                ticker_date_results[key]["metrics_count"] += len(metrics)
                
                # Print summary
                for key, info in ticker_date_results.items():
                    if info["analyst"] == "complete":
                        print(f"\n{info['ticker']} ({info['date']}): Complete analysis completed")
                        if info["has_analyses"]:
                            print(f"  ✓ Total metrics: {info['metrics_count']}")
                    else:
                        print(f"\n{info['ticker']} ({info['date']}): {info['analyst']} analysis completed")
                        if info["metrics_count"] > 0:
                            print(f"  ✓ Metrics: {info['metrics_count']}")
                
                print(f"\nTotal analyses completed: {len(results)}")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error printing summary: {e}")
    
    def close_connections(self):
        """Close all database connections."""
        self.orchestrator.close_connections()
        logger.info("All connections closed")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="AI Portfolio Manager Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete analysis for a single ticker
  python main.py --ticker AAPL --date 2023-12-31

  # Run specific analyst for a ticker
  python main.py --ticker AAPL --date 2023-12-31 --analyst fundamentals

  # Run batch analysis for multiple tickers
  python main.py --tickers AAPL MSFT GOOGL --date 2023-12-31

  # Run batch analysis with specific analyst
  python main.py --tickers AAPL MSFT --date 2023-12-31 --analyst technical

  # Run batch analysis for multiple tickers across multiple dates
  python main.py --tickers AAPL MSFT --dates 2023-12-31 2023-11-30 2023-10-31

  # Run batch ticker-date analysis with specific analyst
  python main.py --tickers AAPL MSFT --dates 2023-12-31 2023-11-30 --analyst technical

  # Save results to file (saved in test folder)
  python main.py --ticker AAPL --date 2023-12-31 --output results.json
        """
    )
    
    # Add arguments
    parser.add_argument(
        "--ticker", 
        type=str, 
        help="Single ticker symbol to analyze"
    )
    parser.add_argument(
        "--date", 
        type=str, 
        help="Analysis date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--dates", 
        nargs="+", 
        help="Multiple analysis dates in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--analyst", 
        type=str, 
        choices=["fundamentals", "valuation", "technical", "risk", "sentiment", "macro"],
        help="Specific analyst to run (if not specified, runs complete analysis)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output file to save results (JSON format, saved in test folder)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for complete analysis"
    )
    parser.add_argument(
        "--signal",
        action="store_true",
        help="Generate trading signals from analysis outputs (requires OpenAI API key)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use for signal generation and portfolio manager (default: gpt-4o). Portfolio Manager supports: gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini, o3-mini, o4-mini, gpt-5."
    )
    
    # Portfolio manager CLI options
    parser.add_argument('--create_portfolio', action='store_true', help='Run portfolio manager')
    parser.add_argument('--portfolio_name', type=str, help='Portfolio name (unique identifier)')
    parser.add_argument('--portfolio_date', type=str, help='Date for portfolio creation/rebalancing (YYYY-MM-DD)')
    parser.add_argument('--investment', type=float, help='Total investment amount (USD)')
    parser.add_argument('--num_tickers', type=int, help='Number of tickers in the portfolio')
    parser.add_argument('--tickers', type=str, nargs='+', help='Tickers to consider for portfolio or batch analysis')
    parser.add_argument('--risk', type=str, choices=['low', 'normal', 'high'], help='Risk preference (low, normal, high)')
    parser.add_argument('--sectors', type=str, nargs='+', help='Sector preferences (optional)')
    parser.add_argument('--portfolio_debug', action='store_true', help='Enable debug mode for portfolio manager (always reruns LLM and saves results to test folder)')
    parser.add_argument('--rebalance_portfolio', action='store_true', help='Rebalance an existing portfolio (requires --portfolio_name, --rebalance_date, --num_tickers, --tickers, --risk)')
    parser.add_argument('--rebalance_date', type=str, help='Date for portfolio rebalancing (YYYY-MM-DD)')
    parser.add_argument('--manage_portfolio', action='store_true', help='Manage an existing portfolio (requires --portfolio_name, --start_date, --end_date, --num_tickers, --tickers, --risk, --investment, --frequency)')
    parser.add_argument('--start_date', type=str, help='Start date for portfolio management (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date for portfolio management (YYYY-MM-DD)')
    parser.add_argument('--frequency', type=str, choices=['monthly', 'quarterly'], help='Frequency for portfolio management (monthly, quarterly)')

    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not (args.create_portfolio or args.portfolio_debug or args.rebalance_portfolio or args.manage_portfolio):
        if not args.ticker and not args.tickers:
            parser.error("Either --ticker or --tickers must be specified")
        if args.ticker and args.tickers:
            parser.error("Cannot specify both --ticker and --tickers")
        if not args.date and not args.dates:
            parser.error("Either --date or --dates must be specified")
        if args.date and args.dates:
            parser.error("Cannot specify both --date and --dates")
        # Validate date format(s)
        if args.date:
            try:
                datetime.strptime(args.date, '%Y-%m-%d')
            except ValueError:
                parser.error("Date must be in YYYY-MM-DD format")
        if args.dates:
            for date in args.dates:
                try:
                    datetime.strptime(date, '%Y-%m-%d')
                except ValueError:
                    parser.error(f"Date {date} must be in YYYY-MM-DD format")

    # Portfolio-specific validation
    if args.create_portfolio:
        if not (args.portfolio_name and args.portfolio_date and args.investment and args.num_tickers and args.tickers and args.risk):
            parser.error("Missing required portfolio arguments. Please provide --portfolio_name, --portfolio_date, --investment, --num_tickers, --tickers, --risk.")
    if args.rebalance_portfolio:
        if not (args.portfolio_name and args.rebalance_date and args.num_tickers and args.tickers and args.risk):
            parser.error("Missing required rebalance portfolio arguments. Please provide --portfolio_name, --rebalance_date, --num_tickers, --tickers, --risk.")
    if args.manage_portfolio:
        if not (args.portfolio_name and args.start_date and args.end_date and args.num_tickers and args.tickers and args.risk and args.investment and args.frequency):
            parser.error("Missing required manage portfolio arguments. Please provide --portfolio_name, --start_date, --end_date, --num_tickers, --tickers, --risk, --investment, --frequency.")

    # Initialize system
    system = AIPortfolioManagerSystem()
    
    try:
        results = []
        
        if args.ticker:
            # Single ticker analysis
            if args.analyst:
                result = system.run_single_analyst(args.analyst, args.ticker, args.date, debug=args.debug)
                analyst_name = args.analyst

                # Save single analyst result to test folder only in debug mode
                if args.debug and not args.output:
                    os.makedirs("test", exist_ok=True)
                    date_str = args.date.replace("-", "")
                    output_file = f"test/{analyst_name}_{args.ticker}_{date_str}_debug.json"
                    with open(output_file, "w") as f:
                        json.dump(result, f, indent=2)
                        logger.info(f"Single analyst debug result saved to {output_file}")
                elif args.output:
                    result = system.run_complete_analysis(args.ticker, args.date, debug=args.debug)
                    analyst_name = "complete"
                    # Complete analysis results are handled by orchestrator
                results = [result]
            
            elif not args.output:
                result = system.run_complete_analysis(args.ticker, args.date, debug=args.debug)
                analyst_name = "complete"
                # Complete analysis results are handled by orchestrator
                results = [result]
        
        elif args.tickers and args.date:
            # Batch analysis for single date
            results = system.run_batch_analysis(args.tickers, args.date, analyst=args.analyst, debug=args.debug)

            # Save results based on analyst type and debug mode
            if not args.output:
                if args.analyst and args.debug:
                    # Single analyst batch - save to test folder only in debug mode
                    os.makedirs("test", exist_ok=True)
                    date_str = args.date.replace("-", "")
                    for i, ticker in enumerate(args.tickers):
                        result = results[i]
                        output_file = f"test/{args.analyst}_{ticker}_{date_str}_debug.json"
                        with open(output_file, "w") as f:
                            json.dump(result, f, indent=2)
                        logger.info(f"Single analyst debug result for {ticker} saved to {output_file}")
                elif not args.analyst:
                    # Complete analysis batch - results are handled by orchestrator
                    logger.info("Complete analysis batch results saved to database (main mode) or test folder (debug mode)")
        
        elif args.tickers and args.dates:
            # Batch analysis for multiple tickers across multiple dates
            results = system.run_batch_ticker_date_analysis(args.tickers, args.dates, analyst=args.analyst, debug=args.debug)
            
            # Save results based on analyst type and debug mode
            if not args.output:
                if args.analyst and args.debug:
                    # Single analyst batch - save to test folder only in debug mode
                    os.makedirs("test", exist_ok=True)
                    for result in results:
                        ticker = result.get('ticker', 'unknown')
                        date = result.get('analysis_date', 'unknown')
                        date_str = date.replace("-", "")
                        output_file = f"test/{args.analyst}_{ticker}_{date_str}_debug.json"
                        with open(output_file, "w") as f:
                            json.dump(result, f, indent=2)
                            logger.info(f"Single analyst debug result for {ticker} on {date} saved to {output_file}")
                elif not args.analyst:
                    # Complete analysis batch - results are handled by orchestrator
                    logger.info("Complete analysis batch results saved to database (main mode) or test folder (debug mode)")
        
        # Handle signal generation if requested
        if args.signal:
            logger.info("Signal generation requested")
            
            if args.ticker:
                # Single ticker signal generation
                signal_results = [system.run_single_signal(args.ticker, args.date, model=args.model, debug=args.debug)]
            elif args.tickers and args.date:
                # Batch signal generation for single date
                signal_results = system.run_batch_signals(args.tickers, args.date, model=args.model, debug=args.debug)
            elif args.tickers and args.dates:
                # Batch signal generation for multiple tickers across multiple dates
                signal_results = system.run_batch_ticker_date_signals(args.tickers, args.dates, model=args.model, debug=args.debug)
            else:
                logger.error("Signal generation requires ticker and date information")
                sys.exit(1)
            
            # Save signal results if output file specified or debug mode
            if args.output:
                system.save_results(signal_results, args.output)
            elif args.debug:
                # Save debug signal results to test folder
                os.makedirs("test", exist_ok=True)
                for result in signal_results:
                    ticker = result.get('ticker', 'unknown')
                    date = result.get('analysis_date', 'unknown')
                    date_str = date.replace("-", "")
                    output_file = f"test/signal_{ticker}_{date_str}_debug.json"
                    with open(output_file, "w") as f:
                        json.dump(result, f, indent=2)
                    logger.info(f"Signal result for {ticker} on {date} saved to {output_file}")
        
            # Print signal summary
            system.print_summary(signal_results)
            
            # Print detailed results if single ticker
            if args.ticker and not args.output and not args.debug:
                print("\nDETAILED SIGNAL RESULTS:")
                print(json.dumps(signal_results[0], indent=2))
            
            # Use signal results for final output
            results = signal_results
        
        # Handle portfolio management if requested
        if args.create_portfolio:
            # Run portfolio creation
            try:
                result = system.orchestrator.run_portfolio(
                    portfolio_name=args.portfolio_name,
                    analysis_date=args.portfolio_date,
                    investment=args.investment,
                    num_tickers=args.num_tickers,
                    tickers=args.tickers,
                    risk_preference=args.risk,
                    sector_preferences=args.sectors,
                    model=args.model,
                    debug=args.portfolio_debug
                )
                
                # Check if there was an error in the result
                if result.get('error'):
                    if "Raw data missing" in result['error']:
                        print(f"\n{'='*80}")
                        print("RAW DATA MISSING FOR PORTFOLIO CREATION")
                        print(f"{'='*80}")
                        print(result['error'])
                        print(f"{'='*80}")
                        exit(1)
                    else:
                        print(f"\nError in portfolio creation: {result['error']}")
                        exit(1)
                
                print("\n==============================")
                print(f"PORTFOLIO MANAGER SUMMARY for {args.portfolio_name} on {args.portfolio_date}")
                print("==============================")
                print("LLM INPUT:")
                print(json.dumps(result['llm_input'], indent=2))
                print("\nLLM RESPONSE:")
                print(json.dumps(result['llm_response'], indent=2))
                print("\nNEW PORTFOLIO STATE:")
                print(json.dumps(result['new_state'], indent=2))
                if result['notes']:
                    print("\nNOTES:")
                    print(result['notes'])
                if args.portfolio_debug and result.get('debug_file'):
                    print(f"\nDebug output saved to: {result['debug_file']}")
                print("==============================\n")
            except RawDataMissingError as e:
                print(f"\n{'='*80}")
                print("RAW DATA MISSING FOR PORTFOLIO CREATION")
                print(f"{'='*80}")
                print(str(e))
                print(f"{'='*80}")
                exit(1)
            except Exception as e:
                print(f"\nError in portfolio creation: {e}")
                exit(1)
            exit(0)
        if args.rebalance_portfolio:
            # Run portfolio rebalancer
            try:
                result = system.orchestrator.run_portfolio_rebalance(
                    portfolio_name=args.portfolio_name,
                    analysis_date=args.rebalance_date,
                    num_tickers=args.num_tickers,
                    tickers=args.tickers,
                    risk_preference=args.risk,
                    sector_preferences=args.sectors,
                    model=args.model,
                    debug=args.portfolio_debug
                )
                
                # Check if there was an error in the result
                if result.get('error'):
                    if "Raw data missing" in result['error']:
                        print(f"\n{'='*80}")
                        print("RAW DATA MISSING FOR PORTFOLIO REBALANCING")
                        print(f"{'='*80}")
                        print(result['error'])
                        print(f"{'='*80}")
                        exit(1)
                    else:
                        print(f"\nError in portfolio rebalancing: {result['error']}")
                        exit(1)
                
                print("\n==============================")
                print(f"PORTFOLIO REBALANCER SUMMARY for {args.portfolio_name} on {args.rebalance_date}")
                print("==============================")
                print("LLM INPUT:")
                print(json.dumps(result['llm_input'], indent=2))
                print("\nLLM RESPONSE:")
                print(json.dumps(result['llm_response'], indent=2))
                print("\nNEW PORTFOLIO STATE:")
                print(json.dumps(result['new_state'], indent=2))
                if result['notes']:
                    print("\nNOTES:")
                    print(result['notes'])
                if args.portfolio_debug and result.get('debug_file'):
                    print(f"\nDebug output saved to: {result['debug_file']}")
                print("==============================\n")
            except RawDataMissingError as e:
                print(f"\n{'='*80}")
                print("RAW DATA MISSING FOR PORTFOLIO REBALANCING")
                print(f"{'='*80}")
                print(str(e))
                print(f"{'='*80}")
                exit(1)
            except Exception as e:
                print(f"\nError in portfolio rebalancing: {e}")
                exit(1)
            exit(0)
        if args.manage_portfolio:
            # Run portfolio manager for management
            try:
                result = system.orchestrator.manage_portfolio(
                    portfolio_name=args.portfolio_name,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    num_tickers=args.num_tickers,
                    tickers=args.tickers,
                    risk_preference=args.risk,
                    investment=args.investment,
                    frequency=args.frequency,
                    model=args.model,
                    sector_preferences=args.sectors,
                    debug=args.portfolio_debug
                )
                
                # Check if there was an error in the result
                if result.get('error'):
                    if "Raw data missing" in result['error']:
                        print(f"\n{'='*80}")
                        print("RAW DATA MISSING FOR PORTFOLIO MANAGEMENT")
                        print(f"{'='*80}")
                        print(result['error'])
                        print(f"{'='*80}")
                        exit(1)
                    else:
                        print(f"\nError in portfolio management: {result['error']}")
                        exit(1)
                
                print("\n==============================")
                print(f"PORTFOLIO MANAGER SUMMARY for {args.portfolio_name} from {args.start_date} to {args.end_date}")
                print("==============================")
                
                if 'initial_state' in result:
                    print("INITIAL STATE:")
                    print(json.dumps(result['initial_state'], indent=2))
                    
                if 'rebalance_results' in result:
                    print("\nREBALANCE RESULTS:")
                    for i, rebalance in enumerate(result['rebalance_results'], 1):
                        print(f"\n--- Rebalance {i} ---")
                        print(json.dumps(rebalance, indent=2))
                    
                if 'final_state' in result:
                    print("\nFINAL STATE:")
                    print(json.dumps(result['final_state'], indent=2))
                    
                if result.get('notes'):
                    print("\nNOTES:")
                    print(result['notes'])
                    
                print("==============================\n")
            except RawDataMissingError as e:
                print(f"\n{'='*80}")
                print("RAW DATA MISSING FOR PORTFOLIO MANAGEMENT")
                print(f"{'='*80}")
                print(str(e))
                print(f"{'='*80}")
                exit(1)
            except Exception as e:
                print(f"\nError in portfolio management: {e}")
                exit(1)
            exit(0)

        # Print results (for non-signal analysis)
        if args.output and not args.signal:
            system.save_results(results, args.output)
        
        # Print summary (for non-signal analysis)
        if not args.signal:
            system.print_summary(results)
        
        # Print detailed results if single ticker
        if args.ticker and not args.output:
            print("\nDETAILED RESULTS:")
            print(json.dumps(results[0], indent=2))
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        system.close_connections()


if __name__ == "__main__":
    main() 