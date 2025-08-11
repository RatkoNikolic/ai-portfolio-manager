#!/usr/bin/env python3
"""
Database Creation Script for AI Portfolio Manager
Standalone script to create and populate the database with market data.
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Main function for database creation."""
    parser = argparse.ArgumentParser(
        description="Create and populate AI Portfolio Manager database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create database with default tickers
  python create_db.py

  # Create database with custom tickers
  python create_db.py --tickers AAPL MSFT GOOGL AMZN TSLA

  # Create database with specific date range
  python create_db.py --tickers AAPL MSFT --start-date 2020-01-01 --end-date 2023-12-31

  # Create database with different name (will update DB_NAME in environment)
  python create_db.py --db-name my_custom_db.db --tickers AAPL MSFT
        """
    )
    
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "WMT"],
        help="List of ticker symbols to include in the database (default: top 10 S&P 500 companies)"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Start date for data collection in YYYY-MM-DD format (default: 2020-01-01)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for data collection in YYYY-MM-DD format (default: yesterday)"
    )
    
    parser.add_argument(
        "--db-name",
        type=str,
        default=None,
        help="Database name (default: from DB_NAME environment variable or 'aipm_thesis.db')"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of database even if it exists"
    )
    
    args = parser.parse_args()
    
    # Validate date formats
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
        if args.end_date:
            datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        print(f"Error: Invalid date format. {e}")
        sys.exit(1)
    
    # Get database name
    db_name = args.db_name or os.getenv('DB_NAME', 'aipm_thesis.db')
    db_path = f'data/{db_name}'
    
    # Check if database exists
    if os.path.exists(db_path) and not args.force:
        print(f"Database '{db_name}' already exists.")
        print("Use --force to recreate it or specify a different name with --db-name")
        sys.exit(1)
    
    # Check for required API keys
    alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    fred_key = os.getenv('FRED_API_KEY')
    
    if not alpha_vantage_key:
        print("Error: ALPHA_VANTAGE_API_KEY not found in environment variables.")
        print("Please add it to your .env file.")
        sys.exit(1)
    
    if not fred_key:
        print("Error: FRED_API_KEY not found in environment variables.")
        print("Please add it to your .env file.")
        sys.exit(1)
    
    print(f"{'='*80}")
    print("AI PORTFOLIO MANAGER - DATABASE CREATION")
    print(f"{'='*80}")
    print(f"Database: {db_name}")
    print(f"Tickers: {', '.join(args.tickers)} ({len(args.tickers)} total)")
    print(f"Date range: {args.start_date} to {args.end_date or 'yesterday'}")
    print(f"Estimated time: {max(10, len(args.tickers) * 2)} minutes")
    print(f"{'='*80}")
    
    # Confirm before proceeding
    if not args.force:
        response = input("Proceed with database creation? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Database creation cancelled.")
            sys.exit(0)
    
    try:
        # Import and run the database creation function
        from data.create_and_populate_db import create_and_populate_db
        
        print("Starting database creation and population...")
        print("This may take several minutes depending on the number of tickers.")
        print("Please be patient and do not interrupt the process.\n")
        
        # Create and populate the database
        create_and_populate_db(
            tickers=args.tickers,
            db_name=db_path,
            sql_file_path="data/db_schema_create_statemets.sql",
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        print(f"\n{'='*80}")
        print("DATABASE CREATION COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"Database: {db_name}")
        print(f"Location: {os.path.abspath(db_path)}")
        print(f"Tickers: {len(args.tickers)} companies")
        print(f"Date range: {args.start_date} to {args.end_date or 'yesterday'}")
        print("")
        print("You can now run analysis commands such as:")
        print(f"python main.py --ticker {args.tickers[0]} --date 2023-12-31")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print("\nDatabase creation interrupted by user.")
        # Clean up partial database file if it exists
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
                print(f"Cleaned up partial database file: {db_name}")
            except:
                print(f"Warning: Could not remove partial database file: {db_name}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError creating database: {e}")
        print("Please check your API keys and internet connection.")
        # Clean up partial database file if it exists
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
                print(f"Cleaned up partial database file: {db_name}")
            except:
                print(f"Warning: Could not remove partial database file: {db_name}")
        sys.exit(1)


if __name__ == "__main__":
    main() 