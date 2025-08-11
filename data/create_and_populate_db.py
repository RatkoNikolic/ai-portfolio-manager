from dotenv import load_dotenv
import os
import time
import sqlite3
import requests
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from datetime import datetime, timedelta


load_dotenv()
db_name = os.getenv("DB_NAME")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
fred_api_key=os.getenv("FRED_API_KEY")
openai_api_key=os.getenv("OPENAI_API_KEY")


def create_db_schema(db_name, sql_file_path="db_schema_create_statemets.sql"):

    # Read the SQL file
    with open(sql_file_path, "r", encoding="utf-8") as f:
        sql_script = f.read()

    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Execute the SQL script (can contain multiple ; delimited statements)
    cursor.executescript(sql_script)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print(f"✅ Database schema created: {tables}")
    
    conn.commit()
    conn.close()


def populate_db_with_stock_prices(tickers, db_name, start_date="2020-01-01", end_date=None):
    
    # Date range
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d') # yesterday

    # Connect to the SQLite DB
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Delete all rows from the table
    cursor.execute(f"DELETE FROM stock_prices_daily")

    # Loop through tickers and save to DB
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            if stock_data.empty:
                print(f"No data found for {ticker}. Skipping.")
                continue

            # Clean and format data
            stock_data.reset_index(inplace=True)
            stock_data.columns = stock_data.columns.get_level_values(0)
            stock_data['ticker'] = ticker
            stock_data.columns = stock_data.columns.str.lower()
            stock_data = stock_data[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
            stock_data.reset_index(drop=True, inplace=True)
            stock_data.index.name = None

            # Append to SQLite table
            stock_data.to_sql("stock_prices_daily", conn, if_exists="append", index=False)
        
        except Exception as e:
            print(f"Failed to fetch or insert data for {ticker}: {e}")

    # Close connection
    conn.close()

    print("Ticker daily price data fetching and storage completed.")


def populate_db_with_financial_statements(tickers, db_name, start_date="2020-01-01", end_date=None, alpha_vantage_api_key=alpha_vantage_api_key):
    
    # API URL template
    base_url = "https://www.alphavantage.co/query"

    # Function to fetch financial statements
    def fetch_statements(ticker, statement_type):
        params = {
            'function': statement_type,
            'symbol': ticker,
            'apikey': alpha_vantage_api_key
        }
        response = requests.get(base_url, params=params)
        return response.json()

    # Flatten nested financial data
    def flatten_statement(ticker, statement_data):
        def unnest(statement, kind):
            rows = []
            for report_type, label in [('annualReports', 'annual'), ('quarterlyReports', 'quarterly')]:
                reports = statement.get(report_type, [])
                for report in reports:
                    flat = report.copy()
                    flat['ticker'] = ticker
                    flat['report_type'] = label
                    flat['report_date'] = report.get('fiscalDateEnding')
                    rows.append(flat)
            return pd.DataFrame(rows)

        income_df = unnest(statement_data['income_statement'], 'income_statement')
        balance_df = unnest(statement_data['balance_sheet'], 'balance_sheet')
        cashflow_df = unnest(statement_data['cash_flow'], 'cash_flow')
        return income_df, balance_df, cashflow_df

    # Accumulate results
    income_all = []
    balance_all = []
    cashflow_all = []

    for ticker in tqdm(tickers):
        print(f"Fetching financial statements for {ticker}...")

        income_statement = fetch_statements(ticker, 'INCOME_STATEMENT')
        balance_sheet = fetch_statements(ticker, 'BALANCE_SHEET')
        cash_flow = fetch_statements(ticker, 'CASH_FLOW')

        record = {
            "ticker": ticker,
            "income_statement": income_statement,
            "balance_sheet": balance_sheet,
            "cash_flow": cash_flow
        }

        income_df, balance_df, cashflow_df = flatten_statement(ticker, record)

        income_all.append(income_df)
        balance_all.append(balance_df)
        cashflow_all.append(cashflow_df)

        time.sleep(1) 

    # Combine all tickers
    final_income_df = pd.concat(income_all, ignore_index=True)
    final_balance_df = pd.concat(balance_all, ignore_index=True)
    final_cashflow_df = pd.concat(cashflow_all, ignore_index=True)

    print("✅ All financial statements fetched and flattened.")

    # Ensure report_date is datetime
    for df in [final_income_df, final_balance_df, final_cashflow_df]:
        df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')

    # Function to transform each df
    def clean_quarterly_df(df):
        df = df[df['report_type'] == 'quarterly']
        start_date_dt = pd.to_datetime(start_date)
        if end_date:
            end_date_dt = pd.to_datetime(end_date)
            df = df[(df['report_date'] >= start_date_dt) & (df['report_date'] <= end_date_dt)]
        else:
            df = df[df['report_date'] >= start_date_dt]
        df = df.drop(columns=['report_type', 'fiscalDateEnding'], errors='ignore')
        
        # Reorder columns: ticker, report_date first
        front_cols = ['ticker', 'report_date']
        other_cols = [col for col in df.columns if col not in front_cols]
        return df[front_cols + other_cols]

    # Apply transformation
    quarterly_income_df = clean_quarterly_df(final_income_df)
    quarterly_balance_df = clean_quarterly_df(final_balance_df)
    quarterly_cashflow_df = clean_quarterly_df(final_cashflow_df)

    # Rename columns to snake_case
    quarterly_income_df.columns = [
        'ticker', 'report_date', 'reported_currency', 'gross_profit',
        'total_revenue', 'cost_of_revenue', 'cost_of_goods_and_services_sold',
        'operating_income', 'selling_general_and_administrative',
        'research_and_development', 'operating_expenses', 'investment_income_net',
        'net_interest_income', 'interest_income', 'interest_expense',
        'non_interest_income', 'other_non_operating_income', 'depreciation',
        'depreciation_and_amortization', 'income_before_tax', 'income_tax_expense',
        'interest_and_debt_expense', 'net_income_from_continuing_operations',
        'comprehensive_income_net_of_tax', 'ebit', 'ebitda', 'net_income'
    ]
    quarterly_balance_df.columns = [
        'ticker', 'report_date', 'reported_currency', 'total_assets',
        'total_current_assets', 'cash_and_cash_equivalents_at_carrying_value',
        'cash_and_short_term_investments', 'inventory', 'current_net_receivables',
        'total_non_current_assets', 'property_plant_equipment',
        'accumulated_depreciation_amortization_ppe', 'intangible_assets',
        'intangible_assets_excluding_goodwill', 'goodwill', 'investments',
        'long_term_investments', 'short_term_investments', 'other_current_assets',
        'other_non_current_assets', 'total_liabilities', 'total_current_liabilities',
        'current_accounts_payable', 'deferred_revenue', 'current_debt',
        'short_term_debt', 'total_non_current_liabilities',
        'capital_lease_obligations', 'long_term_debt', 'current_long_term_debt',
        'long_term_debt_noncurrent', 'short_long_term_debt_total',
        'other_current_liabilities', 'other_non_current_liabilities',
        'total_shareholder_equity', 'treasury_stock', 'retained_earnings',
        'common_stock', 'common_stock_shares_outstanding'
    ]
    quarterly_cashflow_df.columns = [
        'ticker', 'report_date', 'reported_currency', 'operating_cashflow',
        'payments_for_operating_activities', 'proceeds_from_operating_activities',
        'change_in_operating_liabilities', 'change_in_operating_assets',
        'depreciation_depletion_and_amortization', 'capital_expenditures',
        'change_in_receivables', 'change_in_inventory', 'profit_loss',
        'cashflow_from_investment', 'cashflow_from_financing',
        'proceeds_from_repayments_of_short_term_debt',
        'payments_for_repurchase_of_common_stock', 'payments_for_repurchase_of_equity',
        'payments_for_repurchase_of_preferred_stock', 'dividend_payout',
        'dividend_payout_common_stock', 'dividend_payout_preferred_stock',
        'proceeds_from_issuance_of_common_stock',
        'proceeds_from_issuance_of_long_term_debt_and_capital_securities_net',
        'proceeds_from_issuance_of_preferred_stock',
        'proceeds_from_repurchase_of_equity', 'proceeds_from_sale_of_treasury_stock',
        'change_in_cash_and_cash_equivalents', 'change_in_exchange_rate', 'net_income'
    ]

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Delete all rows from the table
    cursor.execute(f"DELETE FROM income_statement_quarterly")
    cursor.execute(f"DELETE FROM balance_statement_quarterly")
    cursor.execute(f"DELETE FROM cashflow_statement_quarterly")

    # Insert data using pandas
    quarterly_income_df.to_sql("income_statement_quarterly", conn, if_exists="append", index=False)
    quarterly_balance_df.to_sql("balance_statement_quarterly", conn, if_exists="append", index=False)
    quarterly_cashflow_df.to_sql("cashflow_statement_quarterly", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()

    print(f"Database {db_name} populated with financial statements.")


def populate_db_with_ticker_sector_information(tickers, db_name):

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Delete all rows from the table
    cursor.execute(f"DELETE FROM sector_information")   

    # Define sector-to-ETF mapping
    sector_to_etf = {
        'Technology': 'XLK',
        'Financial Services': 'XLF',
        'Healthcare': 'XLV',
        'Consumer Cyclical': 'XLY',
        'Consumer Defensive': 'XLP',
        'Industrials': 'XLI',
        'Communication Services': 'XLC',
        'Utilities': 'XLU',
        'Energy': 'XLE',
        'Basic Materials': 'XLB',
        'Real Estate': 'XLRE',
    }

    for ticker in tqdm(tickers):
        print(f"Fetching sector data for {ticker}...")
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        sector = info.get("sector", "")
        industry = info.get("industry", "")
        sector_etf = sector_to_etf.get(sector, "")

        if sector:
            record = (
                ticker,
                info.get("longName", ""),
                sector,
                industry,
                sector_etf,
                info.get("marketCap", None),
                info.get("longBusinessSummary", ""),
                info.get("country", ""),
                info.get("exchange", ""),
                info.get("currency", "")
            )

            cursor.execute("""
                INSERT OR REPLACE INTO sector_information (
                    ticker, name, sector, industry, sector_etf,
                    market_cap, description, country, exchange, currency
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, record)

            print(f"Inserted sector data for {ticker}")
        else:
            print(f"Failed to fetch sector data for {ticker}")

    # Commit and close
    conn.commit()
    conn.close()

    print("✅ Sector information fetching and storing completed.")


def populate_db_with_macro_data(tickers, db_name, start_date="2020-01-01", end_date=None, fred_api_key=fred_api_key):

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    # FRED series IDs - Minimal 7-metric macro set
    SERIES_IDS = {
        'gdp_growth': 'A191RL1Q225SBEA',    # Real GDP growth rate (quarterly)
        'inflation_yoy': 'CPIAUCSL',        # Consumer Price Index (monthly)
        'unemployment_rate': 'UNRATE',      # Unemployment rate (monthly)
        'federal_funds_rate': 'FEDFUNDS',   # Federal funds rate (monthly)
        'yield_10y': 'GS10',                # 10-Year Treasury (for yield spread and credit spread)
        'yield_2y': 'GS2',                  # 2-Year Treasury (for yield curve spread)
        'yield_3m': 'GS3M',                 # 3-Month Treasury (for yield spread)
        'baa_corporate_yield': 'BAA',       # BAA Corporate Bond Yield (for credit spread)
        'vix': 'VIXCLS'                     # CBOE Volatility Index
    }

    def fetch_data(series_id, start_date=start_date, end_date=None):
        params = {
            'series_id': series_id,
            'api_key': fred_api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()['observations']
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            return df[['date', 'value']].rename(columns={'value': series_id})
        except requests.exceptions.HTTPError as e:
            print(f"Error fetching {series_id}: {e}")
            return pd.DataFrame(columns=['date', series_id])

    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d') # yesterday
    inflation_yoy_start_date =  (pd.to_datetime(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Fetch data
    print("Fetching data...")
    gdp_data = fetch_data(SERIES_IDS['gdp_growth'], start_date=start_date, end_date=end_date)
    cpi_data = fetch_data(SERIES_IDS['inflation_yoy'], start_date=inflation_yoy_start_date, end_date=end_date)
    unemployment_data = fetch_data(SERIES_IDS['unemployment_rate'], start_date=start_date, end_date=end_date)
    fed_funds_data = fetch_data(SERIES_IDS['federal_funds_rate'], start_date=start_date, end_date=end_date)
    yield_10y_data = fetch_data(SERIES_IDS['yield_10y'], start_date=start_date, end_date=end_date)
    yield_2y_data = fetch_data(SERIES_IDS['yield_2y'], start_date=start_date, end_date=end_date)
    yield_3m_data = fetch_data(SERIES_IDS['yield_3m'], start_date=start_date, end_date=end_date)
    baa_yield_data = fetch_data(SERIES_IDS['baa_corporate_yield'], start_date=start_date, end_date=end_date)
    vix_data = fetch_data(SERIES_IDS['vix'], start_date=start_date, end_date=end_date)

    # Calculate inflation YoY
    cpi_data = cpi_data.set_index('date').sort_index()
    cpi_data['inflation_yoy'] = cpi_data['CPIAUCSL'].pct_change(periods=12) * 100
    inflation_data = cpi_data[['inflation_yoy']].reset_index()

    # Calculate yield curve spread (10Y - 2Y)
    yield_data = pd.merge(yield_10y_data, yield_2y_data, on='date')
    yield_data['yield_curve_spread'] = yield_data['GS10'] - yield_data['GS2']
    yield_spread_data = yield_data[['date', 'yield_curve_spread']]

    # Convert GDP to monthly
    gdp_monthly = gdp_data.set_index('date').resample('MS').ffill().reset_index()
    gdp_monthly = gdp_monthly.rename(columns={'A191RL1Q225SBEA': 'gdp_growth'})

    # Merge all data
    result = gdp_monthly
    for data in [inflation_data, unemployment_data, fed_funds_data, yield_spread_data, yield_10y_data, yield_3m_data, baa_yield_data, vix_data]:
        result = pd.merge(result, data, on='date', how='outer')

    # Rename columns
    result = result.rename(columns={
        'UNRATE': 'unemployment_rate',
        'FEDFUNDS': 'federal_funds_rate',
        'GS10': 'yield_10y',
        'GS3M': 'yield_3m',
        'BAA': 'baa_corporate_yield',
        'VIXCLS': 'vix'
    })

    # Finalize format
    result = result.sort_values('date')
    result['date'] = result['date'].dt.strftime('%Y-%m-%d')

    # Select and order columns - Minimal 7-metric set
    final_columns = ['date', 'gdp_growth', 'inflation_yoy', 'unemployment_rate', 
                    'federal_funds_rate', 'yield_curve_spread', 'yield_3m', 'yield_10y', 
                    'baa_corporate_yield', 'vix']
    result = result[final_columns]

    # Filter date range
    result = result[(result['date'] >= start_date) & (result['date'] <= end_date)]
    result = result.ffill().infer_objects(copy=False)

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Delete all rows from the table
    cursor.execute(f"DELETE FROM macroeconomic_monthly")

    # Insert data row-by-row with minimal 7-metric set
    for _, row in result.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO macroeconomic_monthly (
                date, gdp_growth, inflation_yoy, unemployment_rate,
                federal_funds_rate, yield_curve_spread, yield_3m, yield_10y,
                baa_corporate_yield, vix
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row['date'],
            row['gdp_growth'],
            row['inflation_yoy'],
            row['unemployment_rate'],
            row['federal_funds_rate'],
            row['yield_curve_spread'],
            row['yield_3m'],
            row['yield_10y'],
            row.get('baa_corporate_yield', None),  # Use get() in case data is missing
            row.get('vix', None)  # Use get() in case data is missing
        ))

    # Commit and close
    conn.commit()
    conn.close()

    print("✅ US macroeconomic data (7 core metrics) successfully inserted into 'macroeconomic_monthly'.")


def create_and_populate_db(tickers, db_name, sql_file_path="db_schema_create_statemets.sql", start_date="2020-01-01", end_date=None, alpha_vantage_api_key=alpha_vantage_api_key, fred_api_key=fred_api_key):

    tickers.extend(["SPY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLC", "XLU", "XLP", "XLB", "XLRE"])
    create_db_schema(db_name, sql_file_path=sql_file_path)
    populate_db_with_stock_prices(tickers=tickers, db_name=db_name, start_date=start_date, end_date=end_date)
    populate_db_with_financial_statements(tickers=tickers, db_name=db_name, start_date=start_date, alpha_vantage_api_key=alpha_vantage_api_key)
    populate_db_with_ticker_sector_information(tickers=tickers, db_name=db_name)
    populate_db_with_macro_data(tickers=tickers, db_name=db_name, start_date=start_date, end_date=end_date, fred_api_key=fred_api_key)

    print(f"Database {db_name} populated with data.")