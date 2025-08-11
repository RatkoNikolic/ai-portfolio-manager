-- Schema for table: stock_prices_daily
CREATE TABLE IF NOT EXISTS stock_prices_daily (
    ticker TEXT NOT NULL,
    date DATE NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    PRIMARY KEY (ticker, date)
);

-- Schema for table: income_statement_quarterly
CREATE TABLE IF NOT EXISTS income_statement_quarterly (
    ticker TEXT,
    report_date DATE,
    reported_currency TEXT,
    gross_profit REAL,
    total_revenue REAL,
    cost_of_revenue REAL,
    cost_of_goods_and_services_sold REAL,
    operating_income REAL,
    selling_general_and_administrative REAL,
    research_and_development REAL,
    operating_expenses REAL,
    investment_income_net REAL,
    net_interest_income REAL,
    interest_income REAL,
    interest_expense REAL,
    non_interest_income REAL,
    other_non_operating_income REAL,
    depreciation REAL,
    depreciation_and_amortization REAL,
    income_before_tax REAL,
    income_tax_expense REAL,
    interest_and_debt_expense REAL,
    net_income_from_continuing_operations REAL,
    comprehensive_income_net_of_tax REAL,
    ebit REAL,
    ebitda REAL,
    net_income REAL,
    PRIMARY KEY (ticker, report_date)
);

-- Schema for table: balance_statement_quarterly
CREATE TABLE IF NOT EXISTS balance_statement_quarterly (
    ticker TEXT,
    report_date DATE,
    reported_currency TEXT,
    total_assets REAL,
    total_current_assets REAL,
    cash_and_cash_equivalents_at_carrying_value REAL,
    cash_and_short_term_investments REAL,
    inventory REAL,
    current_net_receivables REAL,
    total_non_current_assets REAL,
    property_plant_equipment REAL,
    accumulated_depreciation_amortization_ppe REAL,
    intangible_assets REAL,
    intangible_assets_excluding_goodwill REAL,
    goodwill REAL,
    investments REAL,
    long_term_investments REAL,
    short_term_investments REAL,
    other_current_assets REAL,
    other_non_current_assets REAL,
    total_liabilities REAL,
    total_current_liabilities REAL,
    current_accounts_payable REAL,
    deferred_revenue REAL,
    current_debt REAL,
    short_term_debt REAL,
    total_non_current_liabilities REAL,
    capital_lease_obligations REAL,
    long_term_debt REAL,
    current_long_term_debt REAL,
    long_term_debt_noncurrent REAL,
    short_long_term_debt_total REAL,
    other_current_liabilities REAL,
    other_non_current_liabilities REAL,
    total_shareholder_equity REAL,
    treasury_stock REAL,
    retained_earnings REAL,
    common_stock REAL,
    common_stock_shares_outstanding REAL,
    PRIMARY KEY (ticker, report_date)
);

-- Schema for table: cashflow_statement_quarterly
CREATE TABLE IF NOT EXISTS cashflow_statement_quarterly (
    ticker TEXT,
    report_date DATE,
    reported_currency TEXT,
    operating_cashflow REAL,
    payments_for_operating_activities REAL,
    proceeds_from_operating_activities REAL,
    change_in_operating_liabilities REAL,
    change_in_operating_assets REAL,
    depreciation_depletion_and_amortization REAL,
    capital_expenditures REAL,
    change_in_receivables REAL,
    change_in_inventory REAL,
    profit_loss REAL,
    cashflow_from_investment REAL,
    cashflow_from_financing REAL,
    proceeds_from_repayments_of_short_term_debt REAL,
    payments_for_repurchase_of_common_stock REAL,
    payments_for_repurchase_of_equity REAL,
    payments_for_repurchase_of_preferred_stock REAL,
    dividend_payout REAL,
    dividend_payout_common_stock REAL,
    dividend_payout_preferred_stock REAL,
    proceeds_from_issuance_of_common_stock REAL,
    proceeds_from_issuance_of_long_term_debt_and_capital_securities_net REAL,
    proceeds_from_issuance_of_preferred_stock REAL,
    proceeds_from_repurchase_of_equity REAL,
    proceeds_from_sale_of_treasury_stock REAL,
    change_in_cash_and_cash_equivalents REAL,
    change_in_exchange_rate REAL,
    net_income REAL,
    PRIMARY KEY (ticker, report_date)
);

-- Schema for table: sector_information
CREATE TABLE IF NOT EXISTS sector_information (
    ticker TEXT PRIMARY KEY,
    name TEXT,
    sector TEXT,
    industry TEXT,
    sector_etf TEXT,
    market_cap INTEGER,
    description TEXT,
    country TEXT,
    exchange TEXT,
    currency TEXT
);

-- Schema for table: sentiment_monthly
CREATE TABLE IF NOT EXISTS sentiment_monthly (
    ticker TEXT,
    date DATE,
    sentiment INTEGER,
    explanation TEXT,
    PRIMARY KEY (ticker, date)
);

-- Schema for table: analysis_outputs
CREATE TABLE IF NOT EXISTS analysis_outputs (
    ticker TEXT,
    date DATE,
    analysis_output TEXT,
    created_at DATE,
    PRIMARY KEY (ticker, date)
);

-- Schema for table: macroeconomic_monthly
CREATE TABLE IF NOT EXISTS macroeconomic_monthly (
    date TEXT PRIMARY KEY,
    gdp_growth REAL,
    inflation_yoy REAL,
    unemployment_rate REAL,
    federal_funds_rate REAL,
    yield_curve_spread REAL,
    yield_3m REAL,
    yield_10y REAL,  -- 10-Year Treasury yield for credit spread calculation
    baa_corporate_yield REAL,  -- BAA Corporate Bond Yield for credit spread
    vix REAL  -- CBOE Volatility Index
);

-- Schema for table: signals
CREATE TABLE IF NOT EXISTS signals (
    ticker TEXT,
    date TEXT,
    signal TEXT,
    created_at TEXT,
    PRIMARY KEY (ticker, date)
);

-- Schema for table: portfolio
CREATE TABLE IF NOT EXISTS portfolio (
    portfolio_name TEXT,
    date DATE,
    call TEXT,
    response TEXT,
    state TEXT,
    notes TEXT,
    PRIMARY KEY (portfolio_name, date)
);