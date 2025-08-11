#!/usr/bin/env python3
"""
Valuation Analyst for AI Portfolio Manager
Evaluates if the stock is over/under-valued relative to fundamentals and sector.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from typing import Dict, Optional
import yfinance as yf
import time
import os

from data_manager import DataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValuationAnalyst:
    """Evaluates stock valuation ratios."""
    
    def __init__(self, data_manager: DataManager):
        """Initialize the Valuation Analyst."""
        self.dm = data_manager
        logger.info("Valuation Analyst initialized")
    
    def get_historical_market_data(self, ticker: str, analysis_date: str) -> Dict:
        """Get historical market data for a ticker on analysis date."""
        try:
            # Get historical price from the previous trading day
            historical_price = 0
            try:
                from datetime import datetime, timedelta
                analysis_dt = datetime.strptime(analysis_date, '%Y-%m-%d')
                
                # Look back up to 7 days to ensure we find the previous trading day
                start_date = (analysis_dt - timedelta(days=7)).strftime('%Y-%m-%d')
                end_date = (analysis_dt - timedelta(days=1)).strftime('%Y-%m-%d')  # Day before analysis date
                
                # Add time to ensure BETWEEN query includes end date
                end_date = end_date + ' 23:59:59'
                
                prices = self.dm.get_prices(ticker, start_date, end_date)
                if not prices.empty:
                    # Ensure data is sorted chronologically and get the last available price
                    prices = prices.sort_index()
                    historical_price = prices['close'].iloc[-1]
                    price_date = prices.index[-1]
                    logger.info(f"Using previous trading day price for {ticker} on {price_date.date()} (analysis date: {analysis_date}): ${historical_price:.2f}")
                    logger.info(f"Available dates in query: {[d.date() for d in prices.index]}")
                else:
                    logger.warning(f"No price data found before analysis date {analysis_date}")
            except Exception as e:
                logger.warning(f"Could not get previous trading day price from database: {e}")
            
            # Get shares outstanding and other data from yfinance (less time-sensitive)
            stock = yf.Ticker(ticker)
            info = stock.info
            shares_outstanding = info.get('sharesOutstanding', 0)
            
            # Calculate market cap using historical price
            market_cap = historical_price * shares_outstanding if historical_price > 0 and shares_outstanding > 0 else 0
            
            # Estimate enterprise value (Market Cap + Total Debt - Cash)
            # Use approximate values since historical debt/cash is complex to retrieve
            enterprise_value = market_cap * 1.05  # Rough approximation: assume minimal net debt
            
            market_data = {
                'market_cap': market_cap,
                'current_price': historical_price,
                'shares_outstanding': shares_outstanding,
                'enterprise_value': enterprise_value
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting historical market data for {ticker}: {e}")
            return {
                'market_cap': 0,
                'current_price': 0,
                'shares_outstanding': 0,
                'enterprise_value': 0
            }
    
    def calculate_pe_ratio(self, market_data: Dict, eps: float) -> float:
        """Calculate P/E Ratio."""
        try:
            current_price = market_data.get('current_price', 0)
            
            if eps == 0 or current_price == 0:
                return 0.0
            
            pe_ratio = current_price / eps
            return round(pe_ratio, 2)
            
        except Exception as e:
            logger.error(f"Error calculating P/E ratio: {e}")
            return 0.0
    
    def calculate_peg_ratio(self, pe_ratio: float, eps_growth: float) -> float:
        """Calculate PEG Ratio using percentage growth."""
        try:
            if eps_growth == 0:
                return 0.0
            
            # Convert decimal growth to percentage (e.g., 0.248 -> 24.8)
            eps_growth_pct = eps_growth * 100
            
            # Avoid division by very small growth rates
            if abs(eps_growth_pct) < 0.1:
                return 0.0
            
            peg_ratio = pe_ratio / eps_growth_pct
            logger.info(f"PEG calculation: PE={pe_ratio:.2f}, Growth={eps_growth_pct:.1f}%, PEG={peg_ratio:.2f}")
            return round(peg_ratio, 2)
            
        except Exception as e:
            logger.error(f"Error calculating PEG ratio: {e}")
            return 0.0
    
    def calculate_price_to_book(self, market_data: Dict, book_value: float) -> float:
        """Calculate Price-to-Book ratio."""
        try:
            market_cap = market_data.get('market_cap', 0)
            
            if book_value == 0 or market_cap == 0:
                return 0.0
            
            price_to_book = market_cap / book_value
            return round(price_to_book, 2)
            
        except Exception as e:
            logger.error(f"Error calculating Price-to-Book ratio: {e}")
            return 0.0
    
    def calculate_ev_to_ebitda(self, market_data: Dict, ebitda: float) -> float:
        """Calculate EV/EBITDA ratio."""
        try:
            enterprise_value = market_data.get('enterprise_value', 0)
            
            if ebitda == 0 or enterprise_value == 0:
                return 0.0
            
            ev_to_ebitda = enterprise_value / ebitda
            return round(ev_to_ebitda, 2)
            
        except Exception as e:
            logger.error(f"Error calculating EV/EBITDA ratio: {e}")
            return 0.0
    
    def calculate_price_to_sales(self, market_data: Dict, revenue: float) -> float:
        """Calculate Price-to-Sales ratio."""
        try:
            market_cap = market_data.get('market_cap', 0)
            
            if revenue == 0 or market_cap == 0:
                return 0.0
            
            price_to_sales = market_cap / revenue
            return round(price_to_sales, 2)
            
        except Exception as e:
            logger.error(f"Error calculating Price-to-Sales ratio: {e}")
            return 0.0
    
    def calculate_price_to_fcf(self, market_data: Dict, free_cash_flow: float) -> float:
        """Calculate Price/FCF ratio."""
        try:
            market_cap = market_data.get('market_cap', 0)
            
            if free_cash_flow == 0 or market_cap == 0:
                return 0.0
            
            price_to_fcf = market_cap / free_cash_flow
            return round(price_to_fcf, 2)
            
        except Exception as e:
            logger.error(f"Error calculating Price/FCF ratio: {e}")
            return 0.0
    


    def get_financial_metrics(self, financials: Dict) -> Dict:
        """Extract financial metrics needed for valuation."""
        try:
            income_statement = financials.get('income_statement', {})
            balance_sheet = financials.get('balance_sheet', {})
            cash_flow = financials.get('cash_flow', {})
            
            # Get latest quarterly data
            income_quarterly = income_statement.get('quarterlyReports', [])
            balance_quarterly = balance_sheet.get('quarterlyReports', [])
            cash_quarterly = cash_flow.get('quarterlyReports', [])
            
            if not income_quarterly or not balance_quarterly or not cash_quarterly:
                return {
                    'eps': 0.0,
                    'book_value': 0.0,
                    'ebitda': 0.0,
                    'revenue': 0.0,
                    'free_cash_flow': 0.0,
                    'eps_growth': 0.0
                }
            
            latest_income = income_quarterly[0]
            latest_balance = balance_quarterly[0]
            latest_cash = cash_quarterly[0]
            
            # Calculate annualized values (quarterly * 4)
            shares_outstanding = float(latest_balance.get('common_stock_shares_outstanding', 1))
            eps = float(latest_income.get('net_income', 0)) / shares_outstanding * 4 if shares_outstanding != 0 else 0.0
            book_value = float(latest_balance.get('total_shareholder_equity', 0))  # SQLite field name
            ebitda = float(latest_income.get('ebitda', 0)) * 4  # SQLite field name
            revenue = float(latest_income.get('total_revenue', 0)) * 4  # SQLite field name
            free_cash_flow = float(latest_cash.get('operating_cashflow', 0)) * 4  # SQLite field name
            
            # Calculate EPS growth (if we have previous quarter)
            eps_growth = 0.0
            if len(income_quarterly) >= 2 and len(balance_quarterly) >= 2:
                prev_income = income_quarterly[1]
                prev_balance = balance_quarterly[1]
                prev_shares_outstanding = float(prev_balance.get('common_stock_shares_outstanding', 1))
                prev_eps = float(prev_income.get('net_income', 0)) / prev_shares_outstanding * 4 if prev_shares_outstanding != 0 else 0.0
                if prev_eps != 0:
                    eps_growth = (eps - prev_eps) / prev_eps
            
            return {
                'eps': eps,
                'book_value': book_value,
                'ebitda': ebitda,
                'revenue': revenue,
                'free_cash_flow': free_cash_flow,
                'eps_growth': eps_growth
            }
            
        except Exception as e:
            logger.error(f"Error extracting financial metrics: {e}")
            return {
                'eps': 0.0,
                'book_value': 0.0,
                'ebitda': 0.0,
                'revenue': 0.0,
                'free_cash_flow': 0.0,
                'eps_growth': 0.0
            }
    

    
    def analyze(self, ticker: str, analysis_date: str, debug: bool = False) -> Dict:
        """
        Perform valuation analysis for a given ticker and date.
        """
        default_metrics = {
            "pe_ratio": 0.0,
            "peg_ratio": 0.0,
            "price_to_book": 0.0,
            "ev_to_ebitda": 0.0,
            "price_to_sales": 0.0,
            "price_to_fcf": 0.0
        }
        
        try:
            logger.info(f"Starting valuation analysis for {ticker}")
            
            # Get financial data
            financials = self.dm.get_financials(ticker, analysis_date)
            if not financials:
                logger.warning(f"No financial data available for {ticker}")
                return {
                    "analyst": "valuation",
                    "metrics": default_metrics
                }
            
            # Get historical market data for analysis date
            market_data = self.get_historical_market_data(ticker, analysis_date)
            
            # Get financial metrics
            financial_metrics = self.get_financial_metrics(financials)
            

            
            # Calculate valuation metrics
            pe_ratio = self.calculate_pe_ratio(market_data, financial_metrics['eps'])
            metrics = {
                'price_earnings_ratio': pe_ratio,
                'pe_growth_ratio': self.calculate_peg_ratio(pe_ratio, financial_metrics['eps_growth']),
                'price_book_ratio': self.calculate_price_to_book(market_data, financial_metrics['book_value']),
                'ev_ebitda_ratio': self.calculate_ev_to_ebitda(market_data, financial_metrics['ebitda']),
                'price_sales_ratio': self.calculate_price_to_sales(market_data, financial_metrics['revenue']),
                'price_fcf_ratio': self.calculate_price_to_fcf(market_data, financial_metrics['free_cash_flow'])
            }
            
            result = {
                "analyst": "valuation",
                "metrics": metrics
            }
            
            if debug:
                result["data_used"] = {
                    "summary": {
                        "ticker": ticker,
                        "analysis_date": analysis_date,
                        "market_data": market_data,
                        "financial_metrics": financial_metrics
                    }
                }
            
            logger.info(f"Completed valuation analysis for {ticker}")
            return result
            
        except Exception as e:
            logger.error(f"Error in valuation analysis for {ticker}: {e}")
            return {
                "analyst": "valuation",
                "metrics": default_metrics
            }


 