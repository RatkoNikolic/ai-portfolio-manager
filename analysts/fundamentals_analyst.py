#!/usr/bin/env python3
"""
Fundamentals Analyst for AI Portfolio Manager
Analyzes company financial health using TTM (trailing twelve month) calculations from quarterly statements.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from typing import Dict, Optional, List
import os

from data_manager import DataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FundamentalsAnalyst:
    """Analyzes company financial health using TTM-based quarterly statements."""
    
    def __init__(self, data_manager: DataManager):
        """Initialize the Fundamentals Analyst."""
        self.dm = data_manager
        logger.info("Fundamentals Analyst initialized")
    
    def _get_quarterly_reports_by_date(self, quarterly_reports, analysis_date, num_reports=4):
        """Return up to num_reports reports with report_date <= analysis_date, sorted descending."""
        try:
            filtered = [
                r for r in quarterly_reports
                if 'report_date' in r and r['report_date'] <= analysis_date
            ]
            filtered.sort(key=lambda r: r['report_date'], reverse=True)
            return filtered[:num_reports]
        except Exception as e:
            logger.error(f"Error filtering quarterly reports by date: {e}")
            return []

    def _get_value(self, report, field, fallback_report=None, fallback_field=None):
        """Get a value from a report, or fallback to another report/field if missing or None. Always return float."""
        val = report.get(field)
        if val in [None, '', 'None'] and fallback_report and fallback_field:
            val = fallback_report.get(fallback_field, 0)
        if val in [None, '', 'None']:
            return 0.0
        # Handle string with commas or formatting
        if isinstance(val, str):
            val = val.replace(',', '').replace(' ', '')
        try:
            return float(val)
        except Exception:
            return 0.0

    def _sum_ttm_values(self, reports: List[Dict], field: str, num_quarters: int = None) -> float:
        """Sum values from the specified number of quarters."""
        if num_quarters is None:
            num_quarters = 4  # Default TTM
        if len(reports) < num_quarters:
            return 0.0
        return sum(self._get_value(report, field) for report in reports[:num_quarters])

    def _get_weighted_avg_shares_ttm(self, reports: List[Dict]) -> float:
        """Calculate weighted average shares outstanding for TTM EPS calculation."""
        if len(reports) < 4:
            return 0.0
        
        # Use common_stock_shares_outstanding from the most recent 4 quarters
        # For simplicity, we'll use the average of the 4 quarterly values
        # In a more sophisticated implementation, we'd weight by days in each quarter
        shares_values = [self._get_value(report, 'common_stock_shares_outstanding') for report in reports[:4]]
        valid_shares = [s for s in shares_values if s > 0]
        
        if not valid_shares:
            return 0.0
        
        return sum(valid_shares) / len(valid_shares)

    def calculate_revenue_growth_ttm(self, ticker: str, analysis_date: str) -> float:
        """
        Calculate TTM revenue growth: (Σ total_revenue₁→₄ − Σ total_revenue₀→₃) / Σ total_revenue₀→₃
        Requires 8 quarters of data to compare current 4 vs previous 4.
        Fallback: Use available quarters for best possible comparison.
        """
        try:
            # Try to get 8 quarters first
            financials_8q = self.dm.get_financials(ticker, analysis_date, 8)
            if not financials_8q:
                logger.warning(f"Could not retrieve financial data for {ticker}")
                return 0.0
            
            quarterly_reports = financials_8q['income_statement'].get('quarterlyReports', [])
            reports = self._get_quarterly_reports_by_date(quarterly_reports, analysis_date, 8)
            
            if len(reports) >= 8:
                # Full 8-quarter comparison
                current_ttm_revenue = self._sum_ttm_values(reports[:4], 'total_revenue')
                previous_ttm_revenue = self._sum_ttm_values(reports[4:8], 'total_revenue')
                
                if previous_ttm_revenue == 0:
                    return 0.0
                
                growth = (current_ttm_revenue - previous_ttm_revenue) / previous_ttm_revenue
                logger.info(f"TTM revenue growth for {ticker}: {growth:.4f} (8Q: current: {current_ttm_revenue:.0f}, previous: {previous_ttm_revenue:.0f})")
                return round(growth, 4)
                
            elif len(reports) >= 6:
                # Fallback: Compare current 3 quarters vs previous 3 quarters
                current_revenue = self._sum_ttm_values(reports[:3], 'total_revenue', 3)
                previous_revenue = self._sum_ttm_values(reports[3:6], 'total_revenue', 3)
                
                if previous_revenue == 0:
                    return 0.0
                
                growth = (current_revenue - previous_revenue) / previous_revenue
                logger.info(f"TTM revenue growth for {ticker}: {growth:.4f} (6Q fallback: current: {current_revenue:.0f}, previous: {previous_revenue:.0f})")
                return round(growth, 4)
                
            elif len(reports) >= 4:
                # Minimal fallback: Year-over-year single quarter comparison
                if len(reports) >= 4:
                    current_q = self._get_value(reports[0], 'total_revenue')
                    year_ago_q = self._get_value(reports[3], 'total_revenue')
                    
                    if year_ago_q == 0:
                        return 0.0
                    
                    growth = (current_q - year_ago_q) / year_ago_q
                    logger.info(f"TTM revenue growth for {ticker}: {growth:.4f} (4Q YoY fallback: current: {current_q:.0f}, previous: {year_ago_q:.0f})")
                    return round(growth, 4)
            
            logger.warning(f"Insufficient data for revenue growth calculation for {ticker} (only {len(reports)} quarters)")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating TTM revenue growth for {ticker}: {e}")
            return 0.0
    
    def calculate_eps_ttm(self, income_statement: Dict, balance_sheet: Dict, analysis_date: str) -> float:
        """
        Calculate TTM EPS: Σ net_income₁→₄ / weighted_avg_shares_outstandingₜₜₘ
        """
        try:
            quarterly_reports = income_statement.get('quarterlyReports', [])
            balance_quarters = balance_sheet.get('quarterlyReports', [])
            
            income_reports = self._get_quarterly_reports_by_date(quarterly_reports, analysis_date, 4)
            balance_reports = self._get_quarterly_reports_by_date(balance_quarters, analysis_date, 4)
            
            if len(income_reports) < 4:
                logger.warning(f"Only {len(income_reports)} income quarters available, need 4 for TTM EPS")
                return 0.0
            
            # TTM net income (sum of 4 quarters)
            ttm_net_income = self._sum_ttm_values(income_reports, 'net_income')
            
            # Get weighted average shares outstanding from balance sheet (more reliable)
            if len(balance_reports) < 4:
                logger.warning(f"Only {len(balance_reports)} balance quarters available, need 4 for TTM EPS")
                return 0.0
                
            weighted_avg_shares = self._get_weighted_avg_shares_ttm(balance_reports)
            
            if weighted_avg_shares == 0:
                logger.warning(f"Weighted average shares is zero")
                return 0.0
            
            if ttm_net_income == 0:
                logger.warning(f"TTM net income is zero")
                return 0.0
            
            eps = ttm_net_income / weighted_avg_shares
            logger.info(f"TTM EPS calculation: {ttm_net_income:.0f} / {weighted_avg_shares:.0f} = {eps:.4f}")
            return round(eps, 4)
            
        except Exception as e:
            logger.error(f"Error calculating TTM EPS: {e}")
            return 0.0
    
    def calculate_roe_ttm(self, income_statement: Dict, balance_sheet: Dict, analysis_date: str) -> float:
        """
        Calculate TTM ROE: Σ net_income₁→₄ / ((equity_end₄ + equity_end₀) / 2)
        Uses average of opening and closing equity over the TTM period.
        """
        try:
            income_quarterly = income_statement.get('quarterlyReports', [])
            balance_quarterly = balance_sheet.get('quarterlyReports', [])
            
            income_reports = self._get_quarterly_reports_by_date(income_quarterly, analysis_date, 4)
            balance_reports = self._get_quarterly_reports_by_date(balance_quarterly, analysis_date, 5)  # Need 5 to get opening equity
            
            if len(income_reports) < 4 or len(balance_reports) < 4:
                return 0.0
            
            # TTM net income (sum of 4 quarters)
            ttm_net_income = self._sum_ttm_values(income_reports, 'net_income')
            
            # Ending equity (most recent quarter)
            equity_end = self._get_value(balance_reports[0], 'total_shareholder_equity')
            
            # Opening equity (5 quarters ago, or use 4 quarters ago if 5 not available)
            equity_start = self._get_value(balance_reports[4], 'total_shareholder_equity') if len(balance_reports) >= 5 else equity_end
            
            # Average equity
            average_equity = (equity_end + equity_start) / 2
            
            if average_equity == 0:
                return 0.0
            
            roe = ttm_net_income / average_equity
            return round(roe, 4)
            
        except Exception as e:
            logger.error(f"Error calculating TTM ROE: {e}")
            return 0.0
    
    def calculate_roa_ttm(self, income_statement: Dict, balance_sheet: Dict, analysis_date: str) -> float:
        """
        Calculate TTM ROA: Σ net_income₁→₄ / ((assets_end₄ + assets_end₀) / 2)
        Uses average of opening and closing assets over the TTM period.
        """
        try:
            income_quarterly = income_statement.get('quarterlyReports', [])
            balance_quarterly = balance_sheet.get('quarterlyReports', [])
            
            income_reports = self._get_quarterly_reports_by_date(income_quarterly, analysis_date, 4)
            balance_reports = self._get_quarterly_reports_by_date(balance_quarterly, analysis_date, 5)  # Need 5 to get opening assets
            
            if len(income_reports) < 4 or len(balance_reports) < 4:
                return 0.0
            
            # TTM net income (sum of 4 quarters)
            ttm_net_income = self._sum_ttm_values(income_reports, 'net_income')
            
            # Ending assets (most recent quarter)
            assets_end = self._get_value(balance_reports[0], 'total_assets')
            
            # Opening assets (5 quarters ago, or use 4 quarters ago if 5 not available)
            assets_start = self._get_value(balance_reports[4], 'total_assets') if len(balance_reports) >= 5 else assets_end
            
            # Average assets
            average_assets = (assets_end + assets_start) / 2
            
            if average_assets == 0:
                return 0.0
            
            roa = ttm_net_income / average_assets
            return round(roa, 4)
            
        except Exception as e:
            logger.error(f"Error calculating TTM ROA: {e}")
            return 0.0
    
    def calculate_gross_margin_ttm(self, income_statement: Dict, analysis_date: str) -> float:
        """
        Calculate TTM Gross Margin: Σ gross_profit₁→₄ / Σ total_revenue₁→₄
        """
        try:
            quarterly_reports = income_statement.get('quarterlyReports', [])
            reports = self._get_quarterly_reports_by_date(quarterly_reports, analysis_date, 4)
            
            if len(reports) < 4:
                return 0.0
            
            ttm_gross_profit = self._sum_ttm_values(reports, 'gross_profit')
            ttm_total_revenue = self._sum_ttm_values(reports, 'total_revenue')
            
            if ttm_total_revenue == 0:
                return 0.0
            
            margin = ttm_gross_profit / ttm_total_revenue
            return round(margin, 4)
            
        except Exception as e:
            logger.error(f"Error calculating TTM gross margin: {e}")
            return 0.0

    def calculate_operating_margin_ttm(self, income_statement: Dict, analysis_date: str) -> float:
        """
        Calculate TTM Operating Margin: Σ operating_income₁→₄ / Σ total_revenue₁→₄
        """
        try:
            quarterly_reports = income_statement.get('quarterlyReports', [])
            reports = self._get_quarterly_reports_by_date(quarterly_reports, analysis_date, 4)
            
            if len(reports) < 4:
                return 0.0
            
            ttm_operating_income = self._sum_ttm_values(reports, 'operating_income')
            ttm_total_revenue = self._sum_ttm_values(reports, 'total_revenue')
            
            if ttm_total_revenue == 0:
                return 0.0
            
            margin = ttm_operating_income / ttm_total_revenue
            return round(margin, 4)
            
        except Exception as e:
            logger.error(f"Error calculating TTM operating margin: {e}")
            return 0.0

    def calculate_free_cash_flow_margin_ttm(self, cash_flow: Dict, income_statement: Dict, analysis_date: str) -> float:
        """
        Calculate TTM Free Cash Flow Margin: (Σ operating_cashflow₁→₄ − Σ capital_expenditures₁→₄) / Σ total_revenue₁→₄
        """
        try:
            cash_quarterly = cash_flow.get('quarterlyReports', [])
            income_quarterly = income_statement.get('quarterlyReports', [])
            
            cash_reports = self._get_quarterly_reports_by_date(cash_quarterly, analysis_date, 4)
            income_reports = self._get_quarterly_reports_by_date(income_quarterly, analysis_date, 4)
            
            if len(cash_reports) < 4 or len(income_reports) < 4:
                return 0.0
            
            ttm_operating_cashflow = self._sum_ttm_values(cash_reports, 'operating_cashflow')
            ttm_capital_expenditures = self._sum_ttm_values(cash_reports, 'capital_expenditures')
            ttm_total_revenue = self._sum_ttm_values(income_reports, 'total_revenue')
            
            ttm_free_cash_flow = ttm_operating_cashflow - ttm_capital_expenditures
            
            if ttm_total_revenue == 0:
                return 0.0
            
            margin = ttm_free_cash_flow / ttm_total_revenue
            return round(margin, 4)
            
        except Exception as e:
            logger.error(f"Error calculating TTM free cash flow margin: {e}")
            return 0.0

    def calculate_current_ratio(self, balance_sheet: Dict, analysis_date: str) -> float:
        """
        Calculate Current Ratio: current_assets₄ / current_liabilities₄
        Point-in-time balance sheet metric using most recent quarter.
        """
        try:
            quarterly_reports = balance_sheet.get('quarterlyReports', [])
            reports = self._get_quarterly_reports_by_date(quarterly_reports, analysis_date, 1)
            
            if not reports:
                return 0.0
            
            latest = reports[0]
            current_assets = self._get_value(latest, 'total_current_assets')
            current_liabilities = self._get_value(latest, 'total_current_liabilities')
            
            if current_liabilities == 0:
                return 0.0
            
            ratio = current_assets / current_liabilities
            return round(ratio, 4)
            
        except Exception as e:
            logger.error(f"Error calculating current ratio: {e}")
            return 0.0

    def calculate_debt_to_equity(self, balance_sheet: Dict, analysis_date: str) -> float:
        """
        Calculate Debt-to-Equity ratio: total_liabilities₄ / total_shareholder_equity₄
        Point-in-time balance sheet metric using most recent quarter.
        """
        try:
            quarterly_reports = balance_sheet.get('quarterlyReports', [])
            reports = self._get_quarterly_reports_by_date(quarterly_reports, analysis_date, 1)
            
            if not reports:
                return 0.0
            
            latest = reports[0]
            total_liabilities = self._get_value(latest, 'total_liabilities')
            shareholder_equity = self._get_value(latest, 'total_shareholder_equity')
            
            if shareholder_equity == 0:
                return 0.0
            
            ratio = total_liabilities / shareholder_equity
            return round(ratio, 4)
            
        except Exception as e:
            logger.error(f"Error calculating debt-to-equity ratio: {e}")
            return 0.0

    def _get_etf_price_on_or_before(self, ticker, target_date):
        """Get the ETF close price on or before the target date from SQLite price data."""
        try:
            prices = self.dm.get_prices(ticker, '2000-01-01', target_date)
            if prices.empty:
                return None
            # Filter to dates <= target_date and get the last available
            prices = prices[prices.index <= pd.to_datetime(target_date)]
            if prices.empty:
                return None
            return float(prices.iloc[-1]['close'])
        except Exception as e:
            logger.error(f"Error getting ETF price for {ticker} on {target_date}: {e}")
            return None

    def calculate_sector_relative_growth_ttm(self, ticker: str, sector_etf: str, analysis_date: str) -> float:
        """
        Calculate TTM sector relative growth: (revenue₄/revenue₀ − 1) − (ETF_close₄/ETF_close₀ − 1)
        Company's TTM revenue growth vs. sector ETF benchmark over the same period.
        Uses company reporting dates for ETF price lookup to ensure proper alignment.
        """
        try:
            # Get company financial data
            financials_8q = self.dm.get_financials(ticker, analysis_date, 8)
            if not financials_8q:
                logger.warning(f"Could not retrieve financial data for {ticker}")
                return 0.0
                
            quarterly_reports = financials_8q['income_statement'].get('quarterlyReports', [])
            reports = self._get_quarterly_reports_by_date(quarterly_reports, analysis_date, 8)
            
            # Choose comparison period based on available data
            if len(reports) >= 5:  # Use most recent vs 4 quarters ago
                revenue_recent = self._get_value(reports[0], 'total_revenue')
                revenue_base = self._get_value(reports[4], 'total_revenue')
                date_recent = reports[0]['report_date']
                date_base = reports[4]['report_date']
                comparison_desc = "4-quarter TTM"
            elif len(reports) >= 4:  # Use most recent vs 3 quarters ago
                revenue_recent = self._get_value(reports[0], 'total_revenue')
                revenue_base = self._get_value(reports[3], 'total_revenue')
                date_recent = reports[0]['report_date']
                date_base = reports[3]['report_date']
                comparison_desc = "3-quarter fallback"
            else:
                logger.warning(f"Insufficient quarters for sector relative growth: {len(reports)}")
                return 0.0
            
            if revenue_base == 0:
                logger.warning(f"Base revenue is zero for {ticker}")
                return 0.0
            
            company_growth = (revenue_recent / revenue_base - 1)
            
            # Get ETF prices using company's actual reporting dates for alignment
            # This ensures we're comparing the same time periods that the company's
            # financial performance represents
            etf_close_recent = self._get_etf_price_on_or_before(sector_etf, date_recent)
            etf_close_base = self._get_etf_price_on_or_before(sector_etf, date_base)
            
            if etf_close_recent is None or etf_close_base is None or etf_close_base == 0:
                logger.warning(f"Could not get ETF prices for {sector_etf} between {date_base} and {date_recent}")
                return 0.0
            
            etf_growth = (etf_close_recent / etf_close_base - 1)
            relative_growth = company_growth - etf_growth
            
            # Enhanced logging for transparency
            logger.info(f"Sector relative growth for {ticker} ({comparison_desc}): {relative_growth:.4f}")
            logger.info(f"  Company revenue growth: {company_growth:.4f} ({date_base} to {date_recent})")
            logger.info(f"  Sector ETF ({sector_etf}) growth: {etf_growth:.4f} (${etf_close_base:.2f} to ${etf_close_recent:.2f})")
            logger.info(f"  Relative performance: {relative_growth:.4f} ({relative_growth*100:.1f}%)")
            
            return round(relative_growth, 4)
            
        except Exception as e:
            logger.error(f"Error calculating TTM sector relative growth for {ticker}: {e}")
            return 0.0

    def analyze(self, ticker: str, analysis_date: str, debug: bool = False) -> Dict:
        """
        Perform TTM-based fundamental analysis for a given ticker and date.
        """
        default_metrics = {
            "revenue_growth_ttm": 0.0,
            "eps_ttm": 0.0,
            "roe_ttm": 0.0,
            "roa_ttm": 0.0,
            "gross_margin_ttm": 0.0,
            "operating_margin_ttm": 0.0,
            "free_cash_flow_margin_ttm": 0.0,
            "sector_relative_growth_ttm": 0.0,
            "current_ratio": 0.0,
            "debt_to_equity": 0.0
        }
        
        try:
            logger.info(f"Starting TTM fundamental analysis for {ticker}")
            financials = self.dm.get_financials(ticker, analysis_date)
            if not financials:
                logger.warning(f"No financial data available for {ticker}")
                return {
                    "analyst": "fundamentals",
                    "metrics": default_metrics
                }
            
            # Add ticker to financial data for revenue growth calculation
            financials['income_statement']['ticker'] = ticker
            
            # Get financial data
            inc = financials['income_statement']
            bal = financials['balance_sheet']
            cf = financials['cash_flow']
            
            # Get sector ETF info for comparison
            sector_info = self.dm.get_sector_info(ticker)
            sector_etf = sector_info.get('sector_etf', 'SPY')
            
            # Calculate TTM-based metrics
            metrics = {
                'revenue_growth_ttm': self.calculate_revenue_growth_ttm(ticker, analysis_date),
                'eps_ttm': self.calculate_eps_ttm(inc, bal, analysis_date),
                'roe_ttm': self.calculate_roe_ttm(inc, bal, analysis_date),
                'roa_ttm': self.calculate_roa_ttm(inc, bal, analysis_date),
                'gross_margin_ttm': self.calculate_gross_margin_ttm(inc, analysis_date),
                'operating_margin_ttm': self.calculate_operating_margin_ttm(inc, analysis_date),
                'free_cash_flow_margin_ttm': self.calculate_free_cash_flow_margin_ttm(cf, inc, analysis_date),
                'sector_relative_growth_ttm': self.calculate_sector_relative_growth_ttm(ticker, sector_etf, analysis_date),
                'current_ratio': self.calculate_current_ratio(bal, analysis_date),
                'debt_to_equity': self.calculate_debt_to_equity(bal, analysis_date)
            }
            
            result = {
                "analyst": "fundamentals",
                "metrics": metrics
            }
            
            if debug:
                # Get enhanced data information for accurate reporting
                latest_income_date = inc.get('quarterlyReports', [{}])[0].get('report_date') if inc.get('quarterlyReports') else None
                latest_balance_date = bal.get('quarterlyReports', [{}])[0].get('report_date') if bal.get('quarterlyReports') else None
                latest_cash_date = cf.get('quarterlyReports', [{}])[0].get('report_date') if cf.get('quarterlyReports') else None
                
                # Check actual data usage for growth calculations (requires 8 quarters)
                financials_8q = self.dm.get_financials(ticker, analysis_date, 8)
                growth_quarters_available = 0
                if financials_8q:
                    growth_reports = self._get_quarterly_reports_by_date(
                        financials_8q['income_statement'].get('quarterlyReports', []), 
                        analysis_date, 8
                    )
                    growth_quarters_available = len(growth_reports)
                
                # Check if sector ETF price data is available (using previous trading day for backtesting accuracy)
                from datetime import datetime, timedelta
                analysis_dt = datetime.strptime(analysis_date, '%Y-%m-%d')
                end_date = (analysis_dt - timedelta(days=1)).strftime('%Y-%m-%d')
                end_date = end_date + ' 23:59:59'  # Ensure BETWEEN query includes end date
                
                sector_prices = self.dm.get_prices(sector_etf, '2020-01-01', end_date)
                sector_data_available = not sector_prices.empty
                
                result["data_used"] = {
                    "summary": {
                        "ticker": ticker,
                        "analysis_date": analysis_date,
                        "sector": sector_info.get('sector', 'Unknown'),
                        "sector_etf": sector_info.get('sector_etf', 'SPY'),
                        "calculation_method": "TTM (Trailing Twelve Month)",
                        "data_sources": {
                            "income_statement": {
                                "latest_report": latest_income_date,
                                "reports_used_standard": len(inc.get('quarterlyReports', [])),
                                "reports_used_growth_calc": growth_quarters_available,
                                "date_range": f"{inc.get('quarterlyReports', [{}])[-1].get('report_date', 'N/A')} to {latest_income_date}" if inc.get('quarterlyReports') else "N/A"
                            },
                            "balance_sheet": {
                                "latest_report": latest_balance_date,
                                "reports_used": len(bal.get('quarterlyReports', [])),
                                "date_range": f"{bal.get('quarterlyReports', [{}])[-1].get('report_date', 'N/A')} to {latest_balance_date}" if bal.get('quarterlyReports') else "N/A"
                            },
                            "cash_flow": {
                                "latest_report": latest_cash_date,
                                "reports_used": len(cf.get('quarterlyReports', [])),
                                "date_range": f"{cf.get('quarterlyReports', [{}])[-1].get('report_date', 'N/A')} to {latest_cash_date}" if cf.get('quarterlyReports') else "N/A"
                            }
                        },
                        "calculation_notes": {
                            "revenue_growth_ttm": f"Uses {growth_quarters_available} quarters (needs 8 for full TTM comparison)",
                            "standard_metrics": "Use 4 most recent quarters for TTM calculations",
                            "growth_method": "8-quarter comparison when available, fallback logic otherwise"
                        },
                        "sector_data_available": sector_data_available
                    }
                }
            
            logger.info(f"Completed TTM fundamental analysis for {ticker}")
            return result
            
        except Exception as e:
            logger.error(f"Error in TTM fundamental analysis for {ticker}: {e}")
            return {
                "analyst": "fundamentals",
                "metrics": default_metrics
            }