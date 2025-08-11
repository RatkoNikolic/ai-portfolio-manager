#!/usr/bin/env python3
"""
Risk Analyst for AI Portfolio Manager
Quantifies volatility and downside risk.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from typing import Dict, Optional
from scipy import stats
import os

from data_manager import DataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskAnalyst:
    """Quantifies volatility and downside risk."""
    
    def __init__(self, data_manager: DataManager):
        """Initialize the Risk Analyst."""
        self.dm = data_manager
        logger.info("Risk Analyst initialized")
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.Series:
        """Calculate daily returns."""
        try:
            returns = prices['close'].pct_change().dropna()
            return returns
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return pd.Series()
    
    def calculate_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate Beta relative to market."""
        try:
            if len(stock_returns) < 30 or len(market_returns) < 30:
                return 1.0
            
            # Align the series by date
            aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
            
            if len(aligned_data) < 30:
                return 1.0
            
            stock_ret = aligned_data.iloc[:, 0]
            market_ret = aligned_data.iloc[:, 1]
            
            # Calculate covariance and variance
            covariance = np.cov(stock_ret, market_ret)[0, 1]
            market_variance = np.var(market_ret)
            
            if market_variance == 0:
                return 1.0
            
            beta = covariance / market_variance
            return round(beta, 4)
            
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 1.0
    
    def calculate_volatility_30d(self, returns: pd.Series) -> float:
        """Calculate 30-day volatility."""
        try:
            if len(returns) < 30:
                return 0.0
            
            # Calculate 30-day rolling volatility
            volatility_30d = returns.tail(30).std()
            
            # Annualize (multiply by sqrt(252))
            annualized_vol = volatility_30d * np.sqrt(252)
            return round(annualized_vol, 4)
            
        except Exception as e:
            logger.error(f"Error calculating 30-day volatility: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, prices: pd.DataFrame) -> float:
        """Calculate maximum drawdown from peak to trough."""
        try:
            if len(prices) < 2:
                return 0.0
            
            # Ensure prices are sorted by date
            if not prices.index.is_monotonic_increasing:
                prices = prices.sort_index()
            
            # Calculate running maximum (peak)
            running_max = prices['close'].expanding().max()
            
            # Calculate drawdown at each point
            drawdown = (prices['close'] - running_max) / running_max
            
            # Get maximum drawdown (most negative value)
            max_drawdown = drawdown.min()
            
            # Log for debugging
            logger.info(f"Max drawdown calculation: peak={running_max.max():.2f}, trough={prices['close'].loc[drawdown.idxmin()]:.2f}, drawdown={max_drawdown:.4f}")
            
            return round(max_drawdown, 4)
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        try:
            if len(returns) < 30:
                return 0.0
            
            # Calculate annualized return and volatility
            annual_return = returns.mean() * 252
            annual_volatility = returns.std() * np.sqrt(252)
            
            if annual_volatility == 0:
                return 0.0
            
            # Calculate Sharpe ratio
            sharpe = (annual_return - risk_free_rate) / annual_volatility
            return round(sharpe, 4)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        try:
            if len(returns) < 30:
                return 0.0
            
            # Calculate annualized return
            annual_return = returns.mean() * 252
            
            # Calculate downside deviation (only negative returns)
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return 0.0
            
            downside_deviation = downside_returns.std() * np.sqrt(252)
            
            if downside_deviation == 0:
                return 0.0
            
            # Calculate Sortino ratio
            sortino = (annual_return - risk_free_rate) / downside_deviation
            return round(sortino, 4)
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def calculate_cvar_95(self, returns: pd.Series) -> float:
        """Calculate Conditional Value at Risk (CVaR) at 95% confidence."""
        try:
            if len(returns) < 30:
                return 0.0
            
            # Calculate VaR at 95% confidence
            var_95 = np.percentile(returns, 5)
            
            # Calculate CVaR (expected loss beyond VaR)
            cvar_95 = returns[returns <= var_95].mean()
            
            return round(cvar_95, 4)
            
        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return 0.0
    
    def calculate_idiosyncratic_volatility(self, company_vol, beta, sector_vol):
        try:
            return round(company_vol - (beta ** 2 * sector_vol), 6)
        except Exception as e:
            logger.error(f"Error calculating idiosyncratic volatility: {e}")
            return 0.0

    def calculate_drawdown_vs_sector(self, company_drawdown, sector_drawdown):
        try:
            return round(company_drawdown - sector_drawdown, 4)
        except Exception as e:
            logger.error(f"Error calculating drawdown vs sector: {e}")
            return 0.0

    def calculate_tracking_error(self, company_returns, sector_returns):
        """Calculate annualized tracking error between company and sector returns."""
        try:
            if len(company_returns) != len(sector_returns):
                min_len = min(len(company_returns), len(sector_returns))
                company_returns = company_returns[-min_len:]
                sector_returns = sector_returns[-min_len:]
            
            # Calculate difference in returns
            diff = company_returns - sector_returns
            
            # Calculate daily tracking error standard deviation
            daily_te = np.std(diff)
            
            # Annualize by multiplying by sqrt(252) for trading days
            annualized_te = daily_te * np.sqrt(252)
            
            return round(annualized_te, 6)
        except Exception as e:
            logger.error(f"Error calculating tracking error: {e}")
            return 0.0
    
    def get_risk_free_rate(self, analysis_date: str) -> float:
        """Get risk-free rate for the analysis date."""
        try:
            # Get 3-month Treasury yield from macro data
            macro_data = self.dm.get_macro_data('2020-01-01', analysis_date)
            
            if not macro_data.empty:
                # Look for 3-month Treasury yield data
                if 'yield_3m' in macro_data.columns:
                    latest_rate = macro_data['yield_3m'].iloc[-1]
                    if pd.notna(latest_rate):
                        return latest_rate / 100  # Convert to decimal
            
            # Default risk-free rate
            return 0.02
            
        except Exception as e:
            logger.error(f"Error getting risk-free rate: {e}")
            return 0.02
    
    def analyze(self, ticker: str, analysis_date: str, debug: bool = False) -> Dict:
        """
        Perform risk analysis for a given ticker and date.
        
        Args:
            ticker: Stock ticker symbol
            analysis_date: Analysis date in YYYY-MM-DD format
            
        Returns:
            Dict containing risk analysis results
        """
        try:
            logger.info(f"Starting risk analysis for {ticker}")
            
            # Get price data up to the day before analysis date (backtesting accuracy)
            from datetime import datetime, timedelta
            analysis_dt = datetime.strptime(analysis_date, '%Y-%m-%d')
            end_date = (analysis_dt - timedelta(days=1)).strftime('%Y-%m-%d')
            end_date = end_date + ' 23:59:59'  # Ensure BETWEEN query includes end date
            
            start_date = '2020-01-01'  # Fixed start date as per requirements
            stock_prices = self.dm.get_prices(ticker, start_date, end_date)
            market_prices = self.dm.get_market_data('SPY', start_date, end_date)
            
            if stock_prices.empty:
                logger.warning(f"No price data available for {ticker}")
                return {
                    "analyst": "risk",
                    "metrics": {
                        "beta": 1.0,
                        "volatility_30d": 0.0,
                        "max_drawdown": 0.0,
                        "sharpe_ratio": 0.0,
                        "sortino_ratio": 0.0,
                        "cvar_95": 0.0
                    }
                }
            
            # Calculate returns
            stock_returns = self.calculate_returns(stock_prices)
            market_returns = self.calculate_returns(market_prices)
            
            # Get risk-free rate
            risk_free_rate = self.get_risk_free_rate(analysis_date)
            
            # Get sector ETF info and prices (also using previous trading day)
            sector_info = self.dm.get_sector_info(ticker)
            sector_etf = sector_info.get('sector_etf', 'SPY')
            sector_prices = self.dm.get_prices(sector_etf, '2020-01-01', end_date)
            
            # Calculate sector ETF volatility and drawdown
            sector_vol = 0.0
            sector_drawdown = 0.0
            sector_returns = None
            if not sector_prices.empty:
                sector_returns = sector_prices['close'].pct_change().dropna().iloc[-30:]
                sector_vol = np.std(sector_returns)
                roll_max = sector_prices['close'].cummax()
                drawdown = sector_prices['close'] / roll_max - 1.0
                sector_drawdown = drawdown.min()
            
            # Calculate basic risk metrics first
            beta = self.calculate_beta(stock_returns, market_returns)
            volatility_30d = self.calculate_volatility_30d(stock_returns)
            max_drawdown = self.calculate_max_drawdown(stock_prices)
            
            # Calculate risk metrics
            metrics = {
                'market_beta': beta,
                'volatility_30d_pct': volatility_30d,
                'max_drawdown_pct': max_drawdown,
                'sharpe_ratio': self.calculate_sharpe_ratio(stock_returns, risk_free_rate),
                'sortino_ratio': self.calculate_sortino_ratio(stock_returns, risk_free_rate),
                'cvar_95_pct': self.calculate_cvar_95(stock_returns),
                'idiosyncratic_vol_pct': self.calculate_idiosyncratic_volatility(volatility_30d, beta, sector_vol),
                'drawdown_vs_sector_pct': self.calculate_drawdown_vs_sector(max_drawdown, sector_drawdown),
                'tracking_error_pct': self.calculate_tracking_error(stock_returns if not stock_returns.empty else None, sector_returns) if stock_returns is not None and sector_returns is not None else 0.0
            }
            
            result = {
                "analyst": "risk",
                "metrics": metrics
            }
            
            if debug:
                # Include a concise summary of data sources and date ranges
                result["data_used"] = {
                    "summary": {
                        "ticker": ticker,
                        "analysis_date": analysis_date,
                        "sector": sector_info.get('sector', 'Unknown'),
                        "sector_etf": sector_etf,
                        "data_sources": {
                            "stock_price_data": {
                                "date_range": f"{start_date} to {analysis_date}",
                                "data_points": len(stock_prices),
                                "latest_price": stock_prices['close'].iloc[-1] if not stock_prices.empty else None,
                                "earliest_price": stock_prices['close'].iloc[0] if not stock_prices.empty else None
                            },
                            "market_price_data": {
                                "date_range": f"{start_date} to {analysis_date}",
                                "data_points": len(market_prices),
                                "latest_price": market_prices['close'].iloc[-1] if not market_prices.empty else None,
                                "earliest_price": market_prices['close'].iloc[0] if not market_prices.empty else None
                            },
                            "sector_price_data": {
                                "date_range": f"{start_date} to {analysis_date}",
                                "data_points": len(sector_prices),
                                "latest_price": sector_prices['close'].iloc[-1] if not sector_prices.empty else None,
                                "earliest_price": sector_prices['close'].iloc[0] if not sector_prices.empty else None
                            }
                        },
                        "risk_free_rate": risk_free_rate,
                        "sector_data_available": not sector_prices.empty
                    }
            }
            
            logger.info(f"Completed risk analysis for {ticker}")
            return result
            
        except Exception as e:
            logger.error(f"Error in risk analysis for {ticker}: {e}")
            return {
                "analyst": "risk",
                "metrics": {
                    "beta": 1.0,
                    "volatility_30d": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "cvar_95": 0.0
                }
            }


 