#!/usr/bin/env python3
"""
Technical Analyst for AI Portfolio Manager
Analyzes recent price trends and momentum.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from typing import Dict, Optional
import ta
import time
import os

from data_manager import DataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalAnalyst:
    """Analyzes price trends and momentum."""
    
    def __init__(self, data_manager: DataManager):
        """Initialize the Technical Analyst."""
        self.dm = data_manager
        logger.info("Technical Analyst initialized")
    
    def calculate_momentum_12m(self, prices: pd.DataFrame) -> float:
        """Calculate 12-month momentum."""
        try:
            if len(prices) < 252:  # Need at least 252 trading days
                return 0.0
            
            current_price = prices['close'].iloc[-1]
            price_252_days_ago = prices['close'].iloc[-252]
            
            if price_252_days_ago == 0:
                return 0.0
            
            momentum = (current_price - price_252_days_ago) / price_252_days_ago
            return round(momentum, 4)
            
        except Exception as e:
            logger.error(f"Error calculating 12-month momentum: {e}")
            return 0.0
    
    def calculate_ma_ratio(self, prices: pd.DataFrame) -> float:
        """Calculate 50D/200D MA ratio."""
        try:
            if len(prices) < 200:
                return 1.0
            # Ensure prices are sorted by date ascending
            prices = prices.sort_index()
            # Calculate moving averages
            ma_50 = prices['close'].rolling(window=50).mean()
            ma_200 = prices['close'].rolling(window=200).mean()
            # Calculate actual ratio
            current_ma_50 = ma_50.iloc[-1]
            current_ma_200 = ma_200.iloc[-1]
            
            if current_ma_200 == 0:
                return 1.0
                
            ratio = current_ma_50 / current_ma_200
            logger.info(f"MA Ratio Debug: MA50={current_ma_50:.2f}, MA200={current_ma_200:.2f}, Ratio={ratio:.4f}")
            return round(ratio, 4)
        except Exception as e:
            logger.error(f"Error calculating MA ratio: {e}")
            return 1.0
    
    def calculate_rsi(self, prices: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI."""
        try:
            if len(prices) < period:
                return 50.0  # Neutral RSI
            
            # Calculate RSI using ta library
            rsi = ta.momentum.RSIIndicator(prices['close'], window=period)
            rsi_value = rsi.rsi().iloc[-1]
            
            if pd.isna(rsi_value):
                return 50.0
            
            return round(rsi_value, 2)
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def calculate_macd(self, prices: pd.DataFrame) -> float:
        """Calculate MACD."""
        try:
            if len(prices) < 26:
                return 0.0
            
            # Calculate MACD using ta library
            macd = ta.trend.MACD(prices['close'])
            macd_value = macd.macd().iloc[-1]
            
            if pd.isna(macd_value):
                return 0.0
            
            return round(macd_value, 4)
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return 0.0
    
    def calculate_volume_trend(self, prices: pd.DataFrame) -> str:
        """Calculate volume trend."""
        try:
            if len(prices) < 90:
                return "neutral"
            
            # Calculate average volumes
            avg_volume_30d = prices['volume'].tail(30).mean()
            avg_volume_90d = prices['volume'].tail(90).mean()
            
            if avg_volume_90d == 0:
                return "neutral"
            
            volume_ratio = avg_volume_30d / avg_volume_90d
            
            if volume_ratio > 1.1:
                return "up"
            elif volume_ratio < 0.9:
                return "down"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error calculating volume trend: {e}")
            return "neutral"
    
    def calculate_bollinger_band_width(self, prices: pd.DataFrame, period: int = 20) -> float:
        """Calculate Bollinger Band width."""
        try:
            if len(prices) < period:
                return 0.0
            
            # Calculate Bollinger Bands using ta library
            bb = ta.volatility.BollingerBands(prices['close'], window=period)
            
            upper_band = bb.bollinger_hband().iloc[-1]
            lower_band = bb.bollinger_lband().iloc[-1]
            middle_band = bb.bollinger_mavg().iloc[-1]
            
            if pd.isna(upper_band) or pd.isna(lower_band) or pd.isna(middle_band) or middle_band == 0:
                return 0.0
            
            # Calculate width as percentage of middle band
            width = (upper_band - lower_band) / middle_band
            return round(width, 4)
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Band width: {e}")
            return 0.0
    
    def calculate_sector_relative_strength(self, company_prices, sector_prices, analysis_date):
        try:
            if company_prices.empty or sector_prices.empty:
                return 0.0
            company_price = company_prices.loc[analysis_date]['close'] if analysis_date in company_prices.index else company_prices['close'].iloc[-1]
            sector_price = sector_prices.loc[analysis_date]['close'] if analysis_date in sector_prices.index else sector_prices['close'].iloc[-1]
            if sector_price == 0:
                return 0.0
            return round(company_price / sector_price, 4)
        except Exception as e:
            logger.error(f"Error calculating sector relative strength: {e}")
            return 0.0

    def calculate_sector_relative_momentum(self, company_prices, sector_prices, analysis_date):
        try:
            if company_prices.empty or sector_prices.empty:
                return 0.0
            # 12M return: (P_t - P_t-252) / P_t-252
            window = 252
            if len(company_prices) < window + 1 or len(sector_prices) < window + 1:
                return 0.0
            company_return = (company_prices['close'].iloc[-1] - company_prices['close'].iloc[-window-1]) / company_prices['close'].iloc[-window-1]
            sector_return = (sector_prices['close'].iloc[-1] - sector_prices['close'].iloc[-window-1]) / sector_prices['close'].iloc[-window-1]
            return round(company_return - sector_return, 4)
        except Exception as e:
            logger.error(f"Error calculating sector relative momentum: {e}")
            return 0.0

    def calculate_rolling_beta_sector(self, company_prices, sector_prices):
        try:
            if company_prices.empty or sector_prices.empty:
                return 0.0
            # Use daily returns, 1-year window
            window = 252
            if len(company_prices) < window + 1 or len(sector_prices) < window + 1:
                return 0.0
            company_returns = company_prices['close'].pct_change().dropna().iloc[-window:]
            sector_returns = sector_prices['close'].pct_change().dropna().iloc[-window:]
            if len(company_returns) != len(sector_returns):
                min_len = min(len(company_returns), len(sector_returns))
                company_returns = company_returns[-min_len:]
                sector_returns = sector_returns[-min_len:]
            cov = np.cov(company_returns, sector_returns)[0][1]
            var = np.var(sector_returns)
            if var == 0:
                return 0.0
            return round(cov / var, 4)
        except Exception as e:
            logger.error(f"Error calculating rolling beta to sector: {e}")
            return 0.0

    def analyze(self, ticker: str, analysis_date: str, debug: bool = False) -> Dict:
        """
        Perform technical analysis for a given ticker and date.
        
        Args:
            ticker: Stock ticker symbol
            analysis_date: Analysis date in YYYY-MM-DD format
            
        Returns:
            Dict containing technical analysis results
        """
        try:
            logger.info(f"Starting technical analysis for {ticker}")
            
            # Get price data up to the day before analysis date (backtesting accuracy)
            from datetime import datetime, timedelta
            analysis_dt = datetime.strptime(analysis_date, '%Y-%m-%d')
            end_date = (analysis_dt - timedelta(days=1)).strftime('%Y-%m-%d')
            end_date = end_date + ' 23:59:59'  # Ensure BETWEEN query includes end date
            
            start_date = '2020-01-01'  # Fixed start date as per requirements
            prices = self.dm.get_prices(ticker, start_date, end_date)
            
            if prices.empty:
                logger.warning(f"No price data available for {ticker}")
                return {
                    "analyst": "technical",
                    "metrics": {
                        "price_momentum_12m": 0.0,
                        "ma_50_200_ratio": 0,
                        "rsi_14d": 50.0,
                        "macd_line": 0.0,
                        "volume_trend_30d": "neutral",
                        "bb_width_pct": 0.0
                    }
                }
            
            # Get sector ETF info and prices (also using previous trading day)
            sector_info = self.dm.get_sector_info(ticker)
            sector_etf = sector_info.get('sector_etf', 'SPY')
            sector_prices = self.dm.get_prices(sector_etf, '2020-01-01', end_date)

            # Calculate technical metrics
            metrics = {
                'price_momentum_12m': self.calculate_momentum_12m(prices),
                'ma_50_200_ratio': self.calculate_ma_ratio(prices),
                'rsi_14d': self.calculate_rsi(prices, period=14),
                'macd_line': self.calculate_macd(prices),
                'volume_trend_30d': self.calculate_volume_trend(prices),
                'bb_width_pct': self.calculate_bollinger_band_width(prices, period=20),
                'price_vs_sector_ratio': self.calculate_sector_relative_strength(prices, sector_prices, analysis_date),
                'momentum_vs_sector': self.calculate_sector_relative_momentum(prices, sector_prices, analysis_date),
                'beta_vs_sector': self.calculate_rolling_beta_sector(prices, sector_prices)
            }
            
            result = {
                "analyst": "technical",
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
                            "price_data": {
                                "date_range": f"{start_date} to {analysis_date}",
                                "data_points": len(prices),
                                "latest_price": prices['close'].iloc[-1] if not prices.empty else None,
                                "earliest_price": prices['close'].iloc[0] if not prices.empty else None
                            },
                            "sector_price_data": {
                                "date_range": f"{start_date} to {analysis_date}",
                                "data_points": len(sector_prices),
                                "latest_price": sector_prices['close'].iloc[-1] if not sector_prices.empty else None,
                                "earliest_price": sector_prices['close'].iloc[0] if not sector_prices.empty else None
                            }
                        },
                        "sector_data_available": not sector_prices.empty
                    }
                }
            
            logger.info(f"Completed technical analysis for {ticker}")
            return result
            
        except Exception as e:
            logger.error(f"Error in technical analysis for {ticker}: {e}")
            return {
                "analyst": "technical",
                "metrics": {
                    "momentum_12m": 0.0,
                    "ma_50_over_200": 0,
                    "rsi_14": 50.0,
                    "macd": 0.0,
                    "volume_trend": "neutral",
                    "bollinger_band_width": 0.0
                }
            }


 