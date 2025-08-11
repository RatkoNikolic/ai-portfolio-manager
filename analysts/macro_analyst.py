#!/usr/bin/env python3
"""
Macroeconomic Analyst for AI Portfolio Manager
Provides context using economy-wide indicators from precomputed FRED data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from typing import Dict, Optional
import time
import os

from data_manager import DataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MacroAnalyst:
    """Provides macroeconomic context using precomputed FRED indicators."""
    
    def __init__(self, data_manager: DataManager):
        """Initialize the Macro Analyst."""
        self.dm = data_manager
        logger.info("Macro Analyst initialized")
    
    def get_gdp_growth_pct(self, macro_data: pd.DataFrame, analysis_date: str) -> float:
        """Get GDP growth rate for the quarter containing or immediately before the analysis_date."""
        try:
            if macro_data.empty or 'gdp_growth' not in macro_data.columns:
                logger.warning("GDP growth data not available")
                return 0.0
            
            gdp_series = macro_data['gdp_growth'].dropna()
            if len(gdp_series) == 0:
                logger.warning("No valid GDP growth data points")
                return 0.0
            
            # Use explicit date alignment instead of iloc[-1]
            analysis_dt = pd.to_datetime(analysis_date)
            date_idx = gdp_series.index.asof(analysis_dt)
            
            if pd.isna(date_idx):
                logger.warning("No GDP growth data found within lookback window")
                return 0.0
                
            latest_value = gdp_series.loc[date_idx]
            logger.info(f"GDP Growth Pct: {latest_value}")
            return round(float(latest_value), 2)
            
        except Exception as e:
            logger.error(f"Error getting GDP growth: {e}")
            return 0.0
    
    def get_inflation_yoy_pct(self, macro_data: pd.DataFrame, analysis_date: str) -> float:
        """Get year-over-year inflation rate for the month containing the analysis_date."""
        try:
            if macro_data.empty or 'inflation_yoy' not in macro_data.columns:
                logger.warning("Inflation YoY data not available")
                return 0.0
            
            inflation_series = macro_data['inflation_yoy'].dropna()
            if len(inflation_series) == 0:
                logger.warning("No valid inflation YoY data points")
                return 0.0
            
            # Use explicit date alignment instead of iloc[-1]
            analysis_dt = pd.to_datetime(analysis_date)
            date_idx = inflation_series.index.asof(analysis_dt)
            
            if pd.isna(date_idx):
                logger.warning("No inflation YoY data found within lookback window")
                return 0.0
                
            latest_value = inflation_series.loc[date_idx]
            logger.info(f"Inflation YoY Pct: {latest_value}")
            return round(float(latest_value), 2)
            
        except Exception as e:
            logger.error(f"Error getting inflation YoY: {e}")
            return 0.0
    
    def get_unemployment_rate_pct(self, macro_data: pd.DataFrame, analysis_date: str) -> float:
        """Get unemployment rate for the last available month on or before the analysis_date."""
        try:
            if macro_data.empty or 'unemployment_rate' not in macro_data.columns:
                logger.warning("Unemployment rate data not available")
                return 0.0
            
            unemployment_series = macro_data['unemployment_rate'].dropna()
            if len(unemployment_series) == 0:
                logger.warning("No valid unemployment rate data points")
                return 0.0
            
            # Use explicit date alignment instead of iloc[-1]
            analysis_dt = pd.to_datetime(analysis_date)
            date_idx = unemployment_series.index.asof(analysis_dt)
            
            if pd.isna(date_idx):
                logger.warning("No unemployment rate data found within lookback window")
                return 0.0
                
            latest_value = unemployment_series.loc[date_idx]
            logger.info(f"Unemployment Rate Pct: {latest_value}%")
            return round(float(latest_value), 2)
            
        except Exception as e:
            logger.error(f"Error getting unemployment rate: {e}")
            return 0.0
    
    def get_fed_funds_rate_pct(self, macro_data: pd.DataFrame, analysis_date: str) -> float:
        """Get effective federal funds rate for the last available business-day rate on or before the analysis_date."""
        try:
            if macro_data.empty or 'federal_funds_rate' not in macro_data.columns:
                logger.warning("Federal funds rate data not available")
                return 0.0
            
            fed_series = macro_data['federal_funds_rate'].dropna()
            if len(fed_series) == 0:
                logger.warning("No valid federal funds rate data points")
                return 0.0
            
            # Use explicit date alignment instead of iloc[-1]
            analysis_dt = pd.to_datetime(analysis_date)
            date_idx = fed_series.index.asof(analysis_dt)
            
            if pd.isna(date_idx):
                logger.warning("No federal funds rate data found within lookback window")
                return 0.0
                
            latest_value = fed_series.loc[date_idx]
            logger.info(f"Fed Funds Rate Pct: {latest_value}%")
            return round(float(latest_value), 2)
            
        except Exception as e:
            logger.error(f"Error getting federal funds rate: {e}")
            return 0.0
    
    def get_yield_spread_pct(self, macro_data: pd.DataFrame, analysis_date: str) -> float:
        """Get 10-year Treasury yield minus 3-month Treasury yield for the last business day on or before the analysis_date."""
        try:
            if macro_data.empty or 'yield_curve_spread' not in macro_data.columns:
                logger.warning("Yield curve spread data not available")
                return 0.0
            
            spread_series = macro_data['yield_curve_spread'].dropna()
            if len(spread_series) == 0:
                logger.warning("No valid yield curve spread data points")
                return 0.0
            
            # Use explicit date alignment instead of iloc[-1]
            analysis_dt = pd.to_datetime(analysis_date)
            date_idx = spread_series.index.asof(analysis_dt)
            
            if pd.isna(date_idx):
                logger.warning("No yield curve spread data found within lookback window")
                return 0.0
                
            latest_value = spread_series.loc[date_idx]
            logger.info(f"Yield Spread Pct: {latest_value}")
            return round(float(latest_value), 2)
            
        except Exception as e:
            logger.error(f"Error getting yield curve spread: {e}")
            return 0.0
    

    
    def get_credit_spread_bp(self, macro_data: pd.DataFrame, analysis_date: str) -> float:
        """Get credit spread in basis points: BAA corporate yield minus 10-year Treasury yield."""
        try:
            if macro_data.empty:
                logger.warning("Macro data not available for credit spread calculation")
                return 0.0
            
            # Check for required columns
            if 'baa_corporate_yield' not in macro_data.columns or 'yield_10y' not in macro_data.columns:
                logger.warning("BAA corporate yield or 10Y Treasury data not available for credit spread")
                return 0.0
            
            baa_series = macro_data['baa_corporate_yield'].dropna()
            treasury_series = macro_data['yield_10y'].dropna()
            
            if len(baa_series) == 0 or len(treasury_series) == 0:
                logger.warning("No valid BAA corporate or Treasury yield data points")
                return 0.0
            
            # Use explicit date alignment
            analysis_dt = pd.to_datetime(analysis_date)
            baa_date_idx = baa_series.index.asof(analysis_dt)
            treasury_date_idx = treasury_series.index.asof(analysis_dt)
            
            if pd.isna(baa_date_idx) or pd.isna(treasury_date_idx):
                logger.warning("No BAA corporate or Treasury yield data found within lookback window")
                return 0.0
                
            baa_yield = baa_series.loc[baa_date_idx]
            treasury_yield = treasury_series.loc[treasury_date_idx]
            
            # Calculate spread in basis points (multiply by 100)
            credit_spread_bp = (baa_yield - treasury_yield) * 100
            logger.info(f"Credit Spread BP: {credit_spread_bp}")
            return round(float(credit_spread_bp), 2)
            
        except Exception as e:
            logger.error(f"Error getting credit spread: {e}")
            return 0.0
    
    def get_vix_index(self, macro_data: pd.DataFrame, analysis_date: str) -> float:
        """Get VIX (Volatility Index) for the business day containing or before analysis_date."""
        try:
            if macro_data.empty or 'vix' not in macro_data.columns:
                logger.warning("VIX data not available")
                return 0.0
            
            vix_series = macro_data['vix'].dropna()
            if len(vix_series) == 0:
                logger.warning("No valid VIX data points")
                return 0.0
            
            # Use explicit date alignment
            analysis_dt = pd.to_datetime(analysis_date)
            date_idx = vix_series.index.asof(analysis_dt)
            
            if pd.isna(date_idx):
                logger.warning("No VIX data found within lookback window")
                return 0.0
                
            latest_value = vix_series.loc[date_idx]
            logger.info(f"VIX Index: {latest_value}")
            return round(float(latest_value), 2)
            
        except Exception as e:
            logger.error(f"Error getting VIX index: {e}")
            return 0.0
    
    def analyze(self, analysis_date: str, debug: bool = False) -> Dict:
        """
        Perform macroeconomic analysis for a given date using precomputed FRED indicators.
        
        Args:
            analysis_date: Analysis date in YYYY-MM-DD format
            debug: Whether to include debug information
            
        Returns:
            Dict containing macroeconomic analysis results
        """
        default_metrics = {
            "gdp_growth_pct": 0.0,
            "inflation_yoy_pct": 0.0,
            "unemployment_rate_pct": 0.0,
            "fed_funds_rate_pct": 0.0,
            "yield_spread_pct": 0.0,
            "credit_spread_bp": 0.0,
            "vix_index": 0.0
        }
        
        try:
            logger.info(f"Starting macroeconomic analysis for {analysis_date}")
            
            # Get macro data
            start_date = '2020-01-01'  # Fixed start date as per requirements
            macro_data = self.dm.get_macro_data(start_date, analysis_date)
            
            if macro_data.empty:
                logger.warning("No macro data available, returning zeros")
                return {
                    "analyst": "macro",
                    "metrics": default_metrics
                }
            
            # Calculate macro metrics using explicit date alignment for backtesting
            metrics = {
                'gdp_growth_pct': self.get_gdp_growth_pct(macro_data, analysis_date),
                'inflation_yoy_pct': self.get_inflation_yoy_pct(macro_data, analysis_date),
                'unemployment_rate_pct': self.get_unemployment_rate_pct(macro_data, analysis_date),
                'fed_funds_rate_pct': self.get_fed_funds_rate_pct(macro_data, analysis_date),
                'yield_spread_pct': self.get_yield_spread_pct(macro_data, analysis_date),
                'credit_spread_bp': self.get_credit_spread_bp(macro_data, analysis_date),
                'vix_index': self.get_vix_index(macro_data, analysis_date)
            }
            
            result = {
                "analyst": "macro",
                "metrics": metrics
            }
            
            if debug:
                # Provide detailed information about data sources and methodology
                result["data_used"] = {
                    "summary": {
                        "analysis_date": analysis_date,
                        "data_methodology": "Uses precomputed FRED indicators directly",
                        "data_sources": {
                            "macro_data": {
                                "date_range": f"{start_date} to {analysis_date}",
                                "data_points": len(macro_data),
                                "latest_date": macro_data.index[-1].strftime('%Y-%m-%d') if not macro_data.empty else None,
                                "earliest_date": macro_data.index[0].strftime('%Y-%m-%d') if not macro_data.empty else None,
                                "indicators_available": list(macro_data.columns) if not macro_data.empty else []
                            }
                        },
                        "fred_series_mapping": {
                            "gdp_growth_pct": "A191RL1Q225SBEA (Quarterly GDP growth, annualized, seasonally adjusted)",
                            "inflation_yoy_pct": "CPIAUCSL (Consumer Price Index, YoY calculated)",
                            "unemployment_rate_pct": "UNRATE (Unemployment rate, monthly)",
                            "fed_funds_rate_pct": "FEDFUNDS (Federal funds rate, monthly)",
                            "yield_spread_pct": "GS10 - GS3M (10Y minus 3M Treasury rates)",
                            "credit_spread_bp": "BAA - GS10 (BAA corporate yield minus 10Y Treasury, in basis points)",
                            "vix_index": "VIXCLS (CBOE Volatility Index)"
                        },
                        "calculation_notes": {
                            "method": "Explicit date alignment using pandas asof for backtesting accuracy",
                            "reference_date": analysis_date,
                            "data_alignment": "Uses asof method to get values on or before analysis_date within 30-day window",
                            "minimal_set": "Focuses on 7 core metrics: growth, price, labor, policy, curve, credit, and volatility",
                            "unit_consistency": "All percentages rounded to 2 decimal places, credit spread in basis points",
                            "credit_risk": "Credit spread calculated as BAA corporate yield minus 10Y Treasury in basis points",
                            "market_volatility": "VIX index represents market-implied volatility expectations"
                        }
                    }
                }
            
            logger.info(f"Completed macroeconomic analysis for {analysis_date}")
            return result
            
        except Exception as e:
            logger.error(f"Error in macroeconomic analysis: {e}")
            return {
                "analyst": "macro",
                "metrics": default_metrics
            }

