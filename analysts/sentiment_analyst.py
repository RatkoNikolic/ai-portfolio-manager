#!/usr/bin/env python3
"""
Sentiment Analyst for AI Portfolio Manager
Uses LLM to summarize news coverage and infer sentiment.
"""

import json
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import openai
import requests
import pandas as pd
from pathlib import Path

from data_manager import DataManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyst:
    """Uses LLM to analyze news sentiment."""
    
    def __init__(self, data_manager: DataManager):
        """Initialize the Sentiment Analyst."""
        self.dm = data_manager
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            self.client = None
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        
        # Load prompt from file
        prompt_path = os.path.join(os.path.dirname(__file__), '../prompts/sentiment_analyst_prompt.txt')
        if not os.path.exists(prompt_path):
            prompt_path = os.path.join('prompts', 'sentiment_analyst_prompt.txt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()
        logger.info("Sentiment Analyst initialized")

    def _get_sentiment_from_openai_websearch(self, company: str, ticker: str, date: str) -> Optional[str]:
        """Get sentiment analysis from OpenAI web search."""
        if not self.client:
            return None
            
        user_prompt = f"""
        {{
          "company": "{company}",
          "ticker": "{ticker}",
          "date": "{date}"
        }}
        """
        try:
            response = self.client.responses.create(
                model="gpt-4o",
                input=[
                    {
                        "role": "system",
                        "content": [
                            {"type": "input_text", "text": self.system_prompt}
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_prompt}
                        ]
                    },
                ],
                text={"format": {"type": "text"}},
                reasoning={},
                tools=[
                    {
                        "type": "web_search_preview",
                        "user_location": {"type": "approximate", "timezone": "America/New_York"},
                        "search_context_size": "high"
                    }
                ],
                tool_choice={"type": "web_search_preview"},
                temperature=0,
                max_output_tokens=2048,
                top_p=1
            )
            return response.output[1].content[0].text
        except Exception as e:
            logger.error(f"Error fetching sentiment from OpenAI: {e}")
            return None

    def _parse_sentiment_response(self, response: str) -> tuple[Optional[float], Optional[str]]:
        """Parse sentiment response from OpenAI."""
        if not response:
            return 0.0, "No response from OpenAI."
            
        try:
            response_dict = json.loads(response)
            sentiment = response_dict.get('sentiment')
            explanation = response_dict.get('explanation')
            
            # Replace null sentiment with 0 when no relevant news is found
            if sentiment is None:
                sentiment = 0.0
                logger.info("Sentiment was null, setting to 0 (no relevant news)")
                
            return sentiment, explanation
            
        except Exception as e:
            logger.error(f"Error parsing OpenAI response: {e}")
            return 0.0, f"Error parsing response: {e}"

    def _should_generate_new_sentiment(self, analysis_date: str) -> bool:
        """Determine if we should generate new sentiment based on date."""
        analysis_dt = datetime.strptime(analysis_date, '%Y-%m-%d')
        current_date = datetime.now()
        days_diff = (current_date - analysis_dt).days
        
        # Generate new sentiment for:
        # 1. Future dates (more than 30 days ahead)
        # 2. Recent dates (within 30 days of current date)
        # 3. Dates that are in 2024 or later (since our database only has data up to 2023)
        should_generate = days_diff < -30 or days_diff < 30 or analysis_dt.year >= 2024
        logger.info(f"Date check: analysis_date={analysis_date}, current_date={current_date.strftime('%Y-%m-%d')}, days_diff={days_diff}, year={analysis_dt.year}, should_generate={should_generate}")
        return should_generate

    def _save_sentiment_to_db(self, ticker: str, analysis_date: str, sentiment: float, explanation: str) -> bool:
        """Save sentiment data to database."""
        try:
            conn = self.dm.connection
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO sentiment_monthly (ticker, date, sentiment, explanation) VALUES (?, ?, ?, ?)",
                (ticker, analysis_date, sentiment, explanation)
            )
            conn.commit()
            logger.info(f"Sentiment data saved to DB for {ticker} on {analysis_date}")
            return True
        except Exception as e:
            logger.error(f"Error saving sentiment to DB: {e}")
            return False

    def _save_debug_output(self, ticker: str, analysis_date: str, sentiment: float, explanation: str) -> None:
        """Save debug output to test file."""
        try:
            os.makedirs('test', exist_ok=True)
            fname = f"test/sentiment_{ticker}_{analysis_date.replace('-', '')}_debug.json"
            with open(fname, 'w') as f:
                json.dump({
                    'ticker': ticker,
                    'date': analysis_date,
                    'sentiment': sentiment,
                    'explanation': explanation
                }, f, indent=2)
            logger.info(f"Sentiment debug output saved to {fname}")
        except Exception as e:
            logger.error(f"Error saving debug output: {e}")

    def analyze(self, ticker: str, analysis_date: str, debug: bool = False) -> Dict:
        """
        Perform sentiment analysis for a given ticker and date.
        
        Args:
            ticker: Stock ticker symbol
            analysis_date: Analysis date in YYYY-MM-DD format
            debug: Whether to include debug information
            
        Returns:
            Dict containing sentiment analysis results
        """
        try:
            logger.info(f"Starting sentiment analysis for {ticker} on {analysis_date}")
            
            # Check if we should generate new sentiment
            if self._should_generate_new_sentiment(analysis_date):
                logger.info(f"Analysis date {analysis_date} requires new sentiment generation")
                sentiment_data = None
            else:
                # Check if sentiment data exists in DB for historical dates
                sentiment_data = self.dm.get_sentiment_data(ticker, analysis_date)
                if sentiment_data and sentiment_data.get('sentiment') is not None:
                    logger.info(f"Sentiment data found in DB for {ticker} on {analysis_date}")
                    
                    # Replace null sentiment with 0 when retrieved from database
                    sentiment_score = sentiment_data['sentiment']
                    if sentiment_score is None:
                        sentiment_score = 0.0
                        logger.info(f"Sentiment was null in DB for {ticker} on {analysis_date}, setting to 0 (no relevant news)")
                    
                    result = {
                        "analyst": "sentiment",
                        "metrics": {
                            "sentiment_score": sentiment_score,
                            "sentiment_explanation": sentiment_data['explanation']
                        }
                    }
                    
                    if debug:
                        result["data_used"] = {
                            "summary": {
                                "ticker": ticker,
                                "analysis_date": analysis_date,
                                "data_sources": {
                                    "sentiment_data": {
                                        "source": "database",
                                        "sentiment_score": sentiment_score,
                                        "database_available": True
                                    }
                                }
                            }
                        }
                    return result
            
            # Generate new sentiment using OpenAI
            sector_info = self.dm.get_sector_info(ticker)
            company = sector_info.get('name', ticker)
            
            response = self._get_sentiment_from_openai_websearch(company, ticker, analysis_date)
            sentiment, explanation = self._parse_sentiment_response(response)
            
            # Save result
            if debug:
                self._save_debug_output(ticker, analysis_date, sentiment, explanation)
            else:
                self._save_sentiment_to_db(ticker, analysis_date, sentiment, explanation)
            
            result = {
                "analyst": "sentiment",
                "metrics": {
                    "sentiment_score": sentiment,
                    "sentiment_explanation": explanation
                }
            }
            
            if debug:
                result["data_used"] = {
                    "summary": {
                        "ticker": ticker,
                        "analysis_date": analysis_date,
                        "sector": sector_info.get('sector', 'Unknown'),
                        "company_name": company,
                        "data_sources": {
                            "sentiment_data": {
                                "source": "openai_web_search",
                                "sentiment_score": sentiment,
                                "database_available": False,
                                "openai_api_available": self.client is not None
                            }
                        }
                    }
                }
            
            logger.info(f"Completed sentiment analysis for {ticker}")
            return result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis for {ticker}: {e}")
            return {
                "analyst": "sentiment",
                "metrics": {
                    "sentiment_score": 0.0,
                    "sentiment_explanation": f"Error in sentiment analysis: {str(e)}"
                }
            }


 