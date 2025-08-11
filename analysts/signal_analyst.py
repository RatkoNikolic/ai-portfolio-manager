#!/usr/bin/env python3
"""
Signal Analyst for AI Portfolio Manager
Generates trading signals from analysis outputs using LLM.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

SIGNAL_ANALYST_PROMPT_PATH = os.path.join(os.path.dirname(__file__), '../prompts/signal_analyst_prompt.txt')

class SignalAnalyst:
    """Signal Analyst that generates trading signals from analysis outputs."""
    
    def __init__(self, openai_api_key: str = None, model: str = "gpt-4o"):
        """
        Initialize the Signal Analyst.
        
        Args:
            openai_api_key: OpenAI API key (if None, will try to get from environment)
            model: OpenAI model to use for signal generation
        """
        # Validate model supports structured output
        supported_models = ["gpt-4o", "gpt-4o-mini"]
        if model not in supported_models:
            logger.warning(f"Model {model} does not support structured output. Using gpt-4o instead.")
            model = "gpt-4o"
        
        self.model = model
        self.openai_client = None
        self._initialize_openai(openai_api_key)
        logger.info(f"Signal Analyst initialized with model: {model}")
    
    def _initialize_openai(self, api_key: str = None):
        """Initialize OpenAI client with API key."""
        try:
            if not api_key:
                api_key = os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables or parameters")
            
            self.openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            raise
    
    def get_system_prompt(self) -> str:
        """Load the system prompt for the Signal Analyst from file."""
        with open(SIGNAL_ANALYST_PROMPT_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    
    def analyze(self, analysis_output: Dict) -> Optional[Dict]:
        """
        Generate a trading signal from analysis output.
        
        Args:
            analysis_output: Analysis output dictionary from other analysts
            
        Returns:
            Dict containing the signal or None if error
        """
        try:
            # Convert analysis output to JSON string
            analysis_json = json.dumps(analysis_output, indent=2)
            
            # Generate signal using OpenAI
            system_prompt = self.get_system_prompt()
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": analysis_json
                    }
                ],
                temperature=0,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            signal_result = json.loads(response.choices[0].message.content)
            
            # Reorder fields to ensure closing_price appears in correct position
            # Extract closing_price from analysis_output (it might be null)
            closing_price = analysis_output.get('closing_price')
            
            # Create properly ordered signal result
            ordered_signal_result = {
                "ticker": signal_result.get('ticker', analysis_output.get('ticker', 'Unknown')),
                "analysis_date": signal_result.get('analysis_date', analysis_output.get('analysis_date', 'Unknown')),
                "closing_price": closing_price,
                "signal": signal_result.get('signal', 'hold'),
                "confidence": signal_result.get('confidence', 0.5),
                "time_horizon": signal_result.get('time_horizon', '3M'),
                "explanation": signal_result.get('explanation', ''),
                "key_insights": signal_result.get('key_insights', {})
            }
            
            ticker = analysis_output.get('ticker', 'Unknown')
            analysis_date = analysis_output.get('analysis_date', 'Unknown')
            logger.info(f"Signal generated for {ticker} on {analysis_date}: {ordered_signal_result['signal']}")
            return ordered_signal_result
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    def get_summary(self) -> Dict:
        """Get summary information about the Signal Analyst."""
        return {
            "analyst": "signal",
            "model": self.model,
            "description": "LLM-based trading signal generation from multi-analyst outputs",
            "output_format": {
                "signal": "strong_buy|buy|hold|sell|strong_sell",
                "confidence": "float (0.0-1.0)",
                "time_horizon": "1M|3M|6M",
                "explanation": "string",
                "key_insights": "object with analyst categories"
            }
        } 