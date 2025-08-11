#!/usr/bin/env python3
"""
Portfolio Manager for AI Portfolio Manager
Constructs, rebalances, and manages portfolios using LLM-driven optimization.
"""
import os
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
import openai
from dotenv import load_dotenv

from data_manager import DataManager

load_dotenv()
logger = logging.getLogger(__name__)

PORTFOLIO_PROMPT_PATH = os.path.join(os.path.dirname(__file__), '../prompts/portfolio_creation_prompt.txt')

class PortfolioManager:
    def __init__(self, model: str = 'gpt-4o'):
        valid_models = ['gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'o3-mini', 'o4-mini']
        self.model = model if model in valid_models else 'gpt-4o'
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError('OPENAI_API_KEY not set in environment.')
        self.client = openai.OpenAI(api_key=self.openai_api_key)
        self.dm = DataManager()
        self.system_prompt = self._load_system_prompt()
        # Store reference to orchestrator for dependency management
        self.orchestrator = None

    def set_orchestrator(self, orchestrator):
        """Set reference to orchestrator for automatic dependency creation."""
        self.orchestrator = orchestrator

    def _load_system_prompt(self) -> str:
        with open(PORTFOLIO_PROMPT_PATH, 'r', encoding='utf-8') as f:
            return f.read()

    def get_signals(self, tickers: List[str], analysis_date: str) -> List[Dict]:
        """
        Get signals for the given tickers and date.
        Automatically creates missing signals if orchestrator is available.
        Falls back to old behavior if orchestrator is not set.
        """
        if self.orchestrator:
            # Use automatic dependency creation
            try:
                signals = self.orchestrator.ensure_signals_exist(tickers, analysis_date, self.model)
                return signals
            except Exception as e:
                # If automatic creation fails, fall back to old behavior but log the error
                logger.error(f"Automatic signal creation failed: {e}")
                logger.info("Falling back to manual signal retrieval")
        
        # Fallback to old behavior: try to get existing signals
        signals = []
        for ticker in tickers:
            signal = self.dm.get_signal(ticker, analysis_date)
            if not signal:
                # fallback: get latest available signal
                signal = self.dm.get_latest_signal(ticker)
            if signal:
                signals.append(signal)
        return signals

    def get_current_portfolio(self, portfolio_name: str, analysis_date: str) -> Optional[List[Dict]]:
        state = self.dm.get_portfolio_state(portfolio_name, analysis_date)
        if state and isinstance(state, dict) and "holdings" in state:
            return state["holdings"]
        return state  # could be None or already a list

    def build_llm_input(self, investment: float, num_tickers: int, tickers: List[str], signals: List[Dict], risk_preference: str, sector_preferences: Optional[List[str]], current_portfolio: Optional[List[Dict]], analysis_date: str) -> Dict:
        input_dict = {
            "date": analysis_date,
            "num_tickers": num_tickers,
            "risk_preference": risk_preference,
            "sector_preference": sector_preferences if sector_preferences is not None else [],
            "signals": signals
        }
        return input_dict

    def call_llm(self, llm_input: Dict) -> Dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": 'JSON\n' + json.dumps(llm_input)}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def calculate_new_state(self, current_portfolio: Optional[List[Dict]], llm_response: Dict, analysis_date: str, investment: float) -> Dict:
        # Build the new portfolio state based on LLM allocations (no 'shares' field expected)
        allocations = llm_response.get('allocations') or llm_response.get('portfolio', {}).get('allocations', [])
        state_allocations = []
        total_allocated = 0.0
        total_transaction_cost = 0.0
        cash_value = investment
        for alloc in allocations:
            ticker = alloc.get('ticker')
            allocation = alloc.get('allocation')
            # If allocation is a percentage (0-1), convert to amount
            if allocation is not None and allocation <= 1.0:
                allocation_amount = allocation * investment
            else:
                allocation_amount = allocation if allocation is not None else alloc.get('investment', 0)
            if ticker and ticker != 'CASH':
                closing_price = self.dm.get_closing_price(ticker, analysis_date) or 0.0
                shares = allocation_amount / closing_price if closing_price > 0 else 0.0
                value = shares * closing_price
                total_allocated += value
                cash_value -= value
                tx_cost = 0.002 * value
                total_transaction_cost += tx_cost
                state_allocations.append({
                    'ticker': ticker,
                    'allocation': allocation,
                    'stocks': shares,
                    'stock_price': closing_price,
                    'value': value
                })
        # Add CASH allocation
        state_allocations.append({
            'ticker': 'CASH',
            'allocation': next((a.get('allocation', 0) for a in allocations if a.get('ticker') == 'CASH'), 0),
            'stocks': cash_value,
            'stock_price': 1,
            'value': cash_value
        })
        portfolio_value = sum(a['value'] for a in state_allocations)
        return {
            'date': analysis_date,
            'allocations': state_allocations,
            'portfolio_value': portfolio_value,
            'transaction_fees': total_transaction_cost
        }

    def _get_closing_price(self, ticker: str) -> float:
        # Use DataManager to get latest closing price
        return self.dm.get_closing_price(ticker, datetime.now().strftime('%Y-%m-%d')) or 0.0

    def check_llm_vs_state(self, llm_response: Dict, new_state: List[Dict]) -> str:
        # Compare LLM's recommended state to actual state, log discrepancies
        notes = []
        if llm_response.get('action') == 'create':
            for alloc in llm_response.get('allocations', []):
                s = next((h for h in new_state if h['ticker'] == alloc['ticker']), None)
                if not s or s['shares'] != alloc['shares']:
                    notes.append(f"Discrepancy for {alloc['ticker']}: LLM shares {alloc['shares']} vs actual {s['shares'] if s else 0}")
        elif llm_response.get('action') == 'rebalance':
            # Could compare before/after state, but for now just log if any transaction is not feasible
            pass
        return '\n'.join(notes)

    def validate_portfolio_trade(self, llm_input, llm_response, new_state, action, retry_count: int = 1) -> str:
        """Validate that the LLM's recommendations and resulting state are feasible and consistent."""
        notes = []
        investment = llm_input.get('budget', 0)
        cash = new_state.get('cash', 0)
        holdings = new_state.get('holdings', [])
        total_market_value = sum(h.get('market_value', 0) for h in holdings)
        total_value = total_market_value + cash
        
        # Check that allocations sum to 1.0 (for both creation and rebalancing)
        allocations = llm_response.get('allocations', [])
        allocation_sum = sum(a.get('allocation', 0) for a in allocations)
        if abs(allocation_sum - 1.0) > 1e-2:
            notes.append(f"ERROR: Allocations sum to {allocation_sum:.4f}, should be 1.0")
        
        # 1. Budget constraint (only for creation)
        if action == 'create':
            if total_value > investment + 1e-2:  # allow for rounding
                notes.append(f"ERROR: Portfolio value ({total_value:.2f}) exceeds budget ({investment:.2f})")
        # 1b. Cash balance must not be negative
        if cash < -1e-2:
            notes.append(f"ERROR: Cash balance is negative ({cash:.2f})")
        # 2. Creation feasibility
        if action == 'create':
            allocs = llm_response.get('allocations', [])
            alloc_sum = sum(a.get('allocation_amount', 0) for a in allocs)
            if alloc_sum > investment + 1e-2:
                notes.append(f"ERROR: Allocations ({alloc_sum:.2f}) exceed budget ({investment:.2f})")
            for alloc in allocs:
                if alloc.get('shares', 0) < 0:
                    notes.append(f"ERROR: Negative shares for {alloc.get('ticker')}")
                if alloc.get('allocation_amount', 0) < 0:
                    notes.append(f"ERROR: Negative allocation for {alloc.get('ticker')}")
        # 2b. Portfolio value must match budget (creation only)
        if abs(total_value - investment) > 1e-2 and action == 'create':
            notes.append(f"ERROR: Portfolio value ({total_value:.2f}) does not match budget ({investment:.2f}) (may be due to cash reserve or rounding)")
        # 3. Rebalance feasibility
        if action == 'rebalance':
            txs = llm_response.get('transactions', [])
            # Check that sells do not exceed current holdings
            current_holdings = {h['ticker']: h['shares'] for h in llm_input.get('current_portfolio', [])}
            for tx in txs:
                if tx['action'] == 'sell':
                    if tx['shares'] > current_holdings.get(tx['ticker'], 0):
                        notes.append(f"ERROR: Attempt to sell {tx['shares']} shares of {tx['ticker']} but only {current_holdings.get(tx['ticker'], 0)} available.")
                if tx['shares'] < 0:
                    notes.append(f"ERROR: Negative shares in transaction for {tx['ticker']}")
                if tx['amount'] < 0:
                    notes.append(f"ERROR: Negative amount in transaction for {tx['ticker']}")
        # 4. State consistency
        for h in holdings:
            if h.get('shares', 0) < 0:
                notes.append(f"ERROR: Negative shares in holding for {h.get('ticker')}")
            if h.get('market_value', 0) < 0:
                notes.append(f"ERROR: Negative market value in holding for {h.get('ticker')}")
        if not notes:
            notes.append(f"Validation passed: All constraints satisfied. Retries: {retry_count}")
        return '\n'.join(notes)

    def rebalance_portfolio(self, portfolio_name: str, analysis_date: str, num_tickers: int, tickers: List[str], risk_preference: str, sector_preferences: Optional[List[str]], model: str = 'gpt-4o', debug: bool = False) -> Dict:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Load previous state from portfolio table
                previous_state = self.dm.get_portfolio_state(portfolio_name, analysis_date)
                if not previous_state:
                    return {
                        "error": f"No previous state found for portfolio {portfolio_name} before {analysis_date}",
                        "notes": "No previous state to rebalance from."
                    }
                if isinstance(previous_state, str):
                    previous_state = json.loads(previous_state)
                current_state = []
                total_portfolio_value = 0.0
                allocations = previous_state.get('allocations', [])
                for alloc in allocations:
                    ticker = alloc.get('ticker')
                    if ticker == 'CASH':
                        cash_value = alloc.get('value', 0)
                        total_portfolio_value += cash_value
                        current_state.append({'ticker': 'CASH', 'allocation': 0.0})
                    else:
                        current_price = self.dm.get_closing_price(ticker, analysis_date) or 0.0
                        stocks = alloc.get('stocks', 0)
                        current_value = stocks * current_price
                        total_portfolio_value += current_value
                        current_state.append({'ticker': ticker, 'allocation': 0.0})
                for state_item in current_state:
                    ticker = state_item['ticker']
                    if ticker == 'CASH':
                        cash_value = next((a.get('value', 0) for a in allocations if a.get('ticker') == 'CASH'), 0)
                        state_item['allocation'] = cash_value / total_portfolio_value if total_portfolio_value > 0 else 0.0
                    else:
                        prev_alloc = next((a for a in allocations if a.get('ticker') == ticker), None)
                        if prev_alloc:
                            stocks = prev_alloc.get('stocks', 0)
                            current_price = self.dm.get_closing_price(ticker, analysis_date) or 0.0
                            current_value = stocks * current_price
                            state_item['allocation'] = current_value / total_portfolio_value if total_portfolio_value > 0 else 0.0
                signals = self.get_signals(tickers, analysis_date)
                llm_input = {
                    "date": analysis_date,
                    "num_tickers": num_tickers,
                    "risk_preference": risk_preference,
                    "sector_preference": sector_preferences if sector_preferences is not None else [],
                    "current_state": current_state,
                    "signals": signals
                }
                rebalance_prompt_path = os.path.join(os.path.dirname(__file__), '../prompts/portfolio_rebalance_prompt.txt')
                with open(rebalance_prompt_path, 'r', encoding='utf-8') as f:
                    rebalance_system_prompt = f.read()
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": rebalance_system_prompt},
                        {"role": "user", "content": 'JSON\n' + json.dumps(llm_input)}
                    ],
                    response_format={"type": "json_object"}
                )
                llm_response = json.loads(response.choices[0].message.content)
                new_allocations = llm_response.get('allocations', [])
                new_state_allocations = []
                total_transaction_cost = 0.0
                for alloc in new_allocations:
                    ticker = alloc.get('ticker')
                    allocation = alloc.get('allocation', 0)
                    if ticker == 'CASH':
                        cash_value = allocation * total_portfolio_value
                        new_state_allocations.append({'ticker': 'CASH', 'allocation': allocation, 'stocks': cash_value, 'stock_price': 1, 'value': cash_value})
                    else:
                        allocation_value = allocation * total_portfolio_value
                        current_price = self.dm.get_closing_price(ticker, analysis_date) or 0.0
                        shares = allocation_value / current_price if current_price > 0 else 0.0
                        prev_alloc = next((a for a in allocations if a.get('ticker') == ticker), None)
                        if prev_alloc:
                            prev_value = prev_alloc.get('value', 0)
                            value_change = abs(allocation_value - prev_value)
                            transaction_cost = 0.002 * value_change
                            total_transaction_cost += transaction_cost
                        else:
                            transaction_cost = 0.002 * allocation_value
                            total_transaction_cost += transaction_cost
                        new_state_allocations.append({'ticker': ticker, 'allocation': allocation, 'stocks': shares, 'stock_price': current_price, 'value': allocation_value})
                new_portfolio_value = sum(a.get('value', 0) for a in new_state_allocations)
                new_state = {'date': analysis_date, 'allocations': new_state_allocations, 'portfolio_value': new_portfolio_value, 'transaction_fees': total_transaction_cost}
                notes = self.validate_portfolio_trade(llm_input, llm_response, new_state, 'rebalance', attempt + 1)
                debug_file = None
                if debug:
                    os.makedirs('test', exist_ok=True)
                    fname = f"test/portfolio_rebalance_{portfolio_name}_{analysis_date.replace('-', '')}_debug.json"
                    with open(fname, 'w') as f:
                        json.dump({'llm_input': llm_input, 'llm_response': llm_response, 'new_state': new_state, 'notes': notes}, f, indent=2)
                    debug_file = fname
                else:
                    self.dm.save_portfolio(
                        portfolio_name=portfolio_name,
                        date=analysis_date,
                        call=json.dumps(llm_input),
                        response=json.dumps(llm_response),
                        state=json.dumps(new_state),
                        notes=notes
                    )
                if 'ERROR' not in notes:
                    return {
                        "llm_input": llm_input,
                        "llm_response": llm_response,
                        "new_state": new_state,
                        "notes": notes,
                        "debug_file": debug_file
                    }
                else:
                    self.dm.delete_portfolio_row(portfolio_name, analysis_date)
                    if attempt == max_retries - 1:
                        print(f"Validation failed after {max_retries} attempts: {notes}")
                        return {
                            "llm_input": llm_input,
                            "llm_response": llm_response,
                            "new_state": new_state,
                            "notes": notes,
                            "debug_file": debug_file
                        }
            except Exception as e:
                print(f"Error in rebalance attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    return {"error": str(e), "notes": f"Rebalancing failed: {str(e)}"}

    def run(self, portfolio_name: str, analysis_date: str, investment: float, num_tickers: int, tickers: List[str], risk_preference: Optional[str], sector_preferences: Optional[List[str]], debug: bool = False) -> Dict:
        max_retries = 5
        for attempt in range(max_retries):
            signals = self.get_signals(tickers, analysis_date)
            current_portfolio = self.get_current_portfolio(portfolio_name, analysis_date) if not debug else None
            llm_input = self.build_llm_input(investment, num_tickers, tickers, signals, risk_preference, sector_preferences, current_portfolio, analysis_date)
            llm_response = self.call_llm(llm_input)
            new_state = self.calculate_new_state(current_portfolio, llm_response, analysis_date, investment)
            notes = self.validate_portfolio_trade(llm_input, llm_response, new_state, llm_response.get('action'), attempt + 1)
            debug_file = None
            if debug:
                # Save to test folder
                os.makedirs('test', exist_ok=True)
                fname = f"test/portfolio_{portfolio_name}_{analysis_date.replace('-', '')}_debug.json"
                with open(fname, 'w') as f:
                    json.dump({
                        'llm_input': llm_input,
                        'llm_response': llm_response,
                        'new_state': new_state,
                        'notes': notes
                    }, f, indent=2)
                debug_file = fname
            else:
                # Save to DB
                self.dm.save_portfolio(
                    portfolio_name=portfolio_name,
                    date=analysis_date,
                    call=json.dumps(llm_input),
                    response=json.dumps(llm_response),
                    state=json.dumps(new_state),
                    notes=notes
                )
            if 'ERROR' not in notes:
                return {
                    "llm_input": llm_input,
                    "llm_response": llm_response,
                    "new_state": new_state,
                    "notes": notes,
                    "debug_file": debug_file
                }
            else:
                # Delete problematic row and retry
                self.dm.delete_portfolio_row(portfolio_name, analysis_date)
                if attempt == max_retries - 1:
                    print(f"Validation failed after {max_retries} attempts: {notes}")
                    return {
                        "llm_input": llm_input,
                        "llm_response": llm_response,
                        "new_state": new_state,
                        "notes": notes,
                        "debug_file": debug_file
                    }

    def manage_portfolio(self, portfolio_name, start_date, end_date, num_tickers, tickers, risk_preference, investment, frequency, sector_preferences=None, model='gpt-4o', debug=False):
        """
        Manage a portfolio by creating it on start_date and rebalancing it periodically.
        Returns a summary with initial_state, rebalance_results, final_state, and notes.
        """
        from datetime import datetime
        from dateutil.relativedelta import relativedelta
        notes = []
        # 1. Create portfolio on start_date
        creation_result = self.run(
            portfolio_name=portfolio_name,
            analysis_date=start_date,
            investment=investment,
            num_tickers=num_tickers,
            tickers=tickers,
            risk_preference=risk_preference,
            sector_preferences=sector_preferences,
            debug=debug
        )
        initial_state = creation_result.get('new_state')
        if not initial_state:
            return {
                'initial_state': None,
                'rebalance_results': [],
                'final_state': None,
                'notes': 'Portfolio creation failed.'
            }
        # 2. Generate rebalance dates
        rebalance_dates = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        if frequency == "monthly":
            while True:
                current = (current + relativedelta(months=1)).replace(day=1)
                if current > end:
                    break
                rebalance_dates.append(current.strftime("%Y-%m-%d"))
        elif frequency == "quarterly":
            # Find the first quarter start after start_date
            # Quarters start on: January (1), April (4), July (7), October (10)
            quarter_starts = [1, 4, 7, 10]
            current_month = current.month
            
            # Find the next quarter start month after current month
            next_quarter_month = None
            for q_month in quarter_starts:
                if q_month > current_month:
                    next_quarter_month = q_month
                    break
            
            # If no quarter found in current year, start with January of next year
            if next_quarter_month is None:
                current = current.replace(year=current.year + 1, month=1, day=1)
            else:
                current = current.replace(month=next_quarter_month, day=1)
            
            while current <= end:
                rebalance_dates.append(current.strftime("%Y-%m-%d"))
                current = (current + relativedelta(months=3)).replace(day=1)
        else:
            return {
                'initial_state': initial_state,
                'rebalance_results': [],
                'final_state': initial_state,
                'notes': f'Unknown frequency: {frequency}'
            }
        # 3. Rebalance for each date
        rebalance_results = []
        last_state = initial_state
        for rebalance_date in rebalance_dates:
            rebalance_result = self.rebalance_portfolio(
                portfolio_name=portfolio_name,
                analysis_date=rebalance_date,
                num_tickers=num_tickers,
                tickers=tickers,
                risk_preference=risk_preference,
                sector_preferences=sector_preferences,
                model=model,
                debug=debug
            )
            rebalance_results.append(rebalance_result)
            if rebalance_result.get('new_state'):
                last_state = rebalance_result['new_state']
        return {
            'initial_state': initial_state,
            'rebalance_results': rebalance_results,
            'final_state': last_state,
            'notes': 'Portfolio management completed.'
        } 