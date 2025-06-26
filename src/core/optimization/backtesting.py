"""
Comprehensive backtesting system for Investment MCP System.

This module provides backtesting capabilities for portfolio strategies
with historical Swedish market data and performance analysis.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor

from ..config import TRADEABLE_FUNDS

logger = logging.getLogger(__name__)


class PortfolioBacktester:
    """Comprehensive backtesting framework for portfolio strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Backtesting parameters
        self.transaction_costs = 0.0025  # 0.25% per transaction
        self.rebalancing_threshold = 0.05  # 5% drift threshold
        self.min_rebalancing_interval = 21  # Minimum 21 days between rebalancing
        
        # Performance calculation settings
        self.risk_free_rate = 0.02  # Swedish risk-free rate approximation
        self.trading_days_per_year = 252
    
    def run_backtest(
        self,
        strategy_func: Callable,
        historical_data: pd.DataFrame,
        start_date: str,
        end_date: str,
        initial_capital: float = 1000000,
        rebalancing_frequency: str = "quarterly",
        benchmark_allocation: Optional[Dict[str, float]] = None,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive backtest for a portfolio strategy.
        
        Args:
            strategy_func: Function that returns portfolio allocation
            historical_data: Historical price data for all funds
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            initial_capital: Initial investment amount in SEK
            rebalancing_frequency: Rebalancing frequency (daily/weekly/monthly/quarterly/annually)
            benchmark_allocation: Benchmark portfolio allocation for comparison
            strategy_params: Additional parameters for strategy function
            
        Returns:
            Dictionary containing comprehensive backtest results
        """
        
        try:
            self.logger.info(f"Running backtest from {start_date} to {end_date}")
            
            # Validate inputs
            if not self._validate_backtest_inputs(historical_data, start_date, end_date):
                return self._empty_result("Invalid backtest inputs")
            
            # Prepare data
            backtest_data = self._prepare_backtest_data(
                historical_data, start_date, end_date
            )
            
            if backtest_data.empty:
                return self._empty_result("Insufficient historical data for backtest period")
            
            # Set up rebalancing schedule
            rebalancing_dates = self._get_rebalancing_dates(
                backtest_data.index, rebalancing_frequency
            )
            
            # Run strategy backtest
            strategy_results = self._run_strategy_backtest(
                strategy_func, backtest_data, rebalancing_dates,
                initial_capital, strategy_params
            )
            
            if not strategy_results["success"]:
                return strategy_results
            
            # Run benchmark backtest if provided
            benchmark_results = None
            if benchmark_allocation:
                benchmark_results = self._run_benchmark_backtest(
                    benchmark_allocation, backtest_data, rebalancing_dates, initial_capital
                )
            
            # Calculate comprehensive performance metrics
            performance_metrics = self._calculate_performance_metrics(
                strategy_results["portfolio_values"],
                strategy_results["returns"],
                backtest_data.index
            )
            
            # Risk analysis
            risk_analysis = self._calculate_risk_metrics(
                strategy_results["returns"],
                strategy_results["portfolio_values"],
                initial_capital
            )
            
            # Transaction cost analysis
            transaction_analysis = self._analyze_transaction_costs(
                strategy_results["rebalancing_history"],
                strategy_results["portfolio_values"]
            )
            
            # Period analysis (different market regimes)
            period_analysis = self._analyze_performance_by_period(
                strategy_results["portfolio_values"],
                backtest_data.index
            )
            
            # Comparison with benchmark
            benchmark_comparison = None
            if benchmark_results and benchmark_results["success"]:
                benchmark_comparison = self._compare_with_benchmark(
                    strategy_results, benchmark_results, backtest_data.index
                )
            
            # Swedish market context analysis
            swedish_analysis = self._analyze_swedish_context(
                strategy_results, backtest_data.index
            )
            
            return {
                "success": True,
                "backtest_type": "comprehensive_portfolio_strategy",
                "backtest_period": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_days": len(backtest_data),
                    "trading_days": len(backtest_data),
                    "years": len(backtest_data) / self.trading_days_per_year
                },
                "strategy_performance": {
                    "final_portfolio_value": strategy_results["portfolio_values"].iloc[-1],
                    "total_return": (strategy_results["portfolio_values"].iloc[-1] / initial_capital) - 1,
                    "annualized_return": performance_metrics["annualized_return"],
                    "volatility": performance_metrics["volatility"],
                    "sharpe_ratio": performance_metrics["sharpe_ratio"],
                    "max_drawdown": performance_metrics["max_drawdown"]
                },
                "detailed_performance": performance_metrics,
                "risk_analysis": risk_analysis,
                "transaction_analysis": transaction_analysis,
                "period_analysis": period_analysis,
                "benchmark_comparison": benchmark_comparison,
                "swedish_analysis": swedish_analysis,
                "portfolio_evolution": {
                    "dates": strategy_results["portfolio_values"].index.strftime('%Y-%m-%d').tolist(),
                    "values": strategy_results["portfolio_values"].tolist(),
                    "daily_returns": strategy_results["returns"].tolist()
                },
                "rebalancing_history": strategy_results["rebalancing_history"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return self._empty_result(str(e))
    
    def run_strategy_comparison(
        self,
        strategies: Dict[str, Dict[str, Any]],
        historical_data: pd.DataFrame,
        start_date: str,
        end_date: str,
        initial_capital: float = 1000000
    ) -> Dict[str, Any]:
        """
        Compare multiple portfolio strategies using backtesting.
        
        Args:
            strategies: Dictionary of strategy configurations
            historical_data: Historical price data
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Initial investment amount
            
        Returns:
            Dictionary containing strategy comparison results
        """
        
        try:
            self.logger.info(f"Comparing {len(strategies)} strategies")
            
            strategy_results = {}
            
            # Run backtest for each strategy
            for strategy_name, strategy_config in strategies.items():
                self.logger.info(f"Backtesting strategy: {strategy_name}")
                
                result = self.run_backtest(
                    strategy_func=strategy_config["strategy_func"],
                    historical_data=historical_data,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    rebalancing_frequency=strategy_config.get("rebalancing_frequency", "quarterly"),
                    strategy_params=strategy_config.get("params")
                )
                
                if result["success"]:
                    strategy_results[strategy_name] = {
                        "performance": result["strategy_performance"],
                        "risk_metrics": result["risk_analysis"],
                        "portfolio_values": result["portfolio_evolution"]["values"],
                        "transaction_costs": result["transaction_analysis"]["total_transaction_costs"]
                    }
                else:
                    strategy_results[strategy_name] = {"error": result.get("error")}
            
            # Comprehensive comparison
            comparison_analysis = self._compare_strategies(strategy_results, initial_capital)
            
            # Risk-adjusted rankings
            rankings = self._rank_strategies(strategy_results)
            
            return {
                "success": True,
                "comparison_type": "multi_strategy_backtest",
                "strategies_tested": list(strategies.keys()),
                "backtest_period": {"start_date": start_date, "end_date": end_date},
                "strategy_results": strategy_results,
                "comparison_analysis": comparison_analysis,
                "strategy_rankings": rankings,
                "best_strategy": rankings["by_sharpe_ratio"][0] if rankings["by_sharpe_ratio"] else None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Strategy comparison failed: {e}")
            return {"success": False, "error": str(e)}
    
    def run_rolling_window_analysis(
        self,
        strategy_func: Callable,
        historical_data: pd.DataFrame,
        window_years: int = 3,
        step_months: int = 6,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run rolling window analysis to test strategy stability over time.
        
        Args:
            strategy_func: Portfolio strategy function
            historical_data: Historical price data
            window_years: Length of each backtest window in years
            step_months: Step size between windows in months
            strategy_params: Strategy parameters
            
        Returns:
            Rolling window analysis results
        """
        
        try:
            self.logger.info(f"Running rolling window analysis: {window_years}Y windows, {step_months}M steps")
            
            # Calculate window parameters
            window_days = window_years * self.trading_days_per_year
            step_days = step_months * 21  # Approximate trading days per month
            
            data_start = historical_data.index[0]
            data_end = historical_data.index[-1]
            
            # Generate rolling windows
            windows = []
            current_start = data_start
            
            while current_start + timedelta(days=window_days) <= data_end:
                window_end = current_start + timedelta(days=window_days)
                windows.append((current_start, window_end))
                current_start += timedelta(days=step_days)
            
            if len(windows) < 3:
                return {"success": False, "error": "Insufficient data for rolling window analysis"}
            
            # Run backtests for each window
            window_results = []
            
            for i, (start_date, end_date) in enumerate(windows):
                self.logger.debug(f"Window {i+1}/{len(windows)}: {start_date.date()} to {end_date.date()}")
                
                result = self.run_backtest(
                    strategy_func=strategy_func,
                    historical_data=historical_data,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    strategy_params=strategy_params
                )
                
                if result["success"]:
                    window_results.append({
                        "window_id": i + 1,
                        "start_date": start_date.strftime('%Y-%m-%d'),
                        "end_date": end_date.strftime('%Y-%m-%d'),
                        "total_return": result["strategy_performance"]["total_return"],
                        "annualized_return": result["strategy_performance"]["annualized_return"],
                        "volatility": result["strategy_performance"]["volatility"],
                        "sharpe_ratio": result["strategy_performance"]["sharpe_ratio"],
                        "max_drawdown": result["strategy_performance"]["max_drawdown"]
                    })
            
            # Analyze rolling window stability
            stability_analysis = self._analyze_rolling_stability(window_results)
            
            return {
                "success": True,
                "analysis_type": "rolling_window",
                "window_parameters": {
                    "window_years": window_years,
                    "step_months": step_months,
                    "total_windows": len(windows),
                    "successful_windows": len(window_results)
                },
                "window_results": window_results,
                "stability_analysis": stability_analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Rolling window analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _run_strategy_backtest(
        self,
        strategy_func: Callable,
        price_data: pd.DataFrame,
        rebalancing_dates: List[datetime],
        initial_capital: float,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run backtest for a specific strategy."""
        
        try:
            portfolio_values = pd.Series(index=price_data.index, dtype=float)
            portfolio_values.iloc[0] = initial_capital
            
            current_allocation = {}
            rebalancing_history = []
            transaction_costs_total = 0
            
            # Calculate returns
            returns = price_data.pct_change().fillna(0)
            
            for i, date in enumerate(price_data.index):
                # Check if rebalancing is needed
                if date in rebalancing_dates or i == 0:
                    # Get new allocation from strategy
                    if strategy_params:
                        new_allocation = strategy_func(
                            price_data.loc[:date], current_allocation, **strategy_params
                        )
                    else:
                        new_allocation = strategy_func(price_data.loc[:date], current_allocation)
                    
                    # Validate allocation
                    if not self._validate_allocation(new_allocation):
                        new_allocation = current_allocation  # Keep previous allocation
                    
                    # Calculate transaction costs
                    if current_allocation:
                        transaction_cost = self._calculate_transaction_costs(
                            current_allocation, new_allocation, portfolio_values.iloc[i-1] if i > 0 else initial_capital
                        )
                        transaction_costs_total += transaction_cost
                    
                    # Record rebalancing
                    rebalancing_history.append({
                        "date": date.strftime('%Y-%m-%d'),
                        "allocation": new_allocation.copy(),
                        "portfolio_value": portfolio_values.iloc[i] if i > 0 else initial_capital,
                        "transaction_cost": transaction_cost if i > 0 else 0
                    })
                    
                    current_allocation = new_allocation
                
                # Update portfolio value based on daily returns
                if i > 0 and current_allocation:
                    daily_return = 0
                    for fund, weight in current_allocation.items():
                        if fund in returns.columns:
                            daily_return += weight * returns.loc[date, fund]
                    
                    portfolio_values.iloc[i] = portfolio_values.iloc[i-1] * (1 + daily_return)
                    
                    # Subtract transaction costs on rebalancing days
                    if date in rebalancing_dates and i > 0:
                        cost = rebalancing_history[-1]["transaction_cost"]
                        portfolio_values.iloc[i] -= cost
            
            # Calculate portfolio returns
            portfolio_returns = portfolio_values.pct_change().fillna(0)
            
            return {
                "success": True,
                "portfolio_values": portfolio_values,
                "returns": portfolio_returns,
                "rebalancing_history": rebalancing_history,
                "total_transaction_costs": transaction_costs_total
            }
            
        except Exception as e:
            self.logger.error(f"Strategy backtest failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _run_benchmark_backtest(
        self,
        allocation: Dict[str, float],
        price_data: pd.DataFrame,
        rebalancing_dates: List[datetime],
        initial_capital: float
    ) -> Dict[str, Any]:
        """Run backtest for benchmark (buy-and-hold or simple rebalanced) strategy."""
        
        def benchmark_strategy(price_data, current_allocation):
            return allocation  # Static allocation
        
        return self._run_strategy_backtest(
            benchmark_strategy, price_data, rebalancing_dates, initial_capital
        )
    
    def _calculate_performance_metrics(
        self,
        portfolio_values: pd.Series,
        returns: pd.Series,
        dates: pd.DatetimeIndex
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        
        try:
            # Basic metrics
            total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
            n_years = len(portfolio_values) / self.trading_days_per_year
            annualized_return = (1 + total_return) ** (1 / n_years) - 1
            
            # Volatility
            volatility = returns.std() * np.sqrt(self.trading_days_per_year)
            
            # Risk-adjusted metrics
            excess_returns = returns - (self.risk_free_rate / self.trading_days_per_year)
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(self.trading_days_per_year)
            
            # Drawdown analysis
            drawdown_analysis = self._calculate_drawdown_metrics(portfolio_values)
            
            # Other metrics
            positive_days = (returns > 0).sum()
            negative_days = (returns < 0).sum()
            win_rate = positive_days / len(returns) if len(returns) > 0 else 0
            
            # Monthly and annual breakdown
            monthly_returns = self._calculate_monthly_returns(portfolio_values, dates)
            annual_returns = self._calculate_annual_returns(portfolio_values, dates)
            
            return {
                "total_return": float(total_return),
                "annualized_return": float(annualized_return),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": drawdown_analysis["max_drawdown"],
                "avg_drawdown": drawdown_analysis["avg_drawdown"],
                "drawdown_duration": drawdown_analysis["max_drawdown_duration"],
                "win_rate": float(win_rate),
                "positive_days": int(positive_days),
                "negative_days": int(negative_days),
                "best_day": float(returns.max()),
                "worst_day": float(returns.min()),
                "monthly_returns": monthly_returns,
                "annual_returns": annual_returns,
                "sortino_ratio": self._calculate_sortino_ratio(returns),
                "calmar_ratio": annualized_return / abs(drawdown_analysis["max_drawdown"]) if drawdown_analysis["max_drawdown"] != 0 else float('inf')
            }
            
        except Exception as e:
            self.logger.error(f"Performance metrics calculation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_risk_metrics(
        self,
        returns: pd.Series,
        portfolio_values: pd.Series,
        initial_capital: float
    ) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        
        try:
            # Value at Risk (VaR)
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()
            
            # Downside metrics
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(self.trading_days_per_year)
            
            # Skewness and Kurtosis
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Maximum loss periods
            loss_periods = self._analyze_loss_periods(portfolio_values, initial_capital)
            
            # Beta calculation (if benchmark available)
            # This would require market data - simplified for now
            
            return {
                "var_95_daily": float(var_95),
                "var_99_daily": float(var_99),
                "cvar_95_daily": float(cvar_95),
                "cvar_99_daily": float(cvar_99),
                "var_95_annual": float(var_95 * np.sqrt(self.trading_days_per_year)),
                "var_99_annual": float(var_99 * np.sqrt(self.trading_days_per_year)),
                "downside_deviation": float(downside_deviation),
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "loss_periods": loss_periods,
                "tail_ratio": float(abs(returns.quantile(0.95)) / abs(returns.quantile(0.05))),
                "gain_pain_ratio": float(returns[returns > 0].sum() / abs(returns[returns < 0].sum())) if returns[returns < 0].sum() != 0 else float('inf')
            }
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            return {"error": str(e)}
    
    def _analyze_transaction_costs(
        self,
        rebalancing_history: List[Dict[str, Any]],
        portfolio_values: pd.Series
    ) -> Dict[str, Any]:
        """Analyze transaction costs and their impact."""
        
        try:
            total_costs = sum(rebal["transaction_cost"] for rebal in rebalancing_history)
            n_rebalances = len(rebalancing_history)
            
            # Cost as percentage of final portfolio value
            final_value = portfolio_values.iloc[-1]
            cost_percentage = (total_costs / final_value) * 100
            
            # Average cost per rebalancing
            avg_cost_per_rebalance = total_costs / n_rebalances if n_rebalances > 0 else 0
            
            # Cost impact on returns
            initial_value = portfolio_values.iloc[0]
            return_without_costs = (final_value + total_costs) / initial_value - 1
            return_with_costs = final_value / initial_value - 1
            cost_drag = return_without_costs - return_with_costs
            
            return {
                "total_transaction_costs": float(total_costs),
                "number_of_rebalances": int(n_rebalances),
                "cost_percentage_of_portfolio": float(cost_percentage),
                "average_cost_per_rebalance": float(avg_cost_per_rebalance),
                "cost_drag_on_returns": float(cost_drag),
                "cost_drag_annualized": float(cost_drag * self.trading_days_per_year / len(portfolio_values)),
                "rebalancing_frequency_days": float(len(portfolio_values) / n_rebalances) if n_rebalances > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Transaction cost analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_performance_by_period(
        self,
        portfolio_values: pd.Series,
        dates: pd.DatetimeIndex
    ) -> Dict[str, Any]:
        """Analyze performance across different time periods and market conditions."""
        
        try:
            period_analysis = {}
            
            # Yearly performance
            years = dates.year.unique()
            for year in years:
                year_mask = dates.year == year
                if year_mask.sum() > 20:  # At least 20 trading days
                    year_values = portfolio_values[year_mask]
                    year_return = (year_values.iloc[-1] / year_values.iloc[0]) - 1
                    period_analysis[f"year_{year}"] = {
                        "return": float(year_return),
                        "start_value": float(year_values.iloc[0]),
                        "end_value": float(year_values.iloc[-1]),
                        "trading_days": int(year_mask.sum())
                    }
            
            # Quarterly performance
            quarters = [(dates.year, dates.quarter) for dates in dates]
            unique_quarters = list(set(quarters))
            
            for year, quarter in unique_quarters:
                quarter_mask = (dates.year == year) & (dates.quarter == quarter)
                if quarter_mask.sum() > 15:  # At least 15 trading days
                    quarter_values = portfolio_values[quarter_mask]
                    quarter_return = (quarter_values.iloc[-1] / quarter_values.iloc[0]) - 1
                    period_analysis[f"q{quarter}_{year}"] = {
                        "return": float(quarter_return),
                        "start_value": float(quarter_values.iloc[0]),
                        "end_value": float(quarter_values.iloc[-1])
                    }
            
            # Best and worst periods
            returns = portfolio_values.pct_change().fillna(0)
            
            # Best/worst months
            monthly_returns = portfolio_values.resample('M').last().pct_change().dropna()
            if len(monthly_returns) > 0:
                best_month = monthly_returns.idxmax()
                worst_month = monthly_returns.idxmin()
                
                period_analysis["best_month"] = {
                    "date": best_month.strftime('%Y-%m'),
                    "return": float(monthly_returns[best_month])
                }
                period_analysis["worst_month"] = {
                    "date": worst_month.strftime('%Y-%m'),
                    "return": float(monthly_returns[worst_month])
                }
            
            return period_analysis
            
        except Exception as e:
            self.logger.error(f"Period analysis failed: {e}")
            return {"error": str(e)}
    
    def _compare_with_benchmark(
        self,
        strategy_results: Dict[str, Any],
        benchmark_results: Dict[str, Any],
        dates: pd.DatetimeIndex
    ) -> Dict[str, Any]:
        """Compare strategy performance with benchmark."""
        
        try:
            strategy_values = strategy_results["portfolio_values"]
            benchmark_values = benchmark_results["portfolio_values"]
            
            # Calculate excess returns
            strategy_returns = strategy_results["returns"]
            benchmark_returns = benchmark_results["returns"]
            excess_returns = strategy_returns - benchmark_returns
            
            # Tracking error
            tracking_error = excess_returns.std() * np.sqrt(self.trading_days_per_year)
            
            # Information ratio
            information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(self.trading_days_per_year)
            
            # Beta and Alpha (simplified)
            if benchmark_returns.std() > 0:
                beta = np.cov(strategy_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
                alpha = strategy_returns.mean() - beta * benchmark_returns.mean()
                alpha_annualized = alpha * self.trading_days_per_year
            else:
                beta = 1.0
                alpha_annualized = 0.0
            
            # Outperformance periods
            excess_cumulative = (1 + excess_returns).cumprod()
            outperforming_days = (excess_cumulative > 1).sum()
            outperformance_ratio = outperforming_days / len(excess_cumulative)
            
            # Final comparison
            strategy_final = strategy_values.iloc[-1]
            benchmark_final = benchmark_values.iloc[-1]
            total_outperformance = (strategy_final / benchmark_final) - 1
            
            return {
                "total_outperformance": float(total_outperformance),
                "tracking_error": float(tracking_error),
                "information_ratio": float(information_ratio),
                "beta": float(beta),
                "alpha_annualized": float(alpha_annualized),
                "outperformance_ratio": float(outperformance_ratio),
                "outperforming_days": int(outperforming_days),
                "total_days": int(len(excess_cumulative)),
                "strategy_final_value": float(strategy_final),
                "benchmark_final_value": float(benchmark_final),
                "excess_return_stats": {
                    "mean": float(excess_returns.mean()),
                    "std": float(excess_returns.std()),
                    "max": float(excess_returns.max()),
                    "min": float(excess_returns.min())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Benchmark comparison failed: {e}")
            return {"error": str(e)}
    
    def _analyze_swedish_context(
        self,
        strategy_results: Dict[str, Any],
        dates: pd.DatetimeIndex
    ) -> Dict[str, Any]:
        """Analyze performance in Swedish market context."""
        
        try:
            portfolio_values = strategy_results["portfolio_values"]
            
            # Tax efficiency analysis
            rebalancing_history = strategy_results["rebalancing_history"]
            n_rebalances = len(rebalancing_history)
            holding_period = len(portfolio_values) / n_rebalances if n_rebalances > 0 else len(portfolio_values)
            
            # ISK vs regular account simulation
            final_value = portfolio_values.iloc[-1]
            initial_value = portfolio_values.iloc[0]
            total_gain = final_value - initial_value
            
            # Approximate tax calculations
            capital_gains_tax = 0.30
            isk_tax_rate = 0.375  # Approximate schablonbeskattning
            
            # Regular account: tax on realized gains
            regular_account_tax = total_gain * capital_gains_tax if total_gain > 0 else 0
            after_tax_regular = final_value - regular_account_tax
            
            # ISK account: schablonbeskattning on capital
            avg_capital = (initial_value + final_value) / 2
            isk_tax = avg_capital * isk_tax_rate * (len(portfolio_values) / self.trading_days_per_year)
            after_tax_isk = final_value - isk_tax
            
            # Housing market comparison (simplified)
            years = len(portfolio_values) / self.trading_days_per_year
            housing_appreciation = 0.05  # Assume 5% annual
            housing_final_value = initial_value * (1 + housing_appreciation) ** years
            
            portfolio_vs_housing = (final_value / housing_final_value) - 1
            
            return {
                "tax_analysis": {
                    "estimated_capital_gains_tax": float(regular_account_tax),
                    "after_tax_value_regular_account": float(after_tax_regular),
                    "estimated_isk_tax": float(isk_tax),
                    "after_tax_value_isk": float(after_tax_isk),
                    "isk_advantage": float(after_tax_isk - after_tax_regular),
                    "recommended_account_type": "ISK" if after_tax_isk > after_tax_regular else "Regular"
                },
                "rebalancing_efficiency": {
                    "average_holding_period_days": float(holding_period),
                    "tax_efficiency_score": "High" if holding_period > 365 else "Medium" if holding_period > 180 else "Low"
                },
                "housing_market_comparison": {
                    "portfolio_vs_housing_outperformance": float(portfolio_vs_housing),
                    "portfolio_final_value": float(final_value),
                    "housing_final_value_estimate": float(housing_final_value),
                    "recommendation": "Portfolio" if portfolio_vs_housing > 0 else "Housing"
                },
                "swedish_market_insights": self._generate_swedish_insights(
                    strategy_results, dates
                )
            }
            
        except Exception as e:
            self.logger.error(f"Swedish context analysis failed: {e}")
            return {"error": str(e)}
    
    def _generate_swedish_insights(
        self,
        strategy_results: Dict[str, Any],
        dates: pd.DatetimeIndex
    ) -> List[str]:
        """Generate insights specific to Swedish market conditions."""
        
        insights = []
        
        portfolio_values = strategy_results["portfolio_values"]
        returns = strategy_results["returns"]
        
        # Performance insights
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        volatility = returns.std() * np.sqrt(self.trading_days_per_year)
        
        if total_return > 0.08:
            insights.append("Stark avkastning som överträffar svenska obligationer")
        elif total_return > 0.05:
            insights.append("Måttlig avkastning i linje med svenska förväntningar")
        else:
            insights.append("Låg avkastning - överväg alternativa strategier")
        
        if volatility < 0.15:
            insights.append("Låg volatilitet lämplig för svenska konservativa investerare")
        elif volatility > 0.25:
            insights.append("Hög volatilitet - kräver stark risktolerans")
        
        # Rebalancing insights
        n_rebalances = len(strategy_results["rebalancing_history"])
        if n_rebalances > 12:  # More than monthly
            insights.append("Frekvent rebalansering kan påverka skatteeffektiviteten")
        
        return insights
    
    def _compare_strategies(
        self,
        strategy_results: Dict[str, Dict[str, Any]],
        initial_capital: float
    ) -> Dict[str, Any]:
        """Compare multiple strategies comprehensively."""
        
        try:
            comparison = {}
            
            for strategy_name, results in strategy_results.items():
                if "error" not in results:
                    perf = results["performance"]
                    risk = results["risk_metrics"]
                    
                    comparison[strategy_name] = {
                        "total_return": perf["total_return"],
                        "annualized_return": perf["annualized_return"],
                        "volatility": perf["volatility"],
                        "sharpe_ratio": perf["sharpe_ratio"],
                        "max_drawdown": perf["max_drawdown"],
                        "final_value": perf["final_portfolio_value"],
                        "transaction_costs": results["transaction_costs"]
                    }
            
            # Find best performers
            if comparison:
                best_return = max(comparison.items(), key=lambda x: x[1]["total_return"])
                best_sharpe = max(comparison.items(), key=lambda x: x[1]["sharpe_ratio"])
                lowest_risk = min(comparison.items(), key=lambda x: x[1]["volatility"])
                
                return {
                    "strategy_comparison": comparison,
                    "best_total_return": {"strategy": best_return[0], "value": best_return[1]["total_return"]},
                    "best_sharpe_ratio": {"strategy": best_sharpe[0], "value": best_sharpe[1]["sharpe_ratio"]},
                    "lowest_volatility": {"strategy": lowest_risk[0], "value": lowest_risk[1]["volatility"]},
                    "summary": f"Bästa avkastning: {best_return[0]}, Bästa riskjusterad: {best_sharpe[0]}"
                }
            else:
                return {"error": "No successful strategies to compare"}
                
        except Exception as e:
            self.logger.error(f"Strategy comparison failed: {e}")
            return {"error": str(e)}
    
    def _rank_strategies(self, strategy_results: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Rank strategies by different metrics."""
        
        try:
            valid_strategies = {k: v for k, v in strategy_results.items() if "error" not in v}
            
            if not valid_strategies:
                return {}
            
            # Sort by different metrics
            by_return = sorted(valid_strategies.items(), 
                             key=lambda x: x[1]["performance"]["total_return"], reverse=True)
            by_sharpe = sorted(valid_strategies.items(), 
                             key=lambda x: x[1]["performance"]["sharpe_ratio"], reverse=True)
            by_risk = sorted(valid_strategies.items(), 
                           key=lambda x: x[1]["performance"]["volatility"])
            by_drawdown = sorted(valid_strategies.items(), 
                               key=lambda x: x[1]["performance"]["max_drawdown"])
            
            return {
                "by_total_return": [item[0] for item in by_return],
                "by_sharpe_ratio": [item[0] for item in by_sharpe],
                "by_lowest_risk": [item[0] for item in by_risk],
                "by_lowest_drawdown": [item[0] for item in by_drawdown]
            }
            
        except Exception as e:
            self.logger.error(f"Strategy ranking failed: {e}")
            return {}
    
    def _analyze_rolling_stability(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze stability of strategy across rolling windows."""
        
        try:
            if not window_results:
                return {"error": "No window results to analyze"}
            
            # Extract metrics
            returns = [r["total_return"] for r in window_results]
            sharpe_ratios = [r["sharpe_ratio"] for r in window_results]
            volatilities = [r["volatility"] for r in window_results]
            max_drawdowns = [r["max_drawdown"] for r in window_results]
            
            # Calculate stability metrics
            return_stability = np.std(returns) / np.mean(returns) if np.mean(returns) != 0 else float('inf')
            sharpe_stability = np.std(sharpe_ratios)
            
            # Consistency metrics
            positive_periods = sum(1 for r in returns if r > 0)
            consistency_ratio = positive_periods / len(returns)
            
            # Best and worst periods
            best_period = max(window_results, key=lambda x: x["total_return"])
            worst_period = min(window_results, key=lambda x: x["total_return"])
            
            return {
                "stability_metrics": {
                    "return_stability_cv": float(return_stability),
                    "sharpe_stability_std": float(sharpe_stability),
                    "consistency_ratio": float(consistency_ratio),
                    "positive_periods": int(positive_periods),
                    "total_periods": int(len(returns))
                },
                "summary_statistics": {
                    "avg_return": float(np.mean(returns)),
                    "std_return": float(np.std(returns)),
                    "avg_sharpe": float(np.mean(sharpe_ratios)),
                    "avg_volatility": float(np.mean(volatilities)),
                    "avg_max_drawdown": float(np.mean(max_drawdowns))
                },
                "best_period": {
                    "window": f"{best_period['start_date']} to {best_period['end_date']}",
                    "return": best_period["total_return"]
                },
                "worst_period": {
                    "window": f"{worst_period['start_date']} to {worst_period['end_date']}",
                    "return": worst_period["total_return"]
                },
                "stability_assessment": "High" if return_stability < 0.5 and consistency_ratio > 0.7 else "Medium" if consistency_ratio > 0.5 else "Low"
            }
            
        except Exception as e:
            self.logger.error(f"Rolling stability analysis failed: {e}")
            return {"error": str(e)}
    
    # Helper methods for calculations
    def _calculate_drawdown_metrics(self, portfolio_values: pd.Series) -> Dict[str, float]:
        """Calculate drawdown metrics."""
        
        running_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - running_max) / running_max
        
        max_drawdown = drawdowns.min()
        
        # Average drawdown when underwater
        underwater_drawdowns = drawdowns[drawdowns < 0]
        avg_drawdown = underwater_drawdowns.mean() if len(underwater_drawdowns) > 0 else 0
        
        # Maximum drawdown duration
        underwater = drawdowns < -0.01  # More than 1% underwater
        underwater_periods = []
        current_period = 0
        
        for is_underwater in underwater:
            if is_underwater:
                current_period += 1
            else:
                if current_period > 0:
                    underwater_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            underwater_periods.append(current_period)
        
        max_drawdown_duration = max(underwater_periods) if underwater_periods else 0
        
        return {
            "max_drawdown": float(max_drawdown),
            "avg_drawdown": float(avg_drawdown),
            "max_drawdown_duration": int(max_drawdown_duration)
        }
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return float('inf')
        
        excess_return = returns.mean() - (self.risk_free_rate / self.trading_days_per_year)
        sortino = (excess_return / downside_std) * np.sqrt(self.trading_days_per_year)
        
        return float(sortino)
    
    def _calculate_monthly_returns(self, portfolio_values: pd.Series, dates: pd.DatetimeIndex) -> Dict[str, float]:
        """Calculate monthly returns."""
        
        try:
            monthly_values = portfolio_values.resample('M').last()
            monthly_returns = monthly_values.pct_change().dropna()
            
            return {
                date.strftime('%Y-%m'): float(ret) 
                for date, ret in monthly_returns.items()
            }
        except Exception:
            return {}
    
    def _calculate_annual_returns(self, portfolio_values: pd.Series, dates: pd.DatetimeIndex) -> Dict[str, float]:
        """Calculate annual returns."""
        
        try:
            annual_values = portfolio_values.resample('Y').last()
            annual_returns = annual_values.pct_change().dropna()
            
            return {
                str(date.year): float(ret) 
                for date, ret in annual_returns.items()
            }
        except Exception:
            return {}
    
    def _analyze_loss_periods(self, portfolio_values: pd.Series, initial_capital: float) -> Dict[str, Any]:
        """Analyze periods of losses."""
        
        try:
            returns = portfolio_values.pct_change().fillna(0)
            negative_returns = returns[returns < 0]
            
            if len(negative_returns) == 0:
                return {"total_loss_days": 0, "avg_loss": 0, "max_single_day_loss": 0}
            
            return {
                "total_loss_days": int(len(negative_returns)),
                "avg_loss": float(negative_returns.mean()),
                "max_single_day_loss": float(negative_returns.min()),
                "total_negative_return": float(negative_returns.sum())
            }
            
        except Exception:
            return {"error": "Loss period analysis failed"}
    
    def _get_rebalancing_dates(self, dates: pd.DatetimeIndex, frequency: str) -> List[datetime]:
        """Generate rebalancing dates based on frequency."""
        
        if frequency == "daily":
            return dates.tolist()
        elif frequency == "weekly":
            return [date for date in dates if date.weekday() == 0]  # Mondays
        elif frequency == "monthly":
            return [date for date in dates if date.day <= 7 and date.weekday() == 0]  # First Monday
        elif frequency == "quarterly":
            quarters = [1, 4, 7, 10]  # Jan, Apr, Jul, Oct
            return [date for date in dates if date.month in quarters and date.day <= 7 and date.weekday() == 0]
        elif frequency == "annually":
            return [date for date in dates if date.month == 1 and date.day <= 7 and date.weekday() == 0]
        else:
            # Default to quarterly
            quarters = [1, 4, 7, 10]
            return [date for date in dates if date.month in quarters and date.day <= 7 and date.weekday() == 0]
    
    def _calculate_transaction_costs(
        self,
        old_allocation: Dict[str, float],
        new_allocation: Dict[str, float],
        portfolio_value: float
    ) -> float:
        """Calculate transaction costs for rebalancing."""
        
        total_turnover = 0
        all_funds = set(old_allocation.keys()) | set(new_allocation.keys())
        
        for fund in all_funds:
            old_weight = old_allocation.get(fund, 0)
            new_weight = new_allocation.get(fund, 0)
            turnover = abs(new_weight - old_weight)
            total_turnover += turnover
        
        # Only pay transaction costs on the portion that's actually traded
        # Total turnover is divided by 2 because buying and selling sum to total change
        actual_turnover = total_turnover / 2
        transaction_cost = actual_turnover * portfolio_value * self.transaction_costs
        
        return transaction_cost
    
    def _validate_allocation(self, allocation: Dict[str, float]) -> bool:
        """Validate portfolio allocation."""
        
        if not allocation:
            return False
        
        # Check weights sum approximately to 1
        total_weight = sum(allocation.values())
        if abs(total_weight - 1.0) > 0.05:  # Allow 5% tolerance
            return False
        
        # Check all weights are non-negative
        if any(weight < 0 for weight in allocation.values()):
            return False
        
        # Check all funds are in approved universe
        approved_funds = set(TRADEABLE_FUNDS.keys())
        for fund in allocation.keys():
            if fund not in approved_funds:
                return False
        
        return True
    
    def _prepare_backtest_data(
        self,
        historical_data: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Prepare historical data for backtesting."""
        
        try:
            # Convert dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Filter data
            mask = (historical_data.index >= start_dt) & (historical_data.index <= end_dt)
            filtered_data = historical_data.loc[mask]
            
            # Forward fill missing values
            filtered_data = filtered_data.fillna(method='ffill')
            
            # Remove any remaining NaN rows
            filtered_data = filtered_data.dropna()
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            return pd.DataFrame()
    
    def _validate_backtest_inputs(
        self,
        historical_data: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> bool:
        """Validate backtest inputs."""
        
        if historical_data.empty:
            return False
        
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            if start_dt >= end_dt:
                return False
            
            if start_dt < historical_data.index.min() or end_dt > historical_data.index.max():
                return False
            
            return True
            
        except Exception:
            return False
    
    def _empty_result(self, error_message: str) -> Dict[str, Any]:
        """Return empty result with error message."""
        
        return {
            "success": False,
            "error": error_message,
            "backtest_type": "portfolio_strategy",
            "timestamp": datetime.now().isoformat()
        }