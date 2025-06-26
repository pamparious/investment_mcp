"""
Monte Carlo simulation framework for Investment MCP System.

This module provides comprehensive Monte Carlo simulation capabilities
for portfolio risk analysis and scenario testing.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from ..config import TRADEABLE_FUNDS

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """Monte Carlo simulation for portfolio analysis and risk assessment."""
    
    def __init__(self, random_seed: Optional[int] = None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Simulation parameters
        self.default_simulations = 10000
        self.default_time_horizon = 252  # 1 year in trading days
        self.confidence_levels = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    def run_portfolio_simulation(
        self,
        allocation: Dict[str, float],
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        initial_investment: float = 1000000,
        time_horizon_years: float = 1.0,
        n_simulations: int = None,
        include_rebalancing: bool = False,
        rebalancing_frequency: str = "quarterly"
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for portfolio returns.
        
        Args:
            allocation: Portfolio allocation weights
            expected_returns: Expected annual returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            initial_investment: Initial investment amount in SEK
            time_horizon_years: Investment time horizon in years
            n_simulations: Number of Monte Carlo simulations
            include_rebalancing: Whether to include periodic rebalancing
            rebalancing_frequency: Frequency of rebalancing (monthly/quarterly/annually)
            
        Returns:
            Dictionary containing simulation results and statistics
        """
        
        try:
            if n_simulations is None:
                n_simulations = self.default_simulations
            
            self.logger.info(f"Running Monte Carlo simulation: {n_simulations} paths, {time_horizon_years}Y horizon")
            
            # Validate inputs
            if not self._validate_simulation_inputs(allocation, expected_returns, covariance_matrix):
                return self._empty_result("Invalid simulation inputs")
            
            # Filter to available funds
            available_funds = [fund for fund in allocation.keys() if fund in expected_returns.index]
            if len(available_funds) < 2:
                return self._empty_result("Insufficient funds for simulation")
            
            # Prepare simulation parameters
            time_steps = int(time_horizon_years * 252)  # Daily steps
            dt = 1.0 / 252  # Daily time step
            
            # Subset data
            returns = expected_returns[available_funds]
            cov_matrix = covariance_matrix.loc[available_funds, available_funds]
            weights = np.array([allocation[fund] for fund in available_funds])
            
            # Run simulation
            if include_rebalancing:
                simulation_results = self._simulate_with_rebalancing(
                    weights, returns, cov_matrix, initial_investment,
                    time_steps, dt, n_simulations, rebalancing_frequency
                )
            else:
                simulation_results = self._simulate_buy_and_hold(
                    weights, returns, cov_matrix, initial_investment,
                    time_steps, dt, n_simulations
                )
            
            if not simulation_results["success"]:
                return simulation_results
            
            # Calculate comprehensive statistics
            final_values = simulation_results["final_values"]
            path_statistics = self._calculate_path_statistics(
                simulation_results["all_paths"], initial_investment
            )
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(
                final_values, initial_investment, time_horizon_years
            )
            
            # Scenario analysis
            scenario_analysis = self._perform_scenario_analysis(
                final_values, initial_investment
            )
            
            # Swedish-specific analysis
            swedish_analysis = self._analyze_swedish_context(
                final_values, initial_investment, time_horizon_years
            )
            
            return {
                "success": True,
                "simulation_type": "monte_carlo_portfolio",
                "simulation_parameters": {
                    "n_simulations": n_simulations,
                    "time_horizon_years": time_horizon_years,
                    "initial_investment": initial_investment,
                    "include_rebalancing": include_rebalancing,
                    "rebalancing_frequency": rebalancing_frequency if include_rebalancing else None,
                    "funds_included": available_funds
                },
                "final_value_statistics": {
                    "mean": float(np.mean(final_values)),
                    "median": float(np.median(final_values)),
                    "std": float(np.std(final_values)),
                    "min": float(np.min(final_values)),
                    "max": float(np.max(final_values)),
                    "percentiles": {
                        f"p{int(p*100)}": float(np.percentile(final_values, p*100))
                        for p in self.confidence_levels
                    }
                },
                "path_statistics": path_statistics,
                "risk_metrics": risk_metrics,
                "scenario_analysis": scenario_analysis,
                "swedish_analysis": swedish_analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Monte Carlo simulation failed: {e}")
            return self._empty_result(str(e))
    
    def run_stress_test(
        self,
        allocation: Dict[str, float],
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        stress_scenarios: List[Dict[str, Any]],
        initial_investment: float = 1000000,
        n_simulations: int = None
    ) -> Dict[str, Any]:
        """
        Run stress test scenarios using Monte Carlo simulation.
        
        Args:
            allocation: Portfolio allocation weights
            expected_returns: Expected annual returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            stress_scenarios: List of stress scenarios to test
            initial_investment: Initial investment amount
            n_simulations: Number of simulations per scenario
            
        Returns:
            Dictionary containing stress test results
        """
        
        try:
            if n_simulations is None:
                n_simulations = self.default_simulations // 2  # Fewer sims per scenario
            
            self.logger.info(f"Running stress test with {len(stress_scenarios)} scenarios")
            
            stress_results = {}
            
            for i, scenario in enumerate(stress_scenarios):
                scenario_name = scenario.get("name", f"Scenario_{i+1}")
                
                # Modify returns and covariance based on scenario
                stressed_returns, stressed_cov = self._apply_stress_scenario(
                    expected_returns, covariance_matrix, scenario
                )
                
                # Run simulation for this scenario
                scenario_result = self.run_portfolio_simulation(
                    allocation=allocation,
                    expected_returns=stressed_returns,
                    covariance_matrix=stressed_cov,
                    initial_investment=initial_investment,
                    time_horizon_years=scenario.get("time_horizon", 1.0),
                    n_simulations=n_simulations,
                    include_rebalancing=False  # Usually no rebalancing in stress tests
                )
                
                if scenario_result["success"]:
                    stress_results[scenario_name] = {
                        "scenario_description": scenario.get("description", ""),
                        "final_value_stats": scenario_result["final_value_statistics"],
                        "risk_metrics": scenario_result["risk_metrics"],
                        "worst_case_loss": scenario_result["risk_metrics"]["maximum_drawdown"],
                        "probability_of_loss": scenario_result["scenario_analysis"]["probability_of_loss"]
                    }
                else:
                    stress_results[scenario_name] = {
                        "error": scenario_result.get("error", "Simulation failed")
                    }
            
            # Compare scenarios
            scenario_comparison = self._compare_stress_scenarios(stress_results, initial_investment)
            
            return {
                "success": True,
                "stress_test_type": "monte_carlo_scenarios",
                "n_scenarios": len(stress_scenarios),
                "n_simulations_per_scenario": n_simulations,
                "stress_results": stress_results,
                "scenario_comparison": scenario_comparison,
                "overall_assessment": self._assess_overall_stress_resilience(stress_results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Stress test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def calculate_var_and_cvar(
        self,
        allocation: Dict[str, float],
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        confidence_level: float = 0.05,
        time_horizon_days: int = 22,  # ~1 month
        initial_investment: float = 1000000,
        n_simulations: int = None
    ) -> Dict[str, Any]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR) using Monte Carlo.
        
        Args:
            allocation: Portfolio allocation weights
            expected_returns: Expected annual returns
            covariance_matrix: Covariance matrix
            confidence_level: Confidence level for VaR (e.g., 0.05 for 95% VaR)
            time_horizon_days: Time horizon in days
            initial_investment: Initial investment amount
            n_simulations: Number of simulations
            
        Returns:
            Dictionary containing VaR and CVaR metrics
        """
        
        try:
            if n_simulations is None:
                n_simulations = self.default_simulations
            
            # Run short-term simulation
            simulation_result = self.run_portfolio_simulation(
                allocation=allocation,
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                initial_investment=initial_investment,
                time_horizon_years=time_horizon_days / 252.0,
                n_simulations=n_simulations,
                include_rebalancing=False
            )
            
            if not simulation_result["success"]:
                return {"success": False, "error": simulation_result.get("error")}
            
            final_values = simulation_result["final_value_statistics"]
            
            # Calculate VaR and CVaR
            var_level = confidence_level * 100
            var_value = initial_investment - final_values["percentiles"][f"p{int(var_level)}"]
            
            # CVaR: Expected loss beyond VaR
            # Find all values worse than VaR threshold
            var_threshold = final_values["percentiles"][f"p{int(var_level)}"]
            
            # Approximate CVaR using available percentiles
            if var_level <= 5:
                cvar_threshold = final_values["percentiles"]["p1"]
            elif var_level <= 10:
                cvar_threshold = (final_values["percentiles"]["p1"] + final_values["percentiles"]["p5"]) / 2
            else:
                cvar_threshold = final_values["percentiles"]["p5"]
            
            cvar_value = initial_investment - cvar_threshold
            
            # Additional risk metrics
            expected_shortfall = max(0, initial_investment - final_values["mean"])
            maximum_loss = initial_investment - final_values["min"]
            
            return {
                "success": True,
                "var_cvar_type": "monte_carlo",
                "parameters": {
                    "confidence_level": confidence_level,
                    "confidence_level_pct": (1 - confidence_level) * 100,
                    "time_horizon_days": time_horizon_days,
                    "initial_investment": initial_investment,
                    "n_simulations": n_simulations
                },
                "var_metrics": {
                    "var_absolute": float(var_value),
                    "var_percentage": float(var_value / initial_investment * 100),
                    "cvar_absolute": float(cvar_value),
                    "cvar_percentage": float(cvar_value / initial_investment * 100),
                    "expected_shortfall": float(expected_shortfall),
                    "maximum_loss": float(maximum_loss),
                    "maximum_loss_percentage": float(maximum_loss / initial_investment * 100)
                },
                "interpretation": {
                    "var_interpretation": f"Med {(1-confidence_level)*100:.0f}% säkerhet förväntas förlusten inte överstiga {var_value:,.0f} SEK på {time_horizon_days} dagar",
                    "cvar_interpretation": f"Om förlusten överstiger VaR, förväntas den genomsnittliga förlusten vara {cvar_value:,.0f} SEK",
                    "risk_level": self._assess_var_risk_level(var_value, initial_investment)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"VaR/CVaR calculation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _simulate_buy_and_hold(
        self,
        weights: np.ndarray,
        returns: pd.Series,
        cov_matrix: pd.DataFrame,
        initial_investment: float,
        time_steps: int,
        dt: float,
        n_simulations: int
    ) -> Dict[str, Any]:
        """Run buy-and-hold Monte Carlo simulation."""
        
        try:
            # Convert to numpy arrays
            mu = returns.values
            sigma = cov_matrix.values
            
            # Cholesky decomposition for correlated random numbers
            try:
                L = np.linalg.cholesky(sigma)
            except np.linalg.LinAlgError:
                # Add small amount to diagonal if matrix is not positive definite
                sigma_adj = sigma + np.eye(len(sigma)) * 1e-8
                L = np.linalg.cholesky(sigma_adj)
            
            # Pre-allocate arrays
            all_paths = np.zeros((n_simulations, time_steps + 1))
            all_paths[:, 0] = initial_investment
            
            # Generate all random numbers at once for efficiency
            random_normals = np.random.standard_normal((n_simulations, time_steps, len(mu)))
            
            # Run simulations
            for sim in range(n_simulations):
                portfolio_value = initial_investment
                
                for t in range(time_steps):
                    # Generate correlated random returns
                    random_returns = np.dot(L, random_normals[sim, t, :])
                    
                    # Calculate daily returns for each asset
                    daily_returns = mu * dt + np.sqrt(dt) * random_returns
                    
                    # Portfolio return
                    portfolio_return = np.dot(weights, daily_returns)
                    
                    # Update portfolio value
                    portfolio_value *= (1 + portfolio_return)
                    all_paths[sim, t + 1] = portfolio_value
            
            final_values = all_paths[:, -1]
            
            return {
                "success": True,
                "final_values": final_values,
                "all_paths": all_paths,
                "simulation_type": "buy_and_hold"
            }
            
        except Exception as e:
            self.logger.error(f"Buy-and-hold simulation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _simulate_with_rebalancing(
        self,
        target_weights: np.ndarray,
        returns: pd.Series,
        cov_matrix: pd.DataFrame,
        initial_investment: float,
        time_steps: int,
        dt: float,
        n_simulations: int,
        rebalancing_frequency: str
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation with periodic rebalancing."""
        
        try:
            # Determine rebalancing frequency
            rebalancing_intervals = {
                "monthly": 21,      # ~21 trading days per month
                "quarterly": 63,    # ~63 trading days per quarter
                "annually": 252     # 252 trading days per year
            }
            
            rebalance_every = rebalancing_intervals.get(rebalancing_frequency, 63)
            
            # Convert to numpy arrays
            mu = returns.values
            sigma = cov_matrix.values
            
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(sigma)
            except np.linalg.LinAlgError:
                sigma_adj = sigma + np.eye(len(sigma)) * 1e-8
                L = np.linalg.cholesky(sigma_adj)
            
            # Pre-allocate arrays
            all_paths = np.zeros((n_simulations, time_steps + 1))
            all_paths[:, 0] = initial_investment
            
            # Generate random numbers
            random_normals = np.random.standard_normal((n_simulations, time_steps, len(mu)))
            
            # Run simulations with rebalancing
            for sim in range(n_simulations):
                # Initialize asset values
                asset_values = target_weights * initial_investment
                
                for t in range(time_steps):
                    # Generate correlated random returns
                    random_returns = np.dot(L, random_normals[sim, t, :])
                    daily_returns = mu * dt + np.sqrt(dt) * random_returns
                    
                    # Update asset values
                    asset_values *= (1 + daily_returns)
                    
                    # Rebalancing
                    if (t + 1) % rebalance_every == 0:
                        total_value = np.sum(asset_values)
                        asset_values = target_weights * total_value
                    
                    # Record total portfolio value
                    all_paths[sim, t + 1] = np.sum(asset_values)
            
            final_values = all_paths[:, -1]
            
            return {
                "success": True,
                "final_values": final_values,
                "all_paths": all_paths,
                "simulation_type": "rebalanced",
                "rebalancing_frequency": rebalancing_frequency
            }
            
        except Exception as e:
            self.logger.error(f"Rebalanced simulation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_path_statistics(
        self,
        all_paths: np.ndarray,
        initial_investment: float
    ) -> Dict[str, Any]:
        """Calculate statistics for all simulation paths."""
        
        try:
            # Path-based metrics
            drawdowns = self._calculate_drawdowns(all_paths)
            max_drawdowns = np.min(drawdowns, axis=1)  # Most negative drawdown for each path
            
            # Time to recovery (simplified)
            underwater_periods = []
            for sim in range(all_paths.shape[0]):
                path = all_paths[sim, :]
                peak = np.maximum.accumulate(path)
                underwater = path < peak
                if np.any(underwater):
                    # Find longest underwater period
                    underwater_lengths = []
                    current_length = 0
                    for is_underwater in underwater:
                        if is_underwater:
                            current_length += 1
                        else:
                            if current_length > 0:
                                underwater_lengths.append(current_length)
                            current_length = 0
                    if current_length > 0:
                        underwater_lengths.append(current_length)
                    
                    if underwater_lengths:
                        underwater_periods.append(max(underwater_lengths))
                    else:
                        underwater_periods.append(0)
                else:
                    underwater_periods.append(0)
            
            return {
                "maximum_drawdown": {
                    "mean": float(np.mean(max_drawdowns)),
                    "median": float(np.median(max_drawdowns)),
                    "worst": float(np.min(max_drawdowns)),
                    "percentile_5": float(np.percentile(max_drawdowns, 5))
                },
                "underwater_periods": {
                    "mean_days": float(np.mean(underwater_periods)),
                    "median_days": float(np.median(underwater_periods)),
                    "max_days": float(np.max(underwater_periods)) if underwater_periods else 0
                },
                "volatility_of_paths": {
                    "mean_annual_vol": float(np.mean([np.std(path) for path in all_paths]) * np.sqrt(252) / initial_investment),
                    "median_annual_vol": float(np.median([np.std(path) for path in all_paths]) * np.sqrt(252) / initial_investment)
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Path statistics calculation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_drawdowns(self, all_paths: np.ndarray) -> np.ndarray:
        """Calculate drawdowns for all paths."""
        
        # Calculate running maximum for each path
        running_max = np.maximum.accumulate(all_paths, axis=1)
        
        # Calculate drawdowns as percentage from peak
        drawdowns = (all_paths - running_max) / running_max
        
        return drawdowns
    
    def _calculate_risk_metrics(
        self,
        final_values: np.ndarray,
        initial_investment: float,
        time_horizon_years: float
    ) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics from simulation results."""
        
        # Returns
        total_returns = (final_values - initial_investment) / initial_investment
        annualized_returns = (final_values / initial_investment) ** (1 / time_horizon_years) - 1
        
        # Basic statistics
        prob_of_loss = np.mean(final_values < initial_investment)
        prob_of_ruin = np.mean(final_values < initial_investment * 0.5)  # 50% loss
        
        # Downside metrics
        negative_returns = total_returns[total_returns < 0]
        downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0
        
        # Maximum drawdown (approximated from final values)
        max_loss = (initial_investment - np.min(final_values)) / initial_investment
        
        return {
            "probability_of_loss": float(prob_of_loss),
            "probability_of_ruin": float(prob_of_ruin),
            "expected_return": float(np.mean(annualized_returns)),
            "return_volatility": float(np.std(annualized_returns)),
            "downside_deviation": float(downside_deviation),
            "maximum_drawdown": float(max_loss),
            "sharpe_ratio": float(np.mean(annualized_returns) / np.std(annualized_returns)) if np.std(annualized_returns) > 0 else 0,
            "sortino_ratio": float(np.mean(annualized_returns) / downside_deviation) if downside_deviation > 0 else float('inf'),
            "skewness": float(self._calculate_skewness(annualized_returns)),
            "kurtosis": float(self._calculate_kurtosis(annualized_returns))
        }
    
    def _perform_scenario_analysis(
        self,
        final_values: np.ndarray,
        initial_investment: float
    ) -> Dict[str, Any]:
        """Perform scenario analysis on simulation results."""
        
        # Define scenarios based on final values
        scenarios = {
            "bull_market": np.sum(final_values > initial_investment * 1.5),  # >50% gain
            "bear_market": np.sum(final_values < initial_investment * 0.8),  # >20% loss
            "crash": np.sum(final_values < initial_investment * 0.5),        # >50% loss
            "modest_gains": np.sum((final_values > initial_investment) & 
                                 (final_values <= initial_investment * 1.2)),  # 0-20% gain
            "modest_losses": np.sum((final_values < initial_investment) & 
                                  (final_values >= initial_investment * 0.9))   # 0-10% loss
        }
        
        total_sims = len(final_values)
        scenario_probs = {k: v / total_sims for k, v in scenarios.items()}
        
        return {
            "scenario_counts": scenarios,
            "scenario_probabilities": scenario_probs,
            "probability_of_loss": float(np.mean(final_values < initial_investment)),
            "probability_of_gain": float(np.mean(final_values > initial_investment)),
            "expected_final_value": float(np.mean(final_values)),
            "worst_case_1_percent": float(np.percentile(final_values, 1)),
            "best_case_1_percent": float(np.percentile(final_values, 99))
        }
    
    def _analyze_swedish_context(
        self,
        final_values: np.ndarray,
        initial_investment: float,
        time_horizon_years: float
    ) -> Dict[str, Any]:
        """Analyze results in Swedish investment context."""
        
        # Swedish-specific thresholds and considerations
        isk_tax_equivalent = 0.375  # ISK schablonbeskattning approximation
        capital_gains_tax = 0.30
        
        # Calculate tax implications
        gains = final_values - initial_investment
        positive_gains = gains[gains > 0]
        
        if len(positive_gains) > 0:
            # Approximate tax impact
            avg_gain = np.mean(positive_gains)
            tax_on_gains = avg_gain * capital_gains_tax
            after_tax_gain = avg_gain - tax_on_gains
        else:
            after_tax_gain = 0
        
        # Housing market comparison (approximate)
        housing_return_assumption = 0.05  # 5% annual housing appreciation
        housing_final_value = initial_investment * (1 + housing_return_assumption) ** time_horizon_years
        
        portfolio_beats_housing = np.mean(final_values > housing_final_value)
        
        return {
            "tax_considerations": {
                "average_capital_gains_tax": float(tax_on_gains) if len(positive_gains) > 0 else 0,
                "after_tax_expected_gain": float(after_tax_gain),
                "isk_advantage": "ISK-konto kan vara fördelaktigt för skatteeffektivitet"
            },
            "housing_comparison": {
                "probability_beats_housing": float(portfolio_beats_housing),
                "housing_final_value_assumption": float(housing_final_value),
                "investment_vs_housing": "investment" if np.mean(final_values) > housing_final_value else "housing"
            },
            "swedish_risk_assessment": self._assess_swedish_risk_level(final_values, initial_investment),
            "recommendations": self._generate_swedish_recommendations(final_values, initial_investment)
        }
    
    def _apply_stress_scenario(
        self,
        returns: pd.Series,
        cov_matrix: pd.DataFrame,
        scenario: Dict[str, Any]
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Apply stress scenario modifications to returns and covariance."""
        
        stressed_returns = returns.copy()
        stressed_cov = cov_matrix.copy()
        
        # Apply return shocks
        if "return_shocks" in scenario:
            for fund, shock in scenario["return_shocks"].items():
                if fund in stressed_returns.index:
                    stressed_returns[fund] += shock
        
        # Apply volatility multipliers
        if "volatility_multipliers" in scenario:
            for fund, multiplier in scenario["volatility_multipliers"].items():
                if fund in stressed_cov.index:
                    # Scale variance (diagonal elements)
                    current_var = stressed_cov.loc[fund, fund]
                    new_var = current_var * (multiplier ** 2)
                    
                    # Scale covariances proportionally
                    stressed_cov.loc[fund, :] *= multiplier
                    stressed_cov.loc[:, fund] *= multiplier
                    
                    # Reset diagonal to correct variance
                    stressed_cov.loc[fund, fund] = new_var
        
        # Apply correlation changes
        if "correlation_changes" in scenario:
            # This is more complex - simplified implementation
            correlation_factor = scenario["correlation_changes"].get("increase_factor", 1.0)
            if correlation_factor != 1.0:
                # Convert to correlation matrix, modify, convert back
                corr_matrix = self._cov_to_corr(stressed_cov)
                
                # Increase off-diagonal correlations
                for i in range(len(corr_matrix)):
                    for j in range(len(corr_matrix)):
                        if i != j:
                            corr_matrix.iloc[i, j] *= correlation_factor
                            # Ensure correlations stay within [-1, 1]
                            corr_matrix.iloc[i, j] = np.clip(corr_matrix.iloc[i, j], -0.99, 0.99)
                
                # Convert back to covariance
                stressed_cov = self._corr_to_cov(corr_matrix, stressed_cov)
        
        return stressed_returns, stressed_cov
    
    def _compare_stress_scenarios(
        self,
        stress_results: Dict[str, Any],
        initial_investment: float
    ) -> Dict[str, Any]:
        """Compare results across stress scenarios."""
        
        scenario_comparison = {}
        
        for scenario_name, results in stress_results.items():
            if "error" not in results:
                final_stats = results["final_value_stats"]
                scenario_comparison[scenario_name] = {
                    "expected_loss": initial_investment - final_stats["mean"],
                    "worst_case_loss": initial_investment - final_stats["min"],
                    "probability_of_loss": results["probability_of_loss"],
                    "severity_ranking": 0  # Will be filled in below
                }
        
        # Rank scenarios by severity
        scenarios_by_loss = sorted(
            scenario_comparison.items(),
            key=lambda x: x[1]["expected_loss"],
            reverse=True
        )
        
        for i, (scenario_name, _) in enumerate(scenarios_by_loss):
            scenario_comparison[scenario_name]["severity_ranking"] = i + 1
        
        return scenario_comparison
    
    def _assess_overall_stress_resilience(self, stress_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall portfolio resilience to stress scenarios."""
        
        successful_scenarios = [r for r in stress_results.values() if "error" not in r]
        
        if not successful_scenarios:
            return {"assessment": "Unable to assess - all scenarios failed"}
        
        # Average metrics across scenarios
        avg_prob_loss = np.mean([r["probability_of_loss"] for r in successful_scenarios])
        max_worst_case = max([r["worst_case_loss"] for r in successful_scenarios])
        
        # Overall assessment
        if avg_prob_loss < 0.3 and max_worst_case < 0.4:
            resilience = "High"
        elif avg_prob_loss < 0.5 and max_worst_case < 0.6:
            resilience = "Medium"
        else:
            resilience = "Low"
        
        return {
            "overall_resilience": resilience,
            "average_probability_of_loss": float(avg_prob_loss),
            "maximum_worst_case_loss": float(max_worst_case),
            "number_of_scenarios_tested": len(successful_scenarios),
            "key_insight": f"Portföljen visar {resilience.lower()} motståndskraft mot stresscenarion"
        }
    
    def _assess_var_risk_level(self, var_value: float, initial_investment: float) -> str:
        """Assess risk level based on VaR."""
        
        var_pct = var_value / initial_investment
        
        if var_pct < 0.05:
            return "Låg risk"
        elif var_pct < 0.15:
            return "Medel risk"
        elif var_pct < 0.25:
            return "Hög risk"
        else:
            return "Mycket hög risk"
    
    def _assess_swedish_risk_level(
        self,
        final_values: np.ndarray,
        initial_investment: float
    ) -> str:
        """Assess risk level from Swedish investor perspective."""
        
        prob_loss = np.mean(final_values < initial_investment)
        max_loss_pct = (initial_investment - np.min(final_values)) / initial_investment
        
        if prob_loss < 0.2 and max_loss_pct < 0.3:
            return "Låg till medel risk - lämplig för svenska konservativa investerare"
        elif prob_loss < 0.4 and max_loss_pct < 0.5:
            return "Medel risk - balanserad för svenska förhållanden"
        elif prob_loss < 0.6 and max_loss_pct < 0.7:
            return "Hög risk - kräver hög risktolerans"
        else:
            return "Mycket hög risk - endast för erfarna investerare"
    
    def _generate_swedish_recommendations(
        self,
        final_values: np.ndarray,
        initial_investment: float
    ) -> List[str]:
        """Generate Swedish-specific investment recommendations."""
        
        recommendations = []
        
        prob_loss = np.mean(final_values < initial_investment)
        expected_return = (np.mean(final_values) - initial_investment) / initial_investment
        
        if prob_loss > 0.5:
            recommendations.append("Överväg att minska risken genom diversifiering")
        
        if expected_return < 0.05:
            recommendations.append("Förväntad avkastning är låg - överväg högre riskexponering")
        
        recommendations.append("Använd ISK-konto för skatteeffektivitet")
        recommendations.append("Överväg periodisk rebalansering för att hantera risk")
        
        return recommendations
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _cov_to_corr(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """Convert covariance matrix to correlation matrix."""
        std_devs = np.sqrt(np.diag(cov_matrix.values))
        corr_matrix = cov_matrix.copy()
        
        for i in range(len(std_devs)):
            for j in range(len(std_devs)):
                if std_devs[i] > 0 and std_devs[j] > 0:
                    corr_matrix.iloc[i, j] = cov_matrix.iloc[i, j] / (std_devs[i] * std_devs[j])
        
        return corr_matrix
    
    def _corr_to_cov(self, corr_matrix: pd.DataFrame, original_cov: pd.DataFrame) -> pd.DataFrame:
        """Convert correlation matrix back to covariance matrix."""
        std_devs = np.sqrt(np.diag(original_cov.values))
        cov_matrix = corr_matrix.copy()
        
        for i in range(len(std_devs)):
            for j in range(len(std_devs)):
                cov_matrix.iloc[i, j] = corr_matrix.iloc[i, j] * std_devs[i] * std_devs[j]
        
        return cov_matrix
    
    def _validate_simulation_inputs(
        self,
        allocation: Dict[str, float],
        returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> bool:
        """Validate Monte Carlo simulation inputs."""
        
        if not allocation or sum(allocation.values()) <= 0:
            return False
        
        if returns.empty or cov_matrix.empty:
            return False
        
        if len(returns) != len(cov_matrix):
            return False
        
        # Check for NaN or infinite values
        if returns.isna().any() or cov_matrix.isna().any().any():
            return False
        
        if np.isinf(returns.values).any() or np.isinf(cov_matrix.values).any():
            return False
        
        return True
    
    def _empty_result(self, error_message: str) -> Dict[str, Any]:
        """Return empty result with error message."""
        
        return {
            "success": False,
            "error": error_message,
            "simulation_type": "monte_carlo",
            "timestamp": datetime.now().isoformat()
        }