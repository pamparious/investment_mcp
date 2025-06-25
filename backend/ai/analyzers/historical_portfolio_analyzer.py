"""Analyze portfolio allocations using historical data and Swedish economic context."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import logging
import json
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)


class HistoricalPortfolioAnalyzer:
    """Analyze portfolio allocations using historical data and Swedish economic context."""
    
    def __init__(self, ai_provider):
        """Initialize the historical portfolio analyzer."""
        self.ai = ai_provider
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    async def analyze_optimal_allocation(self, 
                                       historical_data: Dict[str, pd.DataFrame],
                                       swedish_economic_data: Dict,
                                       risk_profile: str,
                                       investment_horizon: int = 10) -> Dict[str, Any]:
        """Determine optimal allocation using historical analysis and current Swedish conditions."""
        
        self.logger.info(f"Analyzing optimal allocation for {risk_profile} investor with {investment_horizon}y horizon")
        
        try:
            # 1. Calculate historical performance metrics
            performance_metrics = self.calculate_fund_performance_metrics(historical_data)
            
            # 2. Analyze correlations between funds
            correlation_matrix = self.calculate_correlation_matrix(historical_data)
            
            # 3. Swedish economic cycle analysis
            economic_phase = self.determine_economic_phase(swedish_economic_data)
            
            # 4. Historical regime analysis
            market_regimes = self.identify_market_regimes(historical_data, swedish_economic_data)
            
            # 5. Generate AI-powered allocation with historical context
            allocation = await self.generate_historically_informed_allocation(
                performance_metrics, correlation_matrix, economic_phase, 
                market_regimes, risk_profile, investment_horizon
            )
            
            # 6. Calculate expected portfolio metrics
            portfolio_metrics = self.calculate_portfolio_metrics(allocation["allocations"], performance_metrics, correlation_matrix)
            
            # 7. Stress test the allocation
            stress_test_results = self.stress_test_allocation(allocation["allocations"], historical_data)
            
            # Combine all results
            result = {
                **allocation,
                "portfolio_metrics": portfolio_metrics,
                "stress_test": stress_test_results,
                "performance_metrics": performance_metrics,
                "correlation_matrix": correlation_matrix.to_dict(),
                "economic_phase": economic_phase,
                "market_regimes": market_regimes,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in optimal allocation analysis: {e}")
            return {"error": str(e)}
    
    def calculate_fund_performance_metrics(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Calculate comprehensive performance metrics for each fund."""
        
        self.logger.info("Calculating fund performance metrics")
        metrics = {}
        
        for fund_code, data in historical_data.items():
            if data is None or len(data) < 252:  # Need at least 1 year
                self.logger.warning(f"Insufficient data for {fund_code}")
                continue
                
            try:
                returns = data['daily_return'].dropna()
                
                if len(returns) == 0:
                    continue
                
                # Basic performance metrics
                annual_return = returns.mean() * 252
                annual_volatility = returns.std() * np.sqrt(252)
                sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
                
                # Risk metrics
                max_drawdown = data['drawdown'].min() if 'drawdown' in data.columns else np.nan
                var_95 = returns.quantile(0.05)
                skewness = returns.skew()
                kurtosis = returns.kurtosis()
                
                # Advanced metrics
                calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
                
                metrics[fund_code] = {
                    "annual_return": float(annual_return),
                    "annual_volatility": float(annual_volatility),
                    "sharpe_ratio": float(sharpe_ratio),
                    "max_drawdown": float(max_drawdown) if not np.isnan(max_drawdown) else -0.20,
                    "var_95": float(var_95),
                    "skewness": float(skewness),
                    "kurtosis": float(kurtosis),
                    "calmar_ratio": float(calmar_ratio),
                    "total_return_10y": self.calculate_total_return(data, years=10),
                    "total_return_20y": self.calculate_total_return(data, years=20),
                    "worst_year": self.calculate_worst_year_return(data),
                    "best_year": self.calculate_best_year_return(data),
                    "positive_years_pct": self.calculate_positive_years_percentage(data),
                    "recession_performance": self.calculate_recession_performance(data),
                    "inflation_hedge_score": self.calculate_inflation_hedge_score(data),
                    "data_points": len(data),
                    "data_quality": data.iloc[0]['data_quality'] if 'data_quality' in data.columns else "medium"
                }
                
            except Exception as e:
                self.logger.error(f"Error calculating metrics for {fund_code}: {e}")
                continue
        
        self.logger.info(f"Calculated metrics for {len(metrics)} funds")
        return metrics
    
    def calculate_correlation_matrix(self, historical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation matrix between all funds."""
        
        self.logger.info("Calculating correlation matrix")
        
        # Align all data to common dates
        returns_data = {}
        
        for fund_code, data in historical_data.items():
            if data is not None and len(data) > 0 and 'daily_return' in data.columns:
                returns_data[fund_code] = data['daily_return']
        
        if not returns_data:
            self.logger.warning("No return data available for correlation calculation")
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(returns_data)
        
        # Remove rows where all values are NaN
        returns_df = returns_df.dropna(how='all')
        
        # Fill remaining NaN values with 0 (assuming missing = no change)
        returns_df = returns_df.fillna(0)
        
        correlation_matrix = returns_df.corr()
        
        self.logger.info(f"Calculated correlation matrix for {len(correlation_matrix)} funds")
        return correlation_matrix
    
    def determine_economic_phase(self, swedish_economic_data: Dict) -> str:
        """Determine current Swedish economic cycle phase."""
        
        try:
            # Extract key indicators
            interest_rates = swedish_economic_data.get('interest_rates', {})
            inflation = swedish_economic_data.get('inflation', {})
            gdp_growth = swedish_economic_data.get('gdp_growth', {})
            employment = swedish_economic_data.get('employment', {})
            
            # Analyze trends
            interest_rate_trend = self.analyze_interest_rate_trend(interest_rates)
            inflation_trend = self.analyze_inflation_trend(inflation)
            gdp_trend = self.analyze_gdp_trend(gdp_growth)
            employment_trend = self.analyze_employment_trend(employment)
            
            # Determine phase based on combination of indicators
            if gdp_trend > 0.02 and employment_trend > 0 and inflation_trend < 0.035:
                return "expansion"
            elif gdp_trend > 0.015 and inflation_trend > 0.035:
                return "late_cycle"
            elif gdp_trend < -0.005 and interest_rate_trend < 0:
                return "recession"
            elif gdp_trend < 0.005 and interest_rate_trend < 0:
                return "recovery"
            else:
                return "transition"
                
        except Exception as e:
            self.logger.error(f"Error determining economic phase: {e}")
            return "uncertain"
    
    def identify_market_regimes(self, historical_data: Dict, swedish_economic_data: Dict) -> Dict[str, Dict]:
        """Identify different market regimes and fund performance in each."""
        
        self.logger.info("Identifying market regimes")
        
        regimes = {
            "bull_market": {},
            "bear_market": {},
            "high_volatility": {},
            "low_volatility": {},
            "crisis": {}
        }
        
        # Analyze fund performance in different market conditions
        for fund_code, data in historical_data.items():
            if data is None or len(data) < 252:
                continue
                
            try:
                regimes["bull_market"][fund_code] = self.analyze_regime_performance(
                    data, regime_type="bull_market"
                )
                
                regimes["bear_market"][fund_code] = self.analyze_regime_performance(
                    data, regime_type="bear_market"
                )
                
                regimes["high_volatility"][fund_code] = self.analyze_regime_performance(
                    data, regime_type="high_volatility"
                )
                
                regimes["low_volatility"][fund_code] = self.analyze_regime_performance(
                    data, regime_type="low_volatility"
                )
                
                regimes["crisis"][fund_code] = self.analyze_regime_performance(
                    data, regime_type="crisis"
                )
                
            except Exception as e:
                self.logger.error(f"Error analyzing regimes for {fund_code}: {e}")
                continue
        
        return regimes
    
    def analyze_regime_performance(self, data: pd.DataFrame, regime_type: str) -> Dict[str, float]:
        """Analyze fund performance in specific market regime."""
        
        try:
            if regime_type == "bull_market":
                # Use bull_market column if available, otherwise use trend
                if 'bull_market' in data.columns:
                    regime_data = data[data['bull_market']]
                else:
                    regime_data = data[data['Close'] > data['sma_200']]
                    
            elif regime_type == "bear_market":
                if 'bear_market' in data.columns:
                    regime_data = data[data['bear_market']]
                else:
                    regime_data = data[data['Close'] < data['sma_200']]
                    
            elif regime_type == "high_volatility":
                vol_threshold = data['volatility_30d'].quantile(0.75)
                regime_data = data[data['volatility_30d'] > vol_threshold]
                
            elif regime_type == "low_volatility":
                vol_threshold = data['volatility_30d'].quantile(0.25)
                regime_data = data[data['volatility_30d'] < vol_threshold]
                
            elif regime_type == "crisis":
                # Define crisis as periods with very negative returns
                crisis_threshold = data['daily_return'].quantile(0.05)
                regime_data = data[data['daily_return'] < crisis_threshold]
                
            else:
                return {}
            
            if len(regime_data) == 0:
                return {"average_return": 0.0, "volatility": 0.0, "frequency": 0.0}
            
            regime_returns = regime_data['daily_return'].dropna()
            
            return {
                "average_return": float(regime_returns.mean() * 252),
                "volatility": float(regime_returns.std() * np.sqrt(252)),
                "frequency": float(len(regime_data) / len(data)),
                "max_drawdown": float(regime_data['drawdown'].min()) if 'drawdown' in regime_data.columns else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing {regime_type} performance: {e}")
            return {"average_return": 0.0, "volatility": 0.0, "frequency": 0.0}
    
    async def generate_historically_informed_allocation(self,
                                                      performance_metrics: Dict,
                                                      correlation_matrix: pd.DataFrame,
                                                      economic_phase: str,
                                                      market_regimes: Dict,
                                                      risk_profile: str,
                                                      investment_horizon: int) -> Dict[str, Any]:
        """Generate allocation using AI with comprehensive historical context."""
        
        self.logger.info(f"Generating AI allocation for {risk_profile} profile")
        
        # Prepare detailed historical context for AI
        historical_context = self.format_historical_context(
            performance_metrics, correlation_matrix, market_regimes
        )
        
        swedish_context = f"Current Swedish economic phase: {economic_phase}"
        
        # Get list of available funds with good data
        available_funds = list(performance_metrics.keys())
        
        prompt = f"""You are an expert portfolio manager with access to 20 years of historical data for Swedish tradeable funds.

AVAILABLE FUNDS (use ONLY these):
{available_funds}

HISTORICAL PERFORMANCE ANALYSIS:
{historical_context}

CURRENT SWEDISH ECONOMIC CONDITIONS:
{swedish_context}

INVESTOR PROFILE:
- Risk Profile: {risk_profile}
- Investment Horizon: {investment_horizon} years

ALLOCATION REQUIREMENTS:
1. Use ONLY the funds listed above
2. Allocations must sum to exactly 1.0 (100%)
3. Consider historical performance in different market regimes
4. Account for current Swedish economic phase: {economic_phase}
5. Optimize for risk-adjusted returns over {investment_horizon} years
6. Maximum 8 funds for practical implementation
7. Minimum allocation per fund: 5% (if included)

RISK PROFILE GUIDELINES:
- Conservative: Focus on downside protection, max 60% equities
- Balanced: Balanced growth and protection, max 80% equities  
- Aggressive: Growth focused, can use up to 100% equities

HISTORICAL INSIGHTS TO CONSIDER:
- Which funds performed best during Swedish economic slowdowns?
- Which funds have lowest correlation for diversification?
- Which funds are best inflation hedges given current inflation environment?
- How should allocation change based on current economic phase: {economic_phase}?

Return ONLY valid JSON format:
{{
    "allocations": {{
        "FUND_CODE": 0.XX,
        "FUND_CODE": 0.XX
    }},
    "historical_reasoning": "explanation based on 20-year analysis",
    "swedish_economic_rationale": "how current Swedish conditions influence allocation",
    "expected_annual_return": 0.XX,
    "expected_volatility": 0.XX,
    "worst_case_scenario": "1-year 95% VaR estimate",
    "regime_performance": "how portfolio performs in different economic regimes"
}}"""
        
        try:
            async with self.ai:
                ai_response = await self.ai._generate_completion(prompt)
            
            return self.parse_and_validate_allocation(ai_response, available_funds)
            
        except Exception as e:
            self.logger.error(f"Error generating AI allocation: {e}")
            return self.get_fallback_allocation(available_funds, risk_profile)
    
    def format_historical_context(self, performance_metrics: Dict, 
                                correlation_matrix: pd.DataFrame, 
                                market_regimes: Dict) -> str:
        """Format historical data for AI consumption."""
        
        context = "HISTORICAL PERFORMANCE (20-year analysis):\n\n"
        
        # Top performers by Sharpe ratio
        if performance_metrics:
            sorted_funds = sorted(performance_metrics.items(), 
                                key=lambda x: x[1].get('sharpe_ratio', 0), reverse=True)
            
            context += "Risk-Adjusted Returns (Sharpe Ratio):\n"
            for fund, metrics in sorted_funds[:8]:  # Top 8
                context += f"- {fund}: {metrics.get('sharpe_ratio', 0):.2f} Sharpe (Return: {metrics.get('annual_return', 0):.1%}, Vol: {metrics.get('annual_volatility', 0):.1%})\n"
            
            context += "\nDownside Protection (Max Drawdown):\n"
            best_drawdown = sorted(performance_metrics.items(), 
                                 key=lambda x: x[1].get('max_drawdown', -1), reverse=True)
            for fund, metrics in best_drawdown[:5]:
                context += f"- {fund}: {metrics.get('max_drawdown', 0):.1%} max drawdown\n"
        
        # Diversification benefits
        if not correlation_matrix.empty:
            context += "\nDiversification Analysis:\n"
            low_corr_pairs = self.find_low_correlation_pairs(correlation_matrix)
            for fund1, fund2, corr in low_corr_pairs[:3]:
                context += f"- {fund1} vs {fund2}: {corr:.2f} correlation (good diversification)\n"
        
        # Market regime performance
        context += "\nMarket Regime Performance:\n"
        if 'bear_market' in market_regimes:
            bear_performance = market_regimes['bear_market']
            if bear_performance:
                best_bear_performers = sorted(bear_performance.items(), 
                                            key=lambda x: x[1].get('average_return', -999), reverse=True)
                context += "Best Bear Market Performers:\n"
                for fund, perf in best_bear_performers[:3]:
                    context += f"- {fund}: {perf.get('average_return', 0):.1%} return during bear markets\n"
        
        return context
    
    def find_low_correlation_pairs(self, correlation_matrix: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Find pairs of funds with low correlation."""
        
        low_corr_pairs = []
        
        for i, fund1 in enumerate(correlation_matrix.index):
            for j, fund2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr = correlation_matrix.loc[fund1, fund2]
                    if not np.isnan(corr):
                        low_corr_pairs.append((fund1, fund2, corr))
        
        # Sort by correlation (lowest first)
        low_corr_pairs.sort(key=lambda x: x[2])
        
        return low_corr_pairs
    
    def parse_and_validate_allocation(self, ai_response: str, available_funds: List[str]) -> Dict[str, Any]:
        """Parse and validate AI response."""
        
        try:
            # Try to extract JSON from response
            if '{' in ai_response and '}' in ai_response:
                start = ai_response.find('{')
                end = ai_response.rfind('}') + 1
                json_str = ai_response[start:end]
                
                allocation_data = json.loads(json_str)
                
                # Validate allocations
                allocations = allocation_data.get('allocations', {})
                
                # Check if allocations are valid
                validated_allocations = {}
                total_allocation = 0
                
                for fund, weight in allocations.items():
                    if fund in available_funds and isinstance(weight, (int, float)) and weight > 0:
                        validated_allocations[fund] = float(weight)
                        total_allocation += weight
                
                # Normalize to sum to 1.0
                if total_allocation > 0 and abs(total_allocation - 1.0) > 0.01:
                    for fund in validated_allocations:
                        validated_allocations[fund] /= total_allocation
                
                # Ensure we have at least some allocation
                if not validated_allocations:
                    raise ValueError("No valid allocations found")
                
                allocation_data['allocations'] = validated_allocations
                return allocation_data
                
            else:
                raise ValueError("No JSON found in AI response")
                
        except Exception as e:
            self.logger.error(f"Error parsing AI response: {e}")
            return self.get_fallback_allocation(available_funds, "balanced")
    
    def get_fallback_allocation(self, available_funds: List[str], risk_profile: str) -> Dict[str, Any]:
        """Get fallback allocation if AI fails."""
        
        self.logger.info(f"Using fallback allocation for {risk_profile}")
        
        # Simple rule-based allocation
        if risk_profile == "conservative":
            # Conservative: favor bonds/gold, lower equity allocation
            allocation_weights = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]
        elif risk_profile == "aggressive":
            # Aggressive: favor growth assets
            allocation_weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
        else:  # balanced
            # Balanced allocation
            allocation_weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
        
        # Take up to 6 funds
        selected_funds = available_funds[:min(6, len(available_funds))]
        
        # Create allocation dictionary
        allocations = {}
        for i, fund in enumerate(selected_funds):
            if i < len(allocation_weights):
                allocations[fund] = allocation_weights[i]
        
        # Normalize
        total = sum(allocations.values())
        if total > 0:
            for fund in allocations:
                allocations[fund] /= total
        
        return {
            "allocations": allocations,
            "historical_reasoning": "Fallback rule-based allocation due to AI processing error",
            "swedish_economic_rationale": "Default balanced approach",
            "expected_annual_return": 0.07,
            "expected_volatility": 0.15,
            "worst_case_scenario": "-15% in worst case year",
            "regime_performance": "Moderate performance across different regimes"
        }
    
    def calculate_portfolio_metrics(self, allocations: Dict[str, float], 
                                  performance_metrics: Dict, 
                                  correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate expected portfolio metrics."""
        
        try:
            # Portfolio expected return
            expected_return = sum(
                allocations.get(fund, 0) * performance_metrics.get(fund, {}).get('annual_return', 0)
                for fund in allocations.keys()
            )
            
            # Portfolio volatility (using correlation matrix)
            portfolio_variance = 0
            for fund1, weight1 in allocations.items():
                for fund2, weight2 in allocations.items():
                    if fund1 in performance_metrics and fund2 in performance_metrics:
                        vol1 = performance_metrics[fund1].get('annual_volatility', 0)
                        vol2 = performance_metrics[fund2].get('annual_volatility', 0)
                        
                        if fund1 == fund2:
                            corr = 1.0
                        elif not correlation_matrix.empty and fund1 in correlation_matrix.index and fund2 in correlation_matrix.columns:
                            corr = correlation_matrix.loc[fund1, fund2]
                            if np.isnan(corr):
                                corr = 0.3  # Default correlation
                        else:
                            corr = 0.3  # Default correlation
                        
                        portfolio_variance += weight1 * weight2 * vol1 * vol2 * corr
            
            portfolio_volatility = np.sqrt(max(0, portfolio_variance))
            
            # Portfolio Sharpe ratio
            portfolio_sharpe = expected_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Portfolio max drawdown (weighted average)
            portfolio_max_drawdown = sum(
                allocations.get(fund, 0) * performance_metrics.get(fund, {}).get('max_drawdown', -0.2)
                for fund in allocations.keys()
            )
            
            return {
                "expected_return": float(expected_return),
                "expected_volatility": float(portfolio_volatility),
                "expected_sharpe": float(portfolio_sharpe),
                "expected_max_drawdown": float(portfolio_max_drawdown)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return {
                "expected_return": 0.07,
                "expected_volatility": 0.15,
                "expected_sharpe": 0.47,
                "expected_max_drawdown": -0.20
            }
    
    def stress_test_allocation(self, allocations: Dict[str, float], 
                             historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Stress test the allocation against historical scenarios."""
        
        try:
            stress_scenarios = {
                "2008_crisis": {"start": "2008-01-01", "end": "2009-03-31"},
                "covid_2020": {"start": "2020-02-01", "end": "2020-04-30"},
                "dotcom_2000": {"start": "2000-03-01", "end": "2002-10-31"}
            }
            
            results = {}
            
            for scenario, dates in stress_scenarios.items():
                try:
                    start_date = pd.to_datetime(dates["start"])
                    end_date = pd.to_datetime(dates["end"])
                    
                    portfolio_return = 0
                    portfolio_data_points = 0
                    
                    for fund, weight in allocations.items():
                        if fund in historical_data and historical_data[fund] is not None:
                            fund_data = historical_data[fund]
                            scenario_data = fund_data[(fund_data.index >= start_date) & (fund_data.index <= end_date)]
                            
                            if len(scenario_data) > 0:
                                # Calculate total return during scenario
                                start_price = scenario_data['Close'].iloc[0]
                                end_price = scenario_data['Close'].iloc[-1]
                                fund_return = (end_price / start_price) - 1
                                
                                portfolio_return += weight * fund_return
                                portfolio_data_points += len(scenario_data)
                    
                    results[scenario] = {
                        "portfolio_return": float(portfolio_return),
                        "data_coverage": portfolio_data_points > 0
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error in stress test scenario {scenario}: {e}")
                    results[scenario] = {"portfolio_return": -0.15, "data_coverage": False}
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in stress testing: {e}")
            return {}
    
    # Helper methods for economic analysis
    def analyze_interest_rate_trend(self, interest_data: Dict) -> float:
        """Analyze interest rate trend direction."""
        if not interest_data or "error" in interest_data:
            return 0.0
        return interest_data.get("rate_change_12m", 0.0) / 100.0
    
    def analyze_inflation_trend(self, inflation_data: Dict) -> float:
        """Analyze inflation trend."""
        if not inflation_data or "error" in inflation_data:
            return 0.02  # Default 2%
        return inflation_data.get("current_cpi", 2.0) / 100.0
    
    def analyze_gdp_trend(self, gdp_data: Dict) -> float:
        """Analyze GDP growth trend."""
        if not gdp_data or "error" in gdp_data:
            return 0.02  # Default 2%
        return gdp_data.get("gdp_growth_annual", 2.0) / 100.0
    
    def analyze_employment_trend(self, employment_data: Dict) -> float:
        """Analyze employment trend."""
        if not employment_data or "error" in employment_data:
            return 0.0
        # Convert unemployment rate to employment trend (lower unemployment = positive trend)
        unemployment_rate = employment_data.get("unemployment_rate", 7.0)
        return (8.0 - unemployment_rate) / 100.0  # Normalize around 8% baseline
    
    # Helper methods for historical calculations
    def calculate_total_return(self, data: pd.DataFrame, years: int) -> float:
        """Calculate total return over specified years."""
        try:
            if len(data) < years * 200:  # Not enough data (account for weekends)
                return np.nan
            
            end_price = data['Close'].iloc[-1]
            start_price = data['Close'].iloc[-(years * 200)]
            
            return float((end_price / start_price) - 1)
        except:
            return np.nan
    
    def calculate_worst_year_return(self, data: pd.DataFrame) -> float:
        """Calculate worst calendar year return."""
        try:
            if len(data) < 200:
                return np.nan
            
            yearly_returns = []
            current_year = data.index[-1].year
            
            for year in range(data.index[0].year, current_year + 1):
                year_data = data[data.index.year == year]
                if len(year_data) > 50:  # Sufficient data for the year
                    year_return = (year_data['Close'].iloc[-1] / year_data['Close'].iloc[0]) - 1
                    yearly_returns.append(year_return)
            
            return float(min(yearly_returns)) if yearly_returns else np.nan
        except:
            return np.nan
    
    def calculate_best_year_return(self, data: pd.DataFrame) -> float:
        """Calculate best calendar year return."""
        try:
            if len(data) < 200:
                return np.nan
            
            yearly_returns = []
            current_year = data.index[-1].year
            
            for year in range(data.index[0].year, current_year + 1):
                year_data = data[data.index.year == year]
                if len(year_data) > 50:
                    year_return = (year_data['Close'].iloc[-1] / year_data['Close'].iloc[0]) - 1
                    yearly_returns.append(year_return)
            
            return float(max(yearly_returns)) if yearly_returns else np.nan
        except:
            return np.nan
    
    def calculate_positive_years_percentage(self, data: pd.DataFrame) -> float:
        """Calculate percentage of positive years."""
        try:
            if len(data) < 200:
                return np.nan
            
            yearly_returns = []
            current_year = data.index[-1].year
            
            for year in range(data.index[0].year, current_year + 1):
                year_data = data[data.index.year == year]
                if len(year_data) > 50:
                    year_return = (year_data['Close'].iloc[-1] / year_data['Close'].iloc[0]) - 1
                    yearly_returns.append(year_return)
            
            if not yearly_returns:
                return np.nan
            
            positive_years = sum(1 for ret in yearly_returns if ret > 0)
            return float(positive_years / len(yearly_returns))
        except:
            return np.nan
    
    def calculate_recession_performance(self, data: pd.DataFrame) -> float:
        """Calculate average performance during recession periods."""
        try:
            # Use bear market periods as recession proxy
            if 'bear_market' in data.columns:
                recession_returns = data[data['bear_market']]['daily_return']
            else:
                # Alternative: use periods when price is below 200-day SMA
                recession_returns = data[data['Close'] < data['sma_200']]['daily_return']
            
            return float(recession_returns.mean() * 252) if len(recession_returns) > 0 else np.nan
        except:
            return np.nan
    
    def calculate_inflation_hedge_score(self, data: pd.DataFrame) -> float:
        """Calculate a simple inflation hedge score."""
        try:
            if len(data) < 252:
                return np.nan
            
            # Use rolling correlation with time trend as inflation proxy
            trend = np.arange(len(data))
            returns = data['daily_return'].dropna()
            
            if len(returns) < 100:
                return np.nan
            
            correlation = np.corrcoef(trend[-len(returns):], returns)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0