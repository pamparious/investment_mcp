"""Analysis MCP Agent for AI-powered investment analysis."""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import argparse

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.settings import Settings
from backend.database import get_db_session
from backend.ai.config import AIConfig
from backend.ai.analyzers import MarketAnalyzer, EconomicAnalyzer
from backend.ai.analyzers.constrained_portfolio_analyzer import ConstrainedPortfolioAnalyzer
from backend.data_collectors.swedish_economic_collector import SwedishEconomicCollector
from backend.data_collectors.historical_fund_collector import HistoricalFundCollector
from backend.ai.analyzers.historical_portfolio_analyzer import HistoricalPortfolioAnalyzer

logger = logging.getLogger(__name__)


class AnalysisAgent:
    """MCP Agent for performing AI-powered investment analysis."""
    
    def __init__(self, settings: Settings):
        """
        Initialize the analysis agent.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.ai_config = AIConfig(settings)
        
        # Initialize analyzers
        self.market_analyzer = MarketAnalyzer(self.ai_config)
        self.economic_analyzer = EconomicAnalyzer(self.ai_config)
        self.portfolio_analyzer = ConstrainedPortfolioAnalyzer(self.ai_config)
        
        # Initialize Phase 3 components
        self.swedish_collector = SwedishEconomicCollector()
        self.historical_collector = HistoricalFundCollector()
        self.historical_analyzer = HistoricalPortfolioAnalyzer(self.ai_config.get_provider())
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Cache for historical data
        self._historical_data_cache = None
        self._cache_timestamp = None
    
    async def test_ai_providers(self) -> Dict[str, Any]:
        """
        Test availability of AI providers.
        
        Returns:
            Provider availability test results
        """
        try:
            self.logger.info("Testing AI provider availability")
            
            # Get list of available providers
            available_providers = self.ai_config.list_available_providers()
            
            # Test each provider
            test_results = {}
            for provider_name, is_installed in available_providers.items():
                if is_installed:
                    test_result = await self.ai_config.test_provider(provider_name)
                    test_results[provider_name] = test_result
                else:
                    test_results[provider_name] = {
                        "provider": provider_name,
                        "available": False,
                        "error": "Provider library not installed"
                    }
            
            return {
                "test_timestamp": datetime.now().isoformat(),
                "providers_tested": list(test_results.keys()),
                "test_results": test_results,
                "default_provider": getattr(self.settings, 'AI_PROVIDER', 'ollama'),
                "recommendations": self._generate_provider_recommendations(test_results)
            }
            
        except Exception as e:
            self.logger.error(f"Error testing AI providers: {e}")
            return {"error": str(e)}
    
    def _generate_provider_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on provider test results."""
        recommendations = []
        
        available_providers = [name for name, result in test_results.items() if result.get("available", False)]
        
        if not available_providers:
            recommendations.append("Install at least one AI provider library")
            recommendations.append("Start with: pip install aiohttp (for Ollama)")
        
        if "ollama" in available_providers:
            recommendations.append("Ollama is available - good for local, private analysis")
        
        if "openai" in available_providers:
            recommendations.append("OpenAI is available - excellent for advanced analysis")
        
        if "claude" in available_providers:
            recommendations.append("Claude is available - great for detailed financial insights")
        
        if len(available_providers) > 1:
            recommendations.append("Multiple providers available - consider using different providers for different analysis types")
        
        return recommendations
    
    async def generate_comprehensive_investment_recommendation(self, 
                                                             risk_profile: str, 
                                                             amount: float,
                                                             investment_horizon: int = 10) -> Dict[str, Any]:
        """Generate investment recommendation using comprehensive historical and Swedish data."""
        
        self.logger.info(f"Generating comprehensive recommendation for {amount:,.0f} SEK, {risk_profile} risk, {investment_horizon}y horizon")
        
        try:
            # 1. Collect current Swedish economic data
            self.logger.info("Collecting Swedish economic data")
            swedish_data = await self.swedish_collector.get_comprehensive_economic_data()
            
            # 2. Get historical fund data (cached daily)
            self.logger.info("Loading historical fund data")
            historical_data = await self.get_or_update_historical_data()
            
            # 3. Generate historically-informed allocation
            self.logger.info("Generating AI-powered allocation with historical context")
            allocation_analysis = await self.historical_analyzer.analyze_optimal_allocation(
                historical_data, swedish_data, risk_profile, investment_horizon
            )
            
            # 4. Create actionable investment plan
            investment_plan = self.create_investment_plan(allocation_analysis, amount)
            
            # 5. Generate comprehensive report
            report = await self.generate_investment_report(
                allocation_analysis, investment_plan, swedish_data, historical_data
            )
            
            return {
                "allocation": allocation_analysis.get("allocations", {}),
                "investment_plan": investment_plan,
                "comprehensive_report": report,
                "swedish_economic_context": swedish_data,
                "historical_analysis": allocation_analysis,
                "recommendation_confidence": self.calculate_confidence_score(allocation_analysis),
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive recommendation: {e}")
            return {"error": str(e)}
    
    async def get_or_update_historical_data(self) -> Dict[str, Any]:
        """Get historical data, update if cache is old."""
        
        # Check if cache exists and is recent (update daily)
        cache_age_hours = 24
        
        if (self._historical_data_cache is not None and 
            self._cache_timestamp is not None and 
            (datetime.now() - self._cache_timestamp).total_seconds() < cache_age_hours * 3600):
            
            self.logger.info("Using cached historical data")
            return self._historical_data_cache
        
        else:
            # Collect fresh historical data
            self.logger.info("Collecting fresh historical data (this may take a few minutes)")
            historical_data = await self.historical_collector.collect_all_historical_data(years_back=20)
            
            # Update cache
            self._historical_data_cache = historical_data
            self._cache_timestamp = datetime.now()
            
            self.logger.info(f"Updated historical data cache with {len([d for d in historical_data.values() if d is not None])} funds")
            return historical_data
    
    def create_investment_plan(self, allocation_analysis: Dict[str, Any], amount: float) -> Dict[str, Any]:
        """Create actionable investment plan."""
        
        allocations = allocation_analysis.get("allocations", {})
        
        # Calculate fund amounts
        fund_amounts = {}
        for fund, percentage in allocations.items():
            fund_amounts[fund] = amount * percentage
        
        # Create rebalancing schedule
        rebalancing_schedule = self.suggest_rebalancing_schedule(allocation_analysis)
        
        # Create action items
        action_items = [
            "Open investment accounts if not already available",
            "Transfer funds to investment account",
            f"Execute initial allocation across {len(allocations)} funds",
            f"Set up automatic rebalancing every {rebalancing_schedule.get('frequency', 'quarter')}",
            "Monitor Swedish economic conditions for allocation adjustments"
        ]
        
        # Calculate fees and costs
        estimated_annual_fees = sum(
            amount * 0.005  # Assume 0.5% average annual fee
            for amount in fund_amounts.values()
        )
        
        return {
            "fund_amounts": fund_amounts,
            "total_amount": amount,
            "estimated_annual_fees": estimated_annual_fees,
            "rebalancing_schedule": rebalancing_schedule,
            "action_items": action_items,
            "implementation_timeline": "1-2 weeks for full implementation",
            "minimum_rebalancing_amount": max(1000, amount * 0.01)  # 1% or 1000 SEK minimum
        }
    
    def suggest_rebalancing_schedule(self, allocation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest rebalancing schedule based on allocation and volatility."""
        
        portfolio_metrics = allocation_analysis.get("portfolio_metrics", {})
        expected_volatility = portfolio_metrics.get("expected_volatility", 0.15)
        
        # Higher volatility portfolios need more frequent rebalancing
        if expected_volatility > 0.20:
            frequency = "monthly"
            threshold = 0.05  # 5% deviation
        elif expected_volatility > 0.15:
            frequency = "quarterly"
            threshold = 0.10  # 10% deviation
        else:
            frequency = "semi-annually"
            threshold = 0.15  # 15% deviation
        
        return {
            "frequency": frequency,
            "deviation_threshold": threshold,
            "next_review_date": (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d"),
            "rationale": f"Based on portfolio volatility of {expected_volatility:.1%}"
        }
    
    async def generate_investment_report(self, allocation_analysis: Dict[str, Any], 
                                       investment_plan: Dict[str, Any],
                                       swedish_data: Dict[str, Any],
                                       historical_data: Dict[str, Any]) -> str:
        """Generate comprehensive investment report."""
        
        try:
            provider = self.ai_config.get_provider()
            
            # Prepare context for AI report generation
            context = {
                "allocation": allocation_analysis.get("allocations", {}),
                "portfolio_metrics": allocation_analysis.get("portfolio_metrics", {}),
                "historical_reasoning": allocation_analysis.get("historical_reasoning", ""),
                "swedish_rationale": allocation_analysis.get("swedish_economic_rationale", ""),
                "investment_plan": investment_plan,
                "economic_phase": swedish_data.get("economic_cycle_phase", "uncertain"),
                "stress_test": allocation_analysis.get("stress_test", {}),
                "fund_count": len(historical_data),
                "data_quality": sum(1 for d in historical_data.values() if d is not None)
            }
            
            prompt = f"""Create a comprehensive investment report based on 20 years of historical analysis and current Swedish economic conditions.

PORTFOLIO ALLOCATION:
{json.dumps(context['allocation'], indent=2)}

EXPECTED PERFORMANCE:
- Annual Return: {context['portfolio_metrics'].get('expected_return', 0.07):.1%}
- Volatility: {context['portfolio_metrics'].get('expected_volatility', 0.15):.1%}
- Sharpe Ratio: {context['portfolio_metrics'].get('expected_sharpe', 0.47):.2f}

HISTORICAL ANALYSIS:
{context['historical_reasoning']}

SWEDISH ECONOMIC CONDITIONS:
Phase: {context['economic_phase']}
{context['swedish_rationale']}

STRESS TEST RESULTS:
{json.dumps(context['stress_test'], indent=2)}

Create a professional report with these sections:
1. Executive Summary (2-3 paragraphs)
2. Portfolio Allocation Rationale (why these funds)
3. Expected Performance & Risk Assessment
4. Swedish Economic Context Impact
5. Historical Performance Review
6. Implementation Plan
7. Monitoring & Rebalancing Strategy

Write in clear, professional Swedish-friendly language. Focus on practical implementation."""

            async with provider:
                report = await provider._generate_completion(prompt)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating investment report: {e}")
            return f"""Investment Report Generation Error

An error occurred while generating the comprehensive report: {str(e)}

Key Information:
- Portfolio Allocation: {len(allocation_analysis.get('allocations', {}))} funds
- Expected Annual Return: {allocation_analysis.get('portfolio_metrics', {}).get('expected_return', 0.07):.1%}
- Expected Volatility: {allocation_analysis.get('portfolio_metrics', {}).get('expected_volatility', 0.15):.1%}
- Swedish Economic Phase: {swedish_data.get('economic_cycle_phase', 'uncertain')}

Please contact support for a detailed analysis."""
    
    def calculate_confidence_score(self, allocation_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the recommendation."""
        
        try:
            confidence_factors = []
            
            # Data quality factor
            portfolio_metrics = allocation_analysis.get("portfolio_metrics", {})
            if portfolio_metrics:
                confidence_factors.append(0.8)  # Good metrics available
            
            # Number of funds factor (diversification)
            num_funds = len(allocation_analysis.get("allocations", {}))
            if num_funds >= 4:
                confidence_factors.append(0.9)
            elif num_funds >= 2:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
            
            # Historical data quality
            stress_test = allocation_analysis.get("stress_test", {})
            if stress_test and len(stress_test) >= 2:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.6)
            
            # AI response quality
            if allocation_analysis.get("historical_reasoning") and allocation_analysis.get("swedish_economic_rationale"):
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.6)
            
            # Calculate average confidence
            confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
            
            return round(confidence, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    async def analyze_fund_historical_performance(self, fund_codes: List[str], comparison_period: str = "10y") -> Dict[str, Any]:
        """Analyze historical performance of specific funds."""
        
        self.logger.info(f"Analyzing historical performance for {len(fund_codes)} funds over {comparison_period}")
        
        try:
            # Get historical data
            historical_data = await self.get_or_update_historical_data()
            
            # Filter to requested funds
            fund_data = {code: data for code, data in historical_data.items() 
                        if code in fund_codes and data is not None}
            
            if not fund_data:
                return {"error": "No data available for requested funds"}
            
            # Calculate performance metrics
            performance_analysis = {}
            
            for fund_code, data in fund_data.items():
                try:
                    returns = data['daily_return'].dropna()
                    
                    # Basic performance
                    annual_return = returns.mean() * 252
                    annual_volatility = returns.std() * np.sqrt(252)
                    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
                    
                    # Period-specific returns
                    if comparison_period == "5y" and len(data) >= 5 * 252:
                        period_data = data.tail(5 * 252)
                    elif comparison_period == "10y" and len(data) >= 10 * 252:
                        period_data = data.tail(10 * 252)
                    elif comparison_period == "20y":
                        period_data = data
                    else:
                        period_data = data
                    
                    period_returns = period_data['daily_return'].dropna()
                    total_return = (period_data['Close'].iloc[-1] / period_data['Close'].iloc[0]) - 1
                    
                    performance_analysis[fund_code] = {
                        "annual_return": float(annual_return),
                        "annual_volatility": float(annual_volatility),
                        "sharpe_ratio": float(sharpe_ratio),
                        "total_return": float(total_return),
                        "max_drawdown": float(period_data['drawdown'].min()) if 'drawdown' in period_data.columns else None,
                        "best_year": self.calculate_best_year_return(period_data),
                        "worst_year": self.calculate_worst_year_return(period_data),
                        "data_points": len(period_data),
                        "period_analyzed": comparison_period
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing {fund_code}: {e}")
                    performance_analysis[fund_code] = {"error": str(e)}
            
            return {
                "performance_analysis": performance_analysis,
                "comparison_period": comparison_period,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "summary": self.generate_performance_summary(performance_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Error in fund performance analysis: {e}")
            return {"error": str(e)}
    
    def generate_performance_summary(self, performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of fund performance analysis."""
        
        valid_funds = {k: v for k, v in performance_analysis.items() if "error" not in v}
        
        if not valid_funds:
            return {"message": "No valid fund data for summary"}
        
        # Find best and worst performers
        best_return_fund = max(valid_funds.items(), key=lambda x: x[1].get("annual_return", -999))
        worst_return_fund = min(valid_funds.items(), key=lambda x: x[1].get("annual_return", 999))
        
        best_sharpe_fund = max(valid_funds.items(), key=lambda x: x[1].get("sharpe_ratio", -999))
        lowest_vol_fund = min(valid_funds.items(), key=lambda x: x[1].get("annual_volatility", 999))
        
        return {
            "total_funds_analyzed": len(valid_funds),
            "best_return": {
                "fund": best_return_fund[0],
                "annual_return": best_return_fund[1].get("annual_return", 0)
            },
            "worst_return": {
                "fund": worst_return_fund[0],
                "annual_return": worst_return_fund[1].get("annual_return", 0)
            },
            "best_risk_adjusted": {
                "fund": best_sharpe_fund[0],
                "sharpe_ratio": best_sharpe_fund[1].get("sharpe_ratio", 0)
            },
            "lowest_volatility": {
                "fund": lowest_vol_fund[0],
                "volatility": lowest_vol_fund[1].get("annual_volatility", 0)
            },
            "average_return": sum(v.get("annual_return", 0) for v in valid_funds.values()) / len(valid_funds),
            "average_volatility": sum(v.get("annual_volatility", 0) for v in valid_funds.values()) / len(valid_funds)
        }
    
    def calculate_best_year_return(self, data) -> float:
        """Calculate best calendar year return."""
        try:
            yearly_returns = []
            current_year = data.index[-1].year
            
            for year in range(data.index[0].year, current_year + 1):
                year_data = data[data.index.year == year]
                if len(year_data) > 50:
                    year_return = (year_data['Close'].iloc[-1] / year_data['Close'].iloc[0]) - 1
                    yearly_returns.append(year_return)
            
            return float(max(yearly_returns)) if yearly_returns else 0.0
        except:
            return 0.0
    
    def calculate_worst_year_return(self, data) -> float:
        """Calculate worst calendar year return."""
        try:
            yearly_returns = []
            current_year = data.index[-1].year
            
            for year in range(data.index[0].year, current_year + 1):
                year_data = data[data.index.year == year]
                if len(year_data) > 50:
                    year_return = (year_data['Close'].iloc[-1] / year_data['Close'].iloc[0]) - 1
                    yearly_returns.append(year_return)
            
            return float(min(yearly_returns)) if yearly_returns else 0.0
        except:
            return 0.0


async def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Investment Analysis MCP Agent")
    parser.add_argument("--test-providers", action="store_true", help="Test AI provider availability")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize settings and agent
        settings = Settings()
        agent = AnalysisAgent(settings)
        
        print("Starting Investment Analysis Agent")
        
        if args.test_providers:
            print("Testing AI Providers...")
            results = await agent.test_ai_providers()
            print("Provider test results:")
            
            for provider, result in results.get("test_results", {}).items():
                status = "Available" if result.get("available", False) else "Not Available"
                error = result.get("error", "")
                print(f"  {provider}: {status}")
                if error:
                    print(f"    Error: {error}")
            
            print("\nRecommendations:")
            for rec in results.get("recommendations", []):
                print(f"  - {rec}")
            
            return
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        logger.error(f"Analysis agent error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())