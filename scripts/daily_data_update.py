#!/usr/bin/env python3
"""Daily data update pipeline for Investment MCP system."""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import argparse
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from backend.database import init_database, get_db_session
from backend.data_collectors.swedish_economic_collector import SwedishEconomicCollector
from backend.data_collectors.historical_fund_collector import HistoricalFundCollector
from backend.mcp_agents.analysis_agent import AnalysisAgent
from backend.ai.config import AIConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/daily_update.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DailyDataPipeline:
    """Daily data update pipeline for Investment MCP system."""
    
    def __init__(self, settings: Settings):
        """Initialize the daily data pipeline."""
        self.settings = settings
        self.swedish_collector = SwedishEconomicCollector()
        self.fund_collector = HistoricalFundCollector()
        self.analysis_agent = AnalysisAgent(settings)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
    async def run_daily_update(self, force_full_update: bool = False) -> Dict[str, Any]:
        """Run the complete daily data update pipeline."""
        
        start_time = datetime.now()
        logger.info("Starting daily data update pipeline")
        
        results = {
            "pipeline_start": start_time.isoformat(),
            "steps_completed": [],
            "errors": [],
            "summary": {}
        }
        
        try:
            # Step 1: Update Swedish economic data
            logger.info("Step 1: Updating Swedish economic data")
            economic_result = await self.update_swedish_economic_data()
            results["steps_completed"].append("swedish_economic_data")
            results["swedish_economic_update"] = economic_result
            
            # Step 2: Update recent fund prices
            logger.info("Step 2: Updating recent fund prices")
            fund_result = await self.update_recent_fund_prices()
            results["steps_completed"].append("recent_fund_prices")
            results["fund_price_update"] = fund_result
            
            # Step 3: Update analysis cache
            logger.info("Step 3: Updating analysis cache")
            cache_result = await self.update_analysis_cache(force_full_update)
            results["steps_completed"].append("analysis_cache")
            results["cache_update"] = cache_result
            
            # Step 4: Generate daily market insight
            logger.info("Step 4: Generating daily market insight")
            insight_result = await self.generate_daily_market_insight()
            results["steps_completed"].append("daily_market_insight")
            results["market_insight"] = insight_result
            
            # Calculate total runtime
            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds()
            
            results["pipeline_end"] = end_time.isoformat()
            results["runtime_seconds"] = runtime
            results["success"] = True
            
            logger.info(f"Daily update pipeline completed successfully in {runtime:.1f} seconds")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results["success"] = False
            results["pipeline_error"] = str(e)
            results["errors"].append(f"Pipeline failure: {e}")
        
        # Generate summary
        results["summary"] = self.generate_pipeline_summary(results)
        
        # Save results to file
        await self.save_pipeline_results(results)
        
        return results
    
    async def update_swedish_economic_data(self) -> Dict[str, Any]:
        """Update Swedish economic data."""
        
        try:
            logger.info("Collecting fresh Swedish economic data")
            economic_data = await self.swedish_collector.get_comprehensive_economic_data()
            
            # Store in database if available
            try:
                async with get_db_session() as session:
                    # In a real implementation, save to database
                    pass
            except Exception as db_error:
                logger.warning(f"Database storage failed: {db_error}")
            
            return {
                "success": True,
                "data_points": len([k for k, v in economic_data.items() if not isinstance(v, str) and "error" not in str(v)]),
                "economic_phase": economic_data.get("economic_cycle_phase", "unknown"),
                "collection_time": datetime.now().isoformat(),
                "key_indicators": {
                    "repo_rate": economic_data.get("interest_rates", {}).get("current_repo_rate"),
                    "inflation": economic_data.get("inflation", {}).get("current_cpi"),
                    "gdp_growth": economic_data.get("gdp_growth", {}).get("gdp_growth_annual"),
                    "unemployment": economic_data.get("employment", {}).get("unemployment_rate")
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to update Swedish economic data: {e}")
            return {"success": False, "error": str(e)}
    
    async def update_recent_fund_prices(self, days_back: int = 30) -> Dict[str, Any]:
        """Update recent fund prices for the last 30 days."""
        
        try:
            logger.info(f"Collecting recent {days_back} days of fund data")
            recent_data = await self.fund_collector.collect_recent_data(days=days_back)
            
            successful_updates = 0
            failed_updates = 0
            fund_summary = {}
            
            for fund_code, data in recent_data.items():
                if data is not None and len(data) > 0:
                    successful_updates += 1
                    
                    # Calculate recent performance
                    if len(data) >= 2:
                        latest_price = data['Close'].iloc[-1]
                        prev_price = data['Close'].iloc[-2]
                        daily_change = (latest_price / prev_price) - 1
                        
                        month_change = None
                        if len(data) >= 20:  # Approximately 1 month
                            month_start_price = data['Close'].iloc[-20]
                            month_change = (latest_price / month_start_price) - 1
                    else:
                        daily_change = 0.0
                        month_change = None
                    
                    fund_summary[fund_code] = {
                        "latest_price": float(latest_price) if 'latest_price' in locals() else None,
                        "daily_change": float(daily_change),
                        "month_change": float(month_change) if month_change is not None else None,
                        "data_points": len(data),
                        "last_update": data.index[-1].isoformat() if len(data) > 0 else None
                    }
                else:
                    failed_updates += 1
                    fund_summary[fund_code] = {"error": "No data available"}
            
            return {
                "success": True,
                "successful_updates": successful_updates,
                "failed_updates": failed_updates,
                "total_funds": len(recent_data),
                "fund_summary": fund_summary,
                "update_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to update recent fund prices: {e}")
            return {"success": False, "error": str(e)}
    
    async def update_analysis_cache(self, force_full_update: bool = False) -> Dict[str, Any]:
        """Update analysis cache with fresh data."""
        
        try:
            logger.info("Updating analysis cache")
            
            # Force cache refresh by clearing existing cache
            if force_full_update:
                logger.info("Forcing full cache update")
                self.analysis_agent._historical_data_cache = None
                self.analysis_agent._cache_timestamp = None
            
            # Trigger cache update by requesting fresh historical data
            cache_start = datetime.now()
            historical_data = await self.analysis_agent.get_or_update_historical_data()
            cache_time = (datetime.now() - cache_start).total_seconds()
            
            # Count successful data loads
            successful_funds = len([k for k, v in historical_data.items() if v is not None])
            total_funds = len(historical_data)
            
            # Calculate cache statistics
            total_data_points = 0
            for fund_data in historical_data.values():
                if fund_data is not None:
                    total_data_points += len(fund_data)
            
            return {
                "success": True,
                "cache_update_time": cache_time,
                "successful_funds": successful_funds,
                "total_funds": total_funds,
                "total_data_points": total_data_points,
                "cache_timestamp": datetime.now().isoformat(),
                "forced_update": force_full_update
            }
            
        except Exception as e:
            logger.error(f"Failed to update analysis cache: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_daily_market_insight(self) -> Dict[str, Any]:
        """Generate daily market insight report."""
        
        try:
            logger.info("Generating daily market insight")
            
            # Get current Swedish economic data
            economic_data = await self.swedish_collector.get_comprehensive_economic_data()
            
            # Get recent fund performance data
            recent_fund_data = await self.fund_collector.collect_recent_data(days=5)
            
            # Calculate market sentiment indicators
            market_sentiment = self.calculate_market_sentiment(recent_fund_data)
            
            # Generate AI-powered insight
            insight_text = await self.generate_ai_market_insight(economic_data, market_sentiment)
            
            # Create actionable recommendations
            recommendations = self.generate_daily_recommendations(economic_data, market_sentiment)
            
            insight = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "market_sentiment": market_sentiment,
                "economic_snapshot": {
                    "economic_phase": economic_data.get("economic_cycle_phase", "unknown"),
                    "key_rates": {
                        "repo_rate": economic_data.get("interest_rates", {}).get("current_repo_rate"),
                        "inflation": economic_data.get("inflation", {}).get("current_cpi")
                    }
                },
                "insight_summary": insight_text,
                "daily_recommendations": recommendations,
                "generated_at": datetime.now().isoformat()
            }
            
            # Save insight to file
            await self.save_daily_insight(insight)
            
            return {
                "success": True,
                "insight": insight,
                "insight_length": len(insight_text),
                "recommendations_count": len(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate daily market insight: {e}")
            return {"success": False, "error": str(e)}
    
    def calculate_market_sentiment(self, recent_fund_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall market sentiment from recent fund performance."""
        
        daily_changes = []
        volatilities = []
        
        for fund_code, data in recent_fund_data.items():
            if data is not None and len(data) >= 2:
                # Calculate daily change
                latest = data['Close'].iloc[-1]
                previous = data['Close'].iloc[-2]
                daily_change = (latest / previous) - 1
                daily_changes.append(daily_change)
                
                # Calculate recent volatility
                if len(data) >= 5:
                    returns = data['daily_return'].tail(5).dropna()
                    if len(returns) > 1:
                        volatility = returns.std()
                        volatilities.append(volatility)
        
        if daily_changes:
            avg_daily_change = sum(daily_changes) / len(daily_changes)
            positive_funds = sum(1 for change in daily_changes if change > 0)
            positive_ratio = positive_funds / len(daily_changes)
        else:
            avg_daily_change = 0.0
            positive_ratio = 0.5
        
        if volatilities:
            avg_volatility = sum(volatilities) / len(volatilities)
        else:
            avg_volatility = 0.02
        
        # Determine sentiment
        if avg_daily_change > 0.01 and positive_ratio > 0.7:
            sentiment = "bullish"
        elif avg_daily_change < -0.01 and positive_ratio < 0.3:
            sentiment = "bearish"
        elif avg_volatility > 0.03:
            sentiment = "volatile"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "average_daily_change": float(avg_daily_change),
            "positive_funds_ratio": float(positive_ratio),
            "average_volatility": float(avg_volatility),
            "funds_analyzed": len(daily_changes)
        }
    
    async def generate_ai_market_insight(self, economic_data: Dict, market_sentiment: Dict) -> str:
        """Generate AI-powered market insight text."""
        
        try:
            ai_config = AIConfig(self.settings)
            provider = ai_config.get_provider()
            
            prompt = f"""Generate a concise daily market insight for Swedish investors based on current conditions.

ECONOMIC CONDITIONS:
- Economic Phase: {economic_data.get('economic_cycle_phase', 'unknown')}
- Repo Rate: {economic_data.get('interest_rates', {}).get('current_repo_rate', 'N/A')}%
- Inflation: {economic_data.get('inflation', {}).get('current_cpi', 'N/A')}%
- GDP Growth: {economic_data.get('gdp_growth', {}).get('gdp_growth_annual', 'N/A')}%

MARKET SENTIMENT:
- Overall Sentiment: {market_sentiment.get('sentiment', 'neutral')}
- Average Daily Change: {market_sentiment.get('average_daily_change', 0):.2%}
- Positive Funds Ratio: {market_sentiment.get('positive_funds_ratio', 0.5):.1%}

Generate 2-3 paragraphs covering:
1. Current market conditions summary
2. Key economic factors impacting Swedish investments
3. Brief outlook for the coming days

Keep it professional, actionable, and focused on Swedish market context."""

            async with provider:
                insight = await provider._generate_completion(prompt)
            
            return insight
            
        except Exception as e:
            logger.error(f"Failed to generate AI insight: {e}")
            return f"""Daily Market Update - {datetime.now().strftime('%Y-%m-%d')}

Market sentiment is currently {market_sentiment.get('sentiment', 'neutral')} with Swedish funds showing mixed performance. 

Economic conditions remain influenced by current monetary policy and global developments. Investors should monitor Swedish economic indicators and maintain diversified portfolios.

For detailed analysis, please run a comprehensive investment recommendation."""
    
    def generate_daily_recommendations(self, economic_data: Dict, market_sentiment: Dict) -> List[str]:
        """Generate actionable daily recommendations."""
        
        recommendations = []
        
        # Economic phase based recommendations
        economic_phase = economic_data.get('economic_cycle_phase', 'unknown')
        if economic_phase == "late_cycle_slowdown":
            recommendations.append("Consider defensive positioning as economy shows late-cycle characteristics")
            recommendations.append("Monitor Swedish housing market developments closely")
        elif economic_phase == "expansion":
            recommendations.append("Growth-oriented funds may benefit from economic expansion")
        elif economic_phase == "recession":
            recommendations.append("Focus on quality defensive funds and bonds")
        
        # Market sentiment recommendations
        sentiment = market_sentiment.get('sentiment', 'neutral')
        if sentiment == "volatile":
            recommendations.append("High volatility detected - consider reducing position sizes")
        elif sentiment == "bearish":
            recommendations.append("Bear market conditions - evaluate downside protection strategies")
        elif sentiment == "bullish":
            recommendations.append("Positive momentum - consider rebalancing to target allocations")
        
        # Interest rate recommendations
        repo_rate = economic_data.get('interest_rates', {}).get('current_repo_rate')
        if repo_rate and repo_rate > 3.5:
            recommendations.append("High interest rates favor bond allocations and cash positions")
        
        # Always include general recommendation
        recommendations.append("Review portfolio allocation against long-term investment plan")
        
        return recommendations
    
    async def save_daily_insight(self, insight: Dict[str, Any]) -> None:
        """Save daily insight to file."""
        
        insights_dir = "data/daily_insights"
        os.makedirs(insights_dir, exist_ok=True)
        
        filename = f"market_insight_{insight['date']}.json"
        filepath = os.path.join(insights_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(insight, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved daily insight to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save daily insight: {e}")
    
    async def save_pipeline_results(self, results: Dict[str, Any]) -> None:
        """Save pipeline results to file."""
        
        results_dir = "data/pipeline_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"daily_pipeline_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved pipeline results to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline results: {e}")
    
    def generate_pipeline_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of pipeline execution."""
        
        summary = {
            "total_steps": 4,
            "completed_steps": len(results.get("steps_completed", [])),
            "success_rate": len(results.get("steps_completed", [])) / 4,
            "total_errors": len(results.get("errors", [])),
            "runtime_minutes": results.get("runtime_seconds", 0) / 60
        }
        
        # Add specific step summaries
        if "swedish_economic_update" in results:
            summary["economic_data_success"] = results["swedish_economic_update"].get("success", False)
        
        if "fund_price_update" in results:
            fund_update = results["fund_price_update"]
            summary["funds_updated"] = fund_update.get("successful_updates", 0)
            summary["fund_update_success"] = fund_update.get("success", False)
        
        if "cache_update" in results:
            cache_update = results["cache_update"]
            summary["cache_success"] = cache_update.get("success", False)
            summary["cached_funds"] = cache_update.get("successful_funds", 0)
        
        if "market_insight" in results:
            insight_update = results["market_insight"]
            summary["insight_generated"] = insight_update.get("success", False)
        
        return summary


async def main():
    """Main function for running the daily data pipeline."""
    
    parser = argparse.ArgumentParser(description="Daily Investment MCP Data Pipeline")
    parser.add_argument("--force-full-update", action="store_true", 
                       help="Force full cache refresh")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run without making changes")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize settings
        settings = Settings()
        
        # Initialize database
        await init_database()
        
        # Create and run pipeline
        pipeline = DailyDataPipeline(settings)
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No changes will be made")
            print("Daily pipeline would run with the following steps:")
            print("1. Update Swedish economic data")
            print("2. Update recent fund prices (last 30 days)")
            print("3. Update analysis cache")
            print("4. Generate daily market insight")
            return
        
        # Run the pipeline
        results = await pipeline.run_daily_update(force_full_update=args.force_full_update)
        
        # Print summary
        if results.get("success"):
            print(f"✅ Daily pipeline completed successfully")
            print(f"⏱️  Runtime: {results.get('runtime_seconds', 0):.1f} seconds")
            print(f"📊 Steps completed: {len(results.get('steps_completed', []))}/4")
            
            if results.get("summary"):
                summary = results["summary"]
                print(f"📈 Funds updated: {summary.get('funds_updated', 0)}")
                print(f"💾 Cache funds: {summary.get('cached_funds', 0)}")
                print(f"🔍 Market insight: {'✅' if summary.get('insight_generated') else '❌'}")
        else:
            print(f"❌ Daily pipeline failed: {results.get('pipeline_error', 'Unknown error')}")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())