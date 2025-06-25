"""
Investment MCP API - Portfolio Analysis Examples

This file demonstrates various ways to use the portfolio analysis endpoints
of the Investment MCP API for Swedish investment analysis.
"""

import asyncio
import json
import time
from typing import Dict, Any, List
from datetime import datetime

import aiohttp
import requests


class InvestmentMCPClient:
    """Client for Investment MCP API."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.investment-mcp.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    def analyze_portfolio(self, **kwargs) -> Dict[str, Any]:
        """Synchronous portfolio analysis."""
        response = requests.post(
            f"{self.base_url}/portfolio/analysis",
            headers=self.headers,
            json=kwargs
        )
        response.raise_for_status()
        return response.json()
    
    def stress_test_portfolio(self, allocation: Dict[str, float], scenarios: List[str] = None) -> Dict[str, Any]:
        """Synchronous portfolio stress testing."""
        payload = {
            "allocation": allocation,
            "scenarios": scenarios or ["all"]
        }
        response = requests.post(
            f"{self.base_url}/portfolio/stress-test",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def analyze_portfolio_async(self, **kwargs) -> Dict[str, Any]:
        """Asynchronous portfolio analysis."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/portfolio/analysis",
                headers=self.headers,
                json=kwargs
            ) as response:
                response.raise_for_status()
                return await response.json()


def example_1_basic_portfolio_analysis():
    """Example 1: Basic portfolio analysis for a balanced investor."""
    
    print("=== Example 1: Basic Portfolio Analysis ===")
    
    # Initialize client (replace with your API key)
    client = InvestmentMCPClient(api_key="your-api-key-here")
    
    try:
        # Request portfolio analysis
        result = client.analyze_portfolio(
            risk_profile="balanced",
            investment_amount=500000,  # 500k SEK
            investment_horizon_years=10
        )
        
        print(f"Analysis completed with confidence: {result['confidence_score']:.2%}")
        print("\nRecommended Allocation:")
        for fund, weight in result['allocation'].items():
            print(f"  {fund}: {weight:.1%}")
        
        print(f"\nExpected Annual Return: {result['expected_metrics']['expected_annual_return']:.1%}")
        print(f"Expected Volatility: {result['expected_metrics']['expected_volatility']:.1%}")
        print(f"Expected Sharpe Ratio: {result['expected_metrics']['expected_sharpe_ratio']:.2f}")
        
    except requests.exceptions.HTTPError as e:
        print(f"API Error: {e}")
        if e.response.status_code == 429:
            print("Rate limit exceeded. Please wait and retry.")
    except Exception as e:
        print(f"Unexpected error: {e}")


def example_2_conservative_vs_aggressive():
    """Example 2: Compare conservative vs aggressive risk profiles."""
    
    print("\n=== Example 2: Conservative vs Aggressive Comparison ===")
    
    client = InvestmentMCPClient(api_key="your-api-key-here")
    
    investment_amount = 1000000  # 1M SEK
    horizon = 15  # 15 years
    
    profiles = ["conservative", "aggressive"]
    results = {}
    
    for profile in profiles:
        try:
            result = client.analyze_portfolio(
                risk_profile=profile,
                investment_amount=investment_amount,
                investment_horizon_years=horizon
            )
            results[profile] = result
            
            print(f"\n{profile.upper()} Profile:")
            print(f"  Expected Return: {result['expected_metrics']['expected_annual_return']:.1%}")
            print(f"  Expected Risk: {result['expected_metrics']['expected_volatility']:.1%}")
            print(f"  Sharpe Ratio: {result['expected_metrics']['expected_sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {result['expected_metrics']['expected_max_drawdown']:.1%}")
            
            # Show top 3 allocations
            top_allocations = sorted(
                result['allocation'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            print("  Top 3 Holdings:")
            for fund, weight in top_allocations:
                print(f"    {fund}: {weight:.1%}")
                
        except Exception as e:
            print(f"Error analyzing {profile} profile: {e}")
    
    # Compare results
    if len(results) == 2:
        conservative = results["conservative"]
        aggressive = results["aggressive"]
        
        return_diff = (
            aggressive['expected_metrics']['expected_annual_return'] - 
            conservative['expected_metrics']['expected_annual_return']
        )
        risk_diff = (
            aggressive['expected_metrics']['expected_volatility'] - 
            conservative['expected_metrics']['expected_volatility']
        )
        
        print(f"\nComparison:")
        print(f"  Return Difference: +{return_diff:.1%} for aggressive")
        print(f"  Risk Difference: +{risk_diff:.1%} for aggressive")


def example_3_portfolio_with_constraints():
    """Example 3: Portfolio analysis with custom constraints."""
    
    print("\n=== Example 3: Portfolio with Constraints ===")
    
    client = InvestmentMCPClient(api_key="your-api-key-here")
    
    # Define constraints
    constraints = {
        "max_funds": 5,  # Limit to 5 funds for simplicity
        "min_allocation_per_fund": 0.10,  # Minimum 10% per fund
        "exclude_funds": [
            "VIRTUNE_BITCOIN_PRIME_ETP",  # Exclude cryptocurrency
            "XBT_ETHER_ONE"
        ]
    }
    
    try:
        result = client.analyze_portfolio(
            risk_profile="balanced",
            investment_amount=750000,
            investment_horizon_years=12,
            constraints=constraints,
            include_stress_test=True
        )
        
        print("Portfolio with Constraints:")
        print(f"Number of funds: {len(result['allocation'])}")
        
        for fund, weight in sorted(result['allocation'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {fund}: {weight:.1%}")
        
        # Show stress test results
        if 'stress_test' in result:
            print(f"\nStress Test Results:")
            scenarios = result['stress_test']['scenarios']
            for scenario_name, scenario_data in scenarios.items():
                print(f"  {scenario_data['scenario_name']}: {scenario_data['portfolio_return']:.1%}")
        
    except Exception as e:
        print(f"Error with constrained portfolio: {e}")


def example_4_portfolio_rebalancing():
    """Example 4: Portfolio rebalancing from current allocation."""
    
    print("\n=== Example 4: Portfolio Rebalancing ===")
    
    client = InvestmentMCPClient(api_key="your-api-key-here")
    
    # Current portfolio allocation (example)
    current_allocation = {
        "AVANZA_USA": 0.40,
        "DNB_GLOBAL_INDEKS_S": 0.30,
        "STOREBRAND_EUROPA_A_SEK": 0.20,
        "XETRA_GOLD_ETC": 0.10
    }
    
    try:
        result = client.analyze_portfolio(
            risk_profile="balanced",
            investment_amount=800000,
            investment_horizon_years=8,
            current_allocation=current_allocation
        )
        
        print("Current vs Recommended Allocation:")
        print(f"{'Fund':<35} {'Current':<10} {'Recommended':<12} {'Change'}")
        print("-" * 70)
        
        all_funds = set(current_allocation.keys()) | set(result['allocation'].keys())
        
        for fund in sorted(all_funds):
            current = current_allocation.get(fund, 0)
            recommended = result['allocation'].get(fund, 0)
            change = recommended - current
            
            change_str = f"{change:+.1%}" if abs(change) > 0.001 else "0.0%"
            
            print(f"{fund:<35} {current:<10.1%} {recommended:<12.1%} {change_str}")
        
        # Show rebalancing recommendations from AI reasoning
        if 'ai_reasoning' in result:
            print(f"\nAI Reasoning for Changes:")
            print(f"  {result['ai_reasoning']['allocation_rationale'][:200]}...")
        
    except Exception as e:
        print(f"Error with rebalancing analysis: {e}")


def example_5_stress_testing():
    """Example 5: Detailed stress testing of a portfolio."""
    
    print("\n=== Example 5: Detailed Stress Testing ===")
    
    client = InvestmentMCPClient(api_key="your-api-key-here")
    
    # Define a test portfolio
    test_allocation = {
        "AVANZA_USA": 0.35,
        "DNB_GLOBAL_INDEKS_S": 0.25,
        "AVANZA_EMERGING_MARKETS": 0.15,
        "STOREBRAND_EUROPA_A_SEK": 0.10,
        "XETRA_GOLD_ETC": 0.10,
        "PLUS_FASTIGHETER_SVERIGE_INDEX": 0.05
    }
    
    try:
        # Run stress test on specific scenarios
        stress_result = client.stress_test_portfolio(
            allocation=test_allocation,
            scenarios=["2008_crisis", "covid_2020", "dotcom_2000"]
        )
        
        print("Stress Test Results:")
        print(f"Overall Risk Score: {stress_result['overall_assessment']['risk_metrics']['risk_score']:.1f}/100")
        print(f"Worst Case Loss: {stress_result['scenarios']['2008_crisis']['portfolio_return']:.1%}")
        
        print("\nScenario Breakdown:")
        for scenario_key, scenario in stress_result['scenarios'].items():
            print(f"  {scenario['scenario_name']}:")
            print(f"    Portfolio Return: {scenario['portfolio_return']:.1%}")
            print(f"    Duration: {scenario['duration_days']} days")
            print(f"    Max Drawdown: {scenario['max_drawdown']:.1%}")
            if scenario.get('recovery_time_days'):
                print(f"    Recovery Time: {scenario['recovery_time_days']} days")
        
        # Show recommendations
        if 'recommendations' in stress_result:
            print(f"\nRisk Management Recommendations:")
            for rec in stress_result['recommendations']:
                print(f"  • {rec}")
        
    except Exception as e:
        print(f"Error with stress testing: {e}")


async def example_6_async_batch_analysis():
    """Example 6: Asynchronous batch analysis for multiple scenarios."""
    
    print("\n=== Example 6: Async Batch Analysis ===")
    
    client = InvestmentMCPClient(api_key="your-api-key-here")
    
    # Define multiple analysis scenarios
    scenarios = [
        {
            "name": "Young Professional",
            "params": {
                "risk_profile": "aggressive",
                "investment_amount": 200000,
                "investment_horizon_years": 25
            }
        },
        {
            "name": "Mid-Career Family",
            "params": {
                "risk_profile": "balanced", 
                "investment_amount": 800000,
                "investment_horizon_years": 15
            }
        },
        {
            "name": "Pre-Retirement",
            "params": {
                "risk_profile": "conservative",
                "investment_amount": 1500000,
                "investment_horizon_years": 8
            }
        }
    ]
    
    async def analyze_scenario(scenario):
        """Analyze a single scenario."""
        try:
            result = await client.analyze_portfolio_async(**scenario["params"])
            return {
                "name": scenario["name"],
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "name": scenario["name"],
                "success": False,
                "error": str(e)
            }
    
    # Run all scenarios concurrently
    start_time = time.time()
    results = await asyncio.gather(*[analyze_scenario(s) for s in scenarios])
    end_time = time.time()
    
    print(f"Analyzed {len(scenarios)} scenarios in {end_time - start_time:.1f} seconds")
    
    # Display results
    for result in results:
        if result["success"]:
            analysis = result["result"]
            print(f"\n{result['name']}:")
            print(f"  Expected Return: {analysis['expected_metrics']['expected_annual_return']:.1%}")
            print(f"  Expected Risk: {analysis['expected_metrics']['expected_volatility']:.1%}")
            print(f"  Confidence: {analysis['confidence_score']:.1%}")
            
            # Top 2 holdings
            top_holdings = sorted(
                analysis['allocation'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:2]
            print(f"  Top Holdings: {', '.join([f'{fund} ({weight:.0%})' for fund, weight in top_holdings])}")
        else:
            print(f"\n{result['name']}: ERROR - {result['error']}")


def example_7_economic_context_integration():
    """Example 7: Portfolio analysis with Swedish economic context."""
    
    print("\n=== Example 7: Economic Context Integration ===")
    
    client = InvestmentMCPClient(api_key="your-api-key-here")
    
    try:
        # Get economic overview first
        economic_response = requests.get(
            f"{client.base_url}/economic/sweden/overview",
            headers=client.headers
        )
        economic_data = economic_response.json()
        
        print("Current Swedish Economic Context:")
        print(f"  Economic Phase: {economic_data['economic_phase']}")
        print(f"  Repo Rate: {economic_data['key_indicators']['repo_rate']:.1%}")
        print(f"  Inflation: {economic_data['key_indicators']['inflation_cpi']:.1%}")
        print(f"  GDP Growth: {economic_data['key_indicators']['gdp_growth']:.1%}")
        
        # Now get portfolio recommendation
        portfolio_result = client.analyze_portfolio(
            risk_profile="balanced",
            investment_amount=600000,
            investment_horizon_years=10
        )
        
        print(f"\nPortfolio Recommendation:")
        for fund, weight in sorted(portfolio_result['allocation'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {fund}: {weight:.1%}")
        
        # Show Swedish economic rationale
        if 'ai_reasoning' in portfolio_result:
            print(f"\nSwedish Economic Context in Recommendation:")
            print(f"  {portfolio_result['ai_reasoning']['swedish_economic_rationale']}")
        
        # Show investment implications
        print(f"\nEconomic Investment Implications:")
        for implication in economic_data['investment_implications']:
            print(f"  • {implication}")
        
    except Exception as e:
        print(f"Error with economic context analysis: {e}")


def example_8_performance_monitoring():
    """Example 8: Monitor API performance and implement retry logic."""
    
    print("\n=== Example 8: Performance Monitoring ===")
    
    client = InvestmentMCPClient(api_key="your-api-key-here")
    
    def analyze_with_retry(max_retries=3, backoff_factor=1.5):
        """Analyze portfolio with retry logic."""
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                result = client.analyze_portfolio(
                    risk_profile="balanced",
                    investment_amount=500000,
                    investment_horizon_years=10
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                print(f"✅ Analysis completed in {response_time:.2f} seconds")
                print(f"   Confidence Score: {result['confidence_score']:.1%}")
                print(f"   Data Quality: {result['data_quality_score']:.1%}")
                
                return result
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    retry_after = int(e.response.headers.get('Retry-After', 60))
                    print(f"⚠️  Rate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                elif e.response.status_code >= 500:  # Server error
                    wait_time = backoff_factor ** attempt
                    print(f"⚠️  Server error (attempt {attempt + 1}). Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Client error: {e}")
                    break
                    
            except Exception as e:
                wait_time = backoff_factor ** attempt
                print(f"⚠️  Unexpected error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    print(f"   Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
        
        print("❌ All retry attempts failed")
        return None
    
    # Test with retry logic
    result = analyze_with_retry()
    
    if result:
        print(f"\nFinal allocation summary:")
        total_equity = sum(
            weight for fund, weight in result['allocation'].items()
            if any(keyword in fund for keyword in ['USA', 'GLOBAL', 'EUROPA', 'NORDEN', 'EMERGING', 'JAPAN', 'SMAB'])
        )
        total_alternatives = sum(
            weight for fund, weight in result['allocation'].items()
            if any(keyword in fund for keyword in ['GOLD', 'BITCOIN', 'ETHER', 'FASTIGHETER'])
        )
        
        print(f"  Total Equity Allocation: {total_equity:.1%}")
        print(f"  Total Alternative Allocation: {total_alternatives:.1%}")


def main():
    """Run all examples."""
    
    print("Investment MCP API - Portfolio Analysis Examples")
    print("=" * 60)
    print()
    print("⚠️  Note: Replace 'your-api-key-here' with your actual API key")
    print("⚠️  Some examples may fail without a valid API key")
    print()
    
    # Uncomment the examples you want to run:
    
    # example_1_basic_portfolio_analysis()
    # example_2_conservative_vs_aggressive()
    # example_3_portfolio_with_constraints()
    # example_4_portfolio_rebalancing()
    # example_5_stress_testing()
    # asyncio.run(example_6_async_batch_analysis())
    # example_7_economic_context_integration()
    # example_8_performance_monitoring()
    
    print("\n✅ Examples completed!")
    print("\nNext steps:")
    print("1. Get your API key from the Investment MCP dashboard")
    print("2. Replace 'your-api-key-here' with your actual key")
    print("3. Uncomment and run the examples you're interested in")
    print("4. Explore the full API documentation at /docs")


if __name__ == "__main__":
    main()