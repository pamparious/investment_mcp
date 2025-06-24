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
from backend.database import DatabaseManager
from backend.ai.config import AIConfig
from backend.ai.analyzers import MarketAnalyzer, EconomicAnalyzer, PortfolioAnalyzer

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
        self.db_manager = DatabaseManager(settings.DATABASE_URL)
        self.ai_config = AIConfig(settings)
        
        # Initialize analyzers
        self.market_analyzer = MarketAnalyzer(self.ai_config)
        self.economic_analyzer = EconomicAnalyzer(self.ai_config)
        self.portfolio_analyzer = PortfolioAnalyzer(self.ai_config)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
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