"""Simple test script to chat with local LLM through Investment MCP."""

import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from config.settings import Settings
from backend.ai.config import AIConfig

async def test_simple_chat():
    """Test basic chat functionality with local LLM."""
    print("🤖 Testing LLM Chat Functionality")
    print("=" * 50)
    
    try:
        # Initialize AI config
        settings = Settings()
        ai_config = AIConfig(settings)
        
        # Test Ollama provider
        print("Getting Ollama provider...")
        provider = ai_config.get_provider("ollama")
        
        print(f"Provider: {provider.__class__.__name__}")
        print(f"Model: {provider.model_name}")
        
        # Test availability
        print("\nTesting provider availability...")
        is_available = await provider.is_available()
        print(f"Available: {is_available}")
        
        if not is_available:
            print("❌ LLM not available. Make sure Ollama is running and model is installed.")
            return
        
        # Test simple completion
        print("\n" + "=" * 50)
        print("💬 TESTING SIMPLE CHAT")
        print("=" * 50)
        
        test_prompt = "Hello! Can you briefly explain what technical analysis is in investing?"
        print(f"User: {test_prompt}")
        print(f"AI: ", end="", flush=True)
        
        async with provider:
            response = await provider._generate_completion(test_prompt)
            print(response)
        
        # Test financial analysis
        print("\n" + "=" * 50)
        print("📊 TESTING FINANCIAL ANALYSIS")
        print("=" * 50)
        
        financial_prompt = """
        Analyze this stock data:
        - Current Price: $100
        - 20-day SMA: $95
        - RSI: 75
        - Trend: Bullish
        
        Provide a brief investment recommendation.
        """
        
        print(f"User: {financial_prompt.strip()}")
        print(f"AI: ", end="", flush=True)
        
        async with provider:
            response = await provider._generate_completion(financial_prompt)
            print(response)
        
        # Test market data analysis
        print("\n" + "=" * 50)
        print("📈 TESTING MARKET DATA ANALYSIS")
        print("=" * 50)
        
        market_data = {
            "symbol": "TEST",
            "close_price": 150.50,
            "volume": 1000000
        }
        
        technical_indicators = {
            "sma_20": 145.20,
            "sma_50": 140.80,
            "rsi": 65.5,
            "trend": "bullish"
        }
        
        print("Testing market data analysis method...")
        async with provider:
            analysis = await provider.analyze_market_data("TEST", market_data, technical_indicators)
            print(f"Analysis result: {analysis}")
        
        print("\n✅ LLM Chat Test Complete!")
        
    except Exception as e:
        print(f"❌ Error testing LLM chat: {e}")
        import traceback
        traceback.print_exc()

async def interactive_chat():
    """Interactive chat session with the LLM."""
    print("\n" + "=" * 50)
    print("🔄 INTERACTIVE CHAT MODE")
    print("=" * 50)
    print("Type 'quit' to exit, 'help' for investment analysis examples")
    
    try:
        settings = Settings()
        ai_config = AIConfig(settings)
        provider = ai_config.get_provider("ollama")
        
        # Check availability
        if not await provider.is_available():
            print("❌ LLM not available. Make sure Ollama is running with llama3.2:3b model.")
            return
        
        async with provider:
            while True:
                print(f"\nYou: ", end="")
                user_input = input().strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("""
📊 Investment Analysis Examples:
• "Analyze AAPL with RSI 75, price above SMA"
• "Should I buy when RSI is oversold?"
• "Explain what a golden cross means"
• "What's the risk of high volatility stocks?"
• "How do interest rates affect stock prices?"
                    """)
                    continue
                
                if not user_input:
                    continue
                
                print(f"AI: ", end="", flush=True)
                try:
                    response = await provider._generate_completion(user_input)
                    print(response)
                except Exception as e:
                    print(f"Error: {e}")
    
    except KeyboardInterrupt:
        print("\n👋 Chat interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Error in interactive chat: {e}")

async def main():
    """Main function."""
    await test_simple_chat()
    
    # Ask if user wants interactive mode
    print(f"\n{'='*50}")
    try:
        answer = input("Would you like to try interactive chat? (y/n): ").strip().lower()
        if answer in ['y', 'yes']:
            await interactive_chat()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())