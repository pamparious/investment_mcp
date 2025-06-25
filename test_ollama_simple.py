"""Simple test to verify Ollama works with optimized settings."""

import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from config.settings import Settings
from backend.ai.config import AIConfig

async def test_ollama_basic():
    """Test basic Ollama functionality."""
    print("🧪 TESTING OPTIMIZED OLLAMA SETUP")
    print("=" * 50)
    
    try:
        settings = Settings()
        ai_config = AIConfig(settings)
        
        # Test provider availability
        print("1. Testing provider availability...")
        test_result = await ai_config.test_provider()
        
        print(f"   Provider: {test_result['provider']}")
        print(f"   Model: {test_result['model']}")
        print(f"   Available: {'✅' if test_result['available'] else '❌'}")
        if test_result['error']:
            print(f"   Error: {test_result['error']}")
            return False
        
        # Test simple completion
        print("\n2. Testing simple completion...")
        provider = ai_config.get_provider()
        
        async with provider:
            response = await provider._generate_completion(
                "What is portfolio diversification? Answer in 20 words or less."
            )
            print(f"   Response: {response[:100]}...")
        
        # Test investment question
        print("\n3. Testing investment analysis...")
        async with provider:
            response = await provider._generate_completion(
                "Stock ABC: price $100, RSI 80. Recommendation? Answer: buy/sell/hold and why in 15 words."
            )
            print(f"   Response: {response[:100]}...")
        
        print("\n✅ Ollama test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the test."""
    success = await test_ollama_basic()
    
    if success:
        print("\n🎉 Ollama is working properly with optimized settings!")
        print("   • Model: gemma3:1b (fast, general purpose)")
        print("   • Timeout: 30 seconds (optimized)")  
        print("   • Temperature: 0.3 (balanced creativity)")
        print("   • Max tokens: 1024 (efficient)")
    else:
        print("\n⚠️  Ollama needs troubleshooting:")
        print("   • Check: ollama serve")
        print("   • Check: ollama list")
        print("   • Try: ollama run gemma3:1b")

if __name__ == "__main__":
    asyncio.run(main())