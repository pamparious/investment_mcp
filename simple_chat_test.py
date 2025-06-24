"""Simple chat test with local LLM."""

import asyncio
import aiohttp
import json

async def test_ollama_direct():
    """Test Ollama API directly."""
    print("🤖 Testing Ollama Direct API")
    print("=" * 40)
    
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "llama3.2:3b",
        "prompt": "Hello! Explain technical analysis in 2 sentences.",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 100
        }
    }
    
    print("Sending request to Ollama...")
    print(f"Prompt: {payload['prompt']}")
    
    try:
        timeout = aiohttp.ClientTimeout(total=120)  # 2 minutes
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                print(f"Status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    print(f"\nAI Response:")
                    print(f"'{result.get('response', 'No response')}'")
                    print(f"\nStats:")
                    print(f"- Total duration: {result.get('total_duration', 0) / 1e9:.1f}s")
                    print(f"- Load duration: {result.get('load_duration', 0) / 1e9:.1f}s")
                    print(f"- Eval count: {result.get('eval_count', 0)} tokens")
                    return True
                else:
                    error_text = await response.text()
                    print(f"Error {response.status}: {error_text}")
                    return False
                    
    except asyncio.TimeoutError:
        print("❌ Request timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def test_investment_question():
    """Test with an investment-related question."""
    print("\n" + "=" * 40)
    print("📊 Testing Investment Question")
    print("=" * 40)
    
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "llama3.2:3b",
        "prompt": """Analyze this stock situation:
- Stock price: $100
- RSI: 75 (overbought)
- Price above 20-day moving average
- High volume

Should I buy, sell, or hold? Give a brief recommendation.""",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 150
        }
    }
    
    print("Asking investment question...")
    
    try:
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"\nInvestment Analysis:")
                    print(f"'{result.get('response', 'No response')}'")
                    return True
                else:
                    print(f"Error {response.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def main():
    """Run tests."""
    print("🚀 Local LLM Chat Test")
    print("=" * 50)
    
    # Test 1: Basic connection
    basic_ok = await test_ollama_direct()
    
    if basic_ok:
        # Test 2: Investment analysis
        investment_ok = await test_investment_question()
        
        print(f"\n{'='*50}")
        print("📋 TEST RESULTS")
        print(f"{'='*50}")
        print(f"✅ Basic Chat: {'PASS' if basic_ok else 'FAIL'}")
        print(f"✅ Investment Analysis: {'PASS' if investment_ok else 'FAIL'}")
        
        if basic_ok and investment_ok:
            print(f"\n🎉 SUCCESS! Your local LLM is working for investment analysis!")
            print(f"\nNext steps:")
            print(f"• Use the Investment MCP server to get AI-powered analysis")
            print(f"• Run: python run_demo_analysis.py (will now include AI insights)")
            print(f"• The system can now provide natural language explanations")
        else:
            print(f"\n⚠️  Some tests failed. Check Ollama setup.")
    else:
        print(f"\n❌ Basic connection failed. Troubleshooting:")
        print(f"• Make sure Ollama is running: ollama serve")
        print(f"• Check model is installed: ollama list")
        print(f"• Try: ollama run llama3.2:3b")

if __name__ == "__main__":
    asyncio.run(main())