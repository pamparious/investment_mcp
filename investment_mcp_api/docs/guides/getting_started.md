# Getting Started with Investment MCP API

Welcome to the Investment MCP API - your comprehensive solution for Swedish investment analysis with AI-powered recommendations.

## Quick Start

### 1. Authentication

All API requests require an API key in the header:

```bash
curl -H "X-API-Key: your-api-key-here" \
     https://api.investment-mcp.com/v1/health
```

### 2. Your First Portfolio Analysis

Generate an AI-powered portfolio recommendation:

```bash
curl -X POST "https://api.investment-mcp.com/v1/portfolio/analysis" \
     -H "X-API-Key: your-api-key-here" \
     -H "Content-Type: application/json" \
     -d '{
       "risk_profile": "balanced",
       "investment_amount": 500000,
       "investment_horizon_years": 10
     }'
```

### 3. Explore Swedish Economic Data

Get current Swedish economic indicators:

```bash
curl -H "X-API-Key: your-api-key-here" \
     https://api.investment-mcp.com/v1/economic/sweden/overview
```

## API Overview

### Base URL
- **Production:** `https://api.investment-mcp.com/v1`
- **Staging:** `https://staging-api.investment-mcp.com/v1`
- **Development:** `http://localhost:8000/v1`

### Core Endpoints

| Endpoint | Purpose | Typical Use Case |
|----------|---------|------------------|
| `/portfolio/analysis` | AI-powered portfolio analysis | Generate investment recommendations |
| `/funds` | Swedish fund data | Research investment options |
| `/economic/sweden/*` | Swedish economic indicators | Market context analysis |
| `/market/sentiment` | Market sentiment analysis | Timing and risk assessment |
| `/ai/investment-recommendation` | Comprehensive AI recommendations | Complete investment planning |

### Rate Limits

| Tier | Requests/Minute | Special Limits |
|------|-----------------|----------------|
| Standard | 100 | Portfolio analysis: 10/min |
| Premium | 1000 | Portfolio analysis: 30/min |
| Enterprise | Custom | Custom limits available |

## Core Concepts

### Risk Profiles

- **Conservative**: Focus on capital preservation, max 60% equities
- **Balanced**: Balanced growth and protection, max 80% equities  
- **Aggressive**: Growth-focused, up to 100% equities

### Swedish Investment Funds

The API covers 12 approved Swedish investment funds:

**Equity Funds:**
- `DNB_GLOBAL_INDEKS_S` - Global equity index
- `AVANZA_USA` - US equity market
- `STOREBRAND_EUROPA_A_SEK` - European equity
- `DNB_NORDEN_INDEKS_S` - Nordic equity
- `PLUS_ALLABOLAG_SVERIGE_INDEX` - Swedish equity
- `AVANZA_EMERGING_MARKETS` - Emerging markets
- `STOREBRAND_JAPAN_A_SEK` - Japanese equity
- `HANDELSBANKEN_GLOBAL_SMAB_INDEX` - Global small cap

**Alternative Assets:**
- `XETRA_GOLD_ETC` - Gold exposure
- `VIRTUNE_BITCOIN_PRIME_ETP` - Bitcoin exposure
- `XBT_ETHER_ONE` - Ethereum exposure
- `PLUS_FASTIGHETER_SVERIGE_INDEX` - Swedish real estate

### Data Sources

- **Riksbank**: Swedish central bank data (interest rates, currency)
- **SCB (Statistics Sweden)**: Economic statistics, housing data
- **Yahoo Finance**: Historical fund price data
- **AI Models**: Portfolio optimization and analysis

## Common Workflows

### 1. Investment Planning Workflow

```python
import requests

api_key = "your-api-key"
headers = {"X-API-Key": api_key}
base_url = "https://api.investment-mcp.com/v1"

# Step 1: Get market overview
market = requests.get(f"{base_url}/market/sentiment", headers=headers)

# Step 2: Check Swedish economic conditions  
economy = requests.get(f"{base_url}/economic/sweden/overview", headers=headers)

# Step 3: Generate portfolio recommendation
portfolio = requests.post(f"{base_url}/portfolio/analysis", 
    headers=headers,
    json={
        "risk_profile": "balanced",
        "investment_amount": 1000000,
        "investment_horizon_years": 15
    }
)

# Step 4: Stress test the recommendation
stress_test = requests.post(f"{base_url}/portfolio/stress-test",
    headers=headers, 
    json={
        "allocation": portfolio.json()["allocation"],
        "scenarios": ["all"]
    }
)
```

### 2. Fund Research Workflow

```python
# Get list of all funds
funds = requests.get(f"{base_url}/funds", headers=headers)

# Get detailed fund information
fund_details = requests.get(f"{base_url}/funds/AVANZA_USA", headers=headers)

# Get historical performance
historical = requests.get(
    f"{base_url}/funds/AVANZA_USA/historical?years=10", 
    headers=headers
)

# Compare multiple funds
comparison = requests.post(f"{base_url}/funds/performance/comparison",
    headers=headers,
    json={
        "fund_codes": ["AVANZA_USA", "DNB_GLOBAL_INDEKS_S", "AVANZA_EMERGING_MARKETS"],
        "comparison_period": "5y"
    }
)
```

### 3. Economic Analysis Workflow

```python
# Get comprehensive economic overview
overview = requests.get(f"{base_url}/economic/sweden/overview", headers=headers)

# Get specific economic indicators
rates = requests.get(f"{base_url}/economic/sweden/interest-rates", headers=headers)
inflation = requests.get(f"{base_url}/economic/sweden/inflation", headers=headers)
housing = requests.get(f"{base_url}/economic/sweden/housing", headers=headers)

# Get daily market insights
insights = requests.get(f"{base_url}/market/insights/daily", headers=headers)
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error information:

```json
{
  "error": "Validation failed",
  "details": [
    {
      "field": "investment_amount",
      "message": "Must be between 10,000 and 100,000,000 SEK"
    }
  ],
  "request_id": "req_123456789"
}
```

### Common Error Codes

- `400` - Bad Request (validation errors)
- `401` - Unauthorized (invalid API key)
- `403` - Forbidden (insufficient permissions)
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error

## Response Formats

All responses follow a consistent structure:

```json
{
  "status": "success",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789",
  "data": {
    // Endpoint-specific data
  }
}
```

## Best Practices

### 1. API Key Security
- Never expose API keys in client-side code
- Use environment variables for API keys
- Rotate keys regularly
- Use different keys for different environments

### 2. Rate Limiting
- Implement exponential backoff for rate limit errors
- Cache responses when appropriate
- Use batch endpoints when available
- Monitor your usage against limits

### 3. Error Handling
- Always check response status codes
- Implement retry logic for 5xx errors
- Handle 429 (rate limit) with proper backoff
- Log request IDs for debugging

### 4. Data Freshness
- Check `last_updated` timestamps in responses
- Economic data updates every 30 minutes
- Fund price data updates every 5 minutes
- Portfolio analysis uses cached historical data (daily updates)

## Next Steps

1. **Explore the API**: Try the interactive documentation at `/docs`
2. **Join the Community**: Connect with other developers
3. **Read Advanced Guides**: Dive deeper into specific use cases
4. **Get Support**: Contact our support team for assistance

## Resources

- [API Reference Documentation](../openapi/investment_mcp_api.yaml)
- [Interactive API Explorer](/docs)
- [Code Examples](../examples/)
- [Postman Collection](../examples/postman_collection.json)
- [SDK Documentation](../sdks/)

---

**Need Help?** Contact our support team at support@investment-mcp.com or visit our documentation portal.