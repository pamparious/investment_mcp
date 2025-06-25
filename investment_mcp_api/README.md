# Investment MCP API

> Swedish Investment Analysis API with AI-Powered Recommendations

[![API Version](https://img.shields.io/badge/API-v3.0.0-blue)](https://api.investment-mcp.com/v1/openapi.json)
[![OpenAPI](https://img.shields.io/badge/OpenAPI-3.0.3-green)](docs/openapi/investment_mcp_api.yaml)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)](https://fastapi.tiangolo.com)

## Overview

The Investment MCP API provides comprehensive Swedish investment analysis with AI-powered portfolio optimization, historical data analysis spanning 20+ years, and real-time Swedish economic indicators.

### Key Features

- 🇸🇪 **Swedish Market Focus**: Specialized for Swedish investment funds and economic conditions
- 🤖 **AI-Powered Analysis**: Portfolio optimization using advanced machine learning models
- 📊 **20+ Years Historical Data**: Deep historical analysis for robust recommendations
- 🏦 **Real-time Economic Data**: Live Swedish economic indicators from Riksbank and SCB
- 🎯 **Risk Assessment**: Comprehensive stress testing and risk metrics
- 📈 **Market Regime Analysis**: Performance analysis across different market conditions
- ⚡ **High Performance**: Sub-15 second response times for complex analyses
- 🔒 **Enterprise Security**: API key authentication with rate limiting

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/investment-mcp-api.git
cd investment-mcp-api

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your configuration
```

### 2. Configuration

```bash
# Required environment variables
export API_KEY_HEADER="X-API-Key"
export DATABASE_URL="postgresql://user:pass@localhost/investment_mcp"
export REDIS_URL="redis://localhost:6379/0"
export AI_PROVIDER="ollama"  # or "openai" or "anthropic"
```

### 3. Run the API

```bash
# Development
python main.py

# Production with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. First API Call

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Portfolio analysis
curl -X POST "http://localhost:8000/api/v1/portfolio/analysis" \
     -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "risk_profile": "balanced",
       "investment_amount": 500000,
       "investment_horizon_years": 10
     }'
```

## API Documentation

### Interactive Documentation
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI Spec**: [http://localhost:8000/api/v1/openapi.json](http://localhost:8000/api/v1/openapi.json)

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/portfolio/analysis` | POST | AI-powered portfolio analysis |
| `/portfolio/stress-test` | POST | Portfolio stress testing |
| `/funds` | GET | List available Swedish funds |
| `/funds/{code}/historical` | GET | Historical fund performance |
| `/economic/sweden/overview` | GET | Swedish economic indicators |
| `/market/sentiment` | GET | Market sentiment analysis |
| `/ai/investment-recommendation` | POST | Comprehensive AI recommendations |

## Architecture

### Project Structure

```
investment_mcp_api/
├── api/                          # API implementation
│   ├── common/                   # Shared utilities
│   │   ├── config.py            # Configuration management
│   │   └── exceptions.py        # Custom exceptions
│   ├── middleware/               # FastAPI middleware
│   │   ├── auth.py              # Authentication
│   │   ├── rate_limit.py        # Rate limiting
│   │   └── logging.py           # Request logging
│   ├── models/                   # Pydantic models
│   │   ├── requests.py          # Request schemas
│   │   └── responses.py         # Response schemas
│   ├── services/                 # Business logic
│   │   ├── portfolio/           # Portfolio services
│   │   ├── data/                # Data services
│   │   ├── analysis/            # Analysis services
│   │   └── market/              # Market services
│   └── v1/                      # API version 1
│       └── routers/             # FastAPI routers
├── docs/                        # Documentation
│   ├── openapi/                 # OpenAPI specifications
│   ├── guides/                  # User guides
│   └── examples/                # Code examples
├── tests/                       # Test suites
├── config/                      # Configuration files
├── scripts/                     # Utility scripts
└── storage/                     # Data storage
```

### Technology Stack

- **Framework**: FastAPI 0.104+
- **API Standard**: OpenAPI 3.0.3
- **Database**: PostgreSQL with SQLAlchemy
- **Cache**: Redis
- **AI/ML**: Ollama, OpenAI, or Anthropic
- **Data Sources**: Riksbank API, SCB API, Yahoo Finance
- **Authentication**: API Key based
- **Documentation**: Automatic OpenAPI generation

### Data Flow

```mermaid
graph TD
    A[Client Request] --> B[Authentication Middleware]
    B --> C[Rate Limiting Middleware]
    C --> D[API Router]
    D --> E[Service Layer]
    E --> F[Data Collection]
    F --> G[AI Analysis]
    G --> H[Response Generation]
    H --> I[Client Response]
```

## Swedish Investment Funds

The API covers 12 approved Swedish investment funds:

### Equity Funds
- **DNB_GLOBAL_INDEKS_S**: Global equity index fund
- **AVANZA_USA**: US equity market exposure
- **STOREBRAND_EUROPA_A_SEK**: European equity focus
- **DNB_NORDEN_INDEKS_S**: Nordic equity markets
- **PLUS_ALLABOLAG_SVERIGE_INDEX**: Swedish equity index
- **AVANZA_EMERGING_MARKETS**: Emerging markets equity
- **STOREBRAND_JAPAN_A_SEK**: Japanese equity market
- **HANDELSBANKEN_GLOBAL_SMAB_INDEX**: Global small-cap equity

### Alternative Assets
- **XETRA_GOLD_ETC**: Gold commodity exposure
- **VIRTUNE_BITCOIN_PRIME_ETP**: Bitcoin cryptocurrency
- **XBT_ETHER_ONE**: Ethereum cryptocurrency
- **PLUS_FASTIGHETER_SVERIGE_INDEX**: Swedish real estate

## Usage Examples

### Portfolio Analysis

```python
import requests

# Basic portfolio analysis
response = requests.post(
    "https://api.investment-mcp.com/v1/portfolio/analysis",
    headers={"X-API-Key": "your-api-key"},
    json={
        "risk_profile": "balanced",
        "investment_amount": 1000000,
        "investment_horizon_years": 15,
        "constraints": {
            "max_funds": 6,
            "min_allocation_per_fund": 0.10
        }
    }
)

result = response.json()
print(f"Recommended allocation: {result['allocation']}")
print(f"Expected return: {result['expected_metrics']['expected_annual_return']:.1%}")
```

### Fund Research

```python
# Get fund information
funds = requests.get(
    "https://api.investment-mcp.com/v1/funds",
    headers={"X-API-Key": "your-api-key"}
).json()

# Historical performance
historical = requests.get(
    "https://api.investment-mcp.com/v1/funds/AVANZA_USA/historical?years=10",
    headers={"X-API-Key": "your-api-key"}
).json()

# Compare funds
comparison = requests.post(
    "https://api.investment-mcp.com/v1/funds/performance/comparison",
    headers={"X-API-Key": "your-api-key"},
    json={
        "fund_codes": ["AVANZA_USA", "DNB_GLOBAL_INDEKS_S"],
        "comparison_period": "5y"
    }
).json()
```

### Economic Data

```python
# Swedish economic overview
economy = requests.get(
    "https://api.investment-mcp.com/v1/economic/sweden/overview",
    headers={"X-API-Key": "your-api-key"}
).json()

print(f"Economic phase: {economy['economic_phase']}")
print(f"Repo rate: {economy['key_indicators']['repo_rate']:.1%}")
print(f"GDP growth: {economy['key_indicators']['gdp_growth']:.1%}")
```

## Authentication & Rate Limits

### API Keys

```bash
# Include API key in all requests
curl -H "X-API-Key: your-api-key-here" \
     https://api.investment-mcp.com/v1/endpoint
```

### Rate Limits

| Tier | Requests/Minute | Portfolio Analysis/Min | Price |
|------|-----------------|------------------------|-------|
| Standard | 100 | 10 | $29/month |
| Premium | 1,000 | 30 | $99/month |
| Enterprise | Custom | Custom | Contact us |

## Development

### Local Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black api/
isort api/

# Type checking
mypy api/

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Required variables
DATABASE_URL=postgresql://user:pass@localhost/investment_mcp
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key-here
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434

# Optional variables
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

### Docker Deployment

```bash
# Build image
docker build -t investment-mcp-api .

# Run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f api
```

## Testing

### Test Coverage

```bash
# Run all tests
pytest

# With coverage
pytest --cov=api tests/

# Specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/e2e/          # End-to-end tests
```

### Test API Endpoints

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Test portfolio analysis
curl -X POST "http://localhost:8000/api/v1/portfolio/analysis" \
     -H "X-API-Key: test-key-123" \
     -H "Content-Type: application/json" \
     -d @tests/fixtures/portfolio_request.json
```

## Monitoring & Observability

### Health Checks

- **Basic Health**: `/api/v1/health`
- **Detailed Status**: `/api/v1/status`
- **Database Health**: Automatic in health endpoint
- **External Services**: Riksbank, SCB, AI providers

### Logging

- **Request Logging**: All requests logged with timing
- **Error Tracking**: Detailed error logs with stack traces
- **Performance Metrics**: Response times and throughput
- **Security Events**: Authentication failures and rate limits

### Metrics

- **Response Times**: Average, P95, P99
- **Error Rates**: By endpoint and status code
- **Cache Hit Rates**: Redis cache performance
- **AI Model Performance**: Response times and accuracy

## Security

### Data Protection

- **API Key Authentication**: Secure API key validation
- **Rate Limiting**: Multiple tiers with burst protection
- **Input Validation**: Comprehensive request validation
- **Error Handling**: No sensitive data in error messages

### Financial Data Security

- **No Personal Data Storage**: Only aggregated analysis
- **Encrypted Transport**: HTTPS/TLS for all communications
- **Data Retention**: Configurable retention policies
- **Audit Logging**: Complete request audit trail

## Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Update documentation if needed
7. Submit a pull request

### Code Standards

- **Python Style**: Black formatting + isort imports
- **Type Hints**: Full type annotation coverage
- **Documentation**: Docstrings for all public functions
- **Testing**: Minimum 90% test coverage
- **API Documentation**: OpenAPI specification updates

## Support

### Documentation

- **Getting Started**: [docs/guides/getting_started.md](docs/guides/getting_started.md)
- **API Reference**: [Interactive docs at /docs](http://localhost:8000/docs)
- **Code Examples**: [docs/examples/](docs/examples/)
- **Troubleshooting**: [docs/guides/troubleshooting.md](docs/guides/troubleshooting.md)

### Contact

- **Email**: support@investment-mcp.com
- **Documentation**: https://docs.investment-mcp.com
- **Issues**: https://github.com/your-org/investment-mcp-api/issues
- **Discussions**: https://github.com/your-org/investment-mcp-api/discussions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### v3.0.0 (2024-01-15)
- 🎉 Complete API restructure following OpenAPI standards
- 🚀 New FastAPI-based architecture
- 🤖 Enhanced AI-powered portfolio analysis
- 📊 20+ years historical data integration
- 🇸🇪 Real-time Swedish economic indicators
- 🔒 Improved security and rate limiting
- 📖 Comprehensive API documentation

### Previous Versions
See [CHANGELOG.md](CHANGELOG.md) for complete version history.

---

**Built with ❤️ for Swedish investors**