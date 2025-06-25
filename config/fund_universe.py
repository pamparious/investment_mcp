"""
Approved fund universe - ONLY these funds can be recommended.

This module defines the constrained universe of tradeable funds that the 
Investment MCP system is allowed to recommend. All portfolio recommendations
must use only funds from this approved list.
"""

TRADEABLE_FUNDS = {
    "DNB_GLOBAL_INDEKS_S": {
        "name": "DNB Global Indeks S",
        "type": "global_equity",
        "description": "Global index fund tracking developed markets",
        "currency": "SEK",
        "risk_level": "medium",
        "expense_ratio": 0.15,
        "category": "global_equity"
    },
    "AVANZA_EMERGING_MARKETS": {
        "name": "Avanza Emerging Markets",
        "type": "emerging_markets",
        "description": "Emerging markets equity fund",
        "currency": "SEK", 
        "risk_level": "high",
        "expense_ratio": 0.50,
        "category": "emerging_markets"
    },
    "STOREBRAND_EUROPA_A_SEK": {
        "name": "Storebrand Europa A SEK",
        "type": "european_equity",
        "description": "European equity fund",
        "currency": "SEK",
        "risk_level": "medium",
        "expense_ratio": 0.75,
        "category": "european_equity"
    },
    "DNB_NORDEN_INDEKS_S": {
        "name": "DNB Norden Indeks S", 
        "type": "nordic_equity",
        "description": "Nordic index fund",
        "currency": "SEK",
        "risk_level": "medium",
        "expense_ratio": 0.20,
        "category": "nordic_equity"
    },
    "PLUS_ALLABOLAG_SVERIGE_INDEX": {
        "name": "PLUS Allabolag Sverige Index",
        "type": "swedish_equity", 
        "description": "Swedish equity index fund",
        "currency": "SEK",
        "risk_level": "medium",
        "expense_ratio": 0.10,
        "category": "swedish_equity"
    },
    "AVANZA_USA": {
        "name": "Avanza USA",
        "type": "us_equity",
        "description": "US equity fund",
        "currency": "SEK",
        "risk_level": "medium", 
        "expense_ratio": 0.25,
        "category": "us_equity"
    },
    "STOREBRAND_JAPAN_A_SEK": {
        "name": "Storebrand Japan A SEK",
        "type": "japanese_equity",
        "description": "Japanese equity fund",
        "currency": "SEK",
        "risk_level": "medium",
        "expense_ratio": 0.85,
        "category": "japanese_equity"
    },
    "HANDELSBANKEN_GLOBAL_SMAB_INDEX": {
        "name": "Handelsbanken Global småb index",
        "type": "small_cap_global",
        "description": "Global small cap index fund",
        "currency": "SEK",
        "risk_level": "high",
        "expense_ratio": 0.40,
        "category": "small_cap"
    },
    "XETRA_GOLD_ETC": {
        "name": "Xetra-Gold ETC",
        "type": "commodity",
        "description": "Physical gold ETC",
        "currency": "SEK",
        "risk_level": "medium",
        "expense_ratio": 0.35,
        "category": "precious_metals"
    },
    "VIRTUNE_BITCOIN_PRIME_ETP": {
        "name": "Virtune Bitcoin Prime ETP",
        "type": "cryptocurrency",
        "description": "Bitcoin ETP",
        "currency": "SEK",
        "risk_level": "very_high",
        "expense_ratio": 1.00,
        "category": "cryptocurrency"
    },
    "XBT_ETHER_ONE": {
        "name": "XBT Ether One",
        "type": "cryptocurrency", 
        "description": "Ethereum ETP",
        "currency": "SEK",
        "risk_level": "very_high",
        "expense_ratio": 1.00,
        "category": "cryptocurrency"
    },
    "PLUS_FASTIGHETER_SVERIGE_INDEX": {
        "name": "Plus Fastigheter Sverige Index",
        "type": "real_estate",
        "description": "Swedish real estate index fund",
        "currency": "SEK",
        "risk_level": "medium",
        "expense_ratio": 0.30,
        "category": "real_estate"
    }
}

FUND_CATEGORIES = {
    "global_equity": ["DNB_GLOBAL_INDEKS_S"],
    "emerging_markets": ["AVANZA_EMERGING_MARKETS"],
    "european_equity": ["STOREBRAND_EUROPA_A_SEK"],
    "nordic_equity": ["DNB_NORDEN_INDEKS_S"],
    "swedish_equity": ["PLUS_ALLABOLAG_SVERIGE_INDEX"],
    "us_equity": ["AVANZA_USA"],
    "japanese_equity": ["STOREBRAND_JAPAN_A_SEK"],
    "small_cap": ["HANDELSBANKEN_GLOBAL_SMAB_INDEX"],
    "precious_metals": ["XETRA_GOLD_ETC"],
    "cryptocurrency": ["VIRTUNE_BITCOIN_PRIME_ETP", "XBT_ETHER_ONE"],
    "real_estate": ["PLUS_FASTIGHETER_SVERIGE_INDEX"]
}

RISK_LEVELS = {
    "low": [],
    "medium": [
        "DNB_GLOBAL_INDEKS_S", "STOREBRAND_EUROPA_A_SEK", "DNB_NORDEN_INDEKS_S",
        "PLUS_ALLABOLAG_SVERIGE_INDEX", "AVANZA_USA", "STOREBRAND_JAPAN_A_SEK",
        "XETRA_GOLD_ETC", "PLUS_FASTIGHETER_SVERIGE_INDEX"
    ],
    "high": ["AVANZA_EMERGING_MARKETS", "HANDELSBANKEN_GLOBAL_SMAB_INDEX"],
    "very_high": ["VIRTUNE_BITCOIN_PRIME_ETP", "XBT_ETHER_ONE"]
}

def get_approved_funds():
    """Return list of all approved fund identifiers."""
    return list(TRADEABLE_FUNDS.keys())

def get_fund_info(fund_id):
    """Get detailed information about a specific fund."""
    return TRADEABLE_FUNDS.get(fund_id)

def get_funds_by_category(category):
    """Get all funds in a specific category."""
    return FUND_CATEGORIES.get(category, [])

def get_funds_by_risk_level(risk_level):
    """Get all funds with a specific risk level."""
    return RISK_LEVELS.get(risk_level, [])

def validate_fund_allocation(allocation_dict):
    """
    Validate that allocation only uses approved funds and sums to 100%.
    
    Args:
        allocation_dict: Dictionary of fund_id -> percentage (as decimal)
        
    Returns:
        dict: Validation result with 'valid' boolean and 'errors' list
    """
    errors = []
    
    # Check if all funds are approved
    for fund_id in allocation_dict.keys():
        if fund_id not in TRADEABLE_FUNDS:
            errors.append(f"Fund '{fund_id}' is not in approved universe")
    
    # Check if allocations sum to 100% (allowing 0.1% tolerance)
    total_allocation = sum(allocation_dict.values())
    if abs(total_allocation - 1.0) > 0.001:
        errors.append(f"Allocations sum to {total_allocation:.1%}, must sum to 100%")
    
    # Check for negative allocations
    for fund_id, allocation in allocation_dict.items():
        if allocation < 0:
            errors.append(f"Negative allocation for {fund_id}: {allocation:.1%}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "total_allocation": total_allocation
    }

def get_diversification_suggestions():
    """Get suggestions for well-diversified portfolios using approved funds."""
    return {
        "conservative": {
            "description": "Conservative portfolio for risk-averse investors",
            "allocation": {
                "DNB_GLOBAL_INDEKS_S": 0.40,
                "PLUS_ALLABOLAG_SVERIGE_INDEX": 0.25,
                "XETRA_GOLD_ETC": 0.15,
                "PLUS_FASTIGHETER_SVERIGE_INDEX": 0.20
            }
        },
        "balanced": {
            "description": "Balanced portfolio with moderate risk",
            "allocation": {
                "DNB_GLOBAL_INDEKS_S": 0.30,
                "AVANZA_USA": 0.25,
                "STOREBRAND_EUROPA_A_SEK": 0.15,
                "PLUS_ALLABOLAG_SVERIGE_INDEX": 0.15,
                "AVANZA_EMERGING_MARKETS": 0.10,
                "XETRA_GOLD_ETC": 0.05
            }
        },
        "growth": {
            "description": "Growth-oriented portfolio for higher returns",
            "allocation": {
                "AVANZA_USA": 0.30,
                "DNB_GLOBAL_INDEKS_S": 0.25,
                "HANDELSBANKEN_GLOBAL_SMAB_INDEX": 0.15,
                "AVANZA_EMERGING_MARKETS": 0.15,
                "STOREBRAND_EUROPA_A_SEK": 0.10,
                "VIRTUNE_BITCOIN_PRIME_ETP": 0.05
            }
        }
    }