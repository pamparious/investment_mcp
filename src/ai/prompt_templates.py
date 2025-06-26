"""
Optimized prompt templates for Gemma 3:1B financial analysis.

This module contains prompt templates specifically designed for Gemma 3:1B's
capabilities, focusing on Swedish investment analysis and portfolio optimization.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


class GemmaPrompts:
    """Prompt templates optimized for Gemma 3:1B financial analysis."""
    
    @staticmethod
    def portfolio_optimization_prompt(
        current_allocation: Dict[str, float],
        risk_tolerance: str,
        investment_amount: float,
        market_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate portfolio optimization prompt."""
        
        # Convert to Swedish context
        risk_map = {
            "low": "låg risk",
            "medium": "medel risk", 
            "high": "hög risk",
            "very_high": "mycket hög risk"
        }
        
        risk_swedish = risk_map.get(risk_tolerance, "medel risk")
        amount_formatted = f"{investment_amount:,.0f} SEK"
        
        prompt = f"""Du är en svensk finansiell rådgivare. Optimera denna portfölj:

NUVARANDE PORTFÖLJ:
{GemmaPrompts._format_allocation(current_allocation)}

INVESTERARPROFIL:
- Risktolerans: {risk_swedish}
- Investeringsbelopp: {amount_formatted}
- Marknad: Sverige

TILLGÄNGLIGA FONDER:
- DNB Global Indeks S (global aktier)
- Avanza Emerging Markets (tillväxtmarknader)
- Storebrand Europa A SEK (europeiska aktier)
- DNB Norden Indeks S (nordiska aktier)
- PLUS Allabolag Sverige Index (svenska aktier)
- Avanza USA (amerikanska aktier)
- Storebrand Japan A SEK (japanska aktier)
- Handelsbanken Global småb index (småbolag)
- Xetra-Gold ETC (guld)
- Virtune Bitcoin Prime ETP (bitcoin)
- XBT Ether One (ethereum)
- Plus Fastigheter Sverige Index (fastigheter)

UPPGIFT:
Föreslå optimal fördelning som:
1. Passar {risk_swedish}
2. Diversifierar risker
3. Passar svenska marknaden
4. Håller kostnaderna låga

FORMAT:
Fondnamn: Procent (motivation)

SVAR:"""
        
        return prompt
    
    @staticmethod
    def risk_analysis_prompt(
        portfolio: Dict[str, float],
        risk_metrics: Dict[str, float],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate risk analysis prompt."""
        
        sharpe = risk_metrics.get("sharpe_ratio", 0)
        volatility = risk_metrics.get("annualized_volatility", 0) * 100
        max_drawdown = abs(risk_metrics.get("max_drawdown", 0)) * 100
        
        prompt = f"""Analysera risk för denna svenska portfölj:

PORTFÖLJ:
{GemmaPrompts._format_allocation(portfolio)}

RISKMÅTT:
- Sharpe-kvot: {sharpe:.2f} (högre = bättre)
- Volatilitet: {volatility:.1f}% per år
- Största nedgång: {max_drawdown:.1f}%

UPPGIFT:
1. Bedöm risknivå (Låg/Medel/Hög)
2. Identifiera största riskerna
3. Föreslå riskminskande åtgärder
4. Förklara för privatinvesterare

Skriv på svenska, max 200 ord:"""
        
        return prompt
    
    @staticmethod
    def market_commentary_prompt(
        swedish_data: Dict[str, Any],
        fund_performance: Dict[str, Any],
        current_date: Optional[datetime] = None
    ) -> str:
        """Generate Swedish market commentary prompt."""
        
        if current_date is None:
            current_date = datetime.now()
        
        date_str = current_date.strftime("%Y-%m-%d")
        
        prompt = f"""Datum: {date_str}

Som svensk marknadsanalytiker, kommentera dagens läge:

SVENSK EKONOMI:
{GemmaPrompts._format_economic_data(swedish_data)}

FONDPRESTATION (senaste tiden):
{GemmaPrompts._format_fund_performance(fund_performance)}

UPPGIFT:
Skriv marknadskommentar för svenska investerare:
1. Läget just nu
2. Vad som påverkar marknaden
3. Viktiga faktorer att följa
4. Råd för investerare

Max 250 ord, fokusera på svenska förhållanden:"""
        
        return prompt
    
    @staticmethod
    def rebalancing_prompt(
        current_portfolio: Dict[str, float],
        target_portfolio: Dict[str, float],
        market_conditions: str
    ) -> str:
        """Generate rebalancing recommendation prompt."""
        
        prompt = f"""Ge rebalanseringsråd för svensk investerare:

NUVARANDE PORTFÖLJ:
{GemmaPrompts._format_allocation(current_portfolio)}

MÅLPORTFÖLJ:
{GemmaPrompts._format_allocation(target_portfolio)}

MARKNADSLÄGE: {market_conditions}

UPPGIFT:
1. Vilka förändringar behövs?
2. När ska rebalansering göras?
3. Skattekonsekvenser i Sverige?
4. Steg-för-steg plan

Praktiska råd på svenska, max 200 ord:"""
        
        return prompt
    
    @staticmethod
    def housing_vs_investment_prompt(
        housing_data: Dict[str, Any],
        investment_returns: Dict[str, float],
        personal_situation: Dict[str, Any]
    ) -> str:
        """Generate housing vs investment analysis prompt."""
        
        age = personal_situation.get("age", 35)
        income = personal_situation.get("monthly_income", 45000)
        savings = personal_situation.get("savings", 500000)
        
        prompt = f"""Råd till svensk person: bostad vs investering

PERSONLIGA FÖRHÅLLANDEN:
- Ålder: {age} år
- Månadsinkomst: {income:,.0f} SEK
- Sparkapital: {savings:,.0f} SEK

BOSTADSMARKNADEN:
{GemmaPrompts._format_housing_data(housing_data)}

INVESTERINGSALTERNATIV:
{GemmaPrompts._format_investment_returns(investment_returns)}

UPPGIFT:
Ge personlig rådgivning:
1. Köpa bostad eller fortsätta hyra?
2. Ekonomiska för- och nackdelar
3. Svensk skattemässiga aspekter
4. Konkret handlingsplan

Svar på svenska, max 300 ord:"""
        
        return prompt
    
    @staticmethod
    def sector_rotation_prompt(
        sector_performance: Dict[str, float],
        economic_indicators: Dict[str, Any],
        current_allocation: Dict[str, float]
    ) -> str:
        """Generate sector rotation analysis prompt."""
        
        prompt = f"""Sektoranalys för svensk investerare:

SEKTORPRESTATION (senaste året):
{GemmaPrompts._format_sector_performance(sector_performance)}

EKONOMISKA INDIKATORER:
{GemmaPrompts._format_economic_indicators(economic_indicators)}

NUVARANDE FÖRDELNING:
{GemmaPrompts._format_allocation(current_allocation)}

UPPGIFT:
1. Vilka sektorer ser starka ut?
2. Vilka bör undvikas?
3. Föreslå sektorrotation
4. Svenska specifika faktorer

Analys på svenska, max 200 ord:"""
        
        return prompt
    
    @staticmethod
    def tax_optimization_prompt(
        portfolio: Dict[str, float],
        realized_gains: float,
        investment_horizon: str
    ) -> str:
        """Generate Swedish tax optimization prompt."""
        
        prompt = f"""Skatteoptimering för svensk investerare:

PORTFÖLJ:
{GemmaPrompts._format_allocation(portfolio)}

SKATTELÄGE:
- Realiserade vinster i år: {realized_gains:,.0f} SEK
- Investeringshorisont: {investment_horizon}

SVENSKA SKATTEREGLER:
- Kapitalinkomstskatt: 30%
- ISK-konto: Schablonbeskattning
- Aktie- och fondkonto: Direktbeskattning

UPPGIFT:
1. Skatteeffektiva strategier
2. ISK vs vanligt konto
3. Timing av försäljningar
4. Praktiska skattetips

Råd på svenska, max 250 ord:"""
        
        return prompt
    
    @staticmethod
    def _format_allocation(allocation: Dict[str, float]) -> str:
        """Format portfolio allocation for prompts."""
        lines = []
        for fund, weight in sorted(allocation.items(), key=lambda x: x[1], reverse=True):
            if weight > 0:
                fund_name = fund.replace("_", " ").title()
                lines.append(f"- {fund_name}: {weight*100:.1f}%")
        return "\n".join(lines)
    
    @staticmethod
    def _format_economic_data(data: Dict[str, Any]) -> str:
        """Format economic data for prompts."""
        lines = []
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if "rate" in key.lower() or "ränta" in key.lower():
                    lines.append(f"- {key}: {value:.2f}%")
                else:
                    lines.append(f"- {key}: {value}")
            elif isinstance(value, str):
                lines.append(f"- {key}: {value}")
        return "\n".join(lines[:5])  # Limit for prompt efficiency
    
    @staticmethod
    def _format_fund_performance(performance: Dict[str, Any]) -> str:
        """Format fund performance for prompts."""
        lines = []
        for fund, perf in performance.items():
            if isinstance(perf, dict) and "return" in perf:
                return_pct = perf["return"] * 100
                lines.append(f"- {fund}: {return_pct:+.1f}%")
            elif isinstance(perf, (int, float)):
                lines.append(f"- {fund}: {perf*100:+.1f}%")
        return "\n".join(lines[:6])
    
    @staticmethod
    def _format_housing_data(housing_data: Dict[str, Any]) -> str:
        """Format housing market data for prompts."""
        lines = []
        for key, value in housing_data.items():
            if "price" in key.lower():
                lines.append(f"- {key}: {value:,.0f} SEK/m²")
            elif "rate" in key.lower():
                lines.append(f"- {key}: {value:.2f}%")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines[:4])
    
    @staticmethod
    def _format_investment_returns(returns: Dict[str, float]) -> str:
        """Format investment returns for prompts."""
        lines = []
        for investment, return_rate in returns.items():
            lines.append(f"- {investment}: {return_rate*100:.1f}% per år")
        return "\n".join(lines)
    
    @staticmethod
    def _format_sector_performance(performance: Dict[str, float]) -> str:
        """Format sector performance for prompts."""
        lines = []
        for sector, perf in sorted(performance.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {sector}: {perf*100:+.1f}%")
        return "\n".join(lines)
    
    @staticmethod
    def _format_economic_indicators(indicators: Dict[str, Any]) -> str:
        """Format economic indicators for prompts."""
        lines = []
        for indicator, value in indicators.items():
            if isinstance(value, (int, float)):
                if "rate" in indicator.lower() or "inflation" in indicator.lower():
                    lines.append(f"- {indicator}: {value:.2f}%")
                else:
                    lines.append(f"- {indicator}: {value}")
        return "\n".join(lines[:4])