"""
Response parser for Gemma 3:1B financial analysis outputs.

This module provides structured parsing of Gemma's responses to extract
actionable investment insights and recommendations.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class GemmaResponseParser:
    """Parser for Gemma 3:1B financial analysis responses."""
    
    @staticmethod
    def parse_portfolio_recommendation(response: str) -> Dict[str, Any]:
        """Parse portfolio optimization recommendation from Gemma."""
        
        # Extract allocation percentages
        allocation = GemmaResponseParser._extract_allocation(response)
        
        # Extract reasoning
        reasoning = GemmaResponseParser._extract_reasoning(response)
        
        # Extract risk assessment
        risk_level = GemmaResponseParser._extract_risk_level(response)
        
        # Extract key recommendations
        recommendations = GemmaResponseParser._extract_recommendations(response)
        
        return {
            "allocation": allocation,
            "risk_level": risk_level,
            "reasoning": reasoning,
            "recommendations": recommendations,
            "confidence": GemmaResponseParser._assess_confidence(response),
            "raw_response": response,
            "parsed_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def parse_risk_analysis(response: str) -> Dict[str, Any]:
        """Parse risk analysis response from Gemma."""
        
        # Extract risk level
        risk_level = GemmaResponseParser._extract_risk_level(response)
        
        # Extract risk factors
        risk_factors = GemmaResponseParser._extract_risk_factors(response)
        
        # Extract mitigation strategies
        mitigation = GemmaResponseParser._extract_mitigation_strategies(response)
        
        # Extract key metrics interpretation
        metrics_interpretation = GemmaResponseParser._extract_metrics_interpretation(response)
        
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "mitigation_strategies": mitigation,
            "metrics_interpretation": metrics_interpretation,
            "summary": GemmaResponseParser._extract_summary(response),
            "raw_response": response
        }
    
    @staticmethod
    def parse_market_commentary(response: str) -> Dict[str, Any]:
        """Parse market commentary response from Gemma."""
        
        # Extract market outlook
        outlook = GemmaResponseParser._extract_market_outlook(response)
        
        # Extract key factors
        key_factors = GemmaResponseParser._extract_key_factors(response)
        
        # Extract investor recommendations
        investor_advice = GemmaResponseParser._extract_investor_advice(response)
        
        # Extract Swedish-specific insights
        swedish_context = GemmaResponseParser._extract_swedish_context(response)
        
        return {
            "market_outlook": outlook,
            "key_factors": key_factors,
            "investor_advice": investor_advice,
            "swedish_context": swedish_context,
            "sentiment": GemmaResponseParser._extract_sentiment(response),
            "raw_response": response
        }
    
    @staticmethod
    def parse_rebalancing_advice(response: str) -> Dict[str, Any]:
        """Parse rebalancing recommendation from Gemma."""
        
        # Extract specific actions
        actions = GemmaResponseParser._extract_rebalancing_actions(response)
        
        # Extract timing advice
        timing = GemmaResponseParser._extract_timing_advice(response)
        
        # Extract tax considerations
        tax_advice = GemmaResponseParser._extract_tax_advice(response)
        
        # Extract step-by-step plan
        action_plan = GemmaResponseParser._extract_action_plan(response)
        
        return {
            "actions_needed": actions,
            "timing_advice": timing,
            "tax_considerations": tax_advice,
            "action_plan": action_plan,
            "urgency": GemmaResponseParser._assess_urgency(response),
            "raw_response": response
        }
    
    @staticmethod
    def _extract_allocation(response: str) -> Dict[str, float]:
        """Extract portfolio allocation from response."""
        
        allocation = {}
        
        # Look for percentage patterns
        patterns = [
            r'([A-ZÅÄÖ][a-zåäö\s]+(?:[A-Z][a-z]*)*)\s*[:]\s*(\d+(?:\.\d+)?)\s*%',
            r'([A-ZÅÄÖ][a-zåäö\s]+(?:[A-Z][a-z]*)*)\s*[-–]\s*(\d+(?:\.\d+)?)\s*%',
            r'(\w+(?:\s+\w+)*)\s*:\s*(\d+(?:\.\d+)?)\s*procent',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                fund_name = match.group(1).strip()
                percentage = float(match.group(2))
                
                # Convert to fund ID if possible
                fund_id = GemmaResponseParser._normalize_fund_name(fund_name)
                if fund_id:
                    allocation[fund_id] = percentage / 100
        
        # Normalize allocation to sum to 1.0
        total = sum(allocation.values())
        if total > 0 and abs(total - 1.0) > 0.1:  # Only normalize if significantly off
            allocation = {k: v/total for k, v in allocation.items()}
        
        return allocation
    
    @staticmethod
    def _extract_reasoning(response: str) -> List[str]:
        """Extract reasoning points from response."""
        
        reasoning = []
        
        # Look for bullet points or numbered lists
        patterns = [
            r'[•\-\*]\s*(.+)',
            r'\d+[\.\)]\s*(.+)',
            r'(?:eftersom|för att|på grund av)\s+(.+?)(?:\.|$)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                reason = match.group(1).strip()
                if len(reason) > 10 and reason not in reasoning:  # Avoid duplicates and too short
                    reasoning.append(reason)
        
        return reasoning[:5]  # Limit to top 5 reasons
    
    @staticmethod
    def _extract_risk_level(response: str) -> str:
        """Extract risk level assessment."""
        
        response_lower = response.lower()
        
        # Swedish risk terms
        if any(term in response_lower for term in ['mycket hög risk', 'mycket riskabel']):
            return "Very High"
        elif any(term in response_lower for term in ['hög risk', 'riskabel', 'högrisk']):
            return "High"
        elif any(term in response_lower for term in ['låg risk', 'säker', 'konservativ']):
            return "Low"
        elif any(term in response_lower for term in ['medel', 'medium', 'måttlig']):
            return "Medium"
        
        # English terms as fallback
        if any(term in response_lower for term in ['very high risk', 'extremely risky']):
            return "Very High"
        elif any(term in response_lower for term in ['high risk', 'risky']):
            return "High"
        elif any(term in response_lower for term in ['low risk', 'safe', 'conservative']):
            return "Low"
        
        return "Medium"  # Default
    
    @staticmethod
    def _extract_recommendations(response: str) -> List[str]:
        """Extract specific recommendations."""
        
        recommendations = []
        
        # Look for recommendation patterns
        patterns = [
            r'(?:föreslå|rekommendera|råd|bör)\s+(.+?)(?:\.|$)',
            r'(?:suggest|recommend|advise)\s+(.+?)(?:\.|$)',
            r'(?:överväg|tänk på)\s+(.+?)(?:\.|$)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                rec = match.group(1).strip()
                if len(rec) > 15 and rec not in recommendations:
                    recommendations.append(rec)
        
        return recommendations[:4]  # Limit to top 4
    
    @staticmethod
    def _extract_risk_factors(response: str) -> List[str]:
        """Extract identified risk factors."""
        
        risk_factors = []
        
        # Look for risk-related terms
        risk_terms = [
            'volatilitet', 'volatility', 'svängningar',
            'nedgång', 'förlust', 'drawdown',
            'korrelation', 'concentration', 'koncentration',
            'valuta', 'currency', 'valutarisk',
            'ränterisk', 'interest rate', 'inflation'
        ]
        
        sentences = response.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(term in sentence.lower() for term in risk_terms):
                if len(sentence) > 20:
                    risk_factors.append(sentence)
        
        return risk_factors[:5]
    
    @staticmethod
    def _extract_mitigation_strategies(response: str) -> List[str]:
        """Extract risk mitigation strategies."""
        
        strategies = []
        
        # Look for mitigation patterns
        patterns = [
            r'(?:minska|reducera|minimera)\s+(.+?)(?:\.|$)',
            r'(?:diversifiera|sprida)\s+(.+?)(?:\.|$)',
            r'(?:för att minska|to reduce)\s+(.+?)(?:\.|$)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                strategy = match.group(1).strip()
                if len(strategy) > 10:
                    strategies.append(strategy)
        
        return strategies[:3]
    
    @staticmethod
    def _extract_metrics_interpretation(response: str) -> Dict[str, str]:
        """Extract interpretation of risk metrics."""
        
        interpretations = {}
        
        # Look for Sharpe ratio interpretation
        sharpe_match = re.search(r'sharpe.{0,20}(\d+\.\d+).{0,50}(bra|dålig|hög|låg|bättre|sämre)', response, re.IGNORECASE)
        if sharpe_match:
            interpretations["sharpe_ratio"] = f"Sharpe-kvot {sharpe_match.group(1)} är {sharpe_match.group(2)}"
        
        # Look for volatility interpretation
        vol_match = re.search(r'volatilitet.{0,20}(\d+\.\d+)%.{0,50}(hög|låg|normal|rimlig)', response, re.IGNORECASE)
        if vol_match:
            interpretations["volatility"] = f"Volatilitet {vol_match.group(1)}% är {vol_match.group(2)}"
        
        return interpretations
    
    @staticmethod
    def _extract_summary(response: str) -> str:
        """Extract summary from response."""
        
        # Take first substantial sentence as summary
        sentences = response.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30 and not sentence.startswith(('PORTFÖLJ', 'UPPGIFT', 'FORMAT')):
                return sentence
        
        return response[:200] + "..." if len(response) > 200 else response
    
    @staticmethod
    def _extract_market_outlook(response: str) -> str:
        """Extract market outlook from commentary."""
        
        response_lower = response.lower()
        
        # Swedish sentiment terms
        if any(term in response_lower for term in ['positiv', 'optimistisk', 'uppgång', 'bra']):
            return "Positive"
        elif any(term in response_lower for term in ['negativ', 'pessimistisk', 'nedgång', 'dålig']):
            return "Negative"
        elif any(term in response_lower for term in ['neutral', 'avvaktande', 'osäker']):
            return "Neutral"
        
        return "Neutral"
    
    @staticmethod
    def _extract_key_factors(response: str) -> List[str]:
        """Extract key market factors."""
        
        factors = []
        
        # Economic terms that indicate key factors
        factor_terms = [
            'ränta', 'inflation', 'tillväxt', 'arbetslöshet',
            'bostäder', 'valuta', 'riksbanken', 'fed',
            'energi', 'olja', 'geopolitik'
        ]
        
        sentences = response.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(term in sentence.lower() for term in factor_terms):
                if len(sentence) > 15:
                    factors.append(sentence)
        
        return factors[:4]
    
    @staticmethod
    def _extract_investor_advice(response: str) -> List[str]:
        """Extract specific advice for investors."""
        
        advice = []
        
        # Look for advice patterns
        advice_patterns = [
            r'(?:investerare bör|bör|ska|rekommenderas)\s+(.+?)(?:\.|$)',
            r'(?:råd|tips)\s*[:]\s*(.+?)(?:\.|$)',
            r'(?:överväg|tänk på)\s+(.+?)(?:\.|$)',
        ]
        
        for pattern in advice_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                tip = match.group(1).strip()
                if len(tip) > 10:
                    advice.append(tip)
        
        return advice[:4]
    
    @staticmethod
    def _extract_swedish_context(response: str) -> List[str]:
        """Extract Swedish-specific context and insights."""
        
        swedish_terms = [
            'sverige', 'svensk', 'svenska', 'riksbanken', 'scb',
            'stockholm', 'göteborg', 'malmö', 'sek', 'krona'
        ]
        
        swedish_insights = []
        sentences = response.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(term in sentence.lower() for term in swedish_terms):
                if len(sentence) > 15:
                    swedish_insights.append(sentence)
        
        return swedish_insights[:3]
    
    @staticmethod
    def _extract_sentiment(response: str) -> str:
        """Extract overall sentiment from response."""
        
        response_lower = response.lower()
        
        positive_words = ['bra', 'positiv', 'stark', 'tillväxt', 'möjligheter']
        negative_words = ['dålig', 'negativ', 'svag', 'risk', 'problem']
        
        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)
        
        if positive_count > negative_count:
            return "Positive"
        elif negative_count > positive_count:
            return "Negative"
        else:
            return "Neutral"
    
    @staticmethod
    def _extract_rebalancing_actions(response: str) -> List[str]:
        """Extract specific rebalancing actions."""
        
        actions = []
        
        # Look for action verbs
        action_patterns = [
            r'(?:minska|öka|sälja|köpa|justera)\s+(.+?)(?:\.|$)',
            r'(?:från|till)\s+(\d+\.?\d*%)\s+(.+?)(?:\.|$)',
        ]
        
        for pattern in action_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                action = match.group(0).strip()
                if len(action) > 10:
                    actions.append(action)
        
        return actions[:5]
    
    @staticmethod
    def _extract_timing_advice(response: str) -> str:
        """Extract timing advice for rebalancing."""
        
        timing_terms = [
            'omedelbart', 'direkt', 'snart', 'inom',
            'vänta', 'senare', 'nästa månad', 'kvartal'
        ]
        
        sentences = response.split('.')
        for sentence in sentences:
            if any(term in sentence.lower() for term in timing_terms):
                return sentence.strip()
        
        return "Ingen specifik tidpunkt angiven"
    
    @staticmethod
    def _extract_tax_advice(response: str) -> List[str]:
        """Extract tax-related advice."""
        
        tax_terms = ['skatt', 'tax', 'isk', 'kapitalvinst', 'avdrag']
        
        tax_advice = []
        sentences = response.split('.')
        
        for sentence in sentences:
            if any(term in sentence.lower() for term in tax_terms):
                if len(sentence.strip()) > 15:
                    tax_advice.append(sentence.strip())
        
        return tax_advice[:3]
    
    @staticmethod
    def _extract_action_plan(response: str) -> List[str]:
        """Extract step-by-step action plan."""
        
        plan_steps = []
        
        # Look for numbered or bulleted steps
        step_patterns = [
            r'(\d+[\.\)])\s*(.+?)(?=\d+[\.\)]|$)',
            r'([•\-\*])\s*(.+?)(?=[•\-\*]|$)',
        ]
        
        for pattern in step_patterns:
            matches = re.finditer(pattern, response, re.MULTILINE | re.DOTALL)
            for match in matches:
                step = match.group(2).strip()
                if len(step) > 10:
                    plan_steps.append(step)
        
        return plan_steps[:5]
    
    @staticmethod
    def _assess_confidence(response: str) -> float:
        """Assess confidence level of the response."""
        
        # Simple confidence assessment based on response characteristics
        confidence = 0.5  # Base confidence
        
        # Increase confidence for specific allocations
        if re.search(r'\d+\.\d+%', response):
            confidence += 0.2
        
        # Increase confidence for reasoning
        if any(word in response.lower() for word in ['eftersom', 'för att', 'på grund av']):
            confidence += 0.1
        
        # Decrease confidence for uncertainty terms
        if any(word in response.lower() for word in ['kanske', 'eventuellt', 'osäker']):
            confidence -= 0.2
        
        return max(0.1, min(1.0, confidence))
    
    @staticmethod
    def _assess_urgency(response: str) -> str:
        """Assess urgency of rebalancing recommendation."""
        
        response_lower = response.lower()
        
        if any(term in response_lower for term in ['omedelbart', 'direkt', 'akut']):
            return "High"
        elif any(term in response_lower for term in ['snart', 'inom kort']):
            return "Medium"
        else:
            return "Low"
    
    @staticmethod
    def _normalize_fund_name(fund_name: str) -> Optional[str]:
        """Normalize fund name to fund ID."""
        
        # Mapping from various name formats to fund IDs
        name_mappings = {
            'dnb global': 'DNB_GLOBAL_INDEKS_S',
            'avanza emerging': 'AVANZA_EMERGING_MARKETS',
            'storebrand europa': 'STOREBRAND_EUROPA_A_SEK',
            'dnb norden': 'DNB_NORDEN_INDEKS_S',
            'plus allabolag': 'PLUS_ALLABOLAG_SVERIGE_INDEX',
            'avanza usa': 'AVANZA_USA',
            'storebrand japan': 'STOREBRAND_JAPAN_A_SEK',
            'handelsbanken global': 'HANDELSBANKEN_GLOBAL_SMAB_INDEX',
            'xetra gold': 'XETRA_GOLD_ETC',
            'virtune bitcoin': 'VIRTUNE_BITCOIN_PRIME_ETP',
            'xbt ether': 'XBT_ETHER_ONE',
            'plus fastigheter': 'PLUS_FASTIGHETER_SVERIGE_INDEX'
        }
        
        fund_name_lower = fund_name.lower()
        
        for partial_name, fund_id in name_mappings.items():
            if partial_name in fund_name_lower:
                return fund_id
        
        return None