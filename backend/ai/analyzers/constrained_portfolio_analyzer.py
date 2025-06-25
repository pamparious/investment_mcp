"""
Constrained Portfolio Analyzer for fund-only recommendations.

This analyzer enforces strict constraints to only recommend allocations
using approved tradeable funds from the fund universe.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..config import AIConfig
from ...analysis.patterns import TechnicalAnalyzer
from ...analysis.correlations import CorrelationAnalyzer
from ...analysis.risk_metrics import RiskAnalyzer
from config.fund_universe import (
    TRADEABLE_FUNDS, 
    get_approved_funds,
    get_fund_info,
    get_funds_by_category,
    get_funds_by_risk_level,
    validate_fund_allocation,
    get_diversification_suggestions
)

logger = logging.getLogger(__name__)


class ConstrainedPortfolioAnalyzer:
    """
    Portfolio analyzer that only recommends allocations using approved funds.
    
    This analyzer ensures all portfolio recommendations use only funds from
    the predefined tradeable universe and validates allocations sum to 100%.
    """
    
    def __init__(self, ai_config: AIConfig):
        """
        Initialize the constrained portfolio analyzer.
        
        Args:
            ai_config: AI configuration for provider access
        """
        self.ai_config = ai_config
        self.technical_analyzer = TechnicalAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.approved_funds = get_approved_funds()
    
    async def recommend_portfolio(
        self,
        investment_amount: float,
        risk_tolerance: str,
        investment_horizon: str,
        market_conditions: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate AI-powered portfolio recommendation using only approved funds.
        
        Args:
            investment_amount: Amount to invest in SEK
            risk_tolerance: 'conservative', 'balanced', or 'growth'
            investment_horizon: 'short', 'medium', or 'long'
            market_conditions: Optional current market analysis
            provider_name: AI provider to use
            
        Returns:
            Constrained portfolio recommendation with fund allocations
        """
        try:
            self.logger.info(f"Generating constrained portfolio recommendation")
            self.logger.info(f"Amount: {investment_amount} SEK, Risk: {risk_tolerance}, Horizon: {investment_horizon}")
            
            # Start with base allocation suggestions
            base_suggestions = get_diversification_suggestions()
            base_allocation = base_suggestions.get(risk_tolerance, base_suggestions['balanced'])
            
            # Get fund information for context
            fund_context = self._build_fund_context()
            
            # Get AI-enhanced recommendation
            ai_recommendation = await self._get_ai_portfolio_recommendation(
                investment_amount=investment_amount,
                risk_tolerance=risk_tolerance,
                investment_horizon=investment_horizon,
                base_allocation=base_allocation,
                fund_context=fund_context,
                market_conditions=market_conditions,
                provider_name=provider_name
            )
            
            # Ensure AI recommendation uses only approved funds
            validated_allocation = self._validate_and_constrain_allocation(
                ai_recommendation.get('recommended_allocation', base_allocation['allocation'])
            )
            
            # Calculate fund amounts
            fund_amounts = self._calculate_fund_amounts(validated_allocation, investment_amount)
            
            # Generate comprehensive recommendation
            recommendation = {
                'portfolio_allocation': validated_allocation,
                'fund_amounts': fund_amounts,
                'investment_summary': {
                    'total_amount': investment_amount,
                    'risk_profile': risk_tolerance,
                    'time_horizon': investment_horizon,
                    'number_of_funds': len(validated_allocation),
                    'allocation_strategy': base_allocation['description']
                },
                'fund_details': self._get_allocation_fund_details(validated_allocation),
                'risk_analysis': self._analyze_allocation_risk(validated_allocation),
                'ai_insights': ai_recommendation.get('ai_insights', {}),
                'validation': validate_fund_allocation(validated_allocation),
                'rebalancing_schedule': self._suggest_rebalancing_schedule(investment_horizon),
                'expected_metrics': self._estimate_portfolio_metrics(validated_allocation)
            }
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error in portfolio recommendation: {e}")
            return {
                'error': str(e),
                'fallback_allocation': get_diversification_suggestions()[risk_tolerance]['allocation']
            }
    
    async def validate_portfolio(self, allocation_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate a proposed portfolio allocation against fund constraints.
        
        Args:
            allocation_dict: Dictionary of fund_id -> allocation percentage
            
        Returns:
            Validation results with suggestions for corrections
        """
        try:
            validation_result = validate_fund_allocation(allocation_dict)
            
            # Add detailed analysis
            validation_result.update({
                'detailed_analysis': {
                    'approved_funds_used': [f for f in allocation_dict.keys() if f in self.approved_funds],
                    'unapproved_funds_used': [f for f in allocation_dict.keys() if f not in self.approved_funds],
                    'fund_details': self._get_allocation_fund_details(allocation_dict),
                    'risk_distribution': self._analyze_risk_distribution(allocation_dict),
                    'category_distribution': self._analyze_category_distribution(allocation_dict)
                }
            })
            
            # Generate suggestions if invalid
            if not validation_result['valid']:
                validation_result['suggestions'] = await self._generate_correction_suggestions(allocation_dict)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating portfolio: {e}")
            return {'valid': False, 'errors': [str(e)]}
    
    def list_available_funds(self, category: Optional[str] = None, risk_level: Optional[str] = None) -> Dict[str, Any]:
        """
        List all available funds with optional filtering.
        
        Args:
            category: Optional category filter
            risk_level: Optional risk level filter
            
        Returns:
            Dictionary of available funds with details
        """
        try:
            funds_to_include = self.approved_funds
            
            # Apply filters
            if category:
                category_funds = get_funds_by_category(category)
                funds_to_include = [f for f in funds_to_include if f in category_funds]
            
            if risk_level:
                risk_funds = get_funds_by_risk_level(risk_level)
                funds_to_include = [f for f in funds_to_include if f in risk_funds]
            
            # Build fund list with details
            fund_list = {}
            for fund_id in funds_to_include:
                fund_info = get_fund_info(fund_id)
                if fund_info:
                    fund_list[fund_id] = fund_info
            
            return {
                'total_funds': len(fund_list),
                'filter_applied': {
                    'category': category,
                    'risk_level': risk_level
                },
                'funds': fund_list,
                'categories_available': list(set(fund['category'] for fund in fund_list.values())),
                'risk_levels_available': list(set(fund['risk_level'] for fund in fund_list.values()))
            }
            
        except Exception as e:
            self.logger.error(f"Error listing funds: {e}")
            return {'error': str(e)}
    
    def _build_fund_context(self) -> Dict[str, Any]:
        """Build context about available funds for AI analysis."""
        fund_context = {
            'total_funds': len(self.approved_funds),
            'categories': {},
            'risk_levels': {},
            'expense_ratios': [],
            'fund_details': []
        }
        
        for fund_id in self.approved_funds:
            fund_info = get_fund_info(fund_id)
            if fund_info:
                # Category distribution
                category = fund_info['category']
                if category not in fund_context['categories']:
                    fund_context['categories'][category] = 0
                fund_context['categories'][category] += 1
                
                # Risk level distribution
                risk_level = fund_info['risk_level']
                if risk_level not in fund_context['risk_levels']:
                    fund_context['risk_levels'][risk_level] = 0
                fund_context['risk_levels'][risk_level] += 1
                
                # Expense ratios
                fund_context['expense_ratios'].append(fund_info['expense_ratio'])
                
                # Fund details for AI
                fund_context['fund_details'].append({
                    'id': fund_id,
                    'name': fund_info['name'],
                    'category': fund_info['category'],
                    'risk_level': fund_info['risk_level'],
                    'expense_ratio': fund_info['expense_ratio']
                })
        
        # Calculate averages
        if fund_context['expense_ratios']:
            fund_context['average_expense_ratio'] = sum(fund_context['expense_ratios']) / len(fund_context['expense_ratios'])
        
        return fund_context
    
    async def _get_ai_portfolio_recommendation(
        self,
        investment_amount: float,
        risk_tolerance: str,
        investment_horizon: str,
        base_allocation: Dict[str, Any],
        fund_context: Dict[str, Any],
        market_conditions: Optional[Dict[str, Any]],
        provider_name: Optional[str]
    ) -> Dict[str, Any]:
        """Get AI-enhanced portfolio recommendation."""
        try:
            provider = self.ai_config.get_provider(provider_name)
            
            # Prepare context for AI
            ai_context = {
                'investment_parameters': {
                    'amount': investment_amount,
                    'risk_tolerance': risk_tolerance,
                    'time_horizon': investment_horizon
                },
                'base_suggestion': base_allocation,
                'available_funds': fund_context,
                'market_conditions': market_conditions or {},
                'constraints': {
                    'must_use_only_approved_funds': True,
                    'must_sum_to_100_percent': True,
                    'approved_fund_ids': self.approved_funds
                }
            }
            
            async with provider:
                ai_insights = await provider.generate_insights(ai_context, "constrained_portfolio_recommendation")
            
            return ai_insights
            
        except Exception as e:
            self.logger.error(f"Error getting AI recommendation: {e}")
            return {'error': str(e)}
    
    def _validate_and_constrain_allocation(self, proposed_allocation: Dict[str, float]) -> Dict[str, float]:
        """Validate and constrain allocation to approved funds only."""
        constrained_allocation = {}
        
        # Only include approved funds
        for fund_id, allocation in proposed_allocation.items():
            if fund_id in self.approved_funds and allocation > 0:
                constrained_allocation[fund_id] = allocation
        
        # Normalize to sum to 1.0
        total_allocation = sum(constrained_allocation.values())
        if total_allocation > 0:
            constrained_allocation = {
                fund_id: allocation / total_allocation 
                for fund_id, allocation in constrained_allocation.items()
            }
        else:
            # Fallback to balanced allocation
            constrained_allocation = get_diversification_suggestions()['balanced']['allocation']
        
        return constrained_allocation
    
    def _calculate_fund_amounts(self, allocation: Dict[str, float], total_amount: float) -> Dict[str, float]:
        """Calculate actual fund amounts in SEK."""
        return {
            fund_id: allocation_pct * total_amount
            for fund_id, allocation_pct in allocation.items()
        }
    
    def _get_allocation_fund_details(self, allocation: Dict[str, float]) -> Dict[str, Any]:
        """Get detailed information for funds in allocation."""
        fund_details = {}
        
        for fund_id, allocation_pct in allocation.items():
            fund_info = get_fund_info(fund_id)
            if fund_info:
                fund_details[fund_id] = {
                    **fund_info,
                    'allocation_percentage': allocation_pct * 100,
                    'is_approved': fund_id in self.approved_funds
                }
        
        return fund_details
    
    def _analyze_allocation_risk(self, allocation: Dict[str, float]) -> Dict[str, Any]:
        """Analyze risk characteristics of allocation."""
        risk_analysis = {
            'risk_distribution': {},
            'weighted_average_expense_ratio': 0,
            'category_concentration': {},
            'overall_risk_score': 0
        }
        
        risk_weights = {'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}
        total_risk_score = 0
        total_expense_ratio = 0
        
        for fund_id, allocation_pct in allocation.items():
            fund_info = get_fund_info(fund_id)
            if fund_info:
                # Risk distribution
                risk_level = fund_info['risk_level']
                if risk_level not in risk_analysis['risk_distribution']:
                    risk_analysis['risk_distribution'][risk_level] = 0
                risk_analysis['risk_distribution'][risk_level] += allocation_pct
                
                # Category concentration
                category = fund_info['category']
                if category not in risk_analysis['category_concentration']:
                    risk_analysis['category_concentration'][category] = 0
                risk_analysis['category_concentration'][category] += allocation_pct
                
                # Weighted metrics
                total_risk_score += risk_weights.get(risk_level, 2) * allocation_pct
                total_expense_ratio += fund_info['expense_ratio'] * allocation_pct
        
        risk_analysis['overall_risk_score'] = total_risk_score
        risk_analysis['weighted_average_expense_ratio'] = total_expense_ratio
        
        return risk_analysis
    
    def _analyze_risk_distribution(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Analyze risk level distribution."""
        risk_distribution = {}
        
        for fund_id, allocation_pct in allocation.items():
            fund_info = get_fund_info(fund_id)
            if fund_info:
                risk_level = fund_info['risk_level']
                if risk_level not in risk_distribution:
                    risk_distribution[risk_level] = 0
                risk_distribution[risk_level] += allocation_pct
        
        return risk_distribution
    
    def _analyze_category_distribution(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Analyze category distribution."""
        category_distribution = {}
        
        for fund_id, allocation_pct in allocation.items():
            fund_info = get_fund_info(fund_id)
            if fund_info:
                category = fund_info['category']
                if category not in category_distribution:
                    category_distribution[category] = 0
                category_distribution[category] += allocation_pct
        
        return category_distribution
    
    async def _generate_correction_suggestions(self, invalid_allocation: Dict[str, float]) -> List[str]:
        """Generate suggestions to correct invalid allocation."""
        suggestions = []
        
        # Check for unapproved funds
        unapproved_funds = [f for f in invalid_allocation.keys() if f not in self.approved_funds]
        if unapproved_funds:
            suggestions.append(f"Remove unapproved funds: {', '.join(unapproved_funds)}")
            
            # Suggest approved alternatives
            for unapproved_fund in unapproved_funds:
                # Simple category-based suggestion logic
                suggestions.append(f"Consider approved alternatives from fund universe")
        
        # Check allocation sum
        total = sum(invalid_allocation.values())
        if abs(total - 1.0) > 0.001:
            if total > 1.0:
                suggestions.append(f"Reduce allocations by {(total - 1.0) * 100:.1f}%")
            else:
                suggestions.append(f"Increase allocations by {(1.0 - total) * 100:.1f}%")
        
        return suggestions
    
    def _suggest_rebalancing_schedule(self, investment_horizon: str) -> Dict[str, Any]:
        """Suggest rebalancing schedule based on horizon."""
        schedules = {
            'short': {'frequency': 'monthly', 'threshold': 0.05},
            'medium': {'frequency': 'quarterly', 'threshold': 0.10},
            'long': {'frequency': 'semi-annually', 'threshold': 0.15}
        }
        
        return schedules.get(investment_horizon, schedules['medium'])
    
    def _estimate_portfolio_metrics(self, allocation: Dict[str, float]) -> Dict[str, Any]:
        """Estimate basic portfolio metrics based on allocation."""
        # This is a simplified estimation - in practice you'd use historical data
        risk_weights = {'low': 0.05, 'medium': 0.12, 'high': 0.20, 'very_high': 0.30}
        expense_total = 0
        estimated_volatility = 0
        
        for fund_id, allocation_pct in allocation.items():
            fund_info = get_fund_info(fund_id)
            if fund_info:
                expense_total += fund_info['expense_ratio'] * allocation_pct
                estimated_volatility += risk_weights.get(fund_info['risk_level'], 0.12) * allocation_pct
        
        return {
            'estimated_annual_fee': expense_total,
            'estimated_volatility': estimated_volatility,
            'diversification_score': min(len(allocation) / 6.0, 1.0)  # Max score with 6+ funds
        }