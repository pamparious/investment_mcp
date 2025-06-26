"""
Portfolio constraints for Investment MCP System optimization.

This module defines comprehensive constraints for portfolio optimization
with Swedish market requirements and risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from ..config import TRADEABLE_FUNDS


class PortfolioConstraints:
    """Portfolio constraints for optimization algorithms."""
    
    def __init__(self):
        # Swedish market specific parameters
        self.max_single_position = {
            "low": 0.25,        # 25% max in any single position
            "medium": 0.35,     # 35% max for moderate risk
            "high": 0.50,       # 50% max for aggressive
            "very_high": 0.75   # 75% max for very aggressive
        }
        
        self.min_position_size = {
            "low": 0.02,        # 2% minimum position (focus)
            "medium": 0.01,     # 1% minimum position
            "high": 0.005,      # 0.5% minimum position
            "very_high": 0.001  # 0.1% minimum position
        }
        
        # Swedish regulatory and tax considerations
        self.crypto_max_allocation = {
            "low": 0.0,         # No crypto for conservative
            "medium": 0.05,     # 5% max crypto
            "high": 0.15,       # 15% max crypto
            "very_high": 0.30   # 30% max crypto
        }
        
        # Diversification requirements
        self.min_number_of_positions = {
            "low": 3,           # At least 3 positions
            "medium": 5,        # At least 5 positions
            "high": 5,          # At least 5 positions
            "very_high": 3      # Can concentrate more
        }
        
        self.sector_concentration_limits = {
            "swedish_equity": 0.40,      # Max 40% in Swedish equity
            "cryptocurrency": 0.30,      # Max 30% in crypto
            "precious_metals": 0.20,     # Max 20% in gold/commodities
            "real_estate": 0.25,         # Max 25% in real estate
            "single_country": 0.60       # Max 60% in any single country
        }
    
    def get_weight_bounds(self, risk_tolerance: str, n_assets: int) -> List[Tuple[float, float]]:
        """Get weight bounds for each asset based on risk tolerance."""
        
        min_weight = self.min_position_size.get(risk_tolerance, 0.01)
        max_weight = self.max_single_position.get(risk_tolerance, 0.35)
        
        # Adjust for number of assets - can't have too many minimum positions
        max_positions = min(n_assets, 1.0 / min_weight)
        if n_assets > max_positions:
            min_weight = 0.0  # Allow zero weights if too many assets
        
        bounds = []
        for i in range(n_assets):
            bounds.append((0.0, max_weight))  # Always allow zero, respect max
        
        return bounds
    
    def get_risk_constraints(self, risk_tolerance: str, n_assets: int) -> List[Dict[str, Any]]:
        """Get risk-based constraints for optimization."""
        
        constraints = []
        
        # Maximum concentration constraint
        max_weight = self.max_single_position.get(risk_tolerance, 0.35)
        
        def max_concentration_constraint(weights):
            return max_weight - np.max(weights)
        
        constraints.append({
            'type': 'ineq',
            'fun': max_concentration_constraint
        })
        
        # Minimum diversification constraint (at least N non-zero positions)
        min_positions = self.min_number_of_positions.get(risk_tolerance, 3)
        
        def min_positions_constraint(weights):
            non_zero_positions = np.sum(weights > 0.001)  # Count positions > 0.1%
            return non_zero_positions - min_positions
        
        constraints.append({
            'type': 'ineq', 
            'fun': min_positions_constraint
        })
        
        return constraints
    
    def get_sector_constraints(
        self, 
        fund_ids: List[str], 
        risk_tolerance: str
    ) -> List[Dict[str, Any]]:
        """Get sector-based allocation constraints."""
        
        constraints = []
        
        # Cryptocurrency constraint
        crypto_funds = self._get_funds_by_category(fund_ids, "cryptocurrency")
        if crypto_funds:
            crypto_max = self.crypto_max_allocation.get(risk_tolerance, 0.15)
            
            def crypto_constraint(weights):
                crypto_allocation = sum(weights[i] for i in crypto_funds)
                return crypto_max - crypto_allocation
            
            constraints.append({
                'type': 'ineq',
                'fun': crypto_constraint
            })
        
        # Swedish equity constraint (prevent home bias)
        swedish_funds = self._get_funds_by_category(fund_ids, "swedish_equity")
        if swedish_funds:
            swedish_max = self.sector_concentration_limits["swedish_equity"]
            
            def swedish_constraint(weights):
                swedish_allocation = sum(weights[i] for i in swedish_funds)
                return swedish_max - swedish_allocation
            
            constraints.append({
                'type': 'ineq',
                'fun': swedish_constraint
            })
        
        # Precious metals constraint
        commodity_funds = self._get_funds_by_category(fund_ids, "precious_metals")
        if commodity_funds:
            commodity_max = self.sector_concentration_limits["precious_metals"]
            
            def commodity_constraint(weights):
                commodity_allocation = sum(weights[i] for i in commodity_funds)
                return commodity_max - commodity_allocation
            
            constraints.append({
                'type': 'ineq',
                'fun': commodity_constraint
            })
        
        # Real estate constraint
        real_estate_funds = self._get_funds_by_category(fund_ids, "real_estate")
        if real_estate_funds:
            real_estate_max = self.sector_concentration_limits["real_estate"]
            
            def real_estate_constraint(weights):
                real_estate_allocation = sum(weights[i] for i in real_estate_funds)
                return real_estate_max - real_estate_allocation
            
            constraints.append({
                'type': 'ineq',
                'fun': real_estate_constraint
            })
        
        return constraints
    
    def get_tax_efficiency_constraints(
        self, 
        fund_ids: List[str], 
        account_type: str = "ISK"
    ) -> List[Dict[str, Any]]:
        """Get tax efficiency constraints for Swedish accounts."""
        
        constraints = []
        
        if account_type == "ISK":
            # ISK accounts have limitations on certain instruments
            # Cryptocurrency ETPs are generally OK, but some derivatives are not
            # For simplicity, we'll focus on keeping total allocation reasonable
            
            # No specific constraints needed for ISK with our fund universe
            pass
        
        elif account_type == "AF":
            # AF (pension) accounts have different rules
            # Usually more conservative requirements
            
            # Limit high-risk assets in pension accounts
            high_risk_funds = []
            for i, fund_id in enumerate(fund_ids):
                fund_info = TRADEABLE_FUNDS.get(fund_id, {})
                if fund_info.get("risk_level") in ["very_high", "high"]:
                    high_risk_funds.append(i)
            
            if high_risk_funds:
                def af_risk_constraint(weights):
                    high_risk_allocation = sum(weights[i] for i in high_risk_funds)
                    return 0.25 - high_risk_allocation  # Max 25% high risk in AF
                
                constraints.append({
                    'type': 'ineq',
                    'fun': af_risk_constraint
                })
        
        return constraints
    
    def get_turnover_constraints(
        self, 
        current_weights: np.ndarray, 
        risk_tolerance: str, 
        max_turnover: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get turnover constraints to limit transaction costs."""
        
        if max_turnover is None:
            # Default turnover limits by risk tolerance
            max_turnover = {
                "low": 0.20,        # Very conservative turnover
                "medium": 0.40,     # Moderate turnover
                "high": 0.60,       # Higher turnover acceptable
                "very_high": 1.0    # No turnover constraint
            }.get(risk_tolerance, 0.40)
        
        constraints = []
        
        if max_turnover < 1.0:  # Only apply if there's a limit
            def turnover_constraint(weights):
                turnover = np.sum(np.abs(weights - current_weights))
                return max_turnover - turnover
            
            constraints.append({
                'type': 'ineq',
                'fun': turnover_constraint
            })
        
        return constraints
    
    def get_leverage_constraints(self, risk_tolerance: str) -> List[Dict[str, Any]]:
        """Get leverage constraints (for cash/margin positions)."""
        
        constraints = []
        
        # Most retail Swedish investors should not use leverage
        # Only very aggressive strategies might consider small leverage
        
        max_leverage = {
            "low": 1.0,         # No leverage
            "medium": 1.0,      # No leverage
            "high": 1.1,        # 10% leverage max
            "very_high": 1.25   # 25% leverage max
        }.get(risk_tolerance, 1.0)
        
        def leverage_constraint(weights):
            total_allocation = np.sum(weights)
            return max_leverage - total_allocation
        
        constraints.append({
            'type': 'ineq',
            'fun': leverage_constraint
        })
        
        return constraints
    
    def get_volatility_constraints(
        self, 
        covariance_matrix: pd.DataFrame, 
        max_volatility: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get portfolio volatility constraints."""
        
        constraints = []
        
        if max_volatility is not None:
            def volatility_constraint(weights):
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix.values, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                return max_volatility - portfolio_volatility
            
            constraints.append({
                'type': 'ineq',
                'fun': volatility_constraint
            })
        
        return constraints
    
    def get_return_constraints(
        self, 
        expected_returns: pd.Series, 
        min_expected_return: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get minimum expected return constraints."""
        
        constraints = []
        
        if min_expected_return is not None:
            def return_constraint(weights):
                portfolio_return = np.dot(weights, expected_returns.values)
                return portfolio_return - min_expected_return
            
            constraints.append({
                'type': 'ineq',
                'fun': return_constraint
            })
        
        return constraints
    
    def get_esg_constraints(self, fund_ids: List[str]) -> List[Dict[str, Any]]:
        """Get ESG-based constraints (placeholder for future implementation)."""
        
        # This would require ESG data for each fund
        # For now, return empty constraints
        constraints = []
        
        # Future implementation could include:
        # - Minimum ESG score requirements
        # - Exclusion of certain sectors (tobacco, weapons, etc.)
        # - Minimum sustainable investing allocation
        
        return constraints
    
    def validate_portfolio_allocation(
        self, 
        allocation: Dict[str, float], 
        risk_tolerance: str
    ) -> Dict[str, Any]:
        """Validate a portfolio allocation against all constraints."""
        
        errors = []
        warnings = []
        
        # Basic validation
        total_allocation = sum(allocation.values())
        if abs(total_allocation - 1.0) > 0.001:
            errors.append(f"Total allocation {total_allocation:.1%} != 100%")
        
        # Check individual position limits
        max_position = self.max_single_position.get(risk_tolerance, 0.35)
        for fund, weight in allocation.items():
            if weight > max_position:
                errors.append(f"{fund} allocation {weight:.1%} exceeds max {max_position:.1%}")
            
            if weight < 0:
                errors.append(f"{fund} has negative allocation {weight:.1%}")
        
        # Check sector concentrations
        sector_allocations = self._calculate_sector_allocations(allocation)
        
        for sector, allocation_pct in sector_allocations.items():
            if sector in self.sector_concentration_limits:
                limit = self.sector_concentration_limits[sector]
                if allocation_pct > limit:
                    errors.append(f"{sector} allocation {allocation_pct:.1%} exceeds limit {limit:.1%}")
        
        # Check cryptocurrency limits
        crypto_allocation = sector_allocations.get("cryptocurrency", 0)
        crypto_limit = self.crypto_max_allocation.get(risk_tolerance, 0.15)
        if crypto_allocation > crypto_limit:
            errors.append(f"Cryptocurrency allocation {crypto_allocation:.1%} exceeds limit {crypto_limit:.1%}")
        
        # Check minimum diversification
        min_positions = self.min_number_of_positions.get(risk_tolerance, 3)
        significant_positions = sum(1 for weight in allocation.values() if weight > 0.01)
        if significant_positions < min_positions:
            warnings.append(f"Only {significant_positions} significant positions, recommend at least {min_positions}")
        
        # Check fund eligibility
        approved_funds = set(TRADEABLE_FUNDS.keys())
        for fund in allocation.keys():
            if fund not in approved_funds:
                errors.append(f"Fund {fund} not in approved universe")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "total_allocation": total_allocation,
            "sector_breakdown": sector_allocations,
            "significant_positions": significant_positions
        }
    
    def _get_funds_by_category(self, fund_ids: List[str], category: str) -> List[int]:
        """Get indices of funds in a specific category."""
        
        indices = []
        for i, fund_id in enumerate(fund_ids):
            fund_info = TRADEABLE_FUNDS.get(fund_id, {})
            fund_category = fund_info.get("category", "")
            fund_type = fund_info.get("type", "")
            
            if category == "cryptocurrency" and fund_category == "cryptocurrency":
                indices.append(i)
            elif category == "swedish_equity" and ("swedish" in fund_type or "sverige" in fund_id.lower()):
                indices.append(i)
            elif category == "precious_metals" and fund_category == "precious_metals":
                indices.append(i)
            elif category == "real_estate" and fund_category == "real_estate":
                indices.append(i)
        
        return indices
    
    def _calculate_sector_allocations(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Calculate allocation by sector/category."""
        
        sector_allocations = {
            "swedish_equity": 0,
            "nordic_equity": 0,
            "global_equity": 0,
            "cryptocurrency": 0,
            "precious_metals": 0,
            "real_estate": 0,
            "emerging_markets": 0,
            "other": 0
        }
        
        for fund_id, weight in allocation.items():
            fund_info = TRADEABLE_FUNDS.get(fund_id, {})
            category = fund_info.get("category", "other")
            fund_type = fund_info.get("type", "")
            
            # Map to our sector categories
            if "swedish" in fund_type or "sverige" in fund_id.lower():
                sector_allocations["swedish_equity"] += weight
            elif "nordic" in fund_type or "norden" in fund_id.lower():
                sector_allocations["nordic_equity"] += weight
            elif category == "cryptocurrency":
                sector_allocations["cryptocurrency"] += weight
            elif category == "precious_metals":
                sector_allocations["precious_metals"] += weight
            elif category == "real_estate":
                sector_allocations["real_estate"] += weight
            elif category == "emerging_markets":
                sector_allocations["emerging_markets"] += weight
            elif category in ["global_equity", "us_equity", "european_equity", "japanese_equity"]:
                sector_allocations["global_equity"] += weight
            else:
                sector_allocations["other"] += weight
        
        return sector_allocations
    
    def get_comprehensive_constraints(
        self,
        fund_ids: List[str],
        risk_tolerance: str,
        covariance_matrix: Optional[pd.DataFrame] = None,
        expected_returns: Optional[pd.Series] = None,
        current_weights: Optional[np.ndarray] = None,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get comprehensive set of constraints for optimization."""
        
        all_constraints = []
        
        # Basic risk constraints
        all_constraints.extend(self.get_risk_constraints(risk_tolerance, len(fund_ids)))
        
        # Sector constraints
        all_constraints.extend(self.get_sector_constraints(fund_ids, risk_tolerance))
        
        # Tax efficiency constraints
        all_constraints.extend(self.get_tax_efficiency_constraints(fund_ids))
        
        # Turnover constraints
        if current_weights is not None:
            all_constraints.extend(self.get_turnover_constraints(current_weights, risk_tolerance))
        
        # Leverage constraints
        all_constraints.extend(self.get_leverage_constraints(risk_tolerance))
        
        # Volatility constraints
        if covariance_matrix is not None and custom_constraints and "max_volatility" in custom_constraints:
            all_constraints.extend(self.get_volatility_constraints(
                covariance_matrix, custom_constraints["max_volatility"]
            ))
        
        # Return constraints
        if expected_returns is not None and custom_constraints and "min_expected_return" in custom_constraints:
            all_constraints.extend(self.get_return_constraints(
                expected_returns, custom_constraints["min_expected_return"]
            ))
        
        # Custom constraints
        if custom_constraints and "additional_constraints" in custom_constraints:
            all_constraints.extend(custom_constraints["additional_constraints"])
        
        return all_constraints