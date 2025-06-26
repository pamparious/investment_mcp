"""
Swedish Housing Market Analysis for Investment MCP System.

This module provides comprehensive analysis of Swedish housing market
data and comparison with portfolio investment alternatives.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SwedishMortgageTerms:
    """Swedish mortgage terms and conditions."""
    interest_rate: float
    down_payment_pct: float = 0.15  # 15% minimum down payment
    mortgage_rate_type: str = "variable"  # variable, fixed_3y, fixed_5y
    amortization_required: bool = True
    loan_to_value_limit: float = 0.85  # 85% max LTV
    stress_test_rate: float = None  # If None, calculated as rate + 3%


@dataclass
class SwedishHousingCosts:
    """Comprehensive Swedish housing costs."""
    monthly_mortgage: float
    monthly_interest: float
    monthly_amortization: float
    monthly_hoa_fee: float  # avgift
    monthly_property_tax: float  # fastighetsavgift
    annual_maintenance_pct: float = 0.01  # 1% of property value
    annual_insurance: float = 5000  # SEK
    buying_costs_pct: float = 0.015  # 1.5% buying costs (pantbrev, lagfart, etc.)
    selling_costs_pct: float = 0.05  # 5% selling costs (mäklare, etc.)


@dataclass
class SwedishRegionalData:
    """Regional Swedish housing market data."""
    region: str
    average_price_per_sqm: float
    annual_appreciation_rate: float
    rental_yield: float
    average_hoa_fee_per_sqm: float
    property_tax_rate: float
    market_liquidity: str  # "high", "medium", "low"
    price_volatility: float


class SwedishHousingAnalyzer:
    """Comprehensive Swedish housing market analyzer."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Swedish tax parameters
        self.capital_gains_tax_rate = 0.22  # 22% on housing capital gains
        self.interest_deduction_rate = 0.30  # 30% interest deduction
        self.property_tax_rate = 0.0075  # 0.75% property tax (average)
        
        # Swedish mortgage parameters
        self.riksbank_rate = 0.025  # Current Riksbank rate approximation
        self.mortgage_margin = 0.015  # Typical bank margin
        self.stress_test_addition = 0.03  # 3% stress test addition
        
        # Regional data (example data - would be updated from real sources)
        self.regional_data = self._get_regional_housing_data()
    
    def analyze_housing_vs_investment(
        self,
        property_price: float,
        monthly_rent_savings: float,
        portfolio_allocation: Dict[str, float],
        expected_portfolio_return: float,
        portfolio_volatility: float,
        analysis_period_years: int = 10,
        region: str = "Stockholm",
        mortgage_terms: Optional[SwedishMortgageTerms] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis comparing housing purchase vs portfolio investment.
        
        Args:
            property_price: Property purchase price in SEK
            monthly_rent_savings: Monthly rent that would be saved by owning
            portfolio_allocation: Portfolio allocation for comparison
            expected_portfolio_return: Expected annual portfolio return
            portfolio_volatility: Portfolio volatility
            analysis_period_years: Analysis time horizon
            region: Swedish region for housing data
            mortgage_terms: Mortgage terms (if None, uses defaults)
            
        Returns:
            Comprehensive housing vs investment analysis
        """
        
        try:
            # Set default mortgage terms if not provided
            if mortgage_terms is None:
                current_rate = self.riksbank_rate + self.mortgage_margin
                mortgage_terms = SwedishMortgageTerms(
                    interest_rate=current_rate,
                    stress_test_rate=current_rate + self.stress_test_addition
                )
            
            # Get regional data
            regional_info = self.regional_data.get(region, self.regional_data["Stockholm"])
            
            # Calculate housing scenario
            housing_analysis = self._analyze_housing_scenario(
                property_price, monthly_rent_savings, mortgage_terms,
                regional_info, analysis_period_years
            )
            
            # Calculate investment scenario
            investment_analysis = self._analyze_investment_scenario(
                housing_analysis["initial_capital_required"],
                monthly_rent_savings,
                expected_portfolio_return,
                portfolio_volatility,
                analysis_period_years
            )
            
            # Monte Carlo comparison
            monte_carlo_comparison = self._monte_carlo_housing_vs_investment(
                housing_analysis, investment_analysis, analysis_period_years
            )
            
            # Tax implications analysis
            tax_analysis = self._analyze_tax_implications(
                housing_analysis, investment_analysis, analysis_period_years
            )
            
            # Risk analysis
            risk_analysis = self._analyze_housing_investment_risks(
                housing_analysis, investment_analysis, regional_info
            )
            
            # Swedish-specific considerations
            swedish_considerations = self._analyze_swedish_considerations(
                property_price, monthly_rent_savings, region, mortgage_terms
            )
            
            # Final recommendation
            recommendation = self._generate_recommendation(
                housing_analysis, investment_analysis, monte_carlo_comparison,
                risk_analysis, regional_info
            )
            
            return {
                "success": True,
                "analysis_type": "swedish_housing_vs_investment",
                "analysis_parameters": {
                    "property_price": property_price,
                    "monthly_rent_savings": monthly_rent_savings,
                    "expected_portfolio_return": expected_portfolio_return,
                    "portfolio_volatility": portfolio_volatility,
                    "analysis_period_years": analysis_period_years,
                    "region": region
                },
                "housing_scenario": housing_analysis,
                "investment_scenario": investment_analysis,
                "monte_carlo_comparison": monte_carlo_comparison,
                "tax_analysis": tax_analysis,
                "risk_analysis": risk_analysis,
                "swedish_considerations": swedish_considerations,
                "recommendation": recommendation,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Housing vs investment analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def calculate_mortgage_affordability(
        self,
        annual_income: float,
        monthly_expenses: float,
        existing_debt: float = 0,
        down_payment_available: float = 0,
        mortgage_terms: Optional[SwedishMortgageTerms] = None
    ) -> Dict[str, Any]:
        """Calculate mortgage affordability under Swedish regulations."""
        
        try:
            if mortgage_terms is None:
                current_rate = self.riksbank_rate + self.mortgage_margin
                mortgage_terms = SwedishMortgageTerms(
                    interest_rate=current_rate,
                    stress_test_rate=current_rate + self.stress_test_addition
                )
            
            # Swedish mortgage regulations
            max_debt_to_income = 4.5  # 4.5x annual income max
            max_debt_service_ratio = 0.55  # 55% of income max for all debt service
            stress_test_rate = mortgage_terms.stress_test_rate or (mortgage_terms.interest_rate + 0.03)
            
            # Calculate maximum loan amount
            max_loan_by_income = annual_income * max_debt_to_income
            
            # Calculate based on debt service ratio
            monthly_income = annual_income / 12
            available_for_debt_service = (monthly_income * max_debt_service_ratio) - monthly_expenses - existing_debt
            
            # Calculate maximum loan based on stress test rate
            max_loan_by_stress_test = self._calculate_max_loan_amount(
                available_for_debt_service, stress_test_rate, 50  # 50 year amortization for calculation
            )
            
            # Actual loan limit is the minimum of all constraints
            max_loan_amount = min(max_loan_by_income, max_loan_by_stress_test)
            
            # Calculate maximum property price
            max_property_price = (max_loan_amount + down_payment_available) / mortgage_terms.loan_to_value_limit
            
            # Monthly payment calculations
            monthly_payment_stress = self._calculate_monthly_payment(
                max_loan_amount, stress_test_rate, 25  # 25 year amortization
            )
            
            monthly_payment_actual = self._calculate_monthly_payment(
                max_loan_amount, mortgage_terms.interest_rate, 25
            )
            
            # Affordability analysis
            affordability_ratio = monthly_payment_actual / monthly_income
            stress_test_ratio = monthly_payment_stress / monthly_income
            
            return {
                "success": True,
                "max_loan_amount": float(max_loan_amount),
                "max_property_price": float(max_property_price),
                "down_payment_required": float(max_property_price * mortgage_terms.down_payment_pct),
                "monthly_payment_actual": float(monthly_payment_actual),
                "monthly_payment_stress_test": float(monthly_payment_stress),
                "affordability_ratios": {
                    "debt_service_ratio": float(affordability_ratio),
                    "stress_test_ratio": float(stress_test_ratio),
                    "debt_to_income_ratio": float(max_loan_amount / annual_income)
                },
                "constraints": {
                    "max_loan_by_income": float(max_loan_by_income),
                    "max_loan_by_stress_test": float(max_loan_by_stress_test),
                    "binding_constraint": "income_multiple" if max_loan_by_income < max_loan_by_stress_test else "stress_test"
                },
                "mortgage_terms": {
                    "interest_rate": mortgage_terms.interest_rate,
                    "stress_test_rate": stress_test_rate,
                    "loan_to_value_limit": mortgage_terms.loan_to_value_limit,
                    "down_payment_pct": mortgage_terms.down_payment_pct
                },
                "calculated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Mortgage affordability calculation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def analyze_rent_vs_buy_decision(
        self,
        property_price: float,
        monthly_rent: float,
        region: str = "Stockholm",
        analysis_period_years: int = 5,
        mortgage_terms: Optional[SwedishMortgageTerms] = None
    ) -> Dict[str, Any]:
        """Analyze rent vs buy decision for a specific property."""
        
        try:
            if mortgage_terms is None:
                current_rate = self.riksbank_rate + self.mortgage_margin
                mortgage_terms = SwedishMortgageTerms(interest_rate=current_rate)
            
            regional_info = self.regional_data.get(region, self.regional_data["Stockholm"])
            
            # Calculate ownership costs
            down_payment = property_price * mortgage_terms.down_payment_pct
            loan_amount = property_price - down_payment
            
            # Monthly mortgage payment
            monthly_mortgage = self._calculate_monthly_payment(
                loan_amount, mortgage_terms.interest_rate, 25
            )
            
            # Calculate all housing costs
            housing_costs = SwedishHousingCosts(
                monthly_mortgage=monthly_mortgage,
                monthly_interest=loan_amount * mortgage_terms.interest_rate / 12,
                monthly_amortization=monthly_mortgage - (loan_amount * mortgage_terms.interest_rate / 12),
                monthly_hoa_fee=regional_info.average_hoa_fee_per_sqm * 75,  # Assume 75 sqm
                monthly_property_tax=property_price * self.property_tax_rate / 12,
                annual_insurance=5000
            )
            
            # Total monthly ownership cost
            total_monthly_cost = (
                housing_costs.monthly_mortgage +
                housing_costs.monthly_hoa_fee +
                housing_costs.monthly_property_tax +
                housing_costs.annual_insurance / 12 +
                property_price * housing_costs.annual_maintenance_pct / 12
            )
            
            # Calculate net costs after tax benefits
            annual_interest = housing_costs.monthly_interest * 12
            interest_tax_benefit = annual_interest * self.interest_deduction_rate
            net_annual_ownership_cost = (total_monthly_cost * 12) - interest_tax_benefit
            net_monthly_ownership_cost = net_annual_ownership_cost / 12
            
            # Opportunity cost of down payment
            opportunity_cost_annual = down_payment * 0.07  # Assume 7% investment return
            total_annual_cost_of_ownership = net_annual_ownership_cost + opportunity_cost_annual
            
            # Rental scenario
            annual_rent = monthly_rent * 12
            rental_increases = [(1 + 0.02) ** year for year in range(analysis_period_years)]  # 2% annual increases
            total_rent_paid = sum(annual_rent * increase for increase in rental_increases)
            
            # Ownership scenario over time
            property_values = [property_price * (1 + regional_info.annual_appreciation_rate) ** year 
                             for year in range(analysis_period_years + 1)]
            
            # Calculate equity buildup
            remaining_balances = []
            for year in range(analysis_period_years + 1):
                if year == 0:
                    remaining_balances.append(loan_amount)
                else:
                    # Simplified amortization (would be more complex in reality)
                    annual_amortization = housing_costs.monthly_amortization * 12
                    remaining_balance = max(0, remaining_balances[-1] - annual_amortization)
                    remaining_balances.append(remaining_balance)
            
            # Net worth comparison
            final_property_value = property_values[-1]
            final_mortgage_balance = remaining_balances[-1]
            net_equity = final_property_value - final_mortgage_balance
            
            # Selling costs
            selling_costs = final_property_value * housing_costs.selling_costs_pct
            capital_gains = max(0, final_property_value - property_price)
            capital_gains_tax = capital_gains * self.capital_gains_tax_rate
            
            net_proceeds_from_sale = final_property_value - final_mortgage_balance - selling_costs - capital_gains_tax
            
            # Total cost comparison
            total_ownership_costs = (
                total_annual_cost_of_ownership * analysis_period_years +
                down_payment +
                property_price * housing_costs.buying_costs_pct
            )
            
            net_benefit_of_ownership = net_proceeds_from_sale - total_ownership_costs + (total_rent_paid)
            
            # Calculate break-even point
            breakeven_analysis = self._calculate_breakeven_period(
                property_price, monthly_rent, mortgage_terms, regional_info
            )
            
            # Risk analysis
            risk_factors = self._analyze_rent_buy_risks(property_price, monthly_rent, region)
            
            return {
                "success": True,
                "analysis_type": "rent_vs_buy",
                "property_details": {
                    "property_price": property_price,
                    "monthly_rent": monthly_rent,
                    "region": region,
                    "analysis_period_years": analysis_period_years
                },
                "ownership_scenario": {
                    "down_payment_required": float(down_payment),
                    "monthly_mortgage_payment": float(monthly_mortgage),
                    "monthly_total_cost_gross": float(total_monthly_cost),
                    "monthly_total_cost_net": float(net_monthly_ownership_cost),
                    "annual_interest_deduction": float(interest_tax_benefit),
                    "opportunity_cost_annual": float(opportunity_cost_annual),
                    "final_property_value": float(final_property_value),
                    "net_equity": float(net_equity),
                    "net_proceeds_after_sale": float(net_proceeds_from_sale)
                },
                "rental_scenario": {
                    "current_monthly_rent": float(monthly_rent),
                    "total_rent_paid_period": float(total_rent_paid),
                    "rent_increases_assumed": 0.02  # 2% annual
                },
                "comparison": {
                    "monthly_cost_difference": float(net_monthly_ownership_cost - monthly_rent),
                    "net_benefit_of_ownership": float(net_benefit_of_ownership),
                    "better_option": "buy" if net_benefit_of_ownership > 0 else "rent",
                    "total_savings": float(abs(net_benefit_of_ownership))
                },
                "breakeven_analysis": breakeven_analysis,
                "risk_analysis": risk_factors,
                "swedish_tax_benefits": {
                    "annual_interest_deduction": float(interest_tax_benefit),
                    "capital_gains_tax_on_sale": float(capital_gains_tax),
                    "net_tax_impact": float(interest_tax_benefit * analysis_period_years - capital_gains_tax)
                },
                "calculated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Rent vs buy analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _analyze_housing_scenario(
        self,
        property_price: float,
        monthly_rent_savings: float,
        mortgage_terms: SwedishMortgageTerms,
        regional_info: SwedishRegionalData,
        analysis_period_years: int
    ) -> Dict[str, Any]:
        """Analyze the housing purchase scenario."""
        
        # Initial capital requirements
        down_payment = property_price * mortgage_terms.down_payment_pct
        buying_costs = property_price * 0.015  # 1.5% buying costs
        initial_capital_required = down_payment + buying_costs
        
        # Loan details
        loan_amount = property_price - down_payment
        monthly_mortgage = self._calculate_monthly_payment(
            loan_amount, mortgage_terms.interest_rate, 25
        )
        
        # Monthly costs
        monthly_interest = loan_amount * mortgage_terms.interest_rate / 12
        monthly_amortization = monthly_mortgage - monthly_interest
        monthly_hoa = regional_info.average_hoa_fee_per_sqm * 75  # Assume 75 sqm
        monthly_property_tax = property_price * self.property_tax_rate / 12
        monthly_maintenance = property_price * 0.01 / 12  # 1% annually
        monthly_insurance = 5000 / 12  # 5000 SEK annually
        
        total_monthly_cost = (
            monthly_mortgage + monthly_hoa + monthly_property_tax + 
            monthly_maintenance + monthly_insurance
        )
        
        # Net monthly cost after tax benefits and rent savings
        annual_interest = monthly_interest * 12
        interest_tax_benefit = annual_interest * self.interest_deduction_rate
        monthly_interest_tax_benefit = interest_tax_benefit / 12
        
        net_monthly_cost = total_monthly_cost - monthly_rent_savings - monthly_interest_tax_benefit
        
        # Property value appreciation
        final_property_value = property_price * (1 + regional_info.annual_appreciation_rate) ** analysis_period_years
        
        # Mortgage balance at end
        final_mortgage_balance = self._calculate_remaining_balance(
            loan_amount, mortgage_terms.interest_rate, 25, analysis_period_years
        )
        
        # Net equity
        net_equity = final_property_value - final_mortgage_balance
        
        # Selling costs and taxes
        selling_costs = final_property_value * 0.05  # 5% selling costs
        capital_gains = max(0, final_property_value - property_price)
        capital_gains_tax = capital_gains * self.capital_gains_tax_rate
        
        net_proceeds = final_property_value - final_mortgage_balance - selling_costs - capital_gains_tax
        
        return {
            "initial_capital_required": float(initial_capital_required),
            "down_payment": float(down_payment),
            "buying_costs": float(buying_costs),
            "loan_amount": float(loan_amount),
            "monthly_mortgage_payment": float(monthly_mortgage),
            "monthly_total_cost": float(total_monthly_cost),
            "net_monthly_cost": float(net_monthly_cost),
            "annual_interest_deduction": float(interest_tax_benefit),
            "final_property_value": float(final_property_value),
            "final_mortgage_balance": float(final_mortgage_balance),
            "net_equity": float(net_equity),
            "selling_costs": float(selling_costs),
            "capital_gains_tax": float(capital_gains_tax),
            "net_proceeds_after_sale": float(net_proceeds),
            "total_return": float(net_proceeds - initial_capital_required),
            "annualized_return": float((net_proceeds / initial_capital_required) ** (1/analysis_period_years) - 1)
        }
    
    def _analyze_investment_scenario(
        self,
        initial_investment: float,
        monthly_contribution: float,
        expected_annual_return: float,
        annual_volatility: float,
        analysis_period_years: int
    ) -> Dict[str, Any]:
        """Analyze the portfolio investment scenario."""
        
        # Calculate future value with monthly contributions
        monthly_return = expected_annual_return / 12
        n_months = analysis_period_years * 12
        
        # Future value of initial investment
        fv_initial = initial_investment * (1 + expected_annual_return) ** analysis_period_years
        
        # Future value of monthly contributions (annuity)
        if monthly_return > 0:
            fv_contributions = monthly_contribution * (
                ((1 + monthly_return) ** n_months - 1) / monthly_return
            )
        else:
            fv_contributions = monthly_contribution * n_months
        
        total_future_value = fv_initial + fv_contributions
        total_contributions = initial_investment + (monthly_contribution * n_months)
        total_return = total_future_value - total_contributions
        
        # Tax calculations for investment returns
        capital_gains = max(0, total_return)
        capital_gains_tax = capital_gains * 0.30  # 30% capital gains tax on securities
        
        # ISK account alternative (schablonbeskattning)
        average_capital = (initial_investment + total_future_value) / 2
        isk_tax = average_capital * 0.375 * analysis_period_years  # Approximate ISK tax
        
        # Net returns after tax
        net_return_regular = total_future_value - capital_gains_tax
        net_return_isk = total_future_value - isk_tax
        
        # Volatility impact analysis
        annual_std = annual_volatility
        confidence_intervals = {
            "95%_downside": total_future_value * (1 - 1.96 * annual_std * np.sqrt(analysis_period_years)),
            "95%_upside": total_future_value * (1 + 1.96 * annual_std * np.sqrt(analysis_period_years)),
            "worst_case_5%": total_future_value * (1 - 1.65 * annual_std * np.sqrt(analysis_period_years))
        }
        
        return {
            "initial_investment": float(initial_investment),
            "monthly_contribution": float(monthly_contribution),
            "total_contributions": float(total_contributions),
            "expected_future_value": float(total_future_value),
            "expected_return": float(total_return),
            "annualized_return": float(expected_annual_return),
            "volatility": float(annual_volatility),
            "tax_analysis": {
                "capital_gains_tax_regular": float(capital_gains_tax),
                "isk_tax": float(isk_tax),
                "net_return_regular_account": float(net_return_regular),
                "net_return_isk_account": float(net_return_isk),
                "tax_advantage_isk": float(capital_gains_tax - isk_tax)
            },
            "confidence_intervals": {k: float(v) for k, v in confidence_intervals.items()}
        }
    
    def _monte_carlo_housing_vs_investment(
        self,
        housing_analysis: Dict[str, Any],
        investment_analysis: Dict[str, Any],
        analysis_period_years: int,
        n_simulations: int = 10000
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation comparing housing vs investment."""
        
        try:
            # Housing scenario parameters
            housing_return = housing_analysis["annualized_return"]
            housing_volatility = 0.15  # Assume 15% housing volatility
            
            # Investment scenario parameters
            investment_return = investment_analysis["annualized_return"]
            investment_volatility = investment_analysis["volatility"]
            
            # Monte Carlo simulations
            np.random.seed(42)  # For reproducibility
            
            # Housing simulations
            housing_returns = np.random.normal(
                housing_return, housing_volatility, n_simulations
            )
            housing_final_values = housing_analysis["initial_capital_required"] * (
                1 + housing_returns
            ) ** analysis_period_years
            
            # Investment simulations
            investment_returns = np.random.normal(
                investment_return, investment_volatility, n_simulations
            )
            investment_final_values = investment_analysis["initial_investment"] * (
                1 + investment_returns
            ) ** analysis_period_years
            
            # Compare outcomes
            investment_wins = np.sum(investment_final_values > housing_final_values)
            probability_investment_wins = investment_wins / n_simulations
            
            # Statistics
            housing_stats = {
                "mean": float(np.mean(housing_final_values)),
                "median": float(np.median(housing_final_values)),
                "std": float(np.std(housing_final_values)),
                "percentile_5": float(np.percentile(housing_final_values, 5)),
                "percentile_95": float(np.percentile(housing_final_values, 95))
            }
            
            investment_stats = {
                "mean": float(np.mean(investment_final_values)),
                "median": float(np.median(investment_final_values)),
                "std": float(np.std(investment_final_values)),
                "percentile_5": float(np.percentile(investment_final_values, 5)),
                "percentile_95": float(np.percentile(investment_final_values, 95))
            }
            
            return {
                "n_simulations": n_simulations,
                "probability_investment_wins": float(probability_investment_wins),
                "probability_housing_wins": float(1 - probability_investment_wins),
                "housing_statistics": housing_stats,
                "investment_statistics": investment_stats,
                "expected_difference": float(np.mean(investment_final_values - housing_final_values)),
                "volatility_of_difference": float(np.std(investment_final_values - housing_final_values))
            }
            
        except Exception as e:
            self.logger.error(f"Monte Carlo comparison failed: {e}")
            return {"error": str(e)}
    
    def _analyze_tax_implications(
        self,
        housing_analysis: Dict[str, Any],
        investment_analysis: Dict[str, Any],
        analysis_period_years: int
    ) -> Dict[str, Any]:
        """Analyze comprehensive tax implications of both scenarios."""
        
        # Housing tax implications
        housing_tax_benefits = housing_analysis["annual_interest_deduction"] * analysis_period_years
        housing_capital_gains_tax = housing_analysis["capital_gains_tax"]
        net_housing_tax_impact = housing_tax_benefits - housing_capital_gains_tax
        
        # Investment tax implications
        investment_tax_regular = investment_analysis["tax_analysis"]["capital_gains_tax_regular"]
        investment_tax_isk = investment_analysis["tax_analysis"]["isk_tax"]
        isk_advantage = investment_analysis["tax_analysis"]["tax_advantage_isk"]
        
        # Compare net tax impacts
        if isk_advantage > 0:
            better_investment_option = "ISK account"
            investment_tax_cost = investment_tax_isk
        else:
            better_investment_option = "Regular account"
            investment_tax_cost = investment_tax_regular
        
        net_tax_difference = net_housing_tax_impact + investment_tax_cost
        
        return {
            "housing_tax_impact": {
                "total_interest_deductions": float(housing_tax_benefits),
                "capital_gains_tax": float(housing_capital_gains_tax),
                "net_tax_benefit": float(net_housing_tax_impact)
            },
            "investment_tax_impact": {
                "regular_account_tax": float(investment_tax_regular),
                "isk_account_tax": float(investment_tax_isk),
                "better_option": better_investment_option,
                "isk_advantage": float(isk_advantage),
                "recommended_tax_cost": float(investment_tax_cost)
            },
            "comparative_analysis": {
                "net_tax_difference": float(net_tax_difference),
                "tax_favored_option": "housing" if net_tax_difference > 0 else "investment",
                "tax_advantage_amount": float(abs(net_tax_difference))
            }
        }
    
    def _analyze_housing_investment_risks(
        self,
        housing_analysis: Dict[str, Any],
        investment_analysis: Dict[str, Any],
        regional_info: SwedishRegionalData
    ) -> Dict[str, Any]:
        """Analyze risks associated with housing vs investment."""
        
        housing_risks = {
            "market_risk": {
                "price_volatility": regional_info.price_volatility,
                "market_liquidity": regional_info.market_liquidity,
                "concentration_risk": "High - single property/location"
            },
            "financial_risks": {
                "interest_rate_sensitivity": "High - variable rate mortgage",
                "maintenance_costs": "Unpredictable ongoing costs",
                "transaction_costs": "High costs to buy/sell (6.5%+)"
            },
            "operational_risks": {
                "vacancy_risk": "Not applicable for owner-occupied",
                "maintenance_responsibility": "Full responsibility for repairs",
                "liquidity_risk": "Low - takes time to sell"
            }
        }
        
        investment_risks = {
            "market_risk": {
                "volatility": investment_analysis["volatility"],
                "market_correlation": "Depends on portfolio diversification",
                "concentration_risk": "Low with diversified portfolio"
            },
            "financial_risks": {
                "currency_risk": "Moderate for international investments",
                "inflation_risk": "Moderate protection through equities",
                "liquidity_risk": "High - can sell quickly"
            },
            "operational_risks": {
                "management_fees": "Low ongoing costs",
                "tax_complexity": "Moderate - capital gains reporting",
                "behavioral_risk": "Risk of emotional decisions"
            }
        }
        
        # Risk scoring
        housing_risk_score = self._calculate_risk_score(housing_risks, "housing")
        investment_risk_score = self._calculate_risk_score(investment_risks, "investment")
        
        return {
            "housing_risks": housing_risks,
            "investment_risks": investment_risks,
            "risk_scores": {
                "housing_total_risk": housing_risk_score,
                "investment_total_risk": investment_risk_score,
                "lower_risk_option": "housing" if housing_risk_score < investment_risk_score else "investment"
            },
            "key_risk_differences": [
                "Housing: High concentration, low liquidity, interest rate sensitivity",
                "Investment: Market volatility, higher liquidity, better diversification",
                "Housing: Predictable tax benefits, physical asset control",
                "Investment: Flexible allocation, easier to adjust over time"
            ]
        }
    
    def _analyze_swedish_considerations(
        self,
        property_price: float,
        monthly_rent_savings: float,
        region: str,
        mortgage_terms: SwedishMortgageTerms
    ) -> Dict[str, Any]:
        """Analyze Swedish-specific considerations."""
        
        regional_info = self.regional_data.get(region, self.regional_data["Stockholm"])
        
        # Swedish housing market factors
        market_factors = {
            "regional_outlook": f"{region} market shows {regional_info.market_liquidity} liquidity",
            "price_appreciation_trend": f"Historical appreciation: {regional_info.annual_appreciation_rate:.1%}",
            "rental_market": f"Rental yield: {regional_info.rental_yield:.1%}",
            "regulatory_environment": "Strong tenant protection, regulated rental increases"
        }
        
        # Swedish financial factors
        financial_factors = {
            "riksbank_policy": f"Current Riksbank rate: {self.riksbank_rate:.1%}",
            "mortgage_regulations": "Stress test required, 85% max LTV, amortization required",
            "tax_system": f"30% interest deduction, 22% capital gains tax on housing",
            "housing_policy": "Government supports homeownership through various programs"
        }
        
        # Swedish lifestyle factors
        lifestyle_factors = {
            "housing_culture": "Strong preference for homeownership in Sweden",
            "mobility": f"{region} offers good career opportunities",
            "quality_of_life": "Swedish housing generally high quality",
            "social_factors": "Homeownership seen as financial security"
        }
        
        return {
            "market_factors": market_factors,
            "financial_factors": financial_factors,
            "lifestyle_factors": lifestyle_factors,
            "swedish_recommendations": [
                "Consider ISK account for investment alternative",
                "Factor in Swedish tax benefits for mortgage interest",
                "Evaluate regional job market stability",
                "Consider housing as inflation hedge in Swedish context",
                "Review Riksbank interest rate outlook"
            ]
        }
    
    def _generate_recommendation(
        self,
        housing_analysis: Dict[str, Any],
        investment_analysis: Dict[str, Any],
        monte_carlo_comparison: Dict[str, Any],
        risk_analysis: Dict[str, Any],
        regional_info: SwedishRegionalData
    ) -> Dict[str, Any]:
        """Generate final recommendation based on all analyses."""
        
        # Score each option
        housing_score = 0
        investment_score = 0
        
        # Return comparison
        if housing_analysis["annualized_return"] > investment_analysis["annualized_return"]:
            housing_score += 2
        else:
            investment_score += 2
        
        # Risk comparison
        if risk_analysis["risk_scores"]["housing_total_risk"] < risk_analysis["risk_scores"]["investment_total_risk"]:
            housing_score += 1
        else:
            investment_score += 1
        
        # Monte Carlo probability
        if monte_carlo_comparison["probability_housing_wins"] > 0.5:
            housing_score += 2
        else:
            investment_score += 2
        
        # Liquidity preference
        investment_score += 1  # Investment always more liquid
        
        # Tax efficiency
        if housing_analysis.get("annual_interest_deduction", 0) > 0:
            housing_score += 1
        
        # Regional factors
        if regional_info.market_liquidity == "high":
            housing_score += 1
        elif regional_info.market_liquidity == "low":
            investment_score += 1
        
        # Determine recommendation
        if housing_score > investment_score:
            recommendation = "housing"
            confidence = "high" if housing_score - investment_score >= 3 else "medium"
        elif investment_score > housing_score:
            recommendation = "investment"
            confidence = "high" if investment_score - housing_score >= 3 else "medium"
        else:
            recommendation = "neutral"
            confidence = "low"
        
        return {
            "recommendation": recommendation,
            "confidence_level": confidence,
            "scores": {
                "housing_score": housing_score,
                "investment_score": investment_score
            },
            "key_factors": {
                "return_advantage": "housing" if housing_analysis["annualized_return"] > investment_analysis["annualized_return"] else "investment",
                "risk_advantage": risk_analysis["risk_scores"]["lower_risk_option"],
                "liquidity_advantage": "investment",
                "tax_advantage": "housing" if housing_analysis.get("annual_interest_deduction", 0) > 10000 else "neutral"
            },
            "summary": self._generate_recommendation_summary(
                recommendation, confidence, housing_analysis, investment_analysis
            )
        }
    
    def _generate_recommendation_summary(
        self,
        recommendation: str,
        confidence: str,
        housing_analysis: Dict[str, Any],
        investment_analysis: Dict[str, Any]
    ) -> str:
        """Generate human-readable recommendation summary."""
        
        if recommendation == "housing":
            return f"Baserat på analysen rekommenderas bostadsköp med {confidence} säkerhet. " \
                   f"Förväntad årlig avkastning: {housing_analysis['annualized_return']:.1%}. " \
                   f"Viktiga faktorer: skattefördelar på ränta och långsiktig värdeutveckling."
        
        elif recommendation == "investment":
            return f"Baserat på analysen rekommenderas portföljinvestering med {confidence} säkerhet. " \
                   f"Förväntad årlig avkastning: {investment_analysis['annualized_return']:.1%}. " \
                   f"Viktiga faktorer: högre likviditet, diversifiering och flexibilitet."
        
        else:
            return f"Analysen visar ingen klar fördel för någon av alternativen. " \
                   f"Beslutet bör baseras på personliga preferenser gällande risk, " \
                   f"likviditet och livsstil."
    
    # Helper methods
    def _get_regional_housing_data(self) -> Dict[str, SwedishRegionalData]:
        """Get regional housing market data (example data)."""
        
        return {
            "Stockholm": SwedishRegionalData(
                region="Stockholm",
                average_price_per_sqm=85000,
                annual_appreciation_rate=0.05,
                rental_yield=0.03,
                average_hoa_fee_per_sqm=650,
                property_tax_rate=0.0075,
                market_liquidity="high",
                price_volatility=0.15
            ),
            "Göteborg": SwedishRegionalData(
                region="Göteborg",
                average_price_per_sqm=55000,
                annual_appreciation_rate=0.04,
                rental_yield=0.035,
                average_hoa_fee_per_sqm=500,
                property_tax_rate=0.0075,
                market_liquidity="medium",
                price_volatility=0.12
            ),
            "Malmö": SwedishRegionalData(
                region="Malmö",
                average_price_per_sqm=45000,
                annual_appreciation_rate=0.035,
                rental_yield=0.04,
                average_hoa_fee_per_sqm=450,
                property_tax_rate=0.0075,
                market_liquidity="medium",
                price_volatility=0.10
            )
        }
    
    def _calculate_monthly_payment(self, loan_amount: float, annual_rate: float, years: int) -> float:
        """Calculate monthly mortgage payment."""
        monthly_rate = annual_rate / 12
        n_payments = years * 12
        
        if monthly_rate == 0:
            return loan_amount / n_payments
        
        return loan_amount * (
            monthly_rate * (1 + monthly_rate) ** n_payments
        ) / ((1 + monthly_rate) ** n_payments - 1)
    
    def _calculate_max_loan_amount(self, monthly_payment: float, annual_rate: float, years: int) -> float:
        """Calculate maximum loan amount for given monthly payment."""
        monthly_rate = annual_rate / 12
        n_payments = years * 12
        
        if monthly_rate == 0:
            return monthly_payment * n_payments
        
        return monthly_payment * (
            ((1 + monthly_rate) ** n_payments - 1) / 
            (monthly_rate * (1 + monthly_rate) ** n_payments)
        )
    
    def _calculate_remaining_balance(
        self, 
        loan_amount: float, 
        annual_rate: float, 
        years: int, 
        years_paid: int
    ) -> float:
        """Calculate remaining mortgage balance after years_paid."""
        monthly_rate = annual_rate / 12
        n_payments = years * 12
        payments_made = years_paid * 12
        
        if monthly_rate == 0:
            return loan_amount * (1 - payments_made / n_payments)
        
        monthly_payment = self._calculate_monthly_payment(loan_amount, annual_rate, years)
        
        remaining_balance = loan_amount * (
            (1 + monthly_rate) ** n_payments - (1 + monthly_rate) ** payments_made
        ) / ((1 + monthly_rate) ** n_payments - 1)
        
        return max(0, remaining_balance)
    
    def _calculate_breakeven_period(
        self,
        property_price: float,
        monthly_rent: float,
        mortgage_terms: SwedishMortgageTerms,
        regional_info: SwedishRegionalData
    ) -> Dict[str, Any]:
        """Calculate breakeven period for rent vs buy."""
        
        # Simplified breakeven calculation
        down_payment = property_price * mortgage_terms.down_payment_pct
        loan_amount = property_price - down_payment
        monthly_mortgage = self._calculate_monthly_payment(
            loan_amount, mortgage_terms.interest_rate, 25
        )
        
        # Monthly cost difference
        monthly_ownership_cost = monthly_mortgage + (property_price * 0.01 / 12)  # Include maintenance
        monthly_cost_difference = monthly_ownership_cost - monthly_rent
        
        # Time to recover down payment through appreciation
        if regional_info.annual_appreciation_rate > 0:
            years_to_breakeven = np.log(
                1 + down_payment / property_price
            ) / np.log(1 + regional_info.annual_appreciation_rate)
        else:
            years_to_breakeven = float('inf')
        
        return {
            "breakeven_years": float(min(years_to_breakeven, 30)),  # Cap at 30 years
            "monthly_cost_difference": float(monthly_cost_difference),
            "down_payment_recovery_years": float(years_to_breakeven),
            "interpretation": "Years until housing ownership breaks even with renting"
        }
    
    def _analyze_rent_buy_risks(
        self, 
        property_price: float, 
        monthly_rent: float, 
        region: str
    ) -> Dict[str, Any]:
        """Analyze risks specific to rent vs buy decision."""
        
        regional_info = self.regional_data.get(region, self.regional_data["Stockholm"])
        
        return {
            "ownership_risks": [
                f"Property value volatility: {regional_info.price_volatility:.1%}",
                "Interest rate increases affecting mortgage costs",
                "Maintenance and repair costs",
                "Difficulty selling quickly if needed",
                "Market liquidity in " + region + " is " + regional_info.market_liquidity
            ],
            "rental_risks": [
                "Rent increases over time",
                "Risk of eviction or non-renewal",
                "No equity building",
                "No control over property improvements",
                "No tax benefits"
            ],
            "risk_mitigation": {
                "ownership": [
                    "Fixed-rate mortgage to hedge interest risk",
                    "Emergency fund for maintenance",
                    "Good location choice for liquidity"
                ],
                "rental": [
                    "Invest rent savings in diversified portfolio",
                    "Build emergency fund for moving costs",
                    "Consider long-term rental agreements"
                ]
            }
        }
    
    def _calculate_risk_score(self, risks: Dict[str, Any], scenario: str) -> float:
        """Calculate numerical risk score."""
        # Simplified risk scoring - would be more sophisticated in practice
        base_score = 5.0  # Medium risk
        
        if scenario == "housing":
            # Housing generally higher risk due to concentration
            return base_score + 1.0
        else:
            # Investment risk depends on portfolio
            return base_score - 0.5