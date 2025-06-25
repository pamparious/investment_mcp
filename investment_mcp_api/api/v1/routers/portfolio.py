"""Portfolio analysis endpoints for Investment MCP API."""

import logging
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from ...models.requests import (
    PortfolioAnalysisRequest,
    HistoricalAllocationRequest, 
    StressTestRequest
)
from ...models.responses import (
    PortfolioAnalysisResponse,
    HistoricalAllocationResponse,
    StressTestResponse,
    ResponseStatus
)
from ...services.portfolio.analysis_service import PortfolioAnalysisService
from ...services.portfolio.historical_service import HistoricalAnalysisService
from ...services.portfolio.stress_test_service import StressTestService
from ...middleware.auth import get_current_user, require_permission
from ...common.exceptions import (
    ValidationException,
    InsufficientDataException,
    PortfolioOptimizationException
)


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/analysis",
    response_model=PortfolioAnalysisResponse,
    summary="Generate AI-powered portfolio analysis",
    description="""
    Generate comprehensive portfolio analysis using historical data and Swedish economic context.
    
    This endpoint provides:
    - Optimal allocation recommendations based on risk profile
    - Expected performance metrics (return, volatility, Sharpe ratio)
    - Historical performance analysis with 20+ years of data
    - Stress testing against major market crises
    - Swedish economic context integration
    - AI-powered reasoning and explanations
    
    **Rate Limits:**
    - Standard tier: 10 requests per minute
    - Premium tier: 30 requests per minute
    
    **Typical Response Time:** 5-15 seconds
    """
)
async def analyze_portfolio(
    request: PortfolioAnalysisRequest,
    request_obj: Request,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate comprehensive portfolio analysis."""
    
    try:
        logger.info(f"Portfolio analysis requested by user {user['user_id']}")
        
        # Validate request
        await _validate_portfolio_request(request)
        
        # Get portfolio analysis service
        analysis_service = PortfolioAnalysisService()
        
        # Perform analysis
        analysis_result = await analysis_service.analyze_portfolio(
            risk_profile=request.risk_profile,
            investment_amount=request.investment_amount,
            investment_horizon_years=request.investment_horizon_years,
            current_allocation=request.current_allocation,
            constraints=request.constraints,
            include_stress_test=request.include_stress_test,
            include_historical_analysis=request.include_historical_analysis,
            user_tier=user["tier"]
        )
        
        # Log successful analysis
        background_tasks.add_task(
            _log_portfolio_analysis,
            user["user_id"],
            request.risk_profile,
            request.investment_amount,
            analysis_result.get("confidence_score", 0)
        )
        
        # Create response
        response = PortfolioAnalysisResponse(
            status=ResponseStatus.SUCCESS,
            timestamp=datetime.utcnow(),
            request_id=request_obj.state.request_id,
            **analysis_result
        )
        
        return response
        
    except InsufficientDataException as e:
        logger.warning(f"Insufficient data for portfolio analysis: {e}")
        raise ValidationException(
            detail="Insufficient historical data for reliable analysis",
            errors=[{"issue": "data_quality", "message": str(e)}]
        )
    
    except PortfolioOptimizationException as e:
        logger.error(f"Portfolio optimization failed: {e}")
        raise ValidationException(
            detail="Portfolio optimization failed",
            errors=[{"issue": "optimization", "message": str(e)}]
        )
    
    except Exception as e:
        logger.error(f"Portfolio analysis error: {e}", exc_info=True)
        raise


@router.post(
    "/allocations/historical",
    response_model=HistoricalAllocationResponse,
    summary="Generate historical allocation analysis",
    description="""
    Generate optimal allocation based on 20+ years of historical data analysis.
    
    This endpoint provides:
    - Historically optimal allocations for different market regimes
    - Performance analysis across multiple time periods
    - Correlation analysis between funds
    - Market regime identification and performance
    - Optimization process details and assumptions
    
    **Use Cases:**
    - Long-term investment planning
    - Historical backtesting
    - Understanding fund relationships
    - Market cycle analysis
    """
)
async def generate_historical_allocation(
    request: HistoricalAllocationRequest,
    request_obj: Request,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate optimal allocation based on historical analysis."""
    
    try:
        logger.info(f"Historical allocation analysis requested by user {user['user_id']}")
        
        # Get historical analysis service
        historical_service = HistoricalAnalysisService()
        
        # Perform historical analysis
        analysis_result = await historical_service.analyze_historical_allocation(
            risk_profile=request.risk_profile,
            investment_horizon_years=request.investment_horizon_years,
            historical_years=request.historical_years,
            include_stress_test=request.include_stress_test,
            include_regime_analysis=request.include_regime_analysis,
            constraints=request.constraints
        )
        
        # Create response
        response = HistoricalAllocationResponse(
            status=ResponseStatus.SUCCESS,
            timestamp=datetime.utcnow(),
            request_id=request_obj.state.request_id,
            **analysis_result
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Historical allocation analysis error: {e}", exc_info=True)
        raise


@router.post(
    "/stress-test",
    response_model=StressTestResponse,
    summary="Perform portfolio stress testing",
    description="""
    Test portfolio performance against historical crisis scenarios.
    
    Available stress test scenarios:
    - **2008 Financial Crisis:** September 2008 - March 2009
    - **COVID-19 Pandemic:** February 2020 - April 2020  
    - **Dot-com Bubble:** March 2000 - October 2002
    - **All Scenarios:** Combined analysis of all major crises
    
    The analysis provides:
    - Portfolio returns during each crisis scenario
    - Maximum drawdown and recovery time estimates
    - Risk metrics and Value-at-Risk calculations
    - Risk management recommendations
    
    **Recommended for:**
    - Risk assessment and due diligence
    - Defensive portfolio construction
    - Understanding downside risks
    """
)
async def stress_test_portfolio(
    request: StressTestRequest,
    request_obj: Request,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Perform portfolio stress testing against historical scenarios."""
    
    try:
        logger.info(f"Stress test requested by user {user['user_id']}")
        
        # Validate allocation
        await _validate_allocation(request.allocation)
        
        # Get stress test service
        stress_test_service = StressTestService()
        
        # Perform stress testing
        stress_test_result = await stress_test_service.run_stress_tests(
            allocation=request.allocation,
            scenarios=request.scenarios,
            confidence_levels=request.confidence_levels
        )
        
        # Create response
        response = StressTestResponse(
            status=ResponseStatus.SUCCESS,
            timestamp=datetime.utcnow(),
            request_id=request_obj.state.request_id,
            **stress_test_result
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Stress test error: {e}", exc_info=True)
        raise


@router.get(
    "/optimization/status/{job_id}",
    summary="Get portfolio optimization job status",
    description="""
    Check the status of a long-running portfolio optimization job.
    
    Some complex analyses are processed asynchronously. Use this endpoint
    to check the status and retrieve results when ready.
    """
)
async def get_optimization_status(
    job_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get status of portfolio optimization job."""
    
    try:
        # This would typically check a job queue/database
        # For now, return a placeholder response
        
        return {
            "job_id": job_id,
            "status": "completed",
            "progress": 100,
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "result_available": True,
            "message": "Portfolio analysis completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error checking optimization status: {e}")
        raise


# Utility functions
async def _validate_portfolio_request(request: PortfolioAnalysisRequest):
    """Validate portfolio analysis request."""
    
    # Check investment amount bounds
    if request.investment_amount < 10000:
        raise ValidationException("Minimum investment amount is 10,000 SEK")
    
    if request.investment_amount > 100000000:
        raise ValidationException("Maximum investment amount is 100,000,000 SEK")
    
    # Validate current allocation if provided
    if request.current_allocation:
        await _validate_allocation(request.current_allocation)
    
    # Check constraints
    if request.constraints:
        if request.constraints.max_funds > 12:
            raise ValidationException("Maximum 12 funds allowed in portfolio")
        
        if request.constraints.min_allocation_per_fund > 0.5:
            raise ValidationException("Minimum allocation per fund cannot exceed 50%")


async def _validate_allocation(allocation: Dict[str, float]):
    """Validate portfolio allocation weights."""
    
    if not allocation:
        raise ValidationException("Allocation cannot be empty")
    
    total_weight = sum(allocation.values())
    if not (0.95 <= total_weight <= 1.05):
        raise ValidationException(
            f"Allocation weights must sum to 1.0, got {total_weight:.3f}"
        )
    
    for fund_code, weight in allocation.items():
        if not (0 <= weight <= 1):
            raise ValidationException(
                f"Invalid weight {weight} for fund {fund_code}. Must be between 0 and 1."
            )
    
    # Check for too many funds
    if len(allocation) > 12:
        raise ValidationException("Maximum 12 funds allowed in portfolio")


async def _log_portfolio_analysis(
    user_id: str,
    risk_profile: str, 
    amount: float,
    confidence_score: float
):
    """Log portfolio analysis for monitoring."""
    
    logger.info(
        f"Portfolio analysis completed",
        extra={
            "user_id": user_id,
            "risk_profile": risk_profile,
            "investment_amount": amount,
            "confidence_score": confidence_score,
            "event": "portfolio_analysis_completed"
        }
    )