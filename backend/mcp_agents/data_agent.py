"""Data collection MCP agent."""

import asyncio
import logging
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
import json
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.data_collectors.stock_market import StockMarketCollector
from backend.data_collectors.riksbanken import RiksbankCollector
from backend.data_collectors.scb import SCBCollector
from backend.database import get_db_session, init_database
from backend.models import DataCollectionLog
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class CollectionResult:
    """Result of a data collection operation."""
    source: str
    collection_type: str
    success: bool
    records_collected: int
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DataCollectionAgent:
    """MCP Agent for coordinating data collection."""
    
    def __init__(self):
        self.stock_collector = StockMarketCollector()
        self.last_collection_times = {}
        
    async def collect_stock_data(self, symbols: Optional[List[str]] = None) -> CollectionResult:
        """Collect stock market data."""
        if not symbols:
            symbols = settings.DEFAULT_STOCK_SYMBOLS
        
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info(f"Starting stock data collection for {len(symbols)} symbols")
            
            # Collect data
            data = self.stock_collector.get_market_data(symbols, period="1mo")
            logger.info(f"Retrieved data for {len(data)} symbols")
            
            # Save to database
            records_saved = await self._save_stock_data(data)
            logger.info(f"Saved {records_saved} stock market records")
            
            result = CollectionResult(
                source="stock_market",
                collection_type="daily_update",
                success=True,
                records_collected=records_saved,
                metadata={"symbols": symbols, "period": "1mo"}
            )
            
            logger.info(f"Stock data collection completed: {records_saved} records")
            return result
            
        except Exception as e:
            logger.error(f"Stock data collection failed: {e}")
            return CollectionResult(
                source="stock_market",
                collection_type="daily_update",
                success=False,
                records_collected=0,
                error_message=str(e)
            )
        finally:
            await self._log_collection_activity(
                "stock_market", "daily_update", start_time, 
                result if 'result' in locals() else None
            )
    
    async def collect_riksbank_data(self, days_back: int = 7) -> CollectionResult:
        """Collect Riksbank data."""
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info(f"Starting Riksbank data collection for last {days_back} days")
            
            async with RiksbankCollector() as collector:
                data = await collector.collect_all_series(days_back)
                records_saved = await collector.save_to_database(data)
            
            result = CollectionResult(
                source="riksbank",
                collection_type="periodic_update",
                success=True,
                records_collected=records_saved,
                metadata={"days_back": days_back, "series_count": len(data)}
            )
            
            logger.info(f"Riksbank data collection completed: {records_saved} records")
            return result
            
        except Exception as e:
            logger.error(f"Riksbank data collection failed: {e}")
            return CollectionResult(
                source="riksbank",
                collection_type="periodic_update",
                success=False,
                records_collected=0,
                error_message=str(e)
            )
        finally:
            await self._log_collection_activity(
                "riksbank", "periodic_update", start_time,
                result if 'result' in locals() else None
            )
    
    async def collect_scb_data(self, years_back: int = 2) -> CollectionResult:
        """Collect SCB data."""
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info(f"Starting SCB data collection for last {years_back} years")
            
            async with SCBCollector() as collector:
                data = await collector.collect_all_tables(years_back)
                records_saved = await collector.save_to_database(data)
            
            result = CollectionResult(
                source="scb",
                collection_type="periodic_update",
                success=True,
                records_collected=records_saved,
                metadata={"years_back": years_back, "tables_count": len(data)}
            )
            
            logger.info(f"SCB data collection completed: {records_saved} records")
            return result
            
        except Exception as e:
            logger.error(f"SCB data collection failed: {e}")
            return CollectionResult(
                source="scb",
                collection_type="periodic_update",
                success=False,
                records_collected=0,
                error_message=str(e)
            )
        finally:
            await self._log_collection_activity(
                "scb", "periodic_update", start_time,
                result if 'result' in locals() else None
            )
    
    async def collect_all_data(self) -> List[CollectionResult]:
        """Collect data from all sources."""
        logger.info("Starting comprehensive data collection")
        
        results = []
        
        # Collect stock data
        stock_result = await self.collect_stock_data()
        results.append(stock_result)
        
        # Small delay between collections
        await asyncio.sleep(2)
        
        # Collect Riksbank data (limited days to avoid rate limits)
        riksbank_result = await self.collect_riksbank_data(days_back=3)
        results.append(riksbank_result)
        
        await asyncio.sleep(2)
        
        # Collect SCB data
        scb_result = await self.collect_scb_data()
        results.append(scb_result)
        
        # Summary logging
        total_records = sum(r.records_collected for r in results)
        successful_collections = sum(1 for r in results if r.success)
        
        logger.info(f"Data collection completed: {successful_collections}/{len(results)} successful, "
                   f"{total_records} total records collected")
        
        return results
    
    async def _save_stock_data(self, data: Dict[str, Any]) -> int:
        """Save stock market data to database."""
        # Import here to avoid circular imports
        from backend.models import MarketData
        import pandas as pd
        
        total_saved = 0
        
        with get_db_session() as session:
            for symbol, symbol_data in data.items():
                if 'error' in symbol_data:
                    logger.warning(f"Skipping {symbol} due to error: {symbol_data['error']}")
                    continue
                    
                price_data = symbol_data.get('data', {})
                if not price_data:
                    logger.warning(f"No price data for {symbol}")
                    continue
                
                logger.info(f"Processing {len(price_data)} data points for {symbol}")
                
                # Convert the dictionary back to DataFrame for easier processing
                df = pd.DataFrame.from_dict(price_data, orient='index')
                
                for index, row in df.iterrows():
                    try:
                        # Convert timezone-aware timestamp to date
                        if hasattr(index, 'tz_convert'):
                            # Convert to UTC first to normalize timezone differences
                            date_value = index.tz_convert('UTC').date()
                        else:
                            date_value = index.date()
                        
                        # Check if data point already exists
                        existing = session.query(MarketData).filter(
                            MarketData.symbol == symbol,
                            MarketData.date == date_value
                        ).first()
                        
                        if not existing:
                            db_record = MarketData(
                                symbol=symbol,
                                date=date_value,
                                open_price=float(row['Open']),
                                high_price=float(row['High']),
                                low_price=float(row['Low']),
                                close_price=float(row['Close']),
                                volume=int(row['Volume'])
                            )
                            session.add(db_record)
                            total_saved += 1
                        else:
                            logger.debug(f"Skipping existing data for {symbol} on {date_value}")
                    except Exception as e:
                        logger.warning(f"Error saving data point for {symbol} at {index}: {e}")
                        continue
        
        return total_saved
    
    async def _log_collection_activity(
        self, 
        source: str, 
        collection_type: str, 
        start_time: datetime,
        result: Optional[CollectionResult]
    ):
        """Log collection activity to database."""
        with get_db_session() as session:
            log_entry = DataCollectionLog(
                source=source,
                collection_type=collection_type,
                status="success" if result and result.success else "error",
                records_collected=result.records_collected if result else 0,
                error_message=result.error_message if result else "Unknown error",
                started_at=start_time,
                completed_at=datetime.now(timezone.utc),
                extra_data=json.dumps(result.metadata) if result and result.metadata else None
            )
            session.add(log_entry)
    
    async def get_collection_status(self) -> Dict[str, Any]:
        """Get status of recent data collections."""
        with get_db_session() as session:
            # Get recent logs (last 24 hours)
            recent_logs = session.query(DataCollectionLog).filter(
                DataCollectionLog.started_at >= datetime.now(timezone.utc) - timedelta(days=1)
            ).order_by(DataCollectionLog.started_at.desc()).all()
            
            status = {
                "total_collections_24h": len(recent_logs),
                "successful_collections_24h": len([log for log in recent_logs if log.status == "success"]),
                "total_records_collected_24h": sum(log.records_collected for log in recent_logs),
                "recent_activities": []
            }
            
            for log in recent_logs[:10]:  # Last 10 activities
                status["recent_activities"].append({
                    "source": log.source,
                    "type": log.collection_type,
                    "status": log.status,
                    "records": log.records_collected,
                    "timestamp": log.started_at.isoformat(),
                    "error": log.error_message if log.status == "error" else None
                })
            
            return status

# Scheduled data collection
class DataCollectionScheduler:
    """Scheduler for automated data collection."""
    
    def __init__(self):
        self.agent = DataCollectionAgent()
        self.running = False
    
    async def start_scheduled_collection(self):
        """Start scheduled data collection."""
        self.running = True
        logger.info("Starting scheduled data collection")
        
        while self.running:
            try:
                # Collect all data
                results = await self.agent.collect_all_data()
                
                # Log summary
                successful = sum(1 for r in results if r.success)
                logger.info(f"Scheduled collection completed: {successful}/{len(results)} successful")
                
                # Wait for next collection interval
                await asyncio.sleep(settings.DATA_COLLECTION_INTERVAL)
                
            except Exception as e:
                logger.error(f"Scheduled collection error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def stop_scheduled_collection(self):
        """Stop scheduled data collection."""
        self.running = False
        logger.info("Stopping scheduled data collection")

# CLI interface for testing
async def main():
    """Main function for testing data collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Investment Data Collection Agent")
    parser.add_argument("--source", choices=["stock", "riksbank", "scb", "all"], 
                       default="all", help="Data source to collect from")
    parser.add_argument("--schedule", action="store_true", 
                       help="Run scheduled collection")
    
    args = parser.parse_args()
    
    # Initialize database
    init_database()
    
    agent = DataCollectionAgent()
    
    if args.schedule:
        scheduler = DataCollectionScheduler()
        try:
            await scheduler.start_scheduled_collection()
        except KeyboardInterrupt:
            scheduler.stop_scheduled_collection()
            logger.info("Scheduled collection stopped by user")
    else:
        if args.source == "stock":
            result = await agent.collect_stock_data()
        elif args.source == "riksbank":
            result = await agent.collect_riksbank_data()
        elif args.source == "scb":
            result = await agent.collect_scb_data()
        else:  # all
            results = await agent.collect_all_data()
            for result in results:
                print(f"{result.source}: {'Success' if result.success else 'Failed'} "
                      f"({result.records_collected} records)")
            return
        
        print(f"{result.source}: {'Success' if result.success else 'Failed'} "
              f"({result.records_collected} records)")
        if result.error_message:
            print(f"Error: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())