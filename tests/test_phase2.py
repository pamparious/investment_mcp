"""Phase 2 tests for data collection implementation."""

import pytest
import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.data_collectors.riksbanken import RiksbankCollector
from backend.data_collectors.scb import SCBCollector
from backend.mcp_agents.data_agent import DataCollectionAgent
from backend.database import init_database, get_db_session
from backend.models import RiksbankData, SCBData, MarketData

class TestDataCollectors:
    """Test data collector implementations."""
    
    @pytest.mark.asyncio
    async def test_riksbank_collector(self):
        """Test Riksbank data collection."""
        async with RiksbankCollector() as collector:
            # Test metadata retrieval
            assert collector.SERIES_CONFIG is not None
            assert len(collector.SERIES_CONFIG) > 0
            
            # Test data collection (small sample)
            data = await collector.fetch_series_data("SEKREPORATE")
            assert isinstance(data, list)
            # Note: API might not return data, so we just check structure
    
    @pytest.mark.asyncio
    async def test_scb_collector(self):
        """Test SCB data collection."""
        async with SCBCollector() as collector:
            # Test table configuration
            assert collector.TABLES_CONFIG is not None
            assert len(collector.TABLES_CONFIG) > 0
            
            # Test metadata retrieval for a working table
            metadata = await collector.get_table_metadata("BO/BO0501/BO0501A/FastpiPSRegAr")
            assert isinstance(metadata, dict)
            
            # Test actual data collection
            data = await collector.fetch_table_data("BO/BO0501/BO0501A/FastpiPSRegAr", time_period="2023-2024")
            assert isinstance(data, list)
            print(f"SCB collector fetched {len(data)} data points")
            if data:
                print(f"Sample data point: {data[0]}")
                assert "value" in data[0]
                assert "date" in data[0]
                assert "region" in data[0]
    
    @pytest.mark.asyncio
    async def test_data_agent(self):
        """Test data collection agent."""
        init_database()
        agent = DataCollectionAgent()
        
        # Test status retrieval
        status = await agent.get_collection_status()
        assert isinstance(status, dict)
        assert "total_collections_24h" in status

def test_database_models():
    """Test database model creation."""
    init_database()
    
    with get_db_session() as session:
        # Test RiksbankData model
        riksbank_data = RiksbankData(
            series_id="TEST_SERIES",
            date=datetime.now(),
            value=1.5,
            description="Test data",
            unit="Percent"
        )
        session.add(riksbank_data)
        session.flush()
        assert riksbank_data.id is not None
        
        # Test SCBData model
        scb_data = SCBData(
            table_id="TEST_TABLE",
            region="00",
            date=datetime.now(),
            value=100.0,
            description="Test SCB data",
            unit="Index"
        )
        session.add(scb_data)
        session.flush()
        assert scb_data.id is not None

if __name__ == "__main__":
    # Run individual tests
    print("Testing database models...")
    test_database_models()
    print("✓ Database models test passed")
    
    print("\nTesting SCB collector...")
    asyncio.run(TestDataCollectors().test_scb_collector())
    print("✓ SCB collector test passed")
    
    print("\nTesting data collectors...")
    asyncio.run(TestDataCollectors().test_data_agent())
    print("✓ Data agent test passed")
    
    print("\nPhase 2 testing complete!")