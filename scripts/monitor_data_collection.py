#!/usr/bin/env python3
"""Monitor data collection activities."""

import asyncio
import sys
import os
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.database import get_db_session
from backend.models import DataCollectionLog, MarketData, RiksbankData, SCBData

def print_collection_summary():
    """Print data collection summary."""
    with get_db_session() as session:
        # Recent activity (last 24 hours)
        recent_logs = session.query(DataCollectionLog).filter(
            DataCollectionLog.started_at >= datetime.now(timezone.utc) - timedelta(days=1)
        ).all()
        
        print("📊 Data Collection Summary (Last 24 Hours)")
        print("=" * 50)
        
        if not recent_logs:
            print("No recent collection activities found.")
            return
        
        # Group by source
        by_source = {}
        for log in recent_logs:
            if log.source not in by_source:
                by_source[log.source] = []
            by_source[log.source].append(log)
        
        for source, logs in by_source.items():
            successful = [l for l in logs if l.status == 'success']
            total_records = sum(l.records_collected for l in logs)
            
            print(f"\n🔹 {source.upper()}")
            print(f"   Collections: {len(logs)} total, {len(successful)} successful")
            print(f"   Records: {total_records}")
            
            if logs:
                latest = max(logs, key=lambda x: x.started_at)
                print(f"   Latest: {latest.started_at.strftime('%Y-%m-%d %H:%M')} "
                      f"({latest.status})")
        
        # Data totals
        print(f"\n📈 Current Data Totals")
        print("=" * 30)
        
        market_count = session.query(MarketData).count()
        riksbank_count = session.query(RiksbankData).count()
        scb_count = session.query(SCBData).count()
        
        print(f"Market Data: {market_count:,} records")
        print(f"Riksbank Data: {riksbank_count:,} records")
        print(f"SCB Data: {scb_count:,} records")
        print(f"Total: {market_count + riksbank_count + scb_count:,} records")

def print_error_summary():
    """Print error summary."""
    with get_db_session() as session:
        error_logs = session.query(DataCollectionLog).filter(
            DataCollectionLog.status == 'error',
            DataCollectionLog.started_at >= datetime.now(timezone.utc) - timedelta(days=7)
        ).order_by(DataCollectionLog.started_at.desc()).limit(10).all()
        
        if error_logs:
            print(f"\n❌ Recent Errors (Last 7 Days)")
            print("=" * 40)
            
            for log in error_logs:
                print(f"• {log.source} - {log.collection_type}")
                print(f"  Time: {log.started_at.strftime('%Y-%m-%d %H:%M')}")
                print(f"  Error: {log.error_message}")
                print()

if __name__ == "__main__":
    print_collection_summary()
    print_error_summary()