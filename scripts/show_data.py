#!/usr/bin/env python3
"""Display collected investment data."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.database import get_db_session
from backend.models import SCBData, RiksbankData, MarketData
from datetime import datetime

def main():
    print('📊 INVESTMENT MCP SERVER - DATA SUMMARY')
    print('=' * 60)

    with get_db_session() as session:
        # SCB Data (Swedish Statistics)
        scb_data = session.query(SCBData).order_by(SCBData.date.desc()).limit(10).all()
        print('\n🏠 SCB DATA (Swedish Housing & Economic Statistics)')
        print('-' * 50)
        for record in scb_data:
            print(f'📅 {record.date.strftime("%Y-%m-%d")} | {record.table_id}')
            print(f'   📍 Region: {record.region} | Value: {record.value:.1f} {record.unit}')
            print(f'   📋 {record.description}')
            print()
        
        # Market Data  
        market_data = session.query(MarketData).order_by(MarketData.date.desc()).limit(5).all()
        print('\n📈 MARKET DATA (Stock Prices)')
        print('-' * 50)
        for record in market_data:
            print(f'📅 {record.date.strftime("%Y-%m-%d")} | {record.symbol}')
            print(f'   💰 Close: ${record.close_price:.2f} | Volume: {record.volume:,}')
            print(f'   📊 High: ${record.high_price:.2f} | Low: ${record.low_price:.2f}')
            print()
        
        # Riksbank Data
        riksbank_data = session.query(RiksbankData).order_by(RiksbankData.date.desc()).limit(5).all()
        print('\n🏦 RIKSBANK DATA (Swedish Central Bank)')
        print('-' * 50)
        for record in riksbank_data:
            print(f'📅 {record.date.strftime("%Y-%m-%d")} | {record.series_id}')
            print(f'   📊 Value: {record.value} {record.unit}')
            print(f'   📋 {record.description}')
            print()
        
        # Summary stats
        scb_count = session.query(SCBData).count()
        market_count = session.query(MarketData).count()
        riksbank_count = session.query(RiksbankData).count()
        
        print('\n📈 DATABASE SUMMARY')
        print('-' * 30)
        print(f'SCB Records: {scb_count}')
        print(f'Market Records: {market_count}')
        print(f'Riksbank Records: {riksbank_count}')
        print(f'Total Records: {scb_count + market_count + riksbank_count}')

if __name__ == "__main__":
    main()