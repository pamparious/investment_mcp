"""SCB (Statistics Sweden) API data collection module."""

import aiohttp
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from config.settings import settings
from backend.database import get_db_session
from backend.models import SCBData, DataCollectionLog

logger = logging.getLogger(__name__)

@dataclass
class SCBTable:
    """SCB table configuration."""
    table_id: str
    table_path: str
    description: str
    unit: str
    regions: List[str]
    variables: Dict[str, List[str]]

class SCBCollector:
    """Collector for SCB (Statistics Sweden) data."""
    
    # Key housing and economic tables (focus on working annual data)
    TABLES_CONFIG = {
        # Real estate price index for one/two-dwelling buildings (annual data)
        "FastpiPSRegAr": SCBTable(
            table_id="FastpiPSRegAr",
            table_path="BO/BO0501/BO0501A/FastpiPSRegAr",
            description="Real estate price index for one- or two-dwelling buildings for permanent living (1981=100)",
            unit="Index 1981=100",
            regions=settings.SCB_HOUSING_REGIONS,
            variables={
                "Region": ["00", "0010", "0020", "0030"],  # Sweden, Stockholm, Göteborg, Malmö
                "ContentsCode": ["BO0501N9"]  # Index
            }
        )
    }
    
    def __init__(self):
        self.base_url = "https://api.scb.se/OV0104/v1/doris/en/ssd"
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.REQUEST_TIMEOUT),
            headers={"Content-Type": "application/json"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get_table_metadata(self, table_path: str) -> Dict[str, Any]:
        """Get metadata for a SCB table using the full path."""
        url = f"{self.base_url}/{table_path}"
        
        try:
            if not self.session:
                raise RuntimeError("Session not initialized. Use 'async with SCBCollector()' context manager.")
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"SCB metadata API error: {response.status} for {url}")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching SCB metadata for {table_path}: {e}")
            return {}
    
    async def discover_table_variables(self, table_path: str) -> Dict[str, Any]:
        """
        Discover available variables for a table by getting its metadata.
        
        Args:
            table_path: Full path to SCB table
            
        Returns:
            Dictionary containing table metadata and available variables
        """
        metadata = await self.get_table_metadata(table_path)
        
        if not metadata or "variables" not in metadata:
            logger.error(f"No variables found for table {table_path}")
            return {}
            
        variables_info = {}
        for var in metadata.get("variables", []):
            var_code = var.get("code", "")
            var_text = var.get("text", "")
            var_values = var.get("values", [])
            var_value_texts = var.get("valueTexts", [])
            
            variables_info[var_code] = {
                "text": var_text,
                "values": var_values,
                "valueTexts": var_value_texts
            }
            
        return {
            "title": metadata.get("title", ""),
            "variables": variables_info,
            "updated": metadata.get("updated", "")
        }

    async def fetch_table_data(
        self, 
        table_path: str,
        variables: Optional[Dict[str, List[str]]] = None,
        time_period: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch data from a SCB table.
        
        Args:
            table_path: Full path to SCB table (e.g., "HE/HE0103/HE0103T04")
            variables: Variables to filter by
            time_period: Time period filter (e.g., "2020-2024")
        
        Returns:
            List of data points
        """
        url = f"{self.base_url}/{table_path}"
        
        # First, get table metadata if no variables provided
        if not variables:
            logger.info(f"Discovering variables for table {table_path}")
            table_info = await self.discover_table_variables(table_path)
            if not table_info.get("variables"):
                logger.error(f"Could not discover variables for {table_path}")
                return []
            logger.info(f"Available variables: {list(table_info['variables'].keys())}")
        
        # Construct query payload
        query = {
            "query": [],
            "response": {
                "format": "json"
            }
        }
        
        # Add variable filters
        if variables:
            for var_name, var_values in variables.items():
                query["query"].append({
                    "code": var_name,
                    "selection": {
                        "filter": "item",
                        "values": var_values
                    }
                })
        
        # Add time filter if specified
        if time_period:
            years = time_period.split("-")
            if len(years) == 2:
                start_year, end_year = years
                # Only include years that actually exist (not future years)
                current_year = datetime.now().year
                actual_end_year = min(int(end_year), current_year)
                time_values = [str(year) for year in range(int(start_year), actual_end_year + 1)]
                query["query"].append({
                    "code": "Tid",
                    "selection": {
                        "filter": "item",
                        "values": time_values
                    }
                })
        
        try:
            if not self.session:
                raise RuntimeError("Session not initialized. Use 'async with SCBCollector()' context manager.")
            
            # Validate the query before sending
            if not self._validate_query(query, table_path):
                logger.error(f"Invalid query for {table_path}: {json.dumps(query, indent=2)}")
                return []
            
            logger.debug(f"Posting to {url} with query: {json.dumps(query, indent=2)}")
            async with self.session.post(url, json=query) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_scb_response(data, table_path)
                elif response.status == 400:
                    # Bad Request - log the detailed error
                    try:
                        error_response = await response.text()
                        logger.error(f"SCB API 400 Bad Request for {table_path}:")
                        logger.error(f"URL: {url}")
                        logger.error(f"Query: {json.dumps(query, indent=2)}")
                        logger.error(f"Response: {error_response}")
                    except:
                        logger.error(f"SCB API 400 Bad Request for {table_path} (could not read response)")
                    return []
                else:
                    logger.error(f"SCB data API error: {response.status} for {table_path}")
                    try:
                        error_text = await response.text()
                        logger.error(f"Error response: {error_text}")
                    except:
                        pass
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching SCB data for {table_path}: {e}")
            return []
    
    def _validate_query(self, query: Dict[str, Any], table_path: str) -> bool:
        """
        Validate the SCB API query structure.
        
        Args:
            query: The query dictionary to validate
            table_path: The table path for context
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check basic structure
            if not isinstance(query, dict):
                logger.error(f"Query is not a dictionary for {table_path}")
                return False
            
            if "query" not in query or "response" not in query:
                logger.error(f"Missing 'query' or 'response' keys for {table_path}")
                return False
            
            if not isinstance(query["query"], list):
                logger.error(f"Query 'query' field is not a list for {table_path}")
                return False
            
            # Validate each query item
            for i, item in enumerate(query["query"]):
                if not isinstance(item, dict):
                    logger.error(f"Query item {i} is not a dictionary for {table_path}")
                    return False
                
                if "code" not in item or "selection" not in item:
                    logger.error(f"Query item {i} missing 'code' or 'selection' for {table_path}")
                    return False
                
                selection = item["selection"]
                if not isinstance(selection, dict):
                    logger.error(f"Selection in query item {i} is not a dictionary for {table_path}")
                    return False
                
                if "filter" not in selection or "values" not in selection:
                    logger.error(f"Selection in query item {i} missing 'filter' or 'values' for {table_path}")
                    return False
                
                if not isinstance(selection["values"], list):
                    logger.error(f"Selection values in query item {i} is not a list for {table_path}")
                    return False
                
                if len(selection["values"]) == 0:
                    logger.warning(f"Empty values list in query item {i} for {table_path}")
            
            # Check response format
            response_format = query.get("response", {}).get("format")
            if response_format not in ["json", "csv", "px"]:
                logger.warning(f"Unusual response format '{response_format}' for {table_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating query for {table_path}: {e}")
            return False
    
    def _parse_scb_response(self, response_data: Dict[str, Any], table_path: str) -> List[Dict[str, Any]]:
        """Parse SCB JSON response."""
        try:
            data_points = []
            
            if "data" not in response_data:
                return data_points
            
            columns = response_data.get("columns", [])
            data_rows = response_data.get("data", [])
            
            # Find column indices
            region_idx = None
            time_idx = None
            value_idx = None
            
            for i, col in enumerate(columns):
                if col.get("code") == "Region":
                    region_idx = i
                elif col.get("code") == "Tid":
                    time_idx = i
                elif "values" in col:
                    value_idx = i
            
            # Extract table_id from path for config lookup
            table_id = table_path.split('/')[-1]
            table_config = self.TABLES_CONFIG.get(table_id)
            
            for row in data_rows:
                try:
                    # Extract data point
                    region = row["key"][region_idx] if region_idx is not None else "00"
                    time_str = row["key"][time_idx] if time_idx is not None else ""
                    value = float(row["values"][0]) if row["values"] and row["values"][0] else None
                    
                    if value is not None and time_str:
                        # Parse time (SCB uses various formats)
                        date = self._parse_scb_date(time_str)
                        
                        if date:
                            data_points.append({
                                "table_id": table_id,
                                "region": region,
                                "date": date,
                                "value": value,
                                "description": table_config.description if table_config else "",
                                "unit": table_config.unit if table_config else ""
                            })
                            
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Error parsing SCB data row: {e}")
                    continue
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error parsing SCB response: {e}")
            return []
    
    def _parse_scb_date(self, date_str: str) -> Optional[datetime]:
        """Parse SCB date string to datetime."""
        try:
            # SCB uses various date formats
            if len(date_str) == 4:  # Year only
                return datetime(int(date_str), 1, 1)
            elif len(date_str) == 6:  # YYYYMM
                year = int(date_str[:4])
                month = int(date_str[4:])
                return datetime(year, month, 1)
            elif len(date_str) == 7 and "M" in date_str:  # YYYY-MM
                year, month = date_str.split("M")
                return datetime(int(year), int(month), 1)
            else:
                # Try standard date parsing
                return datetime.strptime(date_str, "%Y-%m-%d")
                
        except (ValueError, TypeError):
            logger.warning(f"Could not parse SCB date: {date_str}")
            return None
    
    async def collect_all_tables(self, years_back: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """Collect data for all configured tables."""
        current_year = datetime.now().year
        start_year = current_year - years_back
        # End year should be current year - 1 since current year data might not be complete
        end_year = current_year - 1
        time_period = f"{start_year}-{end_year}"
        
        results = {}
        
        for table_id, table_config in self.TABLES_CONFIG.items():
            logger.info(f"Collecting SCB data for table: {table_id}")
            data = await self.fetch_table_data(
                table_config.table_path, 
                table_config.variables,
                time_period
            )
            results[table_id] = data
            
            # Delay to respect API limits
            await asyncio.sleep(2)
        
        return results
    
    async def save_to_database(self, data: Dict[str, List[Dict[str, Any]]]) -> int:
        """Save collected data to database."""
        total_saved = 0
        
        with get_db_session() as session:
            for table_id, data_points in data.items():
                for point in data_points:
                    # Check if data point already exists
                    existing = session.query(SCBData).filter(
                        SCBData.table_id == point["table_id"],
                        SCBData.region == point["region"],
                        SCBData.date == point["date"]
                    ).first()
                    
                    if not existing:
                        db_record = SCBData(
                            table_id=point["table_id"],
                            region=point["region"],
                            date=point["date"],
                            value=point["value"],
                            description=point["description"],
                            unit=point["unit"]
                        )
                        session.add(db_record)
                        total_saved += 1
        
        logger.info(f"Saved {total_saved} new SCB data points")
        return total_saved

# Example usage and testing
async def test_scb_collector():
    """Test the SCB collector."""
    async with SCBCollector() as collector:
        print("Testing SCB data collection...")
        
        # Test collecting all configured tables
        results = await collector.collect_all_tables(years_back=1)
        
        for table_id, data_points in results.items():
            print(f"\nTable {table_id}: {len(data_points)} data points")
            if data_points:
                latest = data_points[-1]
                print(f"  Latest: {latest['value']} {latest['unit']} in region {latest['region']} on {latest['date']}")

if __name__ == "__main__":
    # Initialize database first
    from backend.database import init_database
    init_database()
    
    # Run test
    asyncio.run(test_scb_collector())