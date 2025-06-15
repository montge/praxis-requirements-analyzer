import json
from pathlib import Path
from typing import Dict, Any
from neo4j import Driver
from ..utils.logger import setup_logger, handle_exception
from datetime import datetime

logger = setup_logger(__name__)

class SchemaExtractor:
    """
    Extracts and manages Neo4j database schema information using APOC
    
    Args:
        driver (Driver): Neo4j driver instance
        project_name (str): Name of the project for schema context
    """
    
    def __init__(self, driver: Driver, project_name: str):
        self.driver = driver
        self.project_name = project_name
        self.database = getattr(driver, 'database', 'neo4j')
        
    @handle_exception("Failed to extract Neo4j schema")
    def extract_schema(self) -> Dict[str, Any]:
        """
        Extract schema information from Neo4j database using APOC
        
        Returns:
            Dict[str, Any]: Schema information including nodes, relationships, and properties
        """
        schema = {
            "project": self.project_name,
            "timestamp": datetime.now().isoformat(),
            "schema": {}
        }
        
        # Use APOC to get schema information
        query = "CALL apoc.meta.schema()"
        
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query)
                schema_data = result.single()
                if schema_data:
                    schema["schema"] = schema_data[0]  # The schema is in the first column
                    logger.info("Successfully extracted schema using APOC")
                else:
                    logger.warning("No schema data returned from APOC")
                
                return schema
                
            except Exception as e:
                logger.error(f"Error extracting schema: {str(e)}")
                raise
    
    @handle_exception("Failed to save schema to file")
    def save_schema(self, schema: Dict[str, Any], filepath: str) -> None:
        """
        Save schema information to JSON file
        
        Args:
            schema (Dict[str, Any]): Schema information to save
            filepath (str): Path to save the schema file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(schema, f, indent=2)
        logger.info(f"Schema saved to {filepath}") 