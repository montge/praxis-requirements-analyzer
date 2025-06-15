# src/neo4j/requirements_client.py

from typing import Dict, List
from ..utils.logger import setup_logger, handle_exception
from ..models.requirement import Requirement
from .neo4j_client import Neo4jClient

logger = setup_logger(__name__)

class RequirementsClient:
    """Client for fetching requirements from Neo4j"""
    
    def __init__(self, neo4j_client: Neo4jClient, project_name: str):
        self.client = neo4j_client
        self.project_name = project_name
        
    def get_requirements(self) -> Dict[str, List[Requirement]]:
        """
        Fetch source and target requirements from Neo4j that have ground truth links
        
        Returns:
            Dict with 'source' and 'target' lists of requirements that have ground truth links
        """
        try:
            # Validate Neo4j client and driver
            if not self.client or not self.client.driver:
                raise ValueError("Neo4j client or driver is not initialized")
                
            query = """
            MATCH (p:Project {name: $project_name})
            MATCH (p)-[:CONTAINS]->(d:Document)
            WHERE d.type IN ['SOURCE', 'TARGET']
            MATCH (d)-[:CONTAINS]->(r:Requirement)
            WHERE (r)-[:GROUND_TRUTH]->() OR ()-[:GROUND_TRUTH]->(r)
            RETURN 
                r.id as id,
                r.content as content,
                d.type as type,
                r.level as level
            """
            
            requirements = {"source": [], "target": []}
            
            # Use synchronous context manager (no async with)
            with self.client.driver.session(database=self.client.database) as session:
                if not session:
                    raise ValueError("Failed to create Neo4j session")
                    
                try:
                    # Use synchronous session.run (no await)
                    result = session.run(query, project_name=self.project_name)
                    records = result.data()  # No await needed
                    
                    for record in records:
                        req = Requirement(
                            id=record["id"],
                            content=record["content"],
                            type=record["type"],
                            level=record["level"]
                        )
                        
                        if req.type.upper() == "SOURCE":
                            requirements["source"].append(req)
                        elif req.type.upper() == "TARGET":
                            requirements["target"].append(req)
                            
                    logger.info(f"Found {len(requirements['source'])} source and {len(requirements['target'])} target requirements with ground truth links")
                    return requirements
                    
                except Exception as e:
                    logger.error(f"Error executing Neo4j query: {str(e)}")
                    raise
                    
        except Exception as e:
            logger.error(f"Failed to fetch requirements: {str(e)}")
            raise 