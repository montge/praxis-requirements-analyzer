from typing import Optional
from neo4j import GraphDatabase, Driver
from ..utils.logger import setup_logger, handle_exception
import json
from datetime import datetime

logger = setup_logger(__name__)

class Neo4jClient:
    """
    Neo4j database client for managing connections and queries
    
    Args:
        uri (str): Neo4j connection URI
        user (str): Username for authentication
        password (str): Password for authentication
        database (str): Name of the database to connect to
    """
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver: Optional[Driver] = None
        
    @handle_exception("Failed to connect to Neo4j")
    def connect(self) -> None:
        """Establish connection to Neo4j database"""
        try:
            self._driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            # Verify connection
            self.verify_connectivity()
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
            
    @handle_exception("Failed to verify Neo4j connectivity")
    def verify_connectivity(self) -> None:
        """Verify database connection is working"""
        if self._driver:
            self._driver.verify_connectivity()
            
    @handle_exception("Failed to close Neo4j connection")
    def close(self) -> None:
        """Close the database connection"""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")
            
    @property
    def driver(self) -> Optional[Driver]:
        """Get the Neo4j driver instance"""
        return self._driver 

    @handle_exception("Failed to store SIFP estimation results")
    def store_sifp_results(self, requirement_id: str, model_name: str, project_name: str, estimation_results: dict) -> None:
        """
        Store SIFP estimation results as a relationship to the requirement node.
        If a relationship already exists for the given model, it will be updated.
        Otherwise, a new relationship will be created.
        
        Args:
            requirement_id: ID of the requirement (e.g. CC164)
            model_name: Name of the LLM model used
            project_name: Name of the project the requirement belongs to
            estimation_results: Dictionary containing estimation results
            
        Returns:
            None
            
        Raises:
            Exception: If storage fails
        """
        if not self.driver:
            raise RuntimeError("Neo4j driver not connected. Call connect() first.")
            
        try:
            # Extract relevant data from estimation results
            actor_analysis = estimation_results.get('actor_analysis', {})
            final_estimation = estimation_results.get('final_estimation', {})
            
            # Prepare query parameters
            params = {
                "requirement_id": requirement_id,
                "model_name": model_name,
                "project_name": project_name,
                "timestamp": datetime.now().isoformat(),
                "actor_analysis": json.dumps(actor_analysis),
                "judge_score": estimation_results.get('judge_score', 0),
                "judge_confidence": estimation_results.get('judge_confidence', 0),
                "judge_evaluation": json.dumps(estimation_results.get('judge_evaluation', {})),
                "final_estimation": json.dumps(final_estimation),
                "is_valid": estimation_results.get('is_valid', False),
                "confidence": estimation_results.get('confidence', 0),
                "reasoning": estimation_results.get('reasoning', "")
            }

            # Create Cypher query that either updates existing relationship or creates new one
            # Filter by both requirement ID and project name via the document relationship
            query = """
            MATCH (p:Project {name: $project_name})
            MATCH (p)-[:CONTAINS]->(d:Document)-[:CONTAINS]->(r:Requirement {id: $requirement_id})
            MERGE (r)-[rel:SIFP_ESTIMATION {model: $model_name}]->(r)
            SET rel.timestamp = datetime($timestamp),
                rel.actor_analysis = $actor_analysis,
                rel.judge_score = $judge_score,
                rel.judge_confidence = $judge_confidence,
                rel.judge_evaluation = $judge_evaluation,
                rel.final_estimation = $final_estimation,
                rel.is_valid = $is_valid,
                rel.confidence = $confidence,
                rel.reasoning = $reasoning,
                rel.project = $project_name
            RETURN rel
            """
            
            # Execute query
            with self.driver.session() as session:
                result = session.run(query, params)
                result.consume()  # Ensure query completes
                logger.debug(f"Stored/Updated SIFP results for requirement {requirement_id} in project {project_name} using model {model_name}")
                logger.debug(f"SIFP estimation parameters: {params}")
                
        except Exception as e:
            error_msg = f"Failed to store SIFP results for requirement {requirement_id} in project {project_name} with model {model_name}"
            logger.error(f"{error_msg}: {str(e)}")
            raise 

    @handle_exception("Failed to store judge results")
    def store_judge_results(self, source_id: str, target_id: str, model_name: str, judgment_results: dict) -> None:
        """
        Store requirement judging results as a relationship between source and target requirements.
        Creates or updates a LLM_RESULT_META_JUDGE relationship between requirements.
        
        Args:
            source_id: ID of the source requirement
            target_id: ID of the target requirement  
            model_name: Name of the LLM model used
            judgment_results: Dictionary containing judgment results
            
        Returns:
            None
            
        Raises:
            Exception: If storage fails
        """
        if not self.driver:
            raise RuntimeError("Neo4j driver not connected. Call connect() first.")
            
        try:
            # Prepare query parameters
            params = {
                "source_id": source_id,
                "target_id": target_id,
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "actor_score": judgment_results.get('actor_score', 0),
                "actor_reasoning": judgment_results.get('actor_reasoning', ""),
                "judge_score": judgment_results.get('judge_score', 0),
                "judge_reasoning": judgment_results.get('judge_reasoning', ""),
                "semantic_alignment": judgment_results.get('match_quality', {}).get('semantic_alignment', 0),
                "functional_completeness": judgment_results.get('match_quality', {}).get('functional_completeness', 0),
                "non_functional_coverage": judgment_results.get('match_quality', {}).get('non_functional_coverage', 0),
                "final_score": judgment_results.get('final_score', 0),
                "is_traceable": judgment_results.get('is_traceable', False),
                "meta_judge_reasoning": judgment_results.get('meta_judge_reasoning', "")
            }

            # Create Cypher query that either updates existing relationship or creates new one
            query = """
            MATCH (src:Requirement {id: $source_id})
            MATCH (tgt:Requirement {id: $target_id})
            MERGE (src)-[rel:LLM_RESULT_META_JUDGE {model: $model_name}]->(tgt)
            SET rel.timestamp = datetime($timestamp),
                rel.actor_score = $actor_score,
                rel.actor_reasoning = $actor_reasoning,
                rel.judge_score = $judge_score,
                rel.judge_reasoning = $judge_reasoning,
                rel.semantic_alignment = $semantic_alignment,
                rel.functional_completeness = $functional_completeness,
                rel.non_functional_coverage = $non_functional_coverage,
                rel.final_score = $final_score,
                rel.is_traceable = $is_traceable,
                rel.meta_judge_reasoning = $meta_judge_reasoning
            RETURN rel
            """
            
            # Execute query
            with self.driver.session() as session:
                result = session.run(query, params)
                result.consume()  # Ensure query completes
                logger.debug(f"Stored/Updated judge results for {source_id}->{target_id} using model {model_name}")
                logger.debug(f"Judge results parameters: {params}")
                
        except Exception as e:
            error_msg = f"Failed to store judge results for {source_id}->{target_id} with model {model_name}"
            logger.error(f"{error_msg}: {str(e)}")
            raise 