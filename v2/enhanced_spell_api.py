import os
import re
import time
import json
import pickle
import logging
import asyncio
import unicodedata
from typing import List, Dict, Any, Optional, Set
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import copy
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, WebSocketException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, ValidationInfo
from enum import Enum

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("spellchecker-api")

# Import your utilities (must match the classes used to build the pickles)
try:
    from trie_utils import Trie
    from ngram_utils import NgramGenerator
    from edit_distance_utils import EditDistance
except Exception as e:
    raise RuntimeError(f"Failed to import required utilities: {e}")

# Config: resolve paths relative to this file to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRIE_PKL = os.path.join(BASE_DIR, "my_trie_dictionary.pkl")
NGRAM_PKL = os.path.join(BASE_DIR, "loaded_ngrams.pkl")

WORD_PATTERN = re.compile(r"\b[a-zA-Z']+\b")  # must match training/build regex
MAX_EDIT_DISTANCE = 6
CANDIDATE_THRESHOLD = float(os.getenv("CANDIDATE_THRESHOLD", "0.5"))
EXECUTOR_WORKERS = int(os.getenv("EXECUTOR_WORKERS", "4"))
NORMALIZE_INPUT = True  # normalize Unicode punctuation by default

# Enhanced Models with Pydantic V2 validators
class WordCheckRequest(BaseModel):
    word: str
    
    @field_validator('word')
    @classmethod
    def validate_word(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Word cannot be empty')
        if len(v.strip()) > 100:
            raise ValueError('Word too long (max 100 characters)')
        return v.strip()

class WordsCheckRequest(BaseModel):
    words: List[str]
    
    @field_validator('words')
    @classmethod
    def validate_words(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError('Words list cannot be empty')
        if len(v) > 1000:
            raise ValueError('Too many words (max 1000)')
        for word in v:
            if not word or not word.strip():
                raise ValueError('Words cannot be empty')
            if len(word.strip()) > 100:
                raise ValueError('Word too long (max 100 characters)')
        return [word.strip() for word in v]

class SentenceCheckRequest(BaseModel):
    sentence: str
    
    @field_validator('sentence')
    @classmethod
    def validate_sentence(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Sentence cannot be empty')
        if len(v.strip()) > 10000:
            raise ValueError('Sentence too long (max 10000 characters)')
        return v.strip()

# Enhanced WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}  # Use dict with connection IDs
        self.connection_count = 0
        self._lock = asyncio.Lock()  # Add thread safety
    
    async def connect(self, websocket: WebSocket) -> Optional[str]:
        """Connect a WebSocket and return connection ID"""
        try:
            await websocket.accept()
            connection_id = str(uuid.uuid4())
            
            async with self._lock:
                self.active_connections[connection_id] = websocket
                self.connection_count += 1
            
            logger.info(f"WebSocket connected. ID: {connection_id}, Total connections: {len(self.active_connections)}")
            return connection_id
        except Exception as e:
            logger.error(f"Failed to accept WebSocket connection: {e}")
            return None
    
    async def disconnect(self, connection_id: str):
        """Disconnect a WebSocket by connection ID"""
        async with self._lock:
            if connection_id in self.active_connections:
                try:
                    websocket = self.active_connections[connection_id]
                    # Check if connection is still open before closing
                    if websocket.client_state.name == "CONNECTED":
                        await websocket.close()
                except Exception as e:
                    logger.warning(f"Error closing WebSocket {connection_id}: {e}")
                finally:
                    del self.active_connections[connection_id]
                    logger.info(f"WebSocket disconnected. ID: {connection_id}, Total connections: {len(self.active_connections)}")
    
    async def is_connected(self, connection_id: str) -> bool:
        """Check if a connection is still active"""
        async with self._lock:
            if connection_id not in self.active_connections:
                return False
            
            websocket = self.active_connections[connection_id]
            return websocket.client_state.name == "CONNECTED"
    
    async def send_personal_message(self, message: Dict[str, Any], connection_id: str) -> bool:
        """Send message to a specific connection"""
        async with self._lock:
            if connection_id not in self.active_connections:
                logger.warning(f"Attempted to send message to non-existent connection: {connection_id}")
                return False
            
            websocket = self.active_connections[connection_id]
        
        try:
            # Check if connection is still open
            if websocket.client_state.name != "CONNECTED":
                logger.warning(f"Connection {connection_id} is not in CONNECTED state: {websocket.client_state.name}")
                await self.disconnect(connection_id)
                return False
            
            await websocket.send_json(message)
            return True
        except WebSocketDisconnect:
            logger.info(f"WebSocket {connection_id} disconnected during send")
            await self.disconnect(connection_id)
            return False
        except Exception as e:
            logger.error(f"Failed to send WebSocket message to {connection_id}: {e}")
            await self.disconnect(connection_id)
            return False
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all active connections"""
        async with self._lock:
            connection_ids = list(self.active_connections.keys())
        
        # Send to each connection
        for connection_id in connection_ids:
            success = await self.send_personal_message(message, connection_id)
            if not success:
                logger.warning(f"Failed to broadcast to connection {connection_id}")

# Helpers
def load_pickled(path: str):
    logger.info(f"Loading pickle: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Loaded pickle: {path}")
    return obj

def normalize_text(s: str) -> str:
    # Normalize Unicode (e.g., curly apostrophes) so regex and trie match
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("'", "'").replace("'", "'").replace("‛", "'")
    s = s.replace(""", '"').replace(""", '"').replace("„", '"')
    return s

def create_no_cache_response(content: Any) -> JSONResponse:
    """Create a JSONResponse with explicit no-cache headers"""
    response = JSONResponse(content=content)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Request-ID"] = str(uuid.uuid4())
    response.headers["X-Timestamp"] = str(time.time())
    return response

# Enhanced Core Service
class SpellCheckerService:
    def __init__(self):
        self.trie: Optional[Trie] = None
        self.indexed_n_grams: Optional[NgramGenerator] = None
        self.executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)

    async def initialize(self, trie_path: str, ngram_path: str):
        loop = asyncio.get_running_loop()
        if not os.path.exists(trie_path) or not os.path.exists(ngram_path):
            raise FileNotFoundError(f"Missing pickle(s). trie={trie_path} exists={os.path.exists(trie_path)} "
                                    f"ngrams={ngram_path} exists={os.path.exists(ngram_path)}")
        trie_task = loop.run_in_executor(self.executor, load_pickled, trie_path)
        ngram_task = loop.run_in_executor(self.executor, load_pickled, ngram_path)
        self.trie, self.indexed_n_grams = await asyncio.gather(trie_task, ngram_task)

        if not hasattr(self.trie, "search"):
            raise RuntimeError("Loaded trie object missing 'search' method.")
        if not hasattr(self.indexed_n_grams, "get_candidate_words_only"):
            raise RuntimeError("Loaded ngram object missing 'get_candidate_words_only'.")

        logger.info("SpellCheckerService initialized OK.")

    def _create_fresh_edit_distance_instance(self):
        """Create a fresh EditDistance instance for each request to avoid state issues"""
        return EditDistance()

    async def check_single_word(self, word: str, request_id: str = None) -> Dict[str, Any]:
        """Check spelling of a single word with same structure as sentence check"""
        request_id = request_id or str(uuid.uuid4())
        logger.info(f"[{request_id}] Checking single word: '{word}'")
        
        if NORMALIZE_INPUT:
            normalized_word = normalize_text(word)
        else:
            normalized_word = word
        
        # Clean the word (remove non-alphabetic characters except apostrophes)
        clean_word = re.sub(r"[^a-zA-Z']", "", normalized_word)
        if not clean_word:
            result = {
                "index": 1,
                "mispelled_word": word,
                "potential_reccomendations": [],
                "_request_id": request_id,
                "_timestamp": time.time()
            }
            logger.info(f"[{request_id}] Empty word result: {result}")
            return result
        
        loop = asyncio.get_running_loop()
        
        # Check if word exists in trie
        is_correct = await loop.run_in_executor(
            self.executor, 
            self.trie.search, 
            clean_word.lower()
        )
        
        logger.info(f"[{request_id}] Word '{word}' is_correct: {is_correct}")
        
        if is_correct:
            # Word is correct, return empty recommendations
            result = {
                "index": 1,
                "mispelled_word": word,
                "potential_reccomendations": [],
                "_request_id": request_id,
                "_timestamp": time.time()
            }
            logger.info(f"[{request_id}] Correct word result: {result}")
            return result
        
        # Get candidates using ngram for misspelled word
        candidates = await loop.run_in_executor(
            self.executor,
            self.indexed_n_grams.get_candidate_words_only,
            clean_word.lower(),
            CANDIDATE_THRESHOLD
        )
        
        logger.info(f"[{request_id}] Candidates for '{word}': {list(candidates) if candidates else 'None'}")
        
        # Build spell correction map in same format as sentence check
        spell_correction_map = [{
            "index": 1,
            "mispelled_word": word,
            "potential_reccomendations": list(candidates) if candidates else []
        }]
        
        # If no candidates found, return empty recommendations
        if not candidates:
            result = copy.deepcopy(spell_correction_map[0])
            result["_request_id"] = request_id
            result["_timestamp"] = time.time()
            logger.info(f"[{request_id}] No candidates result: {result}")
            return result
        
        # Create a fresh EditDistance instance to avoid state issues
        edit_distance_calculator = self._create_fresh_edit_distance_instance()
        
        # Run through edit distance calculation
        try:
            # Create a completely isolated copy for edit distance calculation
            isolated_spell_map = json.loads(json.dumps(spell_correction_map))
            
            final_result = await loop.run_in_executor(
                self.executor, 
                edit_distance_calculator.calculate_edit_distance,
                isolated_spell_map,
                MAX_EDIT_DISTANCE
            )
            
            logger.info(f"[{request_id}] Final result for '{word}': {final_result}")
            
            if final_result and len(final_result) > 0:
                result = copy.deepcopy(final_result[0])
                result["_request_id"] = request_id
                result["_timestamp"] = time.time()
                logger.info(f"[{request_id}] Returning result for '{word}': {result}")
                return result
            else:
                result = copy.deepcopy(spell_correction_map[0])
                result["_request_id"] = request_id
                result["_timestamp"] = time.time()
                logger.info(f"[{request_id}] Fallback result for '{word}': {result}")
                return result
        except Exception as e:
            logger.error(f"[{request_id}] Error in edit distance calculation for word '{word}': {e}")
            result = copy.deepcopy(spell_correction_map[0])
            result["_request_id"] = request_id
            result["_timestamp"] = time.time()
            return result

    async def check_multiple_words(self, words: List[str]) -> List[Dict[str, Any]]:
        """Check spelling of multiple words"""
        request_id = str(uuid.uuid4())
        logger.info(f"[{request_id}] Checking multiple words: {words}")
        
        # Process each word independently to avoid any cross-contamination
        results = []
        for i, word in enumerate(words):
            try:
                word_request_id = f"{request_id}-{i}"
                result = await self.check_single_word(word, word_request_id)
                # Update index for multiple words
                result["index"] = i + 1
                # Remove internal tracking fields from final response
                result.pop("_request_id", None)
                result.pop("_timestamp", None)
                results.append(result)
            except Exception as e:
                logger.error(f"[{request_id}] Error checking word '{word}': {e}")
                results.append({
                    "index": i + 1,
                    "mispelled_word": word,
                    "potential_reccomendations": []
                })
        
        return results

    def _build_spell_correction_map(self, sentence: str) -> List[Dict[str, Any]]:
        """Build spell correction map for sentence"""
        spell_correction_map: List[Dict[str, Any]] = []
        for i, match in enumerate(WORD_PATTERN.finditer(sentence)):
            word = match.group(0)
            if not self.trie.search(word.lower()):
                candidates = self.indexed_n_grams.get_candidate_words_only(
                    word.lower(), relevance_threshold=CANDIDATE_THRESHOLD
                )
                spell_correction_map.append({
                    "index": i + 1,
                    "mispelled_word": word,
                    "potential_reccomendations": list(candidates) if candidates else []
                })
        return spell_correction_map

    def _run_edit_distance(self, spell_correction_map: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run edit distance calculation with fresh instance"""
        edit_distance_calculator = self._create_fresh_edit_distance_instance()
        return edit_distance_calculator.calculate_edit_distance(spell_correction_map, MAX_EDIT_DISTANCE)

    async def check_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        """Check spelling of a sentence"""
        if NORMALIZE_INPUT:
            sentence = normalize_text(sentence)

        loop = asyncio.get_running_loop()

        # Build map in executor
        spell_map = await loop.run_in_executor(self.executor, self._build_spell_correction_map, sentence)

        if not spell_map:
            return []  # no misspellings

        # Run edit distance in executor with fresh instance
        final = await loop.run_in_executor(self.executor, self._run_edit_distance, spell_map)

        if not isinstance(final, list):
            logger.warning("EditDistance returned non-list; coercing to empty list.")
            return []
        return final

# Global instances
service: Optional[SpellCheckerService] = None
manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    logger.info("Starting Spell Checker API...")
    service = SpellCheckerService()
    try:
        await service.initialize(TRIE_PKL, NGRAM_PKL)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        service = None
    yield
    try:
        if service and service.executor:
            service.executor.shutdown(wait=True)
    except Exception:
        pass
    logger.info("Spell Checker API shutdown complete.")

# App
app = FastAPI(
    title="Enhanced Spell Checker API", 
    version="2.0.4", 
    description="Enterprise-grade spell checker with fixed WebSocket handling",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST Endpoints
@app.get("/health")
async def health():
    """Health check endpoint"""
    return create_no_cache_response({
        "status": "ok" if service and service.trie and service.indexed_n_grams else "not_ready",
        "trie_loaded": bool(service and service.trie),
        "ngrams_loaded": bool(service and service.indexed_n_grams),
        "threshold": CANDIDATE_THRESHOLD,
        "active_websocket_connections": len(manager.active_connections),
        "total_connections_made": manager.connection_count,
        "timestamp": time.time(),
        "version": "2.0.4"
    })

@app.post("/check-word")
async def check_word(req: WordCheckRequest):
    """Check spelling of a single word with same output structure as sentence check"""
    if service is None or service.trie is None or service.indexed_n_grams is None:
        raise HTTPException(status_code=503, detail="Spell checker not initialized. Check /health and pickle paths.")
    
    try:
        request_id = str(uuid.uuid4())
        logger.info(f"[{request_id}] Processing /check-word request for: '{req.word}'")
        
        result = await service.check_single_word(req.word, request_id)
        
        # Remove internal tracking fields before sending response
        clean_result = copy.deepcopy(result)
        clean_result.pop("_request_id", None)
        clean_result.pop("_timestamp", None)
        
        logger.info(f"[{request_id}] Final API response for '{req.word}': {clean_result}")
        
        return create_no_cache_response(clean_result)
        
    except Exception as e:
        logger.exception(f"check_word failed for word: '{req.word}'")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/check-words")
async def check_words(req: WordsCheckRequest):
    """Check spelling of multiple words"""
    if service is None or service.trie is None or service.indexed_n_grams is None:
        raise HTTPException(status_code=503, detail="Spell checker not initialized. Check /health and pickle paths.")
    
    try:
        results = await service.check_multiple_words(req.words)
        return create_no_cache_response(results)
    except Exception as e:
        logger.exception("check_words failed")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/check-sentence")
async def check_sentence(req: SentenceCheckRequest):
    """Check spelling of a sentence (original endpoint)"""
    if service is None or service.trie is None or service.indexed_n_grams is None:
        raise HTTPException(status_code=503, detail="Spell checker not initialized. Check /health and pickle paths.")
    try:
        result = await service.check_sentence(req.sentence)
        return create_no_cache_response(result)
    except Exception as e:
        logger.exception("check_sentence failed")
        raise HTTPException(status_code=500, detail=str(e))

# Fixed WebSocket Endpoint
@app.websocket("/ws/check-word")
async def websocket_spell_check(websocket: WebSocket):
    """WebSocket endpoint for real-time single word spell checking"""
    if service is None or service.trie is None or service.indexed_n_grams is None:
        await websocket.close(code=1008, reason="Service not initialized")
        return
    
    # Connect and get connection ID
    connection_id = await manager.connect(websocket)
    if not connection_id:
        return
    
    try:
        # Send welcome message
        await manager.send_personal_message({
            "type": "connected",
            "data": {
                "message": "Connected to spell checker WebSocket",
                "connection_id": connection_id,
                "timestamp": time.time()
            }
        }, connection_id)
        
        while await manager.is_connected(connection_id):
            try:
                # Check if connection is still valid before receiving
                if not await manager.is_connected(connection_id):
                    logger.info(f"Connection {connection_id} is no longer active")
                    break
                
                # Receive message from client with timeout
                try:
                    data = await asyncio.wait_for(websocket.receive_json(), timeout=60.0)
                except asyncio.TimeoutError:
                    # Send ping to check if connection is alive
                    success = await manager.send_personal_message({
                        "type": "ping",
                        "data": {"timestamp": time.time()}
                    }, connection_id)
                    if not success:
                        break
                    continue
                
                if data.get("type") == "ping":
                    await manager.send_personal_message({
                        "type": "pong",
                        "data": {"timestamp": time.time()}
                    }, connection_id)
                    continue
                
                if data.get("type") == "check_word":
                    word = data.get("data", {}).get("word", "").strip()
                    
                    if not word:
                        await manager.send_personal_message({
                            "type": "error",
                            "data": {
                                "message": "Word cannot be empty",
                                "timestamp": time.time()
                            }
                        }, connection_id)
                        continue
                    
                    if len(word) > 100:
                        await manager.send_personal_message({
                            "type": "error",
                            "data": {
                                "message": "Word too long (max 100 characters)",
                                "timestamp": time.time()
                            }
                        }, connection_id)
                        continue
                    
                    # Check spelling
                    result = await service.check_single_word(word)
                    
                    # Clean result for WebSocket
                    clean_result = copy.deepcopy(result)
                    clean_result.pop("_request_id", None)
                    clean_result.pop("_timestamp", None)
                    
                    # Send result only if connection is still active
                    if await manager.is_connected(connection_id):
                        await manager.send_personal_message({
                            "type": "result",
                            "data": {
                                **clean_result,
                                "timestamp": time.time()
                            }
                        }, connection_id)
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "data": {
                            "message": f"Unknown message type: {data.get('type')}",
                            "timestamp": time.time()
                        }
                    }, connection_id)
            
            except WebSocketDisconnect:
                logger.info(f"WebSocket {connection_id} disconnected by client")
                break
            except json.JSONDecodeError:
                if await manager.is_connected(connection_id):
                    await manager.send_personal_message({
                        "type": "error",
                        "data": {
                            "message": "Invalid JSON format",
                            "timestamp": time.time()
                        }
                    }, connection_id)
                else:
                    break
            except Exception as e:
                logger.error(f"Error processing WebSocket message for {connection_id}: {e}")
                if await manager.is_connected(connection_id):
                    await manager.send_personal_message({
                        "type": "error",
                        "data": {
                            "message": "Internal server error",
                            "timestamp": time.time()
                        }
                    }, connection_id)
                else:
                    break
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket {connection_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
    finally:
        # Ensure cleanup
        await manager.disconnect(connection_id)

@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return create_no_cache_response({
        "service_status": "ready" if service.trie and service.indexed_n_grams else "not_ready",
        "websocket_connections": len(manager.active_connections),
        "total_connections_made": manager.connection_count,
        "executor_workers": EXECUTOR_WORKERS,
        "candidate_threshold": CANDIDATE_THRESHOLD,
        "max_edit_distance": MAX_EDIT_DISTANCE,
        "normalize_input": NORMALIZE_INPUT
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("enhanced_spell_api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
