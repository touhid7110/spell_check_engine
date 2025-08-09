import streamlit as st
import asyncio
import websockets
import json
import re
import time
from typing import Dict, List, Any, Optional
import threading
from collections import defaultdict
import queue

# Configuration
WEBSOCKET_URL = "ws://localhost:8000/ws/check-word"
WORD_PATTERN = re.compile(r"\b[a-zA-Z']+\b")

class WebSocketSpellChecker:
    def __init__(self):
        self.websocket = None
        self.is_connected = False
        self.message_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.connection_thread = None
        self.word_suggestions = {}
        
    async def connect(self):
        """Connect to WebSocket server"""
        try:
            self.websocket = await websockets.connect(WEBSOCKET_URL)
            self.is_connected = True
            return True
        except Exception as e:
            st.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    async def send_message(self, message_type: str, data: Dict[str, Any]):
        """Send message to WebSocket server"""
        if not self.websocket:
            return False
        
        try:
            message = {
                "type": message_type,
                "data": data
            }
            await self.websocket.send(json.dumps(message))
            return True
        except Exception as e:
            st.error(f"Failed to send message: {e}")
            return False
    
    async def receive_messages(self):
        """Listen for incoming messages"""
        if not self.websocket:
            return
        
        try:
            while True:
                message = await self.websocket.recv()
                data = json.loads(message)
                self.response_queue.put(data)
        except websockets.exceptions.ConnectionClosed:
            self.is_connected = False
        except Exception as e:
            st.error(f"Error receiving message: {e}")
            self.is_connected = False
    
    async def check_word_async(self, word: str):
        """Check spelling of a word"""
        success = await self.send_message("check_word", {"word": word})
        return success
    
    def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.websocket:
            asyncio.create_task(self.websocket.close())
        self.is_connected = False

def init_websocket():
    """Initialize WebSocket connection"""
    if 'websocket_checker' not in st.session_state:
        st.session_state.websocket_checker = WebSocketSpellChecker()
    
    return st.session_state.websocket_checker

def extract_words_with_positions(text: str) -> List[Dict[str, Any]]:
    """Extract words with their positions in the text"""
    words = []
    for match in WORD_PATTERN.finditer(text):
        words.append({
            'word': match.group(0),
            'start': match.start(),
            'end': match.end(),
            'original_word': match.group(0)
        })
    return words

async def check_words_websocket(checker: WebSocketSpellChecker, words: List[str]):
    """Check multiple words via WebSocket"""
    if not checker.is_connected:
        connected = await checker.connect()
        if not connected:
            return {}
    
    # Start receiving messages in background
    receive_task = asyncio.create_task(checker.receive_messages())
    
    # Send word check requests
    results = {}
    for word in words:
        await checker.check_word_async(word.lower())
        await asyncio.sleep(0.1)  # Small delay to avoid overwhelming the server
    
    # Collect responses (with timeout)
    start_time = time.time()
    timeout = 10  # 10 seconds timeout
    
    while len(results) < len(words) and (time.time() - start_time) < timeout:
        try:
            response = checker.response_queue.get(timeout=1)
            if response.get('type') == 'result':
                data = response.get('data', {})
                word = data.get('mispelled_word', '').lower()
                if word:
                    results[word] = data.get('potential_reccomendations', [])
        except queue.Empty:
            continue
        except Exception as e:
            st.error(f"Error processing response: {e}")
            break
    
    # Cancel the receive task
    receive_task.cancel()
    
    return results

def run_websocket_check(words: List[str]):
    """Run WebSocket check in event loop"""
    try:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        checker = WebSocketSpellChecker()
        results = loop.run_until_complete(check_words_websocket(checker, words))
        
        # Clean up
        if checker.websocket:
            loop.run_until_complete(checker.websocket.close())
        loop.close()
        
        return results
    except Exception as e:
        st.error(f"WebSocket check failed: {e}")
        return {}

def replace_word_in_text(text: str, word_info: Dict[str, Any], replacement: str) -> str:
    """Replace a word in text at specific position"""
    start, end = word_info['start'], word_info['end']
    new_text = text[:start] + replacement + text[end:]
    return new_text

def get_word_suggestions_display(suggestions: List[Dict[str, Any]]) -> List[str]:
    """Extract candidate words from suggestions for display"""
    candidates = []
    for suggestion in suggestions[:5]:  # Limit to top 5 suggestions
        if isinstance(suggestion, dict):
            candidate = suggestion.get('potential_candidate', '')
            edit_distance = suggestion.get('edit_distance', 0)
            if candidate:
                candidates.append(f"{candidate} (distance: {edit_distance})")
        elif isinstance(suggestion, str):
            candidates.append(suggestion)
    return candidates

def main():
    st.set_page_config(
        page_title="WebSocket Spell Checker",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("🚀 Real-time WebSocket Spell Checker")
    st.markdown("---")
    
    # Initialize session state
    if 'current_text' not in st.session_state:
        st.session_state.current_text = ""
    if 'spell_results' not in st.session_state:
        st.session_state.spell_results = {}
    if 'last_check_time' not in st.session_state:
        st.session_state.last_check_time = 0
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # WebSocket connection status
        st.subheader("Connection Status")
        
        # Test connection button
        if st.button("🔌 Test WebSocket Connection"):
            with st.spinner("Testing connection..."):
                try:
                    checker = WebSocketSpellChecker()
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    connected = loop.run_until_complete(checker.connect())
                    
                    if connected:
                        st.success("✅ WebSocket connection successful!")
                        # Test with a simple word
                        test_results = loop.run_until_complete(
                            check_words_websocket(checker, ["test"])
                        )
                        if test_results:
                            st.info("🎯 Spell checker is working correctly!")
                    else:
                        st.error("❌ Failed to connect to WebSocket")
                    
                    if checker.websocket:
                        loop.run_until_complete(checker.websocket.close())
                    loop.close()
                    
                except Exception as e:
                    st.error(f"Connection test failed: {e}")
        
        st.markdown("---")
        
        # Settings
        auto_check = st.checkbox("🔄 Auto-check on text change", value=True)
        check_delay = st.slider("⏱️ Check delay (seconds)", 0.5, 3.0, 1.0, 0.1)
        max_suggestions = st.slider("📝 Max suggestions per word", 1, 10, 5)
        
        st.markdown("---")
        
        # Statistics
        st.subheader("📊 Statistics")
        if st.session_state.spell_results:
            total_words = len(st.session_state.spell_results)
            misspelled_words = sum(1 for suggestions in st.session_state.spell_results.values() if suggestions)
            correct_words = total_words - misspelled_words
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Words", total_words)
                st.metric("Correct Words", correct_words)
            with col2:
                st.metric("Misspelled Words", misspelled_words)
                accuracy = (correct_words / total_words * 100) if total_words > 0 else 0
                st.metric("Accuracy", f"{accuracy:.1f}%")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📄 Text Editor")
        
        # Text input area
        text_input = st.text_area(
            "Enter your text here:",
            value=st.session_state.current_text,
            height=300,
            key="text_input_area",
            help="Type or paste your text. Misspelled words will be highlighted with suggestions."
        )
        
        # Check if text has changed
        text_changed = text_input != st.session_state.current_text
        current_time = time.time()
        
        # Auto-check logic
        should_check = (
            text_changed and 
            auto_check and 
            (current_time - st.session_state.last_check_time) > check_delay
        )
        
        # Manual check button
        col_check, col_clear = st.columns([1, 1])
        with col_check:
            manual_check = st.button("🔍 Check Spelling", type="primary")
        with col_clear:
            if st.button("🗑️ Clear Text"):
                st.session_state.current_text = ""
                st.session_state.spell_results = {}
                st.rerun()
        
        # Perform spell check
        if should_check or manual_check:
            if text_input.strip():
                with st.spinner("Checking spelling via WebSocket..."):
                    # Extract words from text
                    words_info = extract_words_with_positions(text_input)
                    unique_words = list(set(word_info['word'].lower() for word_info in words_info))
                    
                    if unique_words:
                        # Check words via WebSocket
                        spell_results = run_websocket_check(unique_words)
                        
                        # Update session state
                        st.session_state.current_text = text_input
                        st.session_state.spell_results = spell_results
                        st.session_state.last_check_time = current_time
                        
                        if spell_results:
                            st.success(f"✅ Checked {len(unique_words)} unique words!")
                        else:
                            st.warning("⚠️ No results received from WebSocket. Check connection.")
            else:
                st.session_state.current_text = text_input
                st.session_state.spell_results = {}
    
    with col2:
        st.subheader("🔧 Spelling Suggestions")
        
        if st.session_state.spell_results:
            # Find misspelled words with suggestions
            misspelled_words = {
                word: suggestions 
                for word, suggestions in st.session_state.spell_results.items() 
                if suggestions
            }
            
            if misspelled_words:
                st.info(f"Found {len(misspelled_words)} misspelled word(s)")
                
                for word, suggestions in misspelled_words.items():
                    with st.expander(f"❌ '{word}'" + (f" ({len(suggestions)} suggestions)" if suggestions else " (no suggestions)"), expanded=True):
                        if suggestions:
                            st.write("**Suggestions:**")
                            
                            # Display suggestions with replacement buttons
                            for i, suggestion in enumerate(suggestions[:max_suggestions]):
                                candidate = suggestion.get('potential_candidate', '') if isinstance(suggestion, dict) else str(suggestion)
                                edit_distance = suggestion.get('edit_distance', 'N/A') if isinstance(suggestion, dict) else 'N/A'
                                
                                col_suggestion, col_replace = st.columns([3, 1])
                                
                                with col_suggestion:
                                    st.write(f"• **{candidate}** (edit distance: {edit_distance})")
                                
                                with col_replace:
                                    if st.button(f"Replace", key=f"replace_{word}_{i}", help=f"Replace '{word}' with '{candidate}'"):
                                        # Replace all occurrences of the word in text
                                        words_info = extract_words_with_positions(st.session_state.current_text)
                                        
                                        # Sort by position (descending) to maintain correct positions during replacement
                                        words_to_replace = [w for w in words_info if w['word'].lower() == word.lower()]
                                        words_to_replace.sort(key=lambda x: x['start'], reverse=True)
                                        
                                        new_text = st.session_state.current_text
                                        replacements_made = 0
                                        
                                        for word_info in words_to_replace:
                                            # Preserve original case if possible
                                            if word_info['word'].isupper():
                                                replacement = candidate.upper()
                                            elif word_info['word'].istitle():
                                                replacement = candidate.capitalize()
                                            else:
                                                replacement = candidate
                                            
                                            new_text = replace_word_in_text(new_text, word_info, replacement)
                                            replacements_made += 1
                                        
                                        # Update session state
                                        st.session_state.current_text = new_text
                                        
                                        # Remove the replaced word from spell results
                                        if word.lower() in st.session_state.spell_results:
                                            del st.session_state.spell_results[word.lower()]
                                        
                                        st.success(f"✅ Replaced {replacements_made} occurrence(s) of '{word}' with '{candidate}'")
                                        st.rerun()
                        else:
                            st.write("No suggestions available for this word.")
            else:
                st.success("🎉 No misspelled words found!")
        else:
            st.info("👆 Enter text and click 'Check Spelling' to see suggestions")
    
    # Display current text with highlighting
    if st.session_state.current_text and st.session_state.spell_results:
        st.markdown("---")
        st.subheader("📋 Text with Spell Check Results")
        
        # Create highlighted text
        text = st.session_state.current_text
        words_info = extract_words_with_positions(text)
        
        highlighted_text = ""
        last_end = 0
        
        for word_info in words_info:
            word = word_info['word'].lower()
            
            # Add text before this word
            highlighted_text += text[last_end:word_info['start']]
            
            # Add the word (highlighted if misspelled)
            if word in st.session_state.spell_results and st.session_state.spell_results[word]:
                # Misspelled word - highlight in red
                highlighted_text += f"<span style='background-color: #ffcccc; color: #cc0000; font-weight: bold;'>{word_info['word']}</span>"
            elif word in st.session_state.spell_results:
                # Correct word - highlight in green
                highlighted_text += f"<span style='background-color: #ccffcc; color: #006600;'>{word_info['word']}</span>"
            else:
                # Not checked
                highlighted_text += word_info['word']
            
            last_end = word_info['end']
        
        # Add remaining text
        highlighted_text += text[last_end:]
        
        # Display highlighted text
        st.markdown(highlighted_text, unsafe_allow_html=True)
        
        # Legend
        st.markdown("""
        **Legend:**
        - <span style='background-color: #ffcccc; color: #cc0000; font-weight: bold;'>Misspelled words</span>
        - <span style='background-color: #ccffcc; color: #006600;'>Correct words</span>
        - Normal text: Not checked yet
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
