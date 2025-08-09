import streamlit as st
import asyncio
import websockets
import json
import re
import time
import requests
from typing import Dict, List, Any, Optional
import threading
from collections import defaultdict
import queue

# Configuration
WEBSOCKET_URL = "ws://localhost:8000/ws/check-word"
GENAI_BASE_URL = "http://localhost:7110"  # Adjust based on your GenAI API server
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

class GenAIEnhancer:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
    
    def get_available_styles(self) -> Dict[str, Any]:
        """Get available styles from GenAI API"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/styles", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to get styles: HTTP {response.status_code}")
                return {"available_styles": [], "total_count": 0}
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to GenAI API: {e}")
            return {"available_styles": [], "total_count": 0}
    
    def enhance_text(self, text: str, style: str) -> Dict[str, Any]:
        """Enhance text with selected style"""
        try:
            payload = {
                "text": text,
                "style": style
            }
            response = requests.post(
                f"{self.base_url}/api/v1/enhance", 
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to enhance text: HTTP {response.status_code}")
                return {"enhanced_reccomendations": {"suggestions": []}, "success": False}
        except requests.exceptions.RequestException as e:
            st.error(f"Error enhancing text: {e}")
            return {"enhanced_reccomendations": {"suggestions": []}, "success": False}

def init_websocket():
    """Initialize WebSocket connection"""
    if 'websocket_checker' not in st.session_state:
        st.session_state.websocket_checker = WebSocketSpellChecker()
    
    return st.session_state.websocket_checker

def init_genai_enhancer():
    """Initialize GenAI enhancer"""
    if 'genai_enhancer' not in st.session_state:
        st.session_state.genai_enhancer = GenAIEnhancer(GENAI_BASE_URL)
    
    return st.session_state.genai_enhancer

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

def check_if_all_words_correct():
    """Check if all words in the current text are spelled correctly"""
    if not st.session_state.spell_results:
        return False
    
    # Check if there are any misspelled words
    misspelled_words = {
        word: suggestions 
        for word, suggestions in st.session_state.spell_results.items() 
        if suggestions
    }
    
    return len(misspelled_words) == 0 and len(st.session_state.spell_results) > 0

def main():
    st.set_page_config(
        page_title="AI-Powered Spell Checker & Text Enhancer",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🚀 AI-Powered Spell Checker & Text Enhancer")
    st.markdown("---")
    
    # Initialize session state
    if 'current_text' not in st.session_state:
        st.session_state.current_text = ""
    if 'spell_results' not in st.session_state:
        st.session_state.spell_results = {}
    if 'last_check_time' not in st.session_state:
        st.session_state.last_check_time = 0
    if 'available_styles' not in st.session_state:
        st.session_state.available_styles = []
    if 'show_genai_enhancement' not in st.session_state:
        st.session_state.show_genai_enhancement = False
    if 'selected_style' not in st.session_state:
        st.session_state.selected_style = None
    if 'enhancement_results' not in st.session_state:
        st.session_state.enhancement_results = {}
    if 'enhancement_text' not in st.session_state:
        st.session_state.enhancement_text = ""
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # WebSocket connection status
        st.subheader("🔗 Spell Checker Connection")
        
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
        
        # GenAI API connection status
        st.subheader("🤖 GenAI Enhancement")
        
        if st.button("🧠 Test GenAI API Connection"):
            with st.spinner("Testing GenAI connection..."):
                enhancer = init_genai_enhancer()
                styles_data = enhancer.get_available_styles()
                if styles_data.get("available_styles"):
                    st.success(f"✅ GenAI API connected! {styles_data.get('total_count', 0)} styles available")
                    st.session_state.available_styles = styles_data.get("available_styles", [])
                else:
                    st.error("❌ Failed to connect to GenAI API")
        
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
                st.session_state.show_genai_enhancement = False
                st.session_state.enhancement_results = {}
                st.session_state.enhancement_text = ""
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
        
        # Show GenAI Enhancement section if all words are correct
        if check_if_all_words_correct() and st.session_state.current_text.strip():
            st.markdown("---")
            
            # GenAI Enhancement Button
            col_enhance, col_hide = st.columns([3, 1])
            with col_enhance:
                if st.button("🧠 Enhance with GenAI", type="secondary", use_container_width=True):
                    st.session_state.show_genai_enhancement = True
                    st.session_state.enhancement_text = st.session_state.current_text
                    # Load available styles
                    enhancer = init_genai_enhancer()
                    styles_data = enhancer.get_available_styles()
                    st.session_state.available_styles = styles_data.get("available_styles", [])
                    st.rerun()
            
            with col_hide:
                if st.session_state.show_genai_enhancement and st.button("❌ Hide"):
                    st.session_state.show_genai_enhancement = False
                    st.session_state.enhancement_results = {}
                    st.rerun()
        
        # GenAI Enhancement Section
        if st.session_state.show_genai_enhancement:
            st.subheader("🤖 GenAI Text Enhancement")
            
            # Style selection
            if st.session_state.available_styles:
                st.write("**Available Enhancement Styles:**")
                
                # Create columns for style selection
                style_cols = st.columns(3)
                selected_styles = []
                
                for i, style in enumerate(st.session_state.available_styles):
                    col_idx = i % 3
                    with style_cols[col_idx]:
                        if st.checkbox(f"**{style}**", key=f"style_{style}"):
                            selected_styles.append(style)
                
                # Text to enhance
                st.write("**Text to enhance:**")
                enhancement_text = st.text_area(
                    "Edit text if needed:",
                    value=st.session_state.enhancement_text,
                    height=150,
                    key="enhancement_text_area"
                )
                st.session_state.enhancement_text = enhancement_text
                
                # Enhance button
                if selected_styles and st.button("✨ Enhance Text", type="primary"):
                    if enhancement_text.strip():
                        with st.spinner("Enhancing text with GenAI..."):
                            enhancer = init_genai_enhancer()
                            
                            # Process each selected style
                            enhancement_results = {}
                            for style in selected_styles:
                                try:
                                    result = enhancer.enhance_text(enhancement_text, style)
                                    enhancement_results[style] = result
                                except Exception as e:
                                    st.error(f"Failed to enhance with style '{style}': {e}")
                                    enhancement_results[style] = {"enhanced_reccomendations": {"suggestions": []}, "success": False}
                            
                            st.session_state.enhancement_results = enhancement_results
                            
                            if enhancement_results:
                                st.success("✅ Text enhanced successfully!")
                    else:
                        st.warning("Please enter text to enhance.")
                
                # Display enhancement results
                if st.session_state.enhancement_results:
                    st.markdown("---")
                    st.write("**🎨 Enhancement Results:**")
                    
                    for style, result in st.session_state.enhancement_results.items():
                        with st.expander(f"📝 {style} Style", expanded=True):
                            if result.get("success", True):  # Default to True if key doesn't exist
                                recommendations = result.get("enhanced_reccomendations", {})
                                suggestions = recommendations.get("suggestions", [])
                                
                                if suggestions:
                                    for i, suggestion in enumerate(suggestions):
                                        col_suggestion, col_replace = st.columns([4, 1])
                                        
                                        with col_suggestion:
                                            st.write(f"**Option {i+1}:**")
                                            st.write(suggestion)
                                        
                                        with col_replace:
                                            if st.button("📄 Use This", key=f"use_{style}_{i}"):
                                                st.session_state.current_text = suggestion
                                                st.session_state.enhancement_text = suggestion
                                                st.success(f"✅ Applied {style} enhancement!")
                                                st.rerun()
                                else:
                                    st.write("No suggestions available for this style.")
                            else:
                                st.error(f"Enhancement failed for {style} style.")
            else:
                st.warning("⚠️ No styles available. Please test GenAI API connection first.")
    
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
                if st.session_state.current_text.strip():
                    st.success("🎉 No misspelled words found!")
                    if len(st.session_state.spell_results) > 0:
                        st.info("💡 Your text is ready for GenAI enhancement!")
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
