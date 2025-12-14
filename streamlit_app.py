import streamlit as st
import requests
import json
import time
from typing import List, Dict, Optional
import httpx
from datetime import datetime
import re

# Page configuration
st.set_page_config(
    page_title="TinyAISearch",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API base URL
API_BASE_URL = "http://localhost:5000"

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'sessions' not in st.session_state:
    st.session_state.sessions = []
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'is_loading' not in st.session_state:
    st.session_state.is_loading = False
if 'use_web' not in st.session_state:
    st.session_state.use_web = False
if 'search_steps' not in st.session_state:
    st.session_state.search_steps = []
if 'current_search_steps' not in st.session_state:
    st.session_state.current_search_steps = []
if 'trace_logs' not in st.session_state:
    st.session_state.trace_logs = []
if 'show_trace' not in st.session_state:
    st.session_state.show_trace = True
if 'trace_spans' not in st.session_state:
    st.session_state.trace_spans = {}


def add_trace_log(event_type: str, message: str, data: any = None):
    """Add a trace log entry with OpenTelemetry-style structure"""
    log_entry = {
        'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3],
        'type': event_type,
        'message': message,
        'data': data,
        'level': 'INFO'
    }
    st.session_state.trace_logs.append(log_entry)


def start_span(span_name: str, attributes: dict = None):
    """Start a trace span (OpenTelemetry-style)"""
    span_id = f"{span_name}_{len(st.session_state.trace_spans)}"
    span = {
        'name': span_name,
        'start_time': datetime.now(),
        'attributes': attributes or {},
        'events': []
    }
    st.session_state.trace_spans[span_id] = span
    add_trace_log("span_start", f"→ START: {span_name}", attributes)
    return span_id


def add_span_event(span_id: str, event_name: str, attributes: dict = None):
    """Add an event to a span"""
    if span_id in st.session_state.trace_spans:
        event = {
            'name': event_name,
            'timestamp': datetime.now(),
            'attributes': attributes or {}
        }
        st.session_state.trace_spans[span_id]['events'].append(event)
        add_trace_log("span_event", f"  • {event_name}", attributes)


def end_span(span_id: str, status: str = "OK", attributes: dict = None):
    """End a trace span"""
    if span_id in st.session_state.trace_spans:
        span = st.session_state.trace_spans[span_id]
        span['end_time'] = datetime.now()
        span['duration_ms'] = (span['end_time'] - span['start_time']).total_seconds() * 1000
        span['status'] = status
        if attributes:
            span['attributes'].update(attributes)

        add_trace_log(
            "span_end",
            f"← END: {span['name']} ({span['duration_ms']:.2f}ms) - {status}",
            {
                'duration_ms': span['duration_ms'],
                'status': status,
                'attributes': span['attributes']
            }
        )


def check_backend_connection():
    """Check if backend is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/status", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_sessions(user_id: str) -> List[Dict]:
    """Fetch user sessions from backend"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/sessions",
            params={"user_id": user_id},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Failed to fetch sessions: {e}")
        return []


def get_messages(session_id: str) -> List[Dict]:
    """Fetch messages for a session"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/sessions/{session_id}/messages",
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Failed to fetch messages: {e}")
        return []


def create_session(user_id: str, title: Optional[str] = None) -> Optional[str]:
    """Create a new session"""
    try:
        payload = {"user_id": user_id}
        if title:
            payload["title"] = title
        response = requests.post(
            f"{API_BASE_URL}/session",
            json=payload,
            timeout=5
        )
        if response.status_code == 201:
            data = response.json()
            return data.get("session_id")
        return None
    except Exception as e:
        st.error(f"Failed to create session: {e}")
        return None


def process_streaming_response(response, message_placeholder, search_steps_placeholder):
    """Process streaming response from backend"""
    full_content = ""
    references = []
    current_steps = []
    chunk_count = 0
    answer_chunks = 0
    process_steps = 0

    span_id = start_span("process_streaming_response")

    try:
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8', errors='ignore').strip()
                if not line_str:
                    continue

                try:
                    chunk = json.loads(line_str)
                    chunk_type = chunk.get('type')
                    payload = chunk.get('payload')
                    chunk_count += 1

                    # Log raw chunk for debugging
                    if chunk_count == 1:
                        add_span_event(span_id, "first_chunk_received", {
                            "chunk_type": chunk_type,
                            "chunk": chunk
                        })

                    if chunk_type == 'process':
                        process_steps += 1
                        # Add search step
                        icon = '⏳'
                        step_type = 'processing'

                        if "正在分析问题" in str(payload) or "analyzing" in str(payload).lower():
                            icon = '🤔'
                            step_type = 'analyzing_query'
                        elif "不需要搜索" in str(payload) or "no search" in str(payload).lower():
                            icon = '💬'
                            step_type = 'direct_answer'
                        elif "搜索关键词" in str(payload) or "search keyword" in str(payload).lower():
                            icon = '🔍'
                            step_type = 'search_keyword_extraction'
                        elif "搜索完成" in str(payload) or "search complete" in str(payload).lower():
                            icon = '✅'
                            step_type = 'search_completed'
                        elif "tool" in str(payload).lower() or "function" in str(payload).lower():
                            icon = '🔧'
                            step_type = 'tool_call'

                        current_steps.append({
                            'text': str(payload),
                            'icon': icon,
                            'timestamp': datetime.now().strftime("%H:%M")
                        })
                        st.session_state.current_search_steps = current_steps

                        # Extract detailed information from payload
                        event_data = {
                            "step_number": process_steps,
                            "step_type": step_type,
                            "message": str(payload),
                            "full_payload": payload
                        }

                        # Parse search keywords if present
                        if "搜索关键词" in str(payload) or "search keyword" in str(payload).lower():
                            import re
                            # Try to extract keywords from the message
                            keyword_match = re.search(r'[:：]\s*(.+)', str(payload))
                            if keyword_match:
                                keywords = keyword_match.group(1).strip()
                                event_data["extracted_keywords"] = keywords
                                event_data["tool"] = "web_search"

                        # Parse search results count if present
                        elif "搜索完成" in str(payload) or "找到" in str(payload):
                            import re
                            count_match = re.search(r'(\d+)\s*个', str(payload))
                            if count_match:
                                event_data["documents_found"] = int(count_match.group(1))
                                event_data["tool_result"] = "web_search_completed"

                        add_span_event(span_id, "process_step", event_data)

                        # Update search steps display
                        if search_steps_placeholder:
                            with search_steps_placeholder.container():
                                for step in current_steps:
                                    st.info(f"{step['icon']} **{step['text']}** ({step['timestamp']})")

                    elif chunk_type == 'answer_chunk':
                        answer_chunks += 1
                        full_content += str(payload)
                        # Update message display with markdown
                        message_placeholder.markdown(full_content + "▌")

                        # Log first chunk and then every 20th to avoid spam
                        if answer_chunks == 1:
                            add_span_event(span_id, "answer_streaming_started", {
                                "first_chunk": str(payload)[:100]
                            })
                        elif answer_chunks % 20 == 0:
                            add_span_event(span_id, f"answer_progress", {
                                "chunks_received": answer_chunks,
                                "content_length": len(full_content),
                                "preview": full_content[:200] + "..." if len(full_content) > 200 else full_content
                            })

                    elif chunk_type == 'retrieval_scores':
                        # Handle retrieval scoring information
                        version = payload.get('version', 'unknown')
                        method = payload.get('method', 'unknown')
                        scores = payload.get('scores', [])

                        add_span_event(span_id, "retrieval_scores_received", {
                            "version": version,
                            "method": method,
                            "chunks_count": len(scores),
                            "scores": scores
                        })

                    elif chunk_type == 'reference':
                        references = payload if isinstance(payload, list) else []
                        add_span_event(span_id, "references_received", {
                            "count": len(references),
                            "references": references
                        })

                    elif chunk_type == 'error':
                        add_span_event(span_id, "error_received", {"error": str(payload)})
                        st.error(f"Error: {payload}")
                        end_span(span_id, "ERROR", {"error": str(payload)})
                        return None, None

                except json.JSONDecodeError as e:
                    add_trace_log("error", f"JSON decode error: {e}", {"line": line_str[:200]})
                    continue

        # Final update without cursor
        if full_content:
            message_placeholder.markdown(full_content)

        end_span(span_id, "OK", {
            "total_chunks": chunk_count,
            "answer_chunks": answer_chunks,
            "process_steps": process_steps,
            "content_length": len(full_content),
            "references_count": len(references)
        })

        return full_content, references

    except Exception as e:
        add_trace_log("error", f"Stream processing error: {e}")
        st.error(f"Error processing stream: {e}")
        import traceback
        print(traceback.format_exc())
        end_span(span_id, "ERROR", {"exception": str(e)})
        return None, None


def send_message(query: str, session_id: Optional[str], user_id: str, use_web: bool):
    """Send a message and get streaming response"""
    span_id = start_span("send_message", {
        "function": "send_message",
        "query": query[:100] + "..." if len(query) > 100 else query,
        "user_id": user_id,
        "use_web": use_web,
        "session_id": session_id
    })

    payload = {
        "query": query,
        "user_id": user_id,
        "use_web": use_web
    }
    if session_id:
        payload["session_id"] = session_id

    add_span_event(span_id, "preparing_request", {
        "endpoint": f"{API_BASE_URL}/search",
        "method": "POST",
        "payload": payload
    })

    try:
        add_span_event(span_id, "sending_http_request", {
            "url": f"{API_BASE_URL}/search",
            "stream": True,
            "timeout": 300
        })

        response = requests.post(
            f"{API_BASE_URL}/search",
            json=payload,
            stream=True,
            timeout=300
        )
        response.raise_for_status()

        end_span(span_id, "OK", {
            "status_code": response.status_code,
            "content_type": response.headers.get('content-type'),
            "transfer_encoding": response.headers.get('transfer-encoding')
        })
        return response
    except Exception as e:
        end_span(span_id, "ERROR", {"exception": str(e), "exception_type": type(e).__name__})
        st.error(f"Failed to send message: {e}")
        return None


def update_llm_config(use_openrouter: bool, openrouter_api_key: str, model_name: str):
    """Update LLM configuration in backend"""
    try:
        # Get current settings
        response = requests.get(f"{API_BASE_URL}/api/settings", timeout=5)
        if response.status_code != 200:
            st.error("Failed to fetch current settings")
            return False
        
        current_settings = response.json()
        
        # Update LLM settings based on OpenRouter or direct API
        if use_openrouter:
            # Use OpenRouter API for LLM, Embedding, and Rerank
            current_settings['llm_base_url'] = 'https://openrouter.ai/api/v1'
            current_settings['llm_api_key'] = openrouter_api_key
            current_settings['llm_model_name'] = model_name

            # Also configure embedding and rerank to use OpenRouter with the same key
            current_settings['embedding_base_url'] = 'https://openrouter.ai/api/v1'
            current_settings['embedding_api_key'] = openrouter_api_key
            current_settings['embedding_model_name'] = model_name  # Use same model for embedding

            current_settings['rerank_base_url'] = 'https://openrouter.ai/api/v1'
            current_settings['rerank_api_key'] = openrouter_api_key
            current_settings['rerank_model_name'] = model_name  # Use same model for rerank
        else:
            # Keep existing settings or use provided values
            if model_name:
                current_settings['llm_model_name'] = model_name
            # API key and base_url should be set via config page
        
        # Save settings
        save_response = requests.post(
            f"{API_BASE_URL}/api/settings",
            json={"settings": current_settings},
            timeout=5
        )
        
        if save_response.status_code == 200:
            return True
        else:
            error_msg = save_response.text
            try:
                error_data = save_response.json()
                error_msg = error_data.get('message', error_msg)
            except:
                pass
            st.error(f"Failed to save settings: {error_msg}")
            return False
    
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Make sure the server is running.")
        return False
    except Exception as e:
        st.error(f"Error updating LLM config: {e}")
        return False


# Sidebar Configuration
with st.sidebar:
    st.title("🔍 TinyAISearch")
    st.markdown("---")
    
    # User ID input (simple authentication)
    if not st.session_state.user_id:
        user_id_input = st.text_input("User ID", placeholder="Enter your user ID")
        if st.button("Set User ID"):
            if user_id_input:
                st.session_state.user_id = user_id_input
                st.rerun()
    else:
        st.success(f"Logged in as: **{st.session_state.user_id}**")
        if st.button("Logout"):
            st.session_state.user_id = None
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.rerun()
    
    st.markdown("---")
    
    # OpenRouter Configuration
    st.subheader("🔧 LLM Configuration")
    
    use_openrouter = st.checkbox("Use OpenRouter API", value=False, help="Use OpenRouter to access multiple LLM providers")
    
    if use_openrouter:
        openrouter_api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Get your API key from https://openrouter.ai/keys",
            placeholder="sk-or-v1-..."
        )

        # Model configuration for OpenRouter
        openrouter_model = st.text_input(
            "Model Name",
            value="openai/gpt-5.2-chat",
            help="Enter the model name (e.g., openai/gpt-5.2-chat, anthropic/claude-3.5-sonnet)",
            placeholder="openai/gpt-5.2-chat"
        )

        if st.button("✅ Apply OpenRouter Config", use_container_width=True):
            if openrouter_api_key:
                with st.spinner("Applying configuration..."):
                    if update_llm_config(True, openrouter_api_key, openrouter_model):
                        st.success(f"✅ OpenRouter configured! Using: {openrouter_model}")
                        st.rerun()
            else:
                st.warning("⚠️ Please enter OpenRouter API key")
    else:
        # Direct LLM configuration
        st.info("💡 Configure LLM via backend settings API or use OpenRouter above")
        
        # Show current model if available
        try:
            response = requests.get(f"{API_BASE_URL}/api/settings", timeout=2)
            if response.status_code == 200:
                settings = response.json()
                current_model = settings.get('llm_model_name', 'Not configured')
                st.caption(f"Current model: **{current_model}**")
        except:
            pass
        
        # Allow manual model override
        manual_model = st.text_input(
            "Model Name Override",
            placeholder="e.g., gpt-4, claude-3.5-sonnet",
            help="Override model name (requires backend API key configured)"
        )
        
        if manual_model and st.button("✅ Apply Model", use_container_width=True):
            with st.spinner("Applying model..."):
                if update_llm_config(False, "", manual_model):
                    st.success(f"✅ Model set to: {manual_model}")
                    st.rerun()
    
    st.markdown("---")
    
    # Session Management
    st.subheader("💬 Sessions")
    
    if st.session_state.user_id:
        # Refresh sessions button
        if st.button("🔄 Refresh Sessions"):
            st.session_state.sessions = get_sessions(st.session_state.user_id)
        
        # New chat button
        if st.button("➕ New Chat"):
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.session_state.current_search_steps = []
            st.rerun()
        
        # Display sessions
        if st.session_state.sessions:
            st.markdown("**Recent Sessions:**")
            for session in st.session_state.sessions:
                session_id = session.get('session_id')
                title = session.get('title', 'Untitled')
                is_selected = st.session_state.current_session_id == session_id
                
                if st.button(
                    f"{'✓ ' if is_selected else ''}{title}",
                    key=f"session_{session_id}",
                    use_container_width=True
                ):
                    st.session_state.current_session_id = session_id
                    # Load messages for this session
                    messages_data = get_messages(session_id)
                    st.session_state.messages = []
                    for msg in messages_data:
                        if msg.get('role') == 'user':
                            st.session_state.messages.append({
                                'role': 'user',
                                'content': msg.get('content', '')
                            })
                        else:
                            try:
                                content_data = json.loads(msg.get('content', '{}'))
                                st.session_state.messages.append({
                                    'role': 'assistant',
                                    'content': content_data.get('text', ''),
                                    'references': content_data.get('references', [])
                                })
                            except:
                                st.session_state.messages.append({
                                    'role': 'assistant',
                                    'content': str(msg.get('content', '')),
                                    'references': []
                                })
                    st.rerun()
        else:
            st.info("No sessions yet. Start a new chat!")
    
    st.markdown("---")
    
    # Web search toggle
    st.session_state.use_web = st.checkbox(
        "🌐 Enable Web Search",
        value=st.session_state.use_web,
        help="Enable internet search for queries"
    )


# Main Chat Interface
col_header1, col_header2 = st.columns([3, 1])
with col_header1:
    st.title("🔍 TinyAISearch")
    st.caption("AI-powered search assistant with RAG capabilities")
with col_header2:
    if st.session_state.user_id:
        st.caption(f"👤 {st.session_state.user_id}")

# Toggle for trace window
st.session_state.show_trace = st.checkbox("🔍 Show Trace Window", value=st.session_state.show_trace)

# Check backend connection
if not check_backend_connection():
    st.error("⚠️ Backend server is not running. Please start the backend server at http://localhost:5000")
    st.stop()

# Check if user is logged in
if not st.session_state.user_id:
    st.info("👆 Please set your User ID in the sidebar to start chatting")
    st.stop()

# Load sessions on first load
if not st.session_state.sessions and st.session_state.user_id:
    st.session_state.sessions = get_sessions(st.session_state.user_id)

# Create two-column layout
if st.session_state.show_trace:
    chat_col, trace_col = st.columns([2, 1])
else:
    chat_col = st.container()
    trace_col = None

# Display chat messages in left column
with chat_col:
    st.markdown("### 💬 Chat")
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Render markdown content
            content = message.get("content", "")
            if content:
                st.markdown(content)

            # Display references if available
            references = message.get("references", [])
            if references:
                with st.expander(f"📚 References ({len(references)})", expanded=False):
                    for ref in references:
                        if isinstance(ref, dict):
                            for title, url in ref.items():
                                st.markdown(f"- [{title}]({url})")
                        elif isinstance(ref, str):
                            st.markdown(f"- {ref}")

# Display trace logs in right column
if st.session_state.show_trace and trace_col:
    with trace_col:
        st.markdown("### 🔬 Trace Logs")

        # Add clear button
        if st.button("🗑️ Clear Trace", key="clear_trace"):
            st.session_state.trace_logs = []
            st.rerun()

        # Display trace logs
        trace_container = st.container()
        with trace_container:
            if st.session_state.trace_logs:
                for idx, log in enumerate(st.session_state.trace_logs):
                    log_type = log.get('type', 'info')
                    timestamp = log.get('timestamp', '')
                    message = log.get('message', '')
                    data = log.get('data')

                    # Different styling based on log type
                    if log_type == 'span_start':
                        st.markdown(f"**🟢 {timestamp}** {message}")
                        if data:
                            with st.expander(f"📋 Attributes", expanded=False):
                                st.json(data)

                    elif log_type == 'span_end':
                        st.markdown(f"**🔴 {timestamp}** {message}")
                        if data:
                            with st.expander(f"📊 Results", expanded=False):
                                st.json(data)

                    elif log_type == 'span_event':
                        st.text(f"  {timestamp} {message}")
                        if data:
                            # Special handling for retrieval_scores event
                            if message and "retrieval_scores_received" in message:
                                version = data.get('version', 'unknown')
                                method = data.get('method', 'unknown')
                                scores = data.get('scores', [])
                                chunks_count = data.get('chunks_count', 0)

                                with st.expander(f"📊 Retrieval Scores ({version} - {chunks_count} chunks)", expanded=True):
                                    st.caption(f"**Method**: {method}")

                                    if version == 'v2':
                                        # Display v2 scores with detailed metrics
                                        for i, score_data in enumerate(scores, 1):
                                            query = score_data.get('query', 'N/A')
                                            title = score_data.get('title', 'Unknown')
                                            emb_score = score_data.get('embedding_score', 0)
                                            bm25_score = score_data.get('bm25_score', 0)
                                            combined = score_data.get('combined_score', 0)

                                            st.markdown(f"**{i}. {title}**")
                                            st.caption(f"Query: `{query}`")
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Embedding", f"{emb_score:.4f}")
                                            with col2:
                                                st.metric("BM25", f"{bm25_score:.4f}")
                                            with col3:
                                                st.metric("Combined", f"{combined:.4f}")
                                            st.markdown("---")
                                    elif version == 'v1':
                                        # Display v1 scores with rank
                                        for score_data in scores:
                                            rank = score_data.get('rank', 'N/A')
                                            title = score_data.get('title', 'Unknown')
                                            method_used = score_data.get('method', 'N/A')

                                            st.markdown(f"**Rank {rank}**: {title}")
                                            st.caption(f"Method: {method_used}")
                                            st.markdown("---")
                            else:
                                # Default JSON display for other events
                                with st.expander(f"🔍 Event Data", expanded=False):
                                    st.json(data)

                    elif log_type == 'error':
                        st.error(f"**❌ {timestamp}** {message}")
                        if data:
                            with st.expander(f"⚠️ Error Details", expanded=True):
                                st.json(data)

                    else:
                        st.text(f"{timestamp} {message}")
                        if data:
                            with st.expander(f"Data", expanded=False):
                                st.json(data)

            else:
                st.caption("No trace logs yet. Send a message to see trace information.")

# Chat input - needs to be in the main column
with chat_col:
    if prompt := st.chat_input("Enter your question..."):
        # Clear trace logs for new message
        st.session_state.trace_logs = []
        st.session_state.trace_spans = {}

        # Start main request span
        request_span = start_span("user_query", {
            "query": prompt,
            "query_length": len(prompt),
            "user_id": st.session_state.user_id,
            "use_web": st.session_state.use_web
        })

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get or create session
        session_span = start_span("session_management")
        session_id = st.session_state.current_session_id
        if not session_id:
            session_id = create_session(st.session_state.user_id, prompt[:50])
            if session_id:
                st.session_state.current_session_id = session_id
                st.session_state.sessions = get_sessions(st.session_state.user_id)
                end_span(session_span, "OK", {"action": "created", "session_id": session_id})
        else:
            end_span(session_span, "OK", {"action": "reused", "session_id": session_id})

        # Display assistant response placeholder
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            search_steps_placeholder = st.empty()

            # Reset search steps
            st.session_state.current_search_steps = []

            # Send request span
            api_span = start_span("api_request", {
                "endpoint": f"{API_BASE_URL}/search",
                "method": "POST",
                "session_id": session_id,
                "use_web": st.session_state.use_web
            })

            # Send message and get streaming response
            response = send_message(
                prompt,
                session_id,
                st.session_state.user_id,
                st.session_state.use_web
            )

            if response:
                end_span(api_span, "OK", {
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                })

                st.session_state.is_loading = True
                try:
                    # Process streaming response
                    full_content, references = process_streaming_response(
                        response,
                        message_placeholder,
                        search_steps_placeholder
                    )

                    if full_content is not None:
                        # Add assistant message to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_content,
                            "references": references or []
                        })

                        # Display references
                        if references:
                            with st.expander(f"📚 References ({len(references)})", expanded=False):
                                for ref in references:
                                    if isinstance(ref, dict):
                                        for title, url in ref.items():
                                            st.markdown(f"- [{title}]({url})")

                        end_span(request_span, "OK", {
                            "response_length": len(full_content),
                            "references_count": len(references) if references else 0
                        })
                    else:
                        end_span(request_span, "ERROR", {"error": "No content received"})
                        st.error("Failed to get response from backend")
                finally:
                    st.session_state.is_loading = False
                    # Clear search steps placeholder after response
                    search_steps_placeholder.empty()
            else:
                end_span(api_span, "ERROR", {"error": "Request failed"})
                end_span(request_span, "ERROR", {"error": "Failed to send request"})
                st.error("Failed to send message to backend. Check backend connection.")

# Auto-refresh sessions periodically
if st.session_state.user_id and st.session_state.current_session_id:
    time.sleep(0.1)  # Small delay to prevent excessive API calls

