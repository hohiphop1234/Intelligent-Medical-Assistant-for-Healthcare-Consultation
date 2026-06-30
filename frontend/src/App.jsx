import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize';
import { Send, Activity, Stethoscope, AlertTriangle } from 'lucide-react';
import './index.css';

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'bot',
      content: 'Chào bạn, mình là Trợ lý Y tế AI. Bạn cần tư vấn thông tin gì về sức khỏe hay thuốc men hôm nay?',
      metadata: null
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  useEffect(() => {
    // Fetch stats on load
    axios.get('http://localhost:8000/api/stats')
      .then(res => setStats(res.data.stats))
      .catch(err => console.error("Could not fetch stats:", err));
  }, []);

  const handleInput = (e) => {
    setInput(e.target.value);
    // Auto resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = '24px';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg = input.trim();
    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = '24px';

    setMessages(prev => [...prev, { id: Date.now(), role: 'user', content: userMsg }]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: userMsg, is_emergency: isEmergency })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      
      let isFirstMetadata = true;
      let currentContent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'metadata') {
                if (data.data.type === 'emergency' || data.data.type === 'out_of_scope' || data.data.type === 'insufficient_evidence') {
                  setMessages(prev => [...prev, {
                    id: Date.now() + 1,
                    role: 'bot',
                    content: data.data.message,
                    metadata: { type: data.data.type }
                  }]);
                  return; // Stop processing
                }
                
                if (isFirstMetadata) {
                  setMessages(prev => [...prev, {
                    id: Date.now() + 1,
                    role: 'bot',
                    content: '',
                    metadata: {
                      type: data.data.type,
                      category: data.data.category,
                      risk: data.data.risk_level,
                      route: data.data.route,
                      sources: data.data.sources,
                      disclaimer: data.data.disclaimer
                    }
                  }]);
                  isFirstMetadata = false;
                }
              } else if (data.type === 'token') {
                currentContent += data.content;
                setMessages(prev => {
                  const newMessages = [...prev];
                  const lastMsg = newMessages[newMessages.length - 1];
                  lastMsg.content = currentContent;
                  return newMessages;
                });
              } else if (data.type === 'error') {
                throw new Error(data.content);
              }
            } catch (e) {
              console.error("Lỗi parse SSE:", e, line);
            }
          }
        }
      }
      
      // Chèn disclaimer vào cuối nếu có
      setMessages(prev => {
        const newMessages = [...prev];
        const lastMsgIndex = newMessages.length - 1;
        const lastMsg = newMessages[lastMsgIndex];
        
        if (lastMsg.metadata && lastMsg.metadata.disclaimer) {
          // Tránh lỗi nối chuỗi 2 lần do React StrictMode bằng cách kiểm tra trước khi nối
          if (!lastMsg.content.includes(lastMsg.metadata.disclaimer)) {
            newMessages[lastMsgIndex] = {
              ...lastMsg,
              content: lastMsg.content + "\n\n---\n" + lastMsg.metadata.disclaimer
            };
          }
        }
        return newMessages;
      });

    } catch (error) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'bot',
        content: `**Lỗi kết nối:** Không thể gọi tới server. Vui lòng đảm bảo FastAPI backend đang chạy.\n\nChi tiết: ${error.message}`,
        metadata: { type: 'error' }
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const [isEmergency, setIsEmergency] = useState(false);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const formatMarkdown = (text) => {
    if (!text) return '';
    // Normalize newlines to ensure ReactMarkdown renders them as breaks
    let formatted = text
      .replace(/\r?\n/g, '\n')
      .replace(/\n/g, '\n\n')
      .replace(/\n{3,}/g, '\n\n');
      
    // Format citation references [1], [2] to beautiful badges
    formatted = formatted.replace(/\[(\d+)\]/g, '<span class="citation-badge">[$1]</span>');
    return formatted;
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1><Activity size={28} /> Health AI Assistant</h1>
        <div className="stats">
          <button 
            className={`sos-btn ${isEmergency ? 'active' : ''}`}
            onClick={() => setIsEmergency(!isEmergency)}
            title="Bật/Tắt chế độ cấp cứu"
          >
            🚨 SOS
          </button>
          {stats ? (
            <span>📚 {stats.vi_count} tài liệu | RAG + GGUF Engine</span>
          ) : (
            <span>Đang kết nối...</span>
          )}
        </div>
      </header>

      <main className="chat-window">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.role}`}>
            <div className="msg-content">
              {msg.role === 'bot' ? (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[
                    rehypeRaw,
                    [rehypeSanitize, {
                      ...defaultSchema,
                      attributes: {
                        ...defaultSchema.attributes,
                        span: [...(defaultSchema.attributes.span || []), 'className', 'class']
                      }
                    }]
                  ]}
                >
                  {(() => {
                    let content = formatMarkdown(msg.content);
                    const openCount = (content.match(/<think>/g) || []).length;
                    const closeCount = (content.match(/<\/think>/g) || []).length;
                    if (openCount > closeCount) {
                      content += '\n</think>';
                    }
                    return content
                      .replace(/<think>/g, '> **[AI đang suy luận...]**\n\n```text\n')
                      .replace(/<\/think>/g, '\n```\n\n**[Câu trả lời]**\n\n');
                  })()}
                </ReactMarkdown>
              ) : (
                msg.content
              )}
            </div>
            
            {msg.role === 'bot' && msg.metadata && msg.metadata.type !== 'error' && msg.metadata.route && (
              <div className="metadata">
                {msg.metadata.risk === 'critical' || msg.metadata.risk === 'high' ? (
                  <span style={{color: '#ef4444', display: 'flex', alignItems: 'center', gap: '4px'}}>
                    <AlertTriangle size={12} /> Nguy cơ: {msg.metadata.risk.toUpperCase()}
                  </span>
                ) : (
                  <span>Danh mục: {msg.metadata.category}</span>
                )}
                <span>•</span>
                <span>Luồng: {msg.metadata.route === 'general_qa' ? '🤖 Local LLM' : '🔍 RAG Pipeline'}</span>
              </div>
            )}
            
            {msg.role === 'bot' && msg.metadata && msg.metadata.sources && msg.metadata.sources.length > 0 && (
              <div className="sources-container">
                <details className="sources-details">
                  <summary className="sources-summary">
                    🔍 Xem {msg.metadata.sources.length} tài liệu đối chiếu y khoa
                  </summary>
                  <div className="sources-list">
                    {msg.metadata.sources.map((src) => (
                      <div key={src.index} className="source-card">
                        <div className="source-title">
                          <strong>[{src.index}] {src.title}</strong>
                        </div>
                        <p className="source-content">{src.content}</p>
                      </div>
                    ))}
                  </div>
                </details>
              </div>
            )}
          </div>
        ))}
        
        {isLoading && (
          <div className="typing-indicator">
            <div className="dot"></div>
            <div className="dot"></div>
            <div className="dot"></div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      <footer className="input-area">
        <div className="input-wrapper">
          <Stethoscope size={24} color="var(--text-muted)" />
          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Hỏi về triệu chứng, thuốc, hoặc lời khuyên y tế..."
            rows={1}
            disabled={isLoading}
          />
          <button onClick={handleSend} disabled={!input.trim() || isLoading}>
            <Send size={20} />
          </button>
        </div>
      </footer>
    </div>
  );
}

export default App;
