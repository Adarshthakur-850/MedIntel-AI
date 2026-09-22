import React, { useState } from 'react';
import { BookOpen, Send, ShieldAlert, ExternalLink, Bot, User } from 'lucide-react';

interface Citation {
  doc_id: string;
  doc_title: string;
  category: string;
  snippet: string;
  relevance_score: number;
}

interface Message {
  sender: 'user' | 'bot';
  text: string;
  citations?: Citation[];
}

export const KnowledgeAssistant: React.FC = () => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: 'bot',
      text: "Hello! I am the MedIntel AI Knowledge Assistant. I answer clinical guideline questions based strictly on curated sources (AHA, CDC, ADA, KDIGO). How can I assist your medical research today?"
    }
  ]);

  const handleSend = async () => {
    if (!query.trim()) return;
    const userMsg: Message = { sender: 'user', text: query };
    setMessages((prev) => [...prev, userMsg]);
    setQuery('');
    setLoading(true);

    try {
      const resp = await fetch("http://localhost:8000/api/v1/rag/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMsg.text, top_k: 3 })
      });
      const data = await resp.json();

      const botMsg: Message = {
        sender: 'bot',
        text: data.answer,
        citations: data.citations
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          sender: 'bot',
          text: "Based on AHA/ACC Guidelines on Cardiovascular Risk Assessment: Systolic blood pressure above 130 mmHg or diastolic blood pressure above 80 mmHg indicates elevated blood pressure or stage 1 hypertension. (Simulated RAG Response)",
          citations: [
            {
              doc_id: "DOC_001",
              doc_title: "AHA/ACC Guidelines on Cardiovascular Risk Assessment",
              category: "Cardiology",
              snippet: "Systolic blood pressure above 130 mmHg indicates stage 1 hypertension.",
              relevance_score: 0.92
            }
          ]
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl shadow-xl flex flex-col h-[600px]">
      {/* Header */}
      <div className="p-4 border-b border-slate-700 flex items-center justify-between bg-slate-800/80">
        <div className="flex items-center space-x-3">
          <BookOpen className="w-5 h-5 text-sky-400" />
          <h3 className="text-base font-semibold text-white">Medical Knowledge Assistant (RAG)</h3>
        </div>
        <span className="text-xs bg-slate-700 px-2.5 py-1 rounded-full text-slate-300">Cited Evidence Only</span>
      </div>

      {/* Safety Alert Banner */}
      <div className="bg-amber-500/10 border-b border-amber-500/20 px-4 py-2 flex items-center space-x-2 text-amber-300 text-xs">
        <ShieldAlert className="w-4 h-4 flex-shrink-0" />
        <p>Educational decision-support assistant. Does not prescribe medication or provide medical diagnosis.</p>
      </div>

      {/* Message Chat Container */}
      <div className="flex-1 p-4 overflow-y-auto space-y-4 bg-slate-900/50">
        {messages.map((m, idx) => (
          <div key={idx} className={`flex items-start space-x-3 ${m.sender === 'user' ? 'justify-end' : ''}`}>
            {m.sender === 'bot' && (
              <div className="w-8 h-8 rounded-full bg-sky-500/20 border border-sky-500/40 flex items-center justify-center text-sky-400 flex-shrink-0">
                <Bot className="w-4 h-4" />
              </div>
            )}
            <div className={`max-w-2xl rounded-xl p-4 text-sm ${m.sender === 'user' ? 'bg-sky-600 text-white' : 'bg-slate-800 border border-slate-700 text-slate-200'}`}>
              <p className="whitespace-pre-wrap leading-relaxed">{m.text}</p>
              {m.citations && m.citations.length > 0 && (
                <div className="mt-3 pt-3 border-t border-slate-700 space-y-2">
                  <span className="text-xs font-semibold text-sky-400 uppercase tracking-wider block">Retrieved Sources & Citations:</span>
                  {m.citations.map((c, cIdx) => (
                    <div key={cIdx} className="bg-slate-900/80 p-2.5 rounded border border-slate-700 text-xs">
                      <div className="flex items-center justify-between text-slate-300 font-medium">
                        <span>{c.doc_title}</span>
                        <span className="text-slate-400 font-mono text-[10px]">Score: {c.relevance_score}</span>
                      </div>
                      <p className="text-slate-400 mt-1 text-[11px] italic">"{c.snippet}"</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
            {m.sender === 'user' && (
              <div className="w-8 h-8 rounded-full bg-indigo-500/20 border border-indigo-500/40 flex items-center justify-center text-indigo-400 flex-shrink-0">
                <User className="w-4 h-4" />
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="flex items-center space-x-2 text-slate-400 text-xs italic">
            <Bot className="w-4 h-4 animate-spin text-sky-400" />
            <span>Retrieving medical guidelines and synthesizing answer...</span>
          </div>
        )}
      </div>

      {/* Input Form */}
      <div className="p-4 border-t border-slate-700 bg-slate-800">
        <div className="flex items-center space-x-2">
          <input
            type="text"
            placeholder="Ask a clinical question (e.g., 'What are the diagnostic thresholds for hypertension?')..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-white focus:outline-none focus:border-sky-500"
          />
          <button
            onClick={handleSend}
            disabled={loading || !query.trim()}
            className="px-4 py-2.5 bg-sky-500 hover:bg-sky-400 text-white rounded-lg transition-colors flex items-center space-x-1.5 disabled:opacity-50"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
