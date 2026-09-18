from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from .chatbot import SupportChatbot

@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.bot=SupportChatbot(); yield
app=FastAPI(title='E-commerce RAG Support Chatbot',version='2.0.0',lifespan=lifespan)
class ChatRequest(BaseModel):
    message:str=Field(min_length=1,max_length=2000)
    top_k:int=Field(default=3,ge=1,le=8)
@app.get('/health')
def health(): return {'status':'ok'}
@app.post('/chat')
def chat(req:ChatRequest):
    try: return app.state.bot.answer(req.message,req.top_k)
    except Exception as exc: raise HTTPException(status_code=500,detail=str(exc))
PAGE="""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>RAG Support</title><style>body{font-family:Arial;background:#f4f6fb}.box{max-width:800px;margin:35px auto;background:#fff;padding:24px;border-radius:14px;box-shadow:0 3px 20px #ccd}#chat{min-height:340px;max-height:55vh;overflow:auto}.m{padding:11px;margin:10px;border-radius:10px;white-space:pre-wrap}.u{background:#e5efff;margin-left:18%}.b{background:#edf8ee;margin-right:18%}form{display:flex;gap:8px}input{flex:1;padding:12px}button{padding:12px;background:#2456d6;color:#fff;border:0;border-radius:8px}</style></head><body><div class='box'><h1>E-commerce RAG Support</h1><div id='chat'></div><form id='f'><input id='q' placeholder='Ask about an order, refund, or account' required><button>Send</button></form></div><script>const c=document.getElementById('chat'),f=document.getElementById('f'),q=document.getElementById('q');function add(t,x){let d=document.createElement('div');d.className='m '+x;d.textContent=t;c.appendChild(d)}f.onsubmit=async e=>{e.preventDefault();let x=q.value.trim();if(!x)return;add(x,'u');q.value='';add('Thinking...','b');let w=c.lastChild;try{let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:x})});let j=await r.json();w.textContent=j.answer+'\\n\\n['+j.language+' | '+j.sentiment+' | '+j.intent+' | priority='+j.priority+']'}catch(e){w.textContent='Error: '+e}}</script></body></html>"""
@app.get('/',response_class=HTMLResponse)
def home(): return PAGE
