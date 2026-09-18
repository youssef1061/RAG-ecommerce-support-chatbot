from pathlib import Path
import os, re
import joblib, faiss, pandas as pd, torch
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
load_dotenv()

class SupportChatbot:
    def __init__(self, artifacts_dir=None):
        self.root=Path(artifacts_dir or os.getenv('ARTIFACTS_DIR','artifacts'))
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.language=joblib.load(self.root/'language'/'language_pipeline.joblib')
        self.intent=joblib.load(self.root/'intent'/'intent_pipeline.joblib')
        self.tok=AutoTokenizer.from_pretrained(self.root/'sentiment')
        self.tone=AutoModelForSequenceClassification.from_pretrained(self.root/'sentiment').to(self.device).eval()
        self.kb=pd.read_parquet(self.root/'rag'/'knowledge_base.parquet')
        self.index=faiss.read_index(str(self.root/'rag'/'faiss.index'))
        self.embedder=SentenceTransformer(os.getenv('EMBED_MODEL','sentence-transformers/all-MiniLM-L6-v2'),device=str(self.device))
        key=os.getenv('GROQ_API_KEY')
        if not key: raise RuntimeError('GROQ_API_KEY is missing. Copy .env.example to .env and add it.')
        self.client=Groq(api_key=key)
        self.model=os.getenv('GROQ_MODEL','openai/gpt-oss-20b')
        self.threshold=float(os.getenv('RAG_THRESHOLD','0.22'))

    def llm(self,messages,temperature=0.1,max_tokens=450):
        out=self.client.chat.completions.create(model=self.model,messages=messages,temperature=temperature,max_tokens=max_tokens)
        return out.choices[0].message.content.strip()

    @torch.no_grad()
    def sentiment(self,text):
        batch=self.tok(text,return_tensors='pt',truncation=True,max_length=128).to(self.device)
        probs=torch.softmax(self.tone(**batch).logits,1)[0].cpu()
        idx=int(probs.argmax()); label=self.tone.config.id2label.get(idx,self.tone.config.id2label.get(str(idx),str(idx)))
        return str(label).lower(),float(probs[idx])

    def retrieve(self,query,k=3):
        vector=self.embedder.encode([query],normalize_embeddings=True,convert_to_numpy=True).astype('float32')
        scores,ids=self.index.search(vector,k); rows=[]
        for score,idx in zip(scores[0],ids[0]):
            row=self.kb.iloc[int(idx)].to_dict(); row['score']=float(score); rows.append(row)
        return rows

    def translate(self,text,language):
        if language=='en': return text
        return self.llm([{'role':'system','content':'Translate into concise English. Preserve IDs, numbers, names, and meaning. Return only the translation.'},{'role':'user','content':text}],0.0,200)

    def direct(self,text,language,intent):
        system=f'Reply concisely as an e-commerce support assistant in language code {language}. Route={intent}. If out of scope, say you only handle store support.'
        return self.llm([{'role':'system','content':system},{'role':'user','content':text}],0.2,180)

    def answer(self,message,top_k=3):
        message=re.sub(r'\s+',' ',str(message)).strip()
        if not message: raise ValueError('Message cannot be empty.')
        language=str(self.language.predict([message])[0]); english=self.translate(message,language)
        sentiment,confidence=self.sentiment(english); intent=str(self.intent.predict([english])[0])
        priority=sentiment=='negative' or intent=='complaint'
        if intent in {'greeting_goodbye_gratitude','out_of_scope'}:
            return {'answer':self.direct(message,language,intent),'language':language,'sentiment':sentiment,'sentiment_confidence':confidence,'intent':intent,'priority':priority,'escalate':False,'sources':[]}
        candidates=self.retrieve(english,top_k); relevant=[x for x in candidates if x['score']>=self.threshold]
        if not relevant:
            return {'answer':self.direct('Reliable support information is unavailable. Offer a human agent.',language,'out_of_scope'),'language':language,'sentiment':sentiment,'sentiment_confidence':confidence,'intent':intent,'priority':priority,'escalate':True,'sources':[]}
        context='\n\n'.join(f"Source {i+1}: {x['response']}" for i,x in enumerate(relevant))
        system=(f'Reply as a professional support assistant in language code {language}. Use ONLY retrieved context. '
                'Never invent order/refund status, dates, links, policy, or actions. If insufficient, offer a human agent. '
                f'Sentiment={sentiment}; route={intent}; priority={priority}. Use brief empathy if priority; offer escalation for complaints.')
        answer=self.llm([{'role':'system','content':system},{'role':'user','content':f'Context:\n{context}\n\nCustomer:\n{message}'}])
        sources=[{'instruction':x['instruction'],'intent':x['intent'],'category':x['category'],'score':round(x['score'],4)} for x in relevant]
        return {'answer':answer,'language':language,'sentiment':sentiment,'sentiment_confidence':confidence,'intent':intent,'priority':priority,'escalate':intent=='complaint','sources':sources}
