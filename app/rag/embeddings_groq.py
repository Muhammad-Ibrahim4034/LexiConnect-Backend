"""
Enhanced Law RAG System - Version 3.0
======================================
Fixes applied:
  1. Contextual query rewriting  → retrieval uses full conversation context
  2. LLM-based classification    → replaces brittle keyword routing
  3. Cross-encoder reranking     → replaces score-boosting heuristics
  4. Structured conversation memory → stores facts, not raw text
  5. Simplified prompts          → fewer rules, higher generation quality
  6. RAGAS-compatible evaluation → faithfulness, relevancy, precision, recall
  7. Secrets via env vars        → no hardcoded keys

INSTALL (run once):
  pip install sentence-transformers chromadb groq ragas datasets langchain openai
"""

import os
import pickle
import json
import csv
import re
import time
import warnings
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb import PersistentClient
from groq import Groq

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# ENUMS & DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

class ResponseMode(Enum):
    CRISIS     = "crisis"
    ACCIDENT   = "accident"
    LITIGATION = "litigation"
    ACADEMIC   = "academic"
    CHAT       = "chat"


class UrgencyLevel(Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"


@dataclass
class QueryAnalysis:
    mode:                    ResponseMode
    urgency:                 UrgencyLevel
    intent:                  str
    key_entities:            List[str]
    emotional_tone:          str
    requires_immediate_action: bool
    specific_laws_mentioned: List[str]
    jurisdiction_hints:      List[str]


@dataclass
class StructuredMemory:
    """
    Structured memory — stores extracted facts, NOT raw text.
    This is what gets injected into prompts, not the full conversation history.
    """
    situation:     Optional[str]        = None   # e.g. "domestic_violence"
    actors:        List[str]            = field(default_factory=list)   # e.g. ["husband","wife","child"]
    location:      Optional[str]        = None   # e.g. "Karachi"
    issues:        List[str]            = field(default_factory=list)   # e.g. ["custody","physical abuse"]
    established_rights: List[str]       = field(default_factory=list)   # facts already confirmed
    laws_cited:    List[str]            = field(default_factory=list)   # laws already mentioned

    def to_prompt_string(self) -> str:
        if not self.situation:
            return ""
        parts = [f"ESTABLISHED CONTEXT (from earlier in this conversation):"]
        if self.situation:
            parts.append(f"  Situation: {self.situation}")
        if self.actors:
            parts.append(f"  People involved: {', '.join(self.actors)}")
        if self.location:
            parts.append(f"  Location: {self.location}")
        if self.issues:
            parts.append(f"  Issues raised: {', '.join(self.issues)}")
        if self.established_rights:
            parts.append(f"  Rights already confirmed: {'; '.join(self.established_rights)}")
        if self.laws_cited:
            parts.append(f"  Laws already mentioned: {', '.join(self.laws_cited)}")
        return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedLawRAGSystem:
    """
    Production-grade conversational RAG system for Pakistani law.

    Key differences from v2:
      • Queries are rewritten with full context before retrieval
      • Classification is done by the LLM (not keywords)
      • Cross-encoder reranks semantic results
      • Memory is structured (facts), not raw chat history
      • Prompts are short and focused
    """

    # ── INIT ──────────────────────────────────────────────────────────────────

    def __init__(
        self,
        chunks_file:    str = "semantic_chunk_new.json",
        metadata_file:  str = "pdf_metadata.csv",
        persistent_dir: str = "chroma_store",
        jurisdiction:   str = "Pakistan",
        use_reranker:   bool = True,
    ):
        print("🚀 Initializing Enhanced Law RAG System v3.0 …")
        print("=" * 70)

        self.jurisdiction   = jurisdiction
        self.use_reranker   = use_reranker
        self._conversation_turns: List[Tuple[str, str]] = []  # (query, response)
        self.memory         = StructuredMemory()

        # Secrets from environment — NEVER hardcode
        self._groq_key = os.getenv("GROQ_API_KEY", "")
        self._hf_token = os.getenv("HF_TOKEN", "")
        if self._hf_token:
            os.environ["HF_TOKEN"] = self._hf_token

        print("📋 Loading metadata …")
        self.metadata = self._load_metadata(metadata_file)
        print(f"   ✓ {len(self.metadata)} documents")

        print("📚 Loading chunks …")
        self.chunks = self._load_chunks(chunks_file)
        self.chunk_texts, self.chunk_metadata = self._extract_chunk_data(self.chunks)
        print(f"   ✓ {len(self.chunk_texts)} chunks")

        print("🔤 Loading embedding model …")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        if use_reranker:
            print("⚡ Loading cross-encoder reranker …")
            # This model is ~80 MB and runs on CPU — no GPU needed
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("   ✓ Reranker ready")
        else:
            self.reranker = None

        print("💾 Connecting to ChromaDB …")
        self.persistent_dir = persistent_dir
        self.client     = PersistentClient(path=persistent_dir)
        self.collection = self.client.get_or_create_collection(
            name="law_documents",
            metadata={"hnsw:space": "cosine"},
        )
        print(f"   ✓ ChromaDB at {persistent_dir}")

        self.llm        = None
        self.model_name = None
        self.embeddings = None

        self._initialize_statutory_priorities()

        print("=" * 70)
        print("✅ System ready\n")

    # ── DATA LOADING ──────────────────────────────────────────────────────────

    def _load_chunks(self, chunks_file: str) -> List:
        paths = [
            chunks_file,
            f"checkpoints/{chunks_file}",
            f"/mnt/user-data/uploads/{chunks_file}",
        ]
        for p in paths:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
        print("   ⚠ No chunks found")
        return []

    def _load_metadata(self, metadata_file: str) -> Dict:
        # Build search paths covering all common project layouts
        basename = os.path.basename(metadata_file)
        paths = [
            metadata_file,            # exactly as given
            basename,                 # bare filename, cwd
            f"checkpoints/{basename}",
            f"data/{basename}",
            f"data/metadata/{basename}",
            f"scripts/{basename}",
            f"/mnt/user-data/uploads/{basename}",
        ]
        # Also resolve relative to the script file itself
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            paths += [
                os.path.join(script_dir, basename),
                os.path.join(script_dir, "..", "data", basename),
                os.path.join(script_dir, "..", "data", "metadata", basename),
                os.path.join(script_dir, "..", basename),
            ]
        except NameError:
            pass  # __file__ undefined in some interactive contexts

        tried = []
        for p in paths:
            norm = os.path.normpath(p)
            if norm in tried:
                continue
            tried.append(norm)
            if os.path.exists(norm):
                try:
                    meta = {}
                    with open(norm, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            doc_id = row.get("doc_id", row.get("filename", "")).strip()
                            if not doc_id:
                                continue
                            meta[doc_id] = {
                                "filename":    row.get("filename",    "").strip(),
                                "law_type":    row.get("law_type",    "").strip(),
                                "source":      row.get("source",      "").strip(),
                                "description": row.get("description", "").strip(),
                                "notes":       row.get("notes",       "").strip(),
                            }
                    print(f"   \u2713 Metadata loaded from: {norm} ({len(meta)} rows)")
                    return meta
                except Exception as e:
                    print(f"   \u2717 Found but failed to read {norm}: {e}")

        # Nothing found — tell the user exactly what was tried
        print("   \u2717 Metadata file NOT found. Paths tried:")
        for p in tried:
            print(f"       {p}")
        print("   \u2192 Fix: pass the absolute path as metadata_file= to load_system()")
        return {}

    def _extract_chunk_data(self, chunks) -> Tuple[List[str], List[Dict]]:
        if not chunks:
            return [], []
        texts, metas = [], []
        if isinstance(chunks, dict):
            chunks = list(chunks.values())
        for chunk in chunks:
            text = chunk.get("page_content", str(chunk)) if isinstance(chunk, dict) else str(chunk)
            texts.append(text)
            metas.append(self._get_chunk_metadata(chunk) if isinstance(chunk, dict) else
                         {"source_name": "Unknown", "doc_id": "unknown", "law_type": "unknown"})
        return texts, metas

    def _get_chunk_metadata(self, chunk: Dict) -> Dict:
        cm       = chunk.get("metadata", {})
        raw_id   = cm.get("doc_id", cm.get("filename", "unknown"))
        # Normalise: strip .pdf suffix so "dv_001.pdf" matches key "dv_001"
        doc_id   = raw_id.replace(".pdf", "").strip() if raw_id else "unknown"

        # 1. Direct key hit
        if doc_id in self.metadata:
            m = self.metadata[doc_id]
            return {"source_name": m["source"], "doc_id": doc_id,
                    "filename": m["filename"], "law_type": m["law_type"],
                    "description": m["description"]}

        # 2. Try with .pdf suffix restored
        pdf_id = doc_id + ".pdf"
        if pdf_id in self.metadata:
            m = self.metadata[pdf_id]
            return {"source_name": m["source"], "doc_id": pdf_id,
                    "filename": m["filename"], "law_type": m["law_type"],
                    "description": m["description"]}

        # 3. Match against the filename column (handles path/to/dv_001.pdf)
        filename = cm.get("source", cm.get("filename", ""))
        fname_base = os.path.basename(filename).replace(".pdf", "")
        for mid, m in self.metadata.items():
            if m["filename"].replace(".pdf", "") == fname_base:
                return {"source_name": m["source"], "doc_id": mid,
                        "filename": m["filename"], "law_type": m["law_type"],
                        "description": m["description"]}
        return {"source_name": filename or "Unknown", "doc_id": doc_id,
                "filename": filename, "law_type": "unknown", "description": ""}

    # ── EMBEDDINGS & DB ───────────────────────────────────────────────────────

    def create_embeddings(self):
        if not self.chunk_texts:
            print("⚠ No chunks to embed")
            return None
        print(f"🔄 Embedding {len(self.chunk_texts)} chunks …")
        batch_size, all_embs = 32, []
        for i in range(0, len(self.chunk_texts), batch_size):
            batch = self.chunk_texts[i : i + batch_size]
            embs  = self.embedding_model.encode(
                batch, show_progress_bar=True,
                convert_to_numpy=True, normalize_embeddings=True,
            )
            all_embs.extend(embs)
        self.embeddings = np.array(all_embs)
        print(f"✅ Embeddings shape: {self.embeddings.shape}")
        return self.embeddings

    def populate_vector_db(self):
        if self.embeddings is None:
            print("❌ Run create_embeddings() first")
            return
        if self.collection.count() > 0:
            resp = input("   Collection not empty. Clear and repopulate? (yes/no): ")
            if resp.lower() == "yes":
                self.client.delete_collection("law_documents")
                self.collection = self.client.get_or_create_collection(
                    "law_documents", metadata={"hnsw:space": "cosine"})
            else:
                return
        ids  = [f"chunk_{i}" for i in range(len(self.chunk_texts))]
        embs = self.embeddings.tolist()
        for i in range(0, len(self.chunk_texts), 100):
            end = min(i + 100, len(self.chunk_texts))
            mbs = [{**self.chunk_metadata[j], "chunk_id": j} for j in range(i, end)]
            self.collection.add(
                embeddings=embs[i:end],
                documents=self.chunk_texts[i:end],
                ids=ids[i:end], metadatas=mbs,
            )
        print(f"✅ Added {len(self.chunk_texts)} documents")

    # ── LLM ───────────────────────────────────────────────────────────────────

    def load_llm(self, model_name: str = "llama-3.3-70b-versatile"):
        key = self._groq_key or os.getenv("GROQ_API_KEY", "")
        if not key:
            raise ValueError("Set GROQ_API_KEY environment variable")
        self.llm        = Groq(api_key=key)
        self.model_name = model_name
        print(f"✅ LLM ready: {model_name}")

    def _call_llm(self, messages: List[Dict],
                  max_tokens: int = 1024, temperature: float = 0.3) -> str:
        try:
            resp = self.llm.chat.completions.create(
                model=self.model_name, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"   ❌ LLM error: {e}")
            return ""

    # ── FIX 1: CONTEXTUAL QUERY REWRITING ────────────────────────────────────
    #
    # This is the most important fix. Before ANY retrieval happens, the raw
    # user message is expanded into a fully self-contained query that includes
    # the conversation context. This means "Can he take my kids?" becomes
    # "In a domestic violence situation in Pakistan, can an abusive husband
    # obtain custody of children from the mother?" — which retrieves the
    # right documents every time.

    def rewrite_query_with_context(self, current_query: str) -> str:
        """
        Expand a follow-up question into a fully self-contained query.
        Returns the original query unchanged if there is no prior context.
        """
        if not self._conversation_turns:
            return current_query

        recent = "\n".join(
            f"User: {q}\nAssistant summary: {a[:200]}…"
            for q, a in self._conversation_turns[-3:]
        )

        prompt = f"""You are a query rewriting assistant for a legal RAG system.

Your task: rewrite the user's latest question into a single, fully self-contained legal query.

Rules:
- Include all relevant context from the conversation (people, situation, jurisdiction)
- Do NOT answer the question — only rewrite it
- Output ONE sentence, no preamble

Conversation so far:
{recent}

User's latest question:
{current_query}

Rewritten standalone query:"""

        messages = [
            {"role": "system", "content": "You rewrite follow-up questions into standalone legal queries."},
            {"role": "user",   "content": prompt},
        ]
        rewritten = self._call_llm(messages, max_tokens=120, temperature=0.0)
        return rewritten if rewritten else current_query

    # ── FIX 2: LLM-BASED CLASSIFICATION ──────────────────────────────────────
    #
    # Keyword matching fails on paraphrases. The LLM understands intent even
    # when the user does not use the "right" keywords.

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Use the LLM to classify the query. Falls back to keyword heuristics
        if the LLM is not loaded yet (e.g. during build phase).
        """
        if self.llm:
            return self._llm_classify(query)
        return self._keyword_classify(query)

    def _llm_classify(self, query: str) -> QueryAnalysis:
        prompt = f"""Classify this legal query. Return ONLY valid JSON — no explanation, no markdown.

Query: "{query}"

JSON schema (all fields required):
{{
  "mode": "crisis|accident|litigation|academic|chat",
  "urgency": "critical|high|medium|low",
  "intent": "one short phrase",
  "key_entities": ["list", "of", "entities"],
  "emotional_tone": "distressed|confused|angry|neutral|seeking_clarity",
  "requires_immediate_action": true or false,
  "specific_laws_mentioned": [],
  "jurisdiction_hints": []
}}

mode definitions:
- crisis: physical danger, domestic abuse, threats, immediate safety risk
- accident: traffic collision, motor vehicle incident
- litigation: court procedure, filing cases, hearings
- academic: general legal questions, rights, definitions
- chat: greetings, thanks, small talk"""

        messages = [
            {"role": "system", "content": "You are a query classification system. Output only JSON."},
            {"role": "user",   "content": prompt},
        ]
        raw = self._call_llm(messages, max_tokens=300, temperature=0.0)

        try:
            # Strip markdown fences if present
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data  = json.loads(clean)
            return QueryAnalysis(
                mode=ResponseMode(data.get("mode", "academic")),
                urgency=UrgencyLevel(data.get("urgency", "low")),
                intent=data.get("intent", "general_inquiry"),
                key_entities=data.get("key_entities", []),
                emotional_tone=data.get("emotional_tone", "neutral"),
                requires_immediate_action=data.get("requires_immediate_action", False),
                specific_laws_mentioned=data.get("specific_laws_mentioned", []),
                jurisdiction_hints=data.get("jurisdiction_hints", []),
            )
        except (json.JSONDecodeError, ValueError) as e:
            print(f"   ⚠ Classification parse error: {e} — falling back to keywords")
            return self._keyword_classify(query)

    def _keyword_classify(self, query: str) -> QueryAnalysis:
        """Keyword fallback (used only when LLM is unavailable)."""
        q = query.lower()
        crisis_words   = ["hit", "beat", "attack", "abuse", "rape", "kill", "threat",
                          "hurt", "blackmail", "violence", "assault"]
        accident_words = ["accident", "collision", "crash", "injured", "ambulance"]
        chat_words     = ["hi", "hello", "hey", "thanks", "bye", "how are you"]

        if any(w in q for w in crisis_words):
            mode, urgency = ResponseMode.CRISIS, UrgencyLevel.CRITICAL
        elif any(w in q for w in accident_words):
            mode, urgency = ResponseMode.ACCIDENT, UrgencyLevel.HIGH
        elif any(w in q for w in chat_words):
            mode, urgency = ResponseMode.CHAT, UrgencyLevel.LOW
        else:
            mode, urgency = ResponseMode.ACADEMIC, UrgencyLevel.MEDIUM

        return QueryAnalysis(
            mode=mode, urgency=urgency, intent="general_inquiry",
            key_entities=[], emotional_tone="neutral",
            requires_immediate_action=(mode == ResponseMode.CRISIS),
            specific_laws_mentioned=[], jurisdiction_hints=[],
        )

    # ── FIX 3: RETRIEVAL + CROSS-ENCODER RERANKING ───────────────────────────
    #
    # Step 1: semantic retrieval fetches a broad candidate set (top_k * 4)
    # Step 2: cross-encoder reranks with full query-document attention
    # Step 3: return top_k after reranking
    #
    # The cross-encoder reads both the query and document together, making it
    # far more accurate than embedding cosine distance alone.

    def hybrid_retrieve(
        self, query: str, analysis: QueryAnalysis, top_k: int = 3,
    ) -> Tuple[List[str], List[Dict], List[float], bool]:

        # 1. Semantic retrieval — fetch a wider pool
        q_emb = self.embedding_model.encode(query, convert_to_numpy=True)
        try:
            results  = self.collection.query(
                query_embeddings=[q_emb.tolist()],
                n_results=min(top_k * 4, self.collection.count()),
            )
            docs      = results["documents"][0]
            distances = results["distances"][0]
            metas     = results["metadatas"][0]
        except Exception as e:
            print(f"   ⚠ Retrieval error: {e}")
            return [], [], [], False

        if not docs:
            return [], [], [], False

        # 2. Apply statutory priority boost to semantic scores
        semantic_scores = [1 - d for d in distances]
        boosted_scores  = [
            self._apply_statutory_boost(s, m, analysis)
            for s, m in zip(semantic_scores, metas)
        ]

        # 3. Cross-encoder reranking
        if self.reranker and docs:
            pairs        = [(query, doc) for doc in docs]
            rerank_scores = self.reranker.predict(pairs).tolist()
            # Normalize rerank scores to [0, 1] range for combination
            min_r, max_r = min(rerank_scores), max(rerank_scores)
            span = max_r - min_r if max_r != min_r else 1.0
            norm_rerank = [(r - min_r) / span for r in rerank_scores]
            # Combine: 40% semantic+boost, 60% reranker
            final_scores = [
                0.4 * bs + 0.6 * rs
                for bs, rs in zip(boosted_scores, norm_rerank)
            ]
        else:
            final_scores = boosted_scores

        # 4. Sort and deduplicate
        combined = sorted(zip(docs, metas, final_scores), key=lambda x: x[2], reverse=True)
        seen, deduped = set(), []
        for doc, meta, score in combined:
            h = hash(doc[:150])
            if h not in seen:
                seen.add(h)
                deduped.append((doc, meta, score))
            if len(deduped) >= top_k:
                break

        out_docs   = [d[0] for d in deduped]
        out_metas  = [d[1] for d in deduped]
        out_scores = [d[2] for d in deduped]

        # Confidence: use reranker-based threshold if available, else cosine
        confident = (out_scores[0] >= 0.45) if out_scores else False

        return out_docs, out_metas, out_scores, confident

    def _apply_statutory_boost(self, base_score: float, meta: Dict,
                                analysis: QueryAnalysis) -> float:
        score = base_score
        source = meta.get("source_name", "").lower()
        cat    = None
        if analysis.mode == ResponseMode.CRISIS:
            cat = "domestic_violence"
        elif analysis.mode == ResponseMode.ACCIDENT:
            cat = "traffic"

        if cat and cat in self.statutory_priority:
            pd = self.statutory_priority[cat]
            for law in pd["primary"]:
                if law.lower() in source:
                    score += pd["boost_primary"]
                    break
            else:
                for law in pd["secondary"]:
                    if law.lower() in source:
                        score += pd["boost_secondary"]
                        break
        return min(score, 1.0)

    def _initialize_statutory_priorities(self):
        self.statutory_priority = {
            "domestic_violence": {
                "primary":   ["domestic violence (prevention and protection) act, 2013",
                              "protection of women (criminal laws amendment) act"],
                "secondary":  ["guardians and wards act", "muslim family laws ordinance",
                               "dissolution of muslim marriages act"],
                "boost_primary": 0.20, "boost_secondary": 0.10,
            },
            "traffic": {
                "primary":   ["provincial motor vehicles ordinance", "sindh act no. xiii"],
                "secondary":  ["traffic offences (special courts) ordinance", "motor vehicles act"],
                "boost_primary": 0.20, "boost_secondary": 0.10,
            },
            "family_law": {
                "primary":   ["muslim family laws ordinance", "guardians and wards act"],
                "secondary":  ["west pakistan family courts act", "child marriage restraint"],
                "boost_primary": 0.20, "boost_secondary": 0.10,
            },
        }

    # ── FIX 4: STRUCTURED MEMORY UPDATE ──────────────────────────────────────
    #
    # After each turn, extract structured facts from the conversation.
    # This compact representation is injected into prompts — not raw history.

    def update_memory(self, query: str, response: str, analysis: QueryAnalysis):
        """Extract structured facts from this turn and update memory."""
        if not self.llm:
            return

        prompt = f"""Extract structured facts from this legal conversation turn.
Return ONLY valid JSON.

User query: {query}
Assistant response: {response[:500]}

JSON schema:
{{
  "situation": "one of: domestic_violence|traffic_accident|family_law|criminal|general|null",
  "actors": ["list of people mentioned, e.g. husband, wife, child"],
  "location": "city or region if mentioned, else null",
  "issues": ["list of legal issues, e.g. custody, physical abuse, divorce"],
  "rights_confirmed": ["any legal rights explicitly confirmed in the response"],
  "laws_cited": ["law names mentioned in the response"]
}}"""

        messages = [
            {"role": "system", "content": "Extract structured facts. Output only JSON."},
            {"role": "user",   "content": prompt},
        ]
        raw = self._call_llm(messages, max_tokens=250, temperature=0.0)
        try:
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data  = json.loads(clean)
            if data.get("situation") and data["situation"] != "null":
                self.memory.situation = data["situation"]
            for actor in data.get("actors", []):
                if actor not in self.memory.actors:
                    self.memory.actors.append(actor)
            if data.get("location"):
                self.memory.location = data["location"]
            for issue in data.get("issues", []):
                if issue not in self.memory.issues:
                    self.memory.issues.append(issue)
            for right in data.get("rights_confirmed", []):
                if right not in self.memory.established_rights:
                    self.memory.established_rights.append(right)
            for law in data.get("laws_cited", []):
                if law not in self.memory.laws_cited:
                    self.memory.laws_cited.append(law)
        except Exception:
            pass  # Memory update failure is non-critical

    # ── FIX 5: SIMPLIFIED RESPONSE GENERATION ────────────────────────────────
    #
    # Fewer rules = better generation quality. Each prompt has ONE job.

    def _build_context_string(self, docs: List[str], metas: List[Dict]) -> str:
        parts = []
        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
            parts.append(f"[Reference {i} — {meta.get('source_name', 'Unknown')}]\n{doc.strip()}")
        return "LEGAL REFERENCES (use to verify facts only, do not summarize):\n\n" + "\n\n".join(parts)

    def generate_response(self, query: str, analysis: QueryAnalysis,
                          docs: List[str], metas: List[Dict]) -> str:
        if self.llm is None:
            return self._fallback_response(analysis)
        if analysis.mode == ResponseMode.CHAT:
            return self._generate_chat_response(query)
        if analysis.mode == ResponseMode.CRISIS:
            return self._generate_crisis_response(query, analysis, docs, metas)
        if analysis.mode == ResponseMode.ACCIDENT:
            return self._generate_accident_response(query, analysis, docs, metas)
        return self._generate_general_response(query, analysis, docs, metas)

    def _generate_chat_response(self, query: str) -> str:
        messages = [
            {"role": "system", "content":
             "You are a professional Pakistani legal assistant. Respond briefly and warmly to greetings."},
            {"role": "user", "content": query},
        ]
        return self._call_llm(messages, max_tokens=200, temperature=0.7)

    def _generate_crisis_response(self, query: str, analysis: QueryAnalysis,
                                  docs: List[str], metas: List[Dict]) -> str:
        resources = self.get_jurisdiction_resources()
        context   = self._build_context_string(docs, metas)
        memory    = self.memory.to_prompt_string()

        system = f"""You are a compassionate legal advisor in {self.jurisdiction} helping a domestic violence victim.

Your response must:
1. Acknowledge their situation briefly (one sentence)
2. If there is physical danger, list emergency contacts immediately
3. Answer their specific question with numbered steps they can take right now
4. Name the relevant law (one sentence)

"CRITICAL: Only cite law names and section numbers that appear verbatim in the provided references."
"If you cannot find a specific provision in the references, say 'the specific section is not in the available references' rather than citing from memory."

Emergency: {resources['emergency']} | Helpline: {resources['domestic_violence']} | Shelter: {resources['shelter']}

Keep the response under 300 words. Plain language only."""

        user = f"""{memory}

User's question: {query}

{context}"""

        return self._call_llm(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=700, temperature=0.3,
        )

    def _generate_accident_response(self, query: str, analysis: QueryAnalysis,
                                    docs: List[str], metas: List[Dict]) -> str:
        resources = self.get_jurisdiction_resources()
        context   = self._build_context_string(docs, metas)
        memory    = self.memory.to_prompt_string()

        system = f"""You are a practical legal advisor in {self.jurisdiction} helping with a traffic accident.

Answer the specific question asked. Give numbered action steps. Name the relevant law at the end.

"CRITICAL: Only cite law names and section numbers that appear verbatim in the provided references. "
"If you cannot find a specific provision in the references, say 'the specific section is not in the available references' rather than citing from memory."

Emergency: {resources['emergency']}. Keep under 250 words."""

        user = f"""{memory}

User's question: {query}

{context}"""

        return self._call_llm(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=600, temperature=0.3,
        )

    def _generate_general_response(self, query: str, analysis: QueryAnalysis,
                                   docs: List[str], metas: List[Dict]) -> str:
        resources = self.get_jurisdiction_resources()
        context   = self._build_context_string(docs, metas)
        memory    = self.memory.to_prompt_string()

        system = f"""You are a legal advisor in {self.jurisdiction}.

Answer the specific question directly. Give numbered practical steps. Name the relevant law.

"CRITICAL: Only cite law names and section numbers that appear verbatim in the provided references. "
"If you cannot find a specific provision in the references, say 'the specific section is not in the available references' rather than citing from memory."

If further help is needed: {resources['legal_aid']}. Keep under 250 words. Plain language."""

        user = f"""{memory}

User's question: {query}

{context}"""

        return self._call_llm(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=600, temperature=0.3,
        )

    def _fallback_response(self, analysis: Optional[QueryAnalysis]) -> str:
        r = self.get_jurisdiction_resources()
        if analysis and analysis.mode == ResponseMode.CRISIS:
            return (f"Please seek help immediately.\n\n"
                    f"Emergency: {r['emergency']}\n"
                    f"24/7 Helpline: {r['domestic_violence']}\n"
                    f"Shelter: {r['shelter']}")
        return f"For legal help in {self.jurisdiction}: {r['legal_aid']} | Emergency: {r['emergency']}"

    def get_jurisdiction_resources(self) -> Dict[str, str]:
        # Case-insensitive comparison so "pakistan" and "Pakistan" both work
        if self.jurisdiction.strip().lower() == "pakistan":
            return {
                "emergency":          "15 (Police) / 1122 (Rescue)",
                "domestic_violence":  "Madadgaar: 1098 (24/7)",
                "legal_aid":          "District Legal Aid Office",
                "protection_officer": "Local Protection Officer",
                "shelter":            "Dar-ul-Aman / Women's Shelter",
                "child_helpline":     "1121",
                "human_rights":       "HRCP: (92-51) 2274197",
            }
        # Default fallback always has all keys so KeyError is impossible
        return {
            "emergency":          "911",
            "domestic_violence":  "1-800-799-7233 (24/7)",
            "legal_aid":          "Legal Services Corporation",
            "protection_officer": "Local law enforcement",
            "shelter":            "National DV Hotline: 1-800-799-7233",
            "child_helpline":     "Childhelp: 1-800-422-4453",
            "human_rights":       "ACLU: aclu.org",
        }

    # ── MAIN QUERY INTERFACE ──────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        Full pipeline:
          1. Rewrite query with conversation context
          2. LLM classification on the rewritten query
          3. Retrieval + cross-encoder reranking
          4. Response generation with structured memory
          5. Update structured memory
        """
        if self.llm is None:
            raise ValueError("Call load_llm() first")

        print(f"\n{'=' * 70}")
        print(f"📝 Question: {question}")
        print(f"{'=' * 70}")

        # Step 1: Contextual rewrite
        effective_query = self.rewrite_query_with_context(question)
        if effective_query != question:
            print(f"✏️  Rewritten: {effective_query}")

        # Step 2: LLM classification on the enriched query
        analysis = self.analyze_query(effective_query)
        print(f"🏷️  Mode: {analysis.mode.value} | Urgency: {analysis.urgency.value}")

        # Short-circuit for chat
        if analysis.mode == ResponseMode.CHAT:
            answer = self._generate_chat_response(question)
            self._conversation_turns.append((question, answer))
            return {"answer": answer, "mode": "chat", "confidence": 1.0,
                    "analysis": analysis, "sources": []}

        # Step 3: Retrieval (on rewritten query)
        docs, metas, scores, confident = self.hybrid_retrieve(effective_query, analysis, top_k)
        if docs:
            print(f"📚 Sources: {', '.join(m.get('source_name','?') for m in metas[:3])}")
            print(f"📊 Scores:  {[round(s, 3) for s in scores[:3]]}")
        else:
            print("   ⚠ No sources found")

        if not confident:
            print("   ⚠ Low confidence — returning fallback")
            answer = self._fallback_response(analysis)
            self._conversation_turns.append((question, answer))
            return {"answer": answer, "mode": "fallback", "confidence": 0.0,
                    "analysis": analysis, "sources": []}

        # Step 4: Generation with structured memory
        answer = self.generate_response(question, analysis, docs, metas)
        print("✅ Response generated")

        # Step 5: Update structured memory
        self.update_memory(question, answer, analysis)

        self._conversation_turns.append((question, answer))

        return {
            "answer":           answer,
            "mode":             "rag",
            "analysis":         analysis,
            "effective_query":  effective_query,
            "sources":          docs,
            "source_metadata":  metas,
            "relevance_scores": scores,
            "confidence":       scores[0] if scores else 0.0,
        }

    # ── FIX 6: EVALUATION METRICS ─────────────────────────────────────────────
    #
    # Two-track evaluation:
    #   Track A — Retrieval metrics (no LLM needed, computable on your own)
    #   Track B — RAGAS generation metrics (requires ground truth dataset)

    def evaluate_retrieval(self, test_cases: List[Dict]) -> Dict:
        """
        Track A: Retrieval evaluation.

        test_cases format:
        [
            {
                "query": "What are my rights under domestic violence law?",
                "relevant_doc_ids": ["dv_act_2013", "ppw_act_2006"],
                "relevant_law_names": ["Domestic Violence Act"]
            },
            ...
        ]

        Returns: precision@k, recall@k, MRR, nDCG
        """
        print("\n📊 Running Retrieval Evaluation …")
        print("=" * 60)

        if not test_cases:
            print("⚠ No test cases provided")
            return {}

        k          = 3
        precisions = []
        recalls    = []
        mrrs       = []
        ndcgs      = []

        for tc in test_cases:
            q            = tc["query"]
            relevant_ids = set(tc.get("relevant_doc_ids", []))
            relevant_laws = set(law.lower() for law in tc.get("relevant_law_names", []))

            analysis = self.analyze_query(q)
            docs, metas, scores, _ = self.hybrid_retrieve(q, analysis, top_k=k)

            # Build relevance list (1 = relevant, 0 = not)
            relevance = []
            for meta in metas:
                doc_id  = meta.get("doc_id", "")
                src     = meta.get("source_name", "").lower()
                is_rel  = (doc_id in relevant_ids) or any(law in src for law in relevant_laws)
                relevance.append(1 if is_rel else 0)

            # Precision@k
            p_at_k = sum(relevance) / k if k > 0 else 0
            precisions.append(p_at_k)

            # Recall@k
            r_at_k = sum(relevance) / max(len(relevant_ids | relevant_laws), 1)
            recalls.append(min(r_at_k, 1.0))

            # MRR
            mrr = 0.0
            for rank, rel in enumerate(relevance, 1):
                if rel == 1:
                    mrr = 1.0 / rank
                    break
            mrrs.append(mrr)

            # nDCG@k
            dcg  = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(relevance, 1))
            idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, sum(relevance) + 1))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)

        results = {
            f"Precision@{k}": round(np.mean(precisions), 4),
            f"Recall@{k}":    round(np.mean(recalls),    4),
            "MRR":            round(np.mean(mrrs),        4),
            f"nDCG@{k}":      round(np.mean(ndcgs),       4),
            "n_test_cases":   len(test_cases),
        }

        print(f"  Precision@{k}: {results[f'Precision@{k}']}")
        print(f"  Recall@{k}:    {results[f'Recall@{k}']}")
        print(f"  MRR:           {results['MRR']}")
        print(f"  nDCG@{k}:      {results[f'nDCG@{k}']}")
        return results

    def evaluate_generation_ragas(self, test_cases: List[Dict]) -> Dict:
        """
        Track B: RAGAS generation evaluation.

        Requires: pip install ragas datasets langchain openai

        test_cases format:
        [
            {
                "question":       "Can my husband evict me from our home?",
                "ground_truth":   "Under Pakistani law, a wife cannot be unilaterally evicted ...",
                "relevant_law_names": ["Muslim Family Laws Ordinance"]
            },
            ...
        ]

        Returns: faithfulness, answer_relevancy, context_precision, context_recall
        """
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness, answer_relevancy,
                context_precision, context_recall,
            )
            from datasets import Dataset
        except ImportError:
            print("❌ RAGAS not installed. Run: pip install ragas datasets")
            return {}

        print("\n📊 Running RAGAS Evaluation …")

        ragas_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

        for tc in test_cases:
            q = tc["question"]
            analysis = self.analyze_query(q)
            docs, metas, _, _ = self.hybrid_retrieve(q, analysis, top_k=3)
            answer = self.generate_response(q, analysis, docs, metas)

            ragas_data["question"].append(q)
            ragas_data["answer"].append(answer)
            ragas_data["contexts"].append(docs if docs else ["No context found"])
            ragas_data["ground_truth"].append(tc.get("ground_truth", ""))

        dataset = Dataset.from_dict(ragas_data)
        scores  = evaluate(dataset, metrics=[
            faithfulness, answer_relevancy,
            context_precision, context_recall,
        ])

        results = {
            "faithfulness":       round(float(scores["faithfulness"]),        4),
            "answer_relevancy":   round(float(scores["answer_relevancy"]),    4),
            "context_precision":  round(float(scores["context_precision"]),   4),
            "context_recall":     round(float(scores["context_recall"]),      4),
            "n_test_cases":       len(test_cases),
        }

        print(f"  Faithfulness:      {results['faithfulness']}")
        print(f"  Answer Relevancy:  {results['answer_relevancy']}")
        print(f"  Context Precision: {results['context_precision']}")
        print(f"  Context Recall:    {results['context_recall']}")
        return results

    def human_evaluation_sheet(self, queries: List[str],
                                output_path: str = "human_eval.csv") -> str:
        """
        Generate a CSV for human evaluation.
        Fill in the scores manually, then compute averages.
        """
        import csv
        rows = []
        for q in queries:
            result = self.query(q)
            rows.append({
                "query":           q,
                "response":        result["answer"],
                "mode":            result["mode"],
                "confidence":      round(result.get("confidence", 0), 3),
                "legal_accuracy":  "",   # Human fills 1-5
                "helpfulness":     "",   # Human fills 1-5
                "safety":          "",   # Human fills 1-5
                "groundedness":    "",   # Human fills 1-5
                "clarity":         "",   # Human fills 1-5
                "notes":           "",
            })

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"✅ Human evaluation sheet saved to {output_path}")
        return output_path

    # ── PERSISTENCE ───────────────────────────────────────────────────────────

    def save_system(self, path: str = "law_rag_v3.pkl"):
        if self.embeddings is None:
            print("⚠ No embeddings to save")
            return
        with open(path, "wb") as f:
            pickle.dump({
                "embeddings":     self.embeddings,
                "chunk_texts":    self.chunk_texts,
                "chunk_metadata": self.chunk_metadata,
            }, f)
        print(f"✅ Saved to {path}")

    @staticmethod
    def load_system(state_file:    str = "law_rag_v3.pkl",metadata_file: str = "pdf_metadata.csv",persistent_dir: str = "chroma_store",jurisdiction:  str = "Pakistan",use_reranker:  bool = True,) -> "EnhancedLawRAGSystem":
        print("📂 Loading system …")
        with open(state_file, "rb") as f:
            state = pickle.load(f)

        rag = EnhancedLawRAGSystem.__new__(EnhancedLawRAGSystem)
        rag.jurisdiction        = jurisdiction
        rag.use_reranker        = use_reranker
        rag._conversation_turns = []
        rag.memory              = StructuredMemory()
        rag._groq_key           = os.getenv("GROQ_API_KEY", "")
        rag._hf_token           = os.getenv("HF_TOKEN", "")

        # Pass absolute path if given; _load_metadata will also search common dirs
        rag.metadata         = rag._load_metadata(metadata_file)
        if not rag.metadata:
            print("   ⚠  metadata is empty — source names will show as 'Unknown'.")
            print("   ⚠  Fix: EnhancedLawRAGSystem.load_system(metadata_file=r'C:/absolute/path/to/pdf_metadata.csv')")
        rag.chunk_texts      = state["chunk_texts"]
        rag.embeddings       = state["embeddings"]
        rag.chunk_metadata   = state.get("chunk_metadata", [])

        rag.persistent_dir   = persistent_dir
        rag.embedding_model  = SentenceTransformer("all-MiniLM-L6-v2")

        if use_reranker:
            print("⚡ Loading cross-encoder …")
            rag.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        else:
            rag.reranker = None

        rag.client     = PersistentClient(path=persistent_dir)
        rag.collection = rag.client.get_collection("law_documents")
        rag._initialize_statutory_priorities()

        rag.llm        = None
        rag.model_name = None

        print("✅ System loaded")
        return rag


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE DEMO
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# REPLACE YOUR ENTIRE interactive_demo() FUNCTION WITH THIS
# Drop-in replacement — paste over the old def interactive_demo(rag): block
# ─────────────────────────────────────────────────────────────────────────────

def interactive_demo(rag: "EnhancedLawRAGSystem"):
    # Import eval suite once at the top of the function
    try:
        from eval_suite import EvaluationSuite
        suite = EvaluationSuite(rag, k=3)
        _suite_available = True
    except ImportError:
        print("   ⚠ eval_suite.py not found — hallucination checks and eval disabled.")
        print("   ⚠ Place eval_suite.py in the same directory as this script.")
        suite = None
        _suite_available = False

    print("\n" + "=" * 70)
    print("🎯 LAW RAG v3.0 — Interactive Demo")
    print(f"📍 Jurisdiction: {rag.jurisdiction}")
    print("💡 Type 'quit' to exit | 'memory' to see structured memory")
    print("💡      'eval' → retrieval eval  | 'fulleval' → full suite (slow)")
    print("=" * 70)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            # ── empty input ───────────────────────────────────────────────────
            if not user_input:
                continue

            # ── quit ─────────────────────────────────────────────────────────
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("\n🤖 Take care!")
                break

            # ── memory ───────────────────────────────────────────────────────
            if user_input.lower() == "memory":
                mem_str = rag.memory.to_prompt_string()
                print(f"\n🧠 Current Memory:\n{mem_str if mem_str else '(empty)'}")
                continue

            # ── retrieval-only eval ───────────────────────────────────────────
            if user_input.lower() == "eval":
                if not _suite_available:
                    print("❌ eval_suite.py not available.")
                    continue
                # Ground-truth test cases (original 8, kept for backward compat)
                ground_truth_cases = [
                    {
                        "query": "What legal protection does a domestic violence victim have in Pakistan?",
                        "relevant_doc_ids": ["dv_007"],
                        "relevant_law_names": ["Domestic Violence (Prevention and Protection) Act"],
                    },
                    {
                        "query": "Can a wife file for divorce in Pakistan and on what grounds?",
                        "relevant_doc_ids": ["dv_014", "dv_011"],
                        "relevant_law_names": ["Dissolution of Muslim Marriages Act", "Muslim Family Laws Ordinance"],
                    },
                    {
                        "query": "What are the rules for child custody after divorce in Pakistan?",
                        "relevant_doc_ids": ["dv_008", "dv_020"],
                        "relevant_law_names": ["Guardians and Wards Act", "Family Courts Act"],
                    },
                    {
                        "query": "A driver is charged for a traffic violation he did not commit. What can he do?",
                        "relevant_doc_ids": ["traffic_002", "traffic_005"],
                        "relevant_law_names": ["Provincial Motor Vehicles Ordinance", "Traffic Offences Special Courts Ordinance"],
                    },
                    {
                        "query": "What are the penalties for traffic violations under Pakistani law?",
                        "relevant_doc_ids": ["traffic_001", "traffic_002"],
                        "relevant_law_names": ["Sindh Act XIII of 2014", "Provincial Motor Vehicles Ordinance"],
                    },
                    {
                        "query": "What maintenance rights does a wife have after separation in Pakistan?",
                        "relevant_doc_ids": ["dv_011", "dv_020"],
                        "relevant_law_names": ["Muslim Family Laws Ordinance", "Family Courts Act"],
                    },
                    {
                        "query": "How can a child marriage be prevented or annulled in Pakistan?",
                        "relevant_doc_ids": ["dv_003"],
                        "relevant_law_names": ["Child Marriage Restraint Act"],
                    },
                    {
                        "query": "What constitutional rights protect Pakistani citizens from abuse by authorities?",
                        "relevant_doc_ids": ["dv_001"],
                        "relevant_law_names": ["Constitution of the Islamic Republic of Pakistan"],
                    },
                ]
                print(f"\n   Running retrieval eval on {len(ground_truth_cases)} ground-truth test cases …")
                results = rag.evaluate_retrieval(ground_truth_cases)
                if results:
                    print("\n   Scores above 0.5 indicate good retrieval.")
                    print("   To improve: rebuild embeddings with smaller chunks (300-500 tokens).")
                continue

            # ── full evaluation suite (all 50 retrieval + 20 generation cases) ─
            if user_input.lower() == "fulleval":
                if not _suite_available:
                    print("❌ eval_suite.py not available.")
                    continue
                print("\n⚠  Full evaluation runs ~70 LLM calls and takes several minutes.")
                confirm = input("   Proceed? (yes/no): ").strip().lower()
                if confirm != "yes":
                    print("   Cancelled.")
                    continue
                suite.run_full_evaluation()
                continue

            # ── normal query ──────────────────────────────────────────────────
            result = rag.query(user_input)
            print(f"\n🤖 Assistant:\n{result['answer']}")

            # Only show metadata for RAG-mode responses (not chat / fallback)
            if result["mode"] == "rag":
                print(
                    f"\n   Mode: {result['analysis'].mode.value} | "
                    f"Confidence: {result['confidence']:.3f} | "
                    f"Sources: {len(result['sources'])}"
                )
                if result.get("effective_query") and result["effective_query"] != user_input:
                    print(f"   Rewritten: {result['effective_query']}")

                # Hallucination check — only for RAG responses that have retrieved chunks
                if _suite_available and result.get("sources"):
                    suite.check_response_hallucination(
                        query=user_input,
                        response=result["answer"],
                        retrieved_chunks=result["sources"],
                        verbose=True,
                    )

        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("\n" + "=" * 70)
    print("🎯 ENHANCED LAW RAG SYSTEM v3.0")
    print("=" * 70)
    print("\n1. Build new system")
    print("2. Load existing system")
    print("3. Load + demo")

    choice = input("\nChoice (1/2/3): ").strip()

    if choice == "1":
        jurisdiction = input("Jurisdiction (Pakistan): ").strip() or "Pakistan"
        rag = EnhancedLawRAGSystem(jurisdiction=jurisdiction)
        if not rag.chunk_texts:
            print("❌ No chunks loaded"); sys.exit(1)
        rag.create_embeddings()
        rag.populate_vector_db()
        rag.save_system()

    elif choice in ("2", "3"):
        jurisdiction   = input("Jurisdiction (Pakistan): ").strip() or "Pakistan"
        metadata_input = input("Absolute path to pdf_metadata.csv (leave blank to auto-search): ").strip()
        metadata_path  = metadata_input if metadata_input else "pdf_metadata.csv"
        rag = EnhancedLawRAGSystem.load_system(
            jurisdiction=jurisdiction,
            metadata_file=metadata_path,
        )
        rag.load_llm("llama-3.3-70b-versatile")
        if choice == "3":
            interactive_demo(rag)
    else:
        print("❌ Invalid choice"); sys.exit(1)