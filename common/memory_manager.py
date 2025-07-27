import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer

class MemoryManager:
    def __init__(self, config):
        self.config = config
        self.model = SentenceTransformer(config['memory']['embedding_model'])
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index_path = config['memory']['faiss_index_path']
        self.map_path = config['memory']['text_map_path']
        
        self.index = self._load_index()
        self.text_mapping = self._load_text_map()
        self.next_id = len(self.text_mapping)

    def _load_index(self):
        if os.path.exists(self.index_path):
            print("INFO: Loading existing FAISS index.")
            return faiss.read_index(self.index_path)
        else:
            print("INFO: Creating new FAISS index.")
            return faiss.IndexFlatL2(self.embedding_dim)

    def _load_text_map(self):
        if os.path.exists(self.map_path):
            with open(self.map_path, 'r') as f:
                return {int(k): v for k, v in json.load(f).items()}
        return {}
    
    def save_to_disk(self):
        """将索引和映射持久化到磁盘"""
        print("INFO: Saving FAISS index and text map to disk...")
        faiss.write_index(self.index, self.index_path)
        with open(self.map_path, 'w') as f:
            json.dump(self.text_mapping, f, indent=2)

    def add_to_global_memory(self, text: str, metadata: dict):
        """将观察或结论存入全局记忆"""
        embedding = self.model.encode([text])
        self.index.add(np.array(embedding, dtype=np.float32))
        
        doc_id = self.next_id
        self.text_mapping[doc_id] = {
            "text": text,
            "metadata": metadata
        }
        self.next_id += 1
        print(f"INFO: Added entry {doc_id} to global memory: '{text[:40]}...'")
        self.save_to_disk() # 在科研项目中，每次添加后保存是可接受的

    def search_global_memory(self, query_text: str, n_results: int = 3) -> list:
        """从全局记忆中检索相关信息"""
        if self.index.ntotal == 0:
            return []
        query_embedding = self.model.encode([query_text])
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), n_results)
        
        results = []
        for i in indices[0]:
            if i in self.text_mapping:
                results.append(self.text_mapping[i])
        return results
