# utils/knowledgebase.py

# TODO: 完成 replace_memory 的复杂逻辑

import uuid
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from transformers import CLIPProcessor, CLIPModel


class KnowledgeBase:
    """A class for managing a knowledge base with text and image data using CLIP model."""
    
    def __init__(self, config):
        
        kb_config = config.get("memory", {})
        
        # replace_memory (bool): Strategy for updating memory. If True, removes similar memories (most similar ones -- similarity >= threshold and no more than three altogether) before adding new ones. If False, always add new ones.
        self.replace_memory = kb_config.get("replace_memory", False)
        
        # dimension of embeddings
        self.dimension = kb_config.get("dimension", 768)
        
        # weight of image embeddings
        self.weight_image = kb_config.get("weight_image", 0.5)
        
        # weight of image-text similarity
        self.weight_text = kb_config.get("weight_text", 0.5)
        
        # device
        self.device = kb_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # lambda_sim (float): Weight for combining observation and caption similarity (default: 0.5).
        # self.lambda_sim = kb_config.get("lambda_sim", 0.5)
        
        # Load CLIP model and processor
        self.text_embedder = self.image_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # Initialize FAISS index with ID mapping for robust removal
        base_index = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIDMap(base_index)
        
        # Store data in a dictionary with unique ID as key
        self.data = []


    def _add_to_memory(self, text, image):
        """
        添加新的记忆到知识库中，可以是文本、图像或两者结合。
        
        Args:
            text (str): 文本描述或None
            image (PIL.Image): 图像或None
        """
        # 为每个新记忆生成唯一ID
        memory_id = int(uuid.uuid4().int & 0xFFFFFFFF)  # 生成32位整数ID
        
        # 计算text和image的embedding
        text_vector = None
        image_vector = None
        
        with torch.no_grad():
            # 处理文本
            if text:
                inputs = self.preprocess(
                    text=text,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                text_features = self.text_embedder.get_text_features(**inputs)
                text_vector = F.normalize(text_features, p=2, dim=1).cpu().numpy()[0]
            
            # 处理图像
            if image:
                # 使用CLIP预处理器处理图像
                image_inputs = self.preprocess(
                    images=image, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                image_features = self.image_model.get_image_features(**image_inputs)
                image_vector = F.normalize(image_features, p=2, dim=1).cpu().numpy()[0]
        
        # 计算综合向量
        if text_vector is not None and image_vector is not None:
            # 文本和图像都存在，进行加权融合
            combined_vector = (self.weight_image * image_vector + 
                               self.weight_text * text_vector)
            # 重新归一化
            combined_vector = combined_vector / np.linalg.norm(combined_vector)
        elif text_vector is not None:
            # 只有文本
            combined_vector = text_vector
        elif image_vector is not None:
            # 只有图像
            combined_vector = image_vector
        else:
            # 两者都没有，不添加
            return
        
        # 添加到FAISS索引
        self.index.add_with_ids(
            np.array([combined_vector], dtype=np.float32),
            np.array([memory_id], dtype=np.int64)
        )
        
        # 保存记忆数据
        memory_data = {
            "id": memory_id,
            "image": image,
            "text": text,
            "image_vector": image_vector,
            "text_vector": text_vector,
            "vector": combined_vector
        }
        
        self.data.append(memory_data)
    
    
    def update_memory(self, text, image):
        # 通过 replace_memory 决定是直接添加记忆还是替换相似记忆
        if self.replace_memory:
            # 先忽略这一部分
            pass
        
        else:
            self._add_to_memory(text, image)


    def search(self, text, image, top_k=5):
        """
        在知识库中搜索与文本和/或图像相似的记忆。
        
        Args:
            text (str): 文本查询或None
            image (PIL.Image): 图像查询或None
            top_k (int): 返回结果的数量
        
        Returns:
            list: 包含匹配记忆的字典列表
        """
        # 计算text和image的embedding
        text_vector = None
        image_vector = None
        
        with torch.no_grad():
            # 处理文本
            if text:
                inputs = self.preprocess(
                    text=text,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                text_features = self.text_embedder.get_text_features(**inputs)
                text_vector = F.normalize(text_features, p=2, dim=1).cpu().numpy()[0]
            
            # 处理图像
            if image:
                # 使用CLIP预处理器处理图像
                image_inputs = self.preprocess(
                    images=image, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                image_features = self.image_model.get_image_features(**image_inputs)
                image_vector = F.normalize(image_features, p=2, dim=1).cpu().numpy()[0]
        
        # 计算综合向量
        if text_vector is not None and image_vector is not None:
            # 文本和图像都存在，进行加权融合
            query_vector = (self.weight_image * image_vector + 
                           self.weight_text * text_vector)
            # 重新归一化
            query_vector = query_vector / np.linalg.norm(query_vector)
        elif text_vector is not None:
            # 只有文本
            query_vector = text_vector
        elif image_vector is not None:
            # 只有图像
            query_vector = image_vector
        else:
            # 两者都没有，返回空结果
            return []
        
        # 如果知识库为空，返回空结果
        if len(self.data) == 0:
            return []
        
        # 执行搜索
        distances, indices = self.index.search(
            np.array([query_vector], dtype=np.float32), 
            max(1, min(top_k, len(self.data)))
        )
        
        # 整理搜索结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS会在结果不足时返回-1
                continue
                
            # 找到对应的记忆数据
            memory_data = None
            for item in self.data:
                if item["id"] == idx:
                    memory_data = item
                    break
            
            if memory_data:
                results.append({
                    "id": memory_data["id"],
                    "text": memory_data["text"],
                    "image": memory_data["image"]
                })
        
        return results


    def clear(self):
        """Clear the FAISS index and stored data."""
        self.index.reset()
        self.data = []
