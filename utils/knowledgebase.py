# utils/knowledgebase.py

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
        Add new memory to the knowledge base, which can be text, image, or both.
        
        Args:
            text (str): Text description or None
            image (PIL.Image): Image or None
        """
        # Generate unique ID for each new memory
        memory_id = int(uuid.uuid4().int & 0xFFFFFFFF)  # Generate 32-bit integer ID
        
        # Calculate embeddings for text and image
        text_vector = None
        image_vector = None
        
        with torch.no_grad():
            # Process text
            if text:
                inputs = self.preprocess(
                    text=text,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                text_features = self.text_embedder.get_text_features(**inputs)
                text_vector = F.normalize(text_features, p=2, dim=1).cpu().numpy()[0]
            
            # Process image
            if image:
                # Use CLIP preprocessor to process image
                image_inputs = self.preprocess(
                    images=image, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                image_features = self.image_model.get_image_features(**image_inputs)
                image_vector = F.normalize(image_features, p=2, dim=1).cpu().numpy()[0]
        
        # Calculate combined vector
        if text_vector is not None and image_vector is not None:
            # Both text and image exist, perform weighted fusion
            combined_vector = (self.weight_image * image_vector + 
                               self.weight_text * text_vector)
            # Re-normalize
            combined_vector = combined_vector / np.linalg.norm(combined_vector)
        elif text_vector is not None:
            # Only text
            combined_vector = text_vector
        elif image_vector is not None:
            # Only image
            combined_vector = image_vector
        else:
            # Neither exists, do not add
            return
        
        # Add to FAISS index
        self.index.add_with_ids(
            np.array([combined_vector], dtype=np.float32),
            np.array([memory_id], dtype=np.int64)
        )
        
        # Save memory data
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
        # Decide whether to directly add memory or replace similar memory through replace_memory
        if self.replace_memory:
            pass
        
        else:
            self._add_to_memory(text, image)


    def search(self, text, image, top_k=5):
        """
        Search for memories similar to text and/or image in the knowledge base.
        
        Args:
            text (str): Text query or None
            image (PIL.Image): Image query or None
            top_k (int): Number of results to return
        
        Returns:
            list: List of dictionaries containing matching memories
        """
        # Calculate embeddings for text and image
        text_vector = None
        image_vector = None
        
        with torch.no_grad():
            # Process text
            if text:
                inputs = self.preprocess(
                    text=text,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                text_features = self.text_embedder.get_text_features(**inputs)
                text_vector = F.normalize(text_features, p=2, dim=1).cpu().numpy()[0]
            
            # Process image
            if image:
                # Use CLIP preprocessor to process image
                image_inputs = self.preprocess(
                    images=image, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                image_features = self.image_model.get_image_features(**image_inputs)
                image_vector = F.normalize(image_features, p=2, dim=1).cpu().numpy()[0]
        
        # Calculate combined vector
        if text_vector is not None and image_vector is not None:
            # Both text and image exist, perform weighted fusion
            query_vector = (self.weight_image * image_vector + 
                           self.weight_text * text_vector)
            # Re-normalize
            query_vector = query_vector / np.linalg.norm(query_vector)
        elif text_vector is not None:
            # Only text
            query_vector = text_vector
        elif image_vector is not None:
            # Only image
            query_vector = image_vector
        else:
            # Neither exists, return empty results
            return []
        
        # If knowledge base is empty, return empty results
        if len(self.data) == 0:
            return []
        
        # Execute search
        distances, indices = self.index.search(
            np.array([query_vector], dtype=np.float32), 
            max(1, min(top_k, len(self.data)))
        )
        
        # Organize search results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS returns -1 when results are insufficient
                continue
                
            # Find corresponding memory data
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
