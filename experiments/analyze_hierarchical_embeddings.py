"""
Analyze what Hierarchical Embeddings actually learned.
Investigate if the three layers implicitly capture cognitive mechanisms.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine

from experiments.hybrid_models import GPT_HierarchicalEmbedding
from neural_networks.nlp.tokenizer import CharacterTokenizer
from neural_networks.data import download_tinystories, TinyStoriesDataset
from torch.utils.data import DataLoader

def train_hierarchical_model(config, train_loader, val_loader, epochs=100, device='cuda'):
    """Train hierarchical model to analyze"""
    print("Training Hierarchical Model for Analysis...")
    
    model = GPT_HierarchicalEmbedding(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
        
        if epoch % 20 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y)
                    val_loss += loss.item()
            print(f"Epoch {epoch}: Val Loss = {val_loss/len(val_loader):.4f}")
    
    return model

def analyze_embedding_spaces(model, tokenizer):
    """Analyze the three embedding spaces"""
    print("\n" + "="*60)
    print("EMBEDDING SPACE ANALYSIS")
    print("="*60)
    
    # Extract embeddings
    symbol_emb = model.embeddings.symbol_emb.weight.detach().cpu().numpy()
    concept_emb = model.embeddings.concept_emb.weight.detach().cpu().numpy()
    law_emb = model.embeddings.law_emb.weight.detach().cpu().numpy()
    
    vocab_size, emb_dim = symbol_emb.shape
    
    print(f"\nEmbedding dimensions: {vocab_size} tokens × {emb_dim} dims")
    
    # 1. Linear Separability
    print("\n1. Linear Separability Analysis")
    print("-" * 40)
    
    # Compute pairwise distances within each space
    def compute_avg_distance(embeddings):
        distances = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(dist)
        return np.mean(distances), np.std(distances)
    
    symbol_dist, symbol_std = compute_avg_distance(symbol_emb)
    concept_dist, concept_std = compute_avg_distance(concept_emb)
    law_dist, law_std = compute_avg_distance(law_emb)
    
    print(f"Symbol space:  mean_dist={symbol_dist:.4f}, std={symbol_std:.4f}")
    print(f"Concept space: mean_dist={concept_dist:.4f}, std={concept_std:.4f}")
    print(f"Law space:     mean_dist={law_dist:.4f}, std={law_std:.4f}")
    
    # 2. Correlation between spaces
    print("\n2. Cross-Space Correlation")
    print("-" * 40)
    
    # Compute correlation between spaces
    symbol_flat = symbol_emb.flatten()
    concept_flat = concept_emb.flatten()
    law_flat = law_emb.flatten()
    
    corr_symbol_concept = np.corrcoef(symbol_flat, concept_flat)[0, 1]
    corr_symbol_law = np.corrcoef(symbol_flat, law_flat)[0, 1]
    corr_concept_law = np.corrcoef(concept_flat, law_flat)[0, 1]
    
    print(f"Symbol ↔ Concept: {corr_symbol_concept:.4f}")
    print(f"Symbol ↔ Law:     {corr_symbol_law:.4f}")
    print(f"Concept ↔ Law:    {corr_concept_law:.4f}")
    
    if abs(corr_symbol_concept) < 0.3:
        print("✅ Spaces are relatively independent")
    else:
        print("⚠️  Spaces are correlated (may be redundant)")
    
    # 3. Semantic Clustering
    print("\n3. Semantic Clustering Analysis")
    print("-" * 40)
    
    # Group tokens by type (letters, digits, punctuation, etc.)
    token_groups = {
        'letters': [],
        'digits': [],
        'space': [],
        'punctuation': []
    }
    
    for idx, token in enumerate(tokenizer.idx_to_char):
        if token.isalpha():
            token_groups['letters'].append(idx)
        elif token.isdigit():
            token_groups['digits'].append(idx)
        elif token.isspace():
            token_groups['space'].append(idx)
        else:
            token_groups['punctuation'].append(idx)
    
    # Compute within-group vs between-group distances for concept space
    def compute_clustering_score(embeddings, groups):
        within_dist = []
        between_dist = []
        
        for group_name, indices in groups.items():
            if len(indices) < 2:
                continue
            # Within-group distances
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    dist = np.linalg.norm(embeddings[indices[i]] - embeddings[indices[j]])
                    within_dist.append(dist)
        
        # Between-group distances
        group_list = [indices for indices in groups.values() if len(indices) > 0]
        for i in range(len(group_list)):
            for j in range(i+1, len(group_list)):
                for idx1 in group_list[i]:
                    for idx2 in group_list[j]:
                        dist = np.linalg.norm(embeddings[idx1] - embeddings[idx2])
                        between_dist.append(dist)
        
        return np.mean(within_dist), np.mean(between_dist)
    
    concept_within, concept_between = compute_clustering_score(concept_emb, token_groups)
    
    print(f"Concept space clustering:")
    print(f"  Within-group distance: {concept_within:.4f}")
    print(f"  Between-group distance: {concept_between:.4f}")
    print(f"  Clustering ratio: {concept_between/concept_within:.2f}x")
    
    if concept_between / concept_within > 1.2:
        print("✅ Concepts cluster by semantic type")
    else:
        print("⚠️  No clear semantic clustering")
    
    # 4. Learned Hierarchy Weights
    print("\n4. Learned Hierarchy Weights")
    print("-" * 40)
    
    alpha = torch.sigmoid(model.embeddings.alpha_logit).item()
    beta = torch.sigmoid(model.embeddings.beta_logit).item()
    
    print(f"Symbol weight: 1.0000 (base)")
    print(f"Concept weight (α): {alpha:.4f}")
    print(f"Law weight (β): {beta:.4f}")
    
    print(f"\nInterpretation:")
    if alpha > 0.4:
        print(f"  ✅ Concepts are important ({alpha:.2f})")
    else:
        print(f"  ⚠️  Concepts underutilized ({alpha:.2f})")
    
    if beta > 0.3:
        print(f"  ✅ Laws are important ({beta:.2f})")
    else:
        print(f"  ⚠️  Laws underutilized ({beta:.2f})")
    
    return {
        'symbol_emb': symbol_emb,
        'concept_emb': concept_emb,
        'law_emb': law_emb,
        'alpha': alpha,
        'beta': beta,
        'clustering_ratio': concept_between/concept_within
    }

def visualize_embeddings(embeddings_dict, tokenizer, save_path='experiments/embedding_analysis'):
    """Visualize embedding spaces with PCA"""
    print("\n5. Visualizing Embedding Spaces")
    print("-" * 40)
    
    os.makedirs(save_path, exist_ok=True)
    
    symbol_emb = embeddings_dict['symbol_emb']
    concept_emb = embeddings_dict['concept_emb']
    law_emb = embeddings_dict['law_emb']
    
    # PCA to 2D
    pca = PCA(n_components=2)
    
    symbol_2d = pca.fit_transform(symbol_emb)
    concept_2d = pca.fit_transform(concept_emb)
    law_2d = pca.fit_transform(law_emb)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, emb_2d, title in zip(axes, [symbol_2d, concept_2d, law_2d], 
                                   ['Symbol Space', 'Concept Space', 'Law Space']):
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.6)
        
        # Label a few tokens
        for i in range(min(10, len(emb_2d))):
            token = tokenizer.idx_to_char[i]
            ax.annotate(repr(token), (emb_2d[i, 0], emb_2d[i, 1]), fontsize=8)
        
        ax.set_title(title)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/embedding_spaces.png', dpi=150)
    print(f"✅ Saved visualization to {save_path}/embedding_spaces.png")
    
    plt.close()

def main():
    # Setup
    config = {
        'vocab_size': None,
        'n_embd': 64,
        'n_head': 4,
        'n_layer': 2,
        'block_size': 32,
        'dropout': 0.1
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    json_path, _ = download_tinystories()
    
    with open(json_path, 'r') as f:
        import json
        data = json.load(f)
        text = ' '.join(data['stories'])
    
    tokenizer = CharacterTokenizer()
    tokenizer.train(text)
    config['vocab_size'] = tokenizer.vocab_size
    
    dataset = TinyStoriesDataset(json_path, tokenizer, block_size=config['block_size'])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Train model
    model = train_hierarchical_model(config, train_loader, val_loader, epochs=100, device=device)
    
    # Analyze
    embeddings_dict = analyze_embedding_spaces(model, tokenizer)
    
    # Visualize
    visualize_embeddings(embeddings_dict, tokenizer)
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nHierarchy Weights:")
    print(f"  Symbol: 1.00 (base)")
    print(f"  Concept: {embeddings_dict['alpha']:.2f}")
    print(f"  Law: {embeddings_dict['beta']:.2f}")
    
    print(f"\nClustering:")
    print(f"  Ratio: {embeddings_dict['clustering_ratio']:.2f}x")
    
    print(f"\nConclusion:")
    if embeddings_dict['alpha'] > 0.4 and embeddings_dict['clustering_ratio'] > 1.2:
        print("  ✅ Hierarchical embeddings learn meaningful structure")
        print("  ✅ Three layers capture different aspects")
    else:
        print("  ⚠️  Hierarchy may be underutilized")

if __name__ == "__main__":
    main()
