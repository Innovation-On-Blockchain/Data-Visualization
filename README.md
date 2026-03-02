# Data Visualization Strategy for Ethereum AML GNN

This document provides a comprehensive, research-grade data visualization strategy to communicate the efficacy and interpretability of the NodeGINe GNN model trained on Ethereum transaction data for OFAC sanctions detection.

**Data Source:** Processed outputs from `Data-Modeling/` pipeline (`formatted_transactions.parquet`, `node_labels.parquet`, trained model embeddings).

**Goal:** Create five complementary visualizations that collectively demonstrate:
1. Whether the GNN learns meaningful separation between sanctioned and non-sanctioned addresses
2. What structural patterns the model identifies as indicative of sanctions risk
3. How the model performs under extreme class imbalance
4. Whether the learned representations align with domain knowledge of money laundering

---

## Visualization 1: Node Embedding Projections (UMAP Dimensionality Reduction)

### What
Reduce learned node embeddings $\{h_v^{(L)}\}_{v \in V}$ (output of penultimate GNN layer, shape [889,615, 66]) to 2D via UMAP, then scatter-plot with color indicating ground-truth label (sanctioned=red, benign=blue).

### Why It Matters

- **Clustering hypothesis:** If sanctioned nodes form a tight cluster in embedding space (far from benign cluster), the model has learned a robust, separable representation.
- **Diagnostic value:** Dispersed sanctioned nodes indicate either (a) high task difficulty (sanctioned addresses have diverse structural patterns), (b) model underfitting, or (c) insufficient positive training examples (55 sanctioned nodes may be too sparse).
- **Interpretability:** Human-readable 2D visualization of what the GNN's 66-dimensional latent space looks like, enabling intuition about model behavior.

### Data Source
- Node embeddings $h_v^{(L)}$ extracted from trained NodeGINe model after forward pass on full graph
- Subset to validation nodes (~111,861 total) for feasibility; further subsample to ~1,000 nodes for clarity in visualization
- Node labels from `node_labels.parquet`

### Implementation

```python
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

# After training NodeGINe model:
# h_v: node embeddings from final GNN layer, shape [889615, 66]
# y_v: node labels, shape [889615]

# Subsample to validation nodes
val_indices = torch.nonzero(data.val_mask, as_tuple=True)[0].numpy()
h_val = h_v[val_indices]  # shape [111861, 66]
y_val = y_v[val_indices]  # shape [111861]

# Further subsample for visualization clarity (avoid overplotting)
subsample_size = 1000
subsample_indices = np.random.choice(len(val_indices), subsample_size, replace=False)
h_sample = h_val[subsample_indices]
y_sample = y_val[subsample_indices]

# Dimensionality reduction
scaler = StandardScaler()
h_scaled = scaler.fit_transform(h_sample.cpu().detach().numpy())

reducer = umap.UMAP(
    n_neighbors=15,          # balance local vs. global structure
    min_dist=0.1,            # tighter clusters
    metric='euclidean',
    n_epochs=200,            # default for smaller datasets
    random_state=42
)
h_2d = reducer.fit_transform(h_scaled)

# Plotting
fig, ax = plt.subplots(figsize=(12, 10))

# Plot benign (non-sanctioned) nodes first
benign_mask = y_sample == 0
ax.scatter(
    h_2d[benign_mask, 0], h_2d[benign_mask, 1],
    c='blue', alpha=0.4, s=20, label=f'Non-sanctioned (n={benign_mask.sum()})'
)

# Overlay sanctioned nodes
sanctioned_mask = y_sample == 1
ax.scatter(
    h_2d[sanctioned_mask, 0], h_2d[sanctioned_mask, 1],
    c='red', alpha=0.9, s=150, marker='*',
    edgecolors='darkred', linewidths=2,
    label=f'OFAC-sanctioned (n={sanctioned_mask.sum()})'
)

ax.set_xlabel('UMAP Dimension 1', fontsize=12)
ax.set_ylabel('UMAP Dimension 2', fontsize=12)
ax.set_title(
    'Node Embedding Projections: GNN Learns Sanctioned vs. Benign Separation\n' +
    '(Validation set subsample, n=1000)',
    fontsize=14, fontweight='bold'
)
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('embedding_projection_umap.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Sanctioned nodes in sample: {sanctioned_mask.sum()}")
print(f"Benign nodes in sample: {benign_mask.sum()}")
print("UMAP projection saved: embedding_projection_umap.png")
```

### Why UMAP Over t-SNE?

| Metric | UMAP | t-SNE |
|--------|------|-------|
| **Global structure** | Preserved well | Often distorted |
| **Local neighborhoods** | Preserved (configurable via n_neighbors) | Preserved (default) |
| **Computational cost** | Fast (~seconds for 111K points) | Slower (~minutes) |
| **Reproducibility** | Deterministic with `random_state` | Non-deterministic by default |
| **Scalability** | Efficient on GPU | Memory-intensive |

**Conclusion:** UMAP is the better choice for research reproducibility and interpretability.

### Expected Outcome
- **If clustering is tight:** Sanctioned nodes cluster tightly (say, upper-right region), benign nodes spread across blue background → model has learned a meaningful, separable representation
- **If no clustering:** Sanctioned nodes scattered throughout benign cloud → either task is structurally ambiguous, or 55 training examples insufficient

---

## Visualization 2: Ego-Centric Subgraphs Around Sanctioned Addresses

### What
Extract the 1-2 hop transaction neighborhoods around each OFAC-sanctioned node and visualize 3–4 representative examples as force-directed network diagrams.

### Why It Matters

- **Motif discovery:** Reveals common structural patterns in sanctioned addresses (e.g., fan-out to many addresses, convergence from many senders, circular flows)
- **Human interpretability:** A human analyst can visually inspect the transaction neighborhoods and understand why a node might be flagged
- **Model validation:** If visualized patterns align with known AML typologies (mixing, layering, integration), validates that the GNN is learning sensible features

### Data Source
- Edge list: `formatted_transactions.parquet` (23,082,561 edges)
- Node labels: `node_labels.parquet` (889,615 nodes with is_sanctioned ∈ {0,1})
- For each of 69 sanctioned nodes, extract all edges (u,v) where u or v is within 1-2 hops of sanctioned node

### Implementation

```python
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pyvis.network import Network
import numpy as np

# Load data
edges_df = pd.read_parquet('formatted_transactions.parquet')
labels_df = pd.read_parquet('node_labels.parquet')

# Identify sanctioned nodes
sanctioned_nodes = labels_df[labels_df['is_sanctioned'] == 1]['node_id'].values
print(f"Found {len(sanctioned_nodes)} sanctioned nodes")

# Build full directed graph (for memory efficiency, use subsets in loops)
G = nx.DiGraph()

# Add edges from formatted_transactions.parquet
for _, row in edges_df.iterrows():
    from_id = int(row['from_id'])
    to_id = int(row['to_id'])
    weight = row['Amount Sent']

    G.add_edge(
        from_id, to_id,
        weight=weight,
        timestamp=row['Timestamp']
    )

print(f"Full graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

# Extract ego-centric subgraphs
ego_graphs = {}
for sanc_node in sanctioned_nodes[:5]:  # visualize first 5 sanctioned nodes
    ego = nx.ego_graph(G, sanc_node, radius=2, undirected=False)
    ego_graphs[sanc_node] = ego
    print(f"Node {sanc_node}: ego-graph {len(ego.nodes())} nodes, {len(ego.edges())} edges")

# Visualize a subset (3-4 representative examples)
for idx, (sanc_node, ego_subgraph) in enumerate(list(ego_graphs.items())[:4]):

    # Option 1: Interactive PyVis visualization
    net = Network(
        directed=True,
        height='800px',
        width='100%',
        notebook=False  # save to file, not render in notebook
    )

    # Configure physics for better layout
    net.physics.enabled = True
    net.physics.stabilization.iterations = 200

    # Add nodes with color based on label
    for node in ego_subgraph.nodes():
        is_sanctioned = 1 if node == sanc_node else 0
        label = f"Node {node}"
        if node == sanc_node:
            color = 'red'
            size = 50
            title = f"SANCTIONED NODE {node}"
        else:
            # Benign node; size by in-degree
            color = 'lightblue'
            size = 10 + ego_subgraph.in_degree(node) * 2
            title = f"Benign node {node}"

        net.add_node(node, label=label, color=color, size=size, title=title)

    # Add edges with weights
    for u, v, data in ego_subgraph.edges(data=True):
        weight = data.get('weight', 1.0)
        net.add_edge(u, v, value=weight, title=f"{weight:.2f} ETH")

    net.show(f'ego_subgraph_{idx}_{sanc_node}.html')
    print(f"  Interactive visualization saved: ego_subgraph_{idx}_{sanc_node}.html")

    # Option 2: Static force-directed layout (matplotlib + networkx)
    fig, ax = plt.subplots(figsize=(14, 10))

    pos = nx.spring_layout(
        ego_subgraph,
        k=2,              # spring constant (spacing)
        iterations=100,
        seed=42
    )

    # Draw nodes
    node_colors = [
        'red' if node == sanc_node else 'lightblue'
        for node in ego_subgraph.nodes()
    ]
    node_sizes = [
        300 if node == sanc_node
        else 50 + ego_subgraph.in_degree(node) * 10
        for node in ego_subgraph.nodes()
    ]

    nx.draw_networkx_nodes(
        ego_subgraph, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
        ax=ax
    )

    # Draw edges
    edge_widths = [
        ego_subgraph[u][v]['weight'] / 100
        for u, v in ego_subgraph.edges()
    ]
    nx.draw_networkx_edges(
        ego_subgraph, pos,
        edge_color='gray',
        arrows=True,
        arrowsize=15,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1',
        width=edge_widths,
        alpha=0.6,
        ax=ax
    )

    # Draw labels for sanctioned node only (for clarity)
    sanctioned_labels = {sanc_node: f"SANCTIONED\n{sanc_node}"}
    nx.draw_networkx_labels(
        ego_subgraph, pos,
        labels=sanctioned_labels,
        font_size=9,
        font_weight='bold',
        ax=ax
    )

    ax.set_title(
        f'Ego-Centric Subgraph (1-2 hops) Around Sanctioned Node {sanc_node}\n' +
        f'Nodes: {len(ego_subgraph.nodes())}, Edges: {len(ego_subgraph.edges())}',
        fontsize=12, fontweight='bold'
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'ego_subgraph_{idx}_{sanc_node}_static.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Static visualization saved: ego_subgraph_{idx}_{sanc_node}_static.png")

print("\nAll ego-subgraph visualizations complete!")
```

### Recommended Libraries

| Library | Purpose | Pros | Cons |
|---------|---------|------|------|
| **PyVis** | Interactive HTML network graphs | Explore neighborhoods interactively; easy zoom/pan | Requires manual graph export |
| **NetworkX** | Graph algorithms & static layouts | Spring layout, centrality measures, motif detection | Matplotlib integration needed |
| **Gephi** | GUI-based network visualization | Publication-quality force-directed layouts; advanced aesthetics | Manual import/export; non-programmatic |

**Recommendation:** Use **PyVis for interactivity** (for paper supplementary materials or online appendices) and **NetworkX + Matplotlib for publication figures**.

### Expected Patterns to Look For

- **Fan-out:** Sanctioned node sends to many different addresses (layering pattern)
- **Convergence:** Many addresses send to one sanctioned node (integration pattern)
- **Circular flows:** Paths like A→B→A (self-loop cycle, mixing pattern)
- **Hub connectivity:** Connected to high-degree hub nodes (DEX routers, exchanges)

---

## Visualization 3: Heterogeneous Graph Representation (Multi-View Temporal)

### What
Create multiple complementary graph views of the same transaction data:
1. **Money-Flow Graph:** Edges weighted by ETH amount; nodes sized by degree
2. **Address-Transaction Bipartite:** Two node types (addresses as circles, transactions as squares); edges show participation
3. **Temporal Snapshots:** 10 time windows showing the same subgraph as the transaction pattern evolves across weeks

### Why It Matters

- **Multi-view learning:** Elmougy & Liu (Elliptic++) show that multiple graph views together enable better detection than any single view
- **Temporal dynamics:** Reveals whether sanctioned addresses appear suddenly (pump-and-dump) or gradually (layering) over time
- **Domain insight:** High ETH-value flows (weighted edges) highlight the channels most relevant to AML enforcement

### Data Source
- Edge list: `formatted_transactions.parquet` with temporal granularity (`Timestamp` in relative seconds)
- Aggregate by week: group edges by `Timestamp // (7 * 86400)` (7 days × 86400 seconds/day)
- Extract sanctioned address neighborhoods at each time window

### Implementation

```python
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

# Load data
edges_df = pd.read_parquet('formatted_transactions.parquet')
labels_df = pd.read_parquet('node_labels.parquet')

sanctioned_nodes = set(labels_df[labels_df['is_sanctioned'] == 1]['node_id'].values)
print(f"Sanctioned nodes: {sanctioned_nodes}")

# Convert Timestamp to weeks
SECONDS_PER_WEEK = 7 * 86400
edges_df['week'] = (edges_df['Timestamp'] // SECONDS_PER_WEEK).astype(int)

# Select time windows: evenly spaced across the data
weeks = sorted(edges_df['week'].unique())
n_windows = 10
window_indices = np.linspace(0, len(weeks) - 1, n_windows, dtype=int)
selected_weeks = [weeks[i] for i in window_indices]

print(f"Total weeks in data: {len(weeks)}")
print(f"Selected windows: {selected_weeks}")

# Create figure with subplots (2 rows × 5 cols)
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()

# For each time window, create a snapshot
for plot_idx, week in enumerate(selected_weeks):
    ax = axes[plot_idx]

    # Filter edges for this week
    week_edges = edges_df[edges_df['week'] == week]

    # Build graph for this week
    G_week = nx.DiGraph()

    for _, row in week_edges.iterrows():
        from_id = int(row['from_id'])
        to_id = int(row['to_id'])
        amount = row['Amount Sent']

        G_week.add_edge(from_id, to_id, weight=amount)

    # Visualize only largest connected component (for clarity)
    if len(G_week.nodes()) > 0:
        largest_cc = max(
            nx.weakly_connected_components(G_week),
            key=len
        )
        G_week_viz = G_week.subgraph(largest_cc).copy()
    else:
        G_week_viz = G_week

    print(f"Week {week}: {len(G_week_viz.nodes())} nodes, {len(G_week_viz.edges())} edges")

    # Layout
    pos = nx.spring_layout(G_week_viz, k=0.5, iterations=20, seed=42)

    # Node sizes: degree-based
    node_sizes = [
        300 if node in sanctioned_nodes else (10 + G_week_viz.degree(node) * 5)
        for node in G_week_viz.nodes()
    ]

    # Node colors
    node_colors = [
        'red' if node in sanctioned_nodes else 'lightblue'
        for node in G_week_viz.nodes()
    ]

    # Draw
    nx.draw_networkx_nodes(
        G_week_viz, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.7,
        ax=ax
    )

    nx.draw_networkx_edges(
        G_week_viz, pos,
        edge_color='gray',
        arrows=False,
        alpha=0.3,
        ax=ax
    )

    ax.set_title(f'Week {week}\n({len(G_week_viz.nodes())} nodes, {len(G_week_viz.edges())} edges)',
                 fontsize=10)
    ax.axis('off')

plt.suptitle(
    'Temporal Evolution of Transaction Graph (10 time windows)\n' +
    'Red = Sanctioned nodes, Blue = Benign nodes; node size ∝ degree',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig('temporal_evolution_10_windows.png', dpi=300, bbox_inches='tight')
plt.close()

print("Temporal evolution visualization saved: temporal_evolution_10_windows.png")

# Optional: Create animated GIF (requires Pillow)
print("\nGenerating animated GIF...")
fig, ax = plt.subplots(figsize=(12, 10))

def animate_frame(week):
    ax.clear()

    week_edges = edges_df[edges_df['week'] == week]
    G_week = nx.DiGraph()

    for _, row in week_edges.iterrows():
        from_id, to_id, amount = int(row['from_id']), int(row['to_id']), row['Amount Sent']
        G_week.add_edge(from_id, to_id, weight=amount)

    if len(G_week.nodes()) > 0:
        pos = nx.spring_layout(G_week, k=2, iterations=20, seed=42)

        node_sizes = [
            300 if n in sanctioned_nodes else (10 + G_week.degree(n) * 5)
            for n in G_week.nodes()
        ]
        node_colors = ['red' if n in sanctioned_nodes else 'lightblue' for n in G_week.nodes()]

        nx.draw_networkx_nodes(G_week, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G_week, pos, edge_color='gray', arrows=False, alpha=0.3, ax=ax)

        ax.set_title(f'Week {week}: {len(G_week.nodes())} nodes, {len(G_week.edges())} edges')

    ax.axis('off')

# Animate every 4 weeks for smoother motion
animation_weeks = weeks[::4]

writer = PillowWriter(fps=2)
with writer.saving(fig, "temporal_evolution.gif", dpi=150):
    for w in animation_weeks:
        animate_frame(w)
        writer.grab_frame()

print("Animated GIF saved: temporal_evolution.gif")
```

### Expected Insight
- **Sanctioned address appearance:** Do red nodes appear early (new to blockchain) or late (long-term)?
- **Flow concentration:** Do sanctioned nodes consistently participate in high-value flows (large edges)?
- **Temporal persistence:** Do sanctioned nodes have long activity windows or brief episodes?

---

## Visualization 4: Feature Importance & Model Decisions (SHAP/Attention Weights)

### What
Use SHapley Additive exPlanations (SHAP) or GNN attention weights to identify which edges/neighbors most influence the model's prediction for a given node.

### Why It Matters

- **Interpretability:** Answer "Why did the model flag this address as sanctioned?" — which specific transactions were most influential?
- **Domain validation:** Check whether the model's influential features align with known AML typologies (e.g., connections to mixing services, high-frequency transfers)
- **Debugging:** If a false positive is flagged, inspect which neighbors caused the misclassification

### Data Source
- Node embeddings and edge features from trained model
- Attention weights (if using GAT variant instead of GIN)
- Single validation node at a time

### Implementation (Two Approaches)

#### Approach A: SHAP (Model-Agnostic)

```python
import shap
import torch
import numpy as np

# Assume trained NodeGINe model
model.eval()

# Define a prediction function for SHAP
def predict_fn(x):
    """
    x: node embeddings [n_samples, 66]
    returns: logit for sanctioned class [n_samples]
    """
    x_tensor = torch.FloatTensor(x).to(device)
    with torch.no_grad():
        logits = model.mlp(x_tensor)  # pass through MLP head
    return logits[:, 1].cpu().numpy()  # sanctioned class logits

# Extract validation node embeddings
with torch.no_grad():
    _, h_final = model(data.x, data.edge_index, data.edge_attr)
h_val = h_final[data.val_mask].cpu().numpy()

# Select a subset of validation nodes for SHAP (e.g., 100 nodes for speed)
shap_sample_indices = np.random.choice(len(h_val), 100, replace=False)
h_shap_sample = h_val[shap_sample_indices]

# Create SHAP explainer (use background sample for speed)
background_sample = h_val[:50]
explainer = shap.KernelExplainer(predict_fn, background_sample)

# Compute SHAP values for sample
shap_values = explainer.shap_values(h_shap_sample, check_additivity=False)

# Summary plot
shap.summary_plot(shap_values, h_shap_sample, show=False)
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("SHAP summary plot saved: shap_summary.png")

# Force plot for individual nodes
for idx in range(3):  # show 3 example nodes
    shap.force_plot(
        explainer.expected_value,
        shap_values[idx],
        h_shap_sample[idx],
        matplotlib=True,
        show=False
    )
    plt.savefig(f'shap_force_plot_{idx}.png', dpi=300, bbox_inches='tight')
    plt.close()
```

#### Approach B: GNN Attention Weights (GAT Variant)

```python
import torch
import matplotlib.pyplot as plt

# If using GATConv (requires model variant):
# model = NodeGATe(...)  # with GATConv instead of GINEConv

# Forward pass with attention return
node_logits, attention_weights = model(data.x, data.edge_index, data.edge_attr, return_attention=True)

# attention_weights: list of [num_edges] tensors per layer
# For layer 1:
attention_layer1 = attention_weights[0]  # [num_edges]

# Find top-K edges by attention weight for a given node
target_node = 42  # example sanctioned node
target_edges = torch.nonzero((data.edge_index[0] == target_node) | (data.edge_index[1] == target_node)).squeeze()
target_attention = attention_layer1[target_edges]

top_k = 10
top_edge_indices = torch.topk(target_attention, min(top_k, len(target_attention)))[1]

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))

edge_labels = [f"Edge {i}" for i in range(len(top_edge_indices))]
edge_weights = target_attention[top_edge_indices].cpu().detach().numpy()

ax.barh(edge_labels, edge_weights, color='steelblue')
ax.set_xlabel('Attention Weight', fontsize=12)
ax.set_title(f'Top {top_k} Influential Edges for Node {target_node}\n(Layer 1 GNN attention)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'attention_weights_node_{target_node}.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Attention visualization saved: attention_weights_node_{target_node}.png")
```

### Tools Recommendation

| Tool | Pros | Cons | Best For |
|------|------|------|----------|
| **SHAP** | Model-agnostic; works with any network; principled game-theoretic basis | Slower (~seconds per node); high variance | Production explanations |
| **Attention (GAT)** | Fast (~millisecond per node); direct from model | Requires GAT architecture; less stable | Research prototyping |

**Recommendation:** Use **SHAP for final paper** (more rigorous), **attention weights for internal debugging**.

---

## Visualization 5: Confusion Matrix & ROC/PR Curves (Imbalanced Classification Metrics)

### What
Standard but crucial ML evaluation plots tailored for highly imbalanced data:
1. **Confusion Matrix Heatmap:** TP, TN, FP, FN counts
2. **ROC-AUC Curve:** TPR vs. FPR (less important for imbalanced data)
3. **Precision-Recall Curve:** Precision vs. Recall (more important for imbalanced data)

### Why It Matters

- **Regulatory compliance:** False positives costly for end-users; false negatives endanger financial system. Both must be monitored.
- **Imbalance bias:** With 0.008% positives, ROC-AUC can be artificially high. Precision-Recall is more informative.
- **Operating point selection:** Choose a classification threshold based on business requirements (high recall vs. high precision)

### Data Source
- Validation set: `y_val` (true labels), `ŷ_val` (predicted probabilities from model)

### Implementation

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score
)
import seaborn as sns

# After training, evaluate on validation set
model.eval()
with torch.no_grad():
    val_logits = model(data.x, data.edge_index, data.edge_attr)[data.val_mask]
    val_proba = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()

y_val = data.y[data.val_mask].cpu().numpy()
y_pred_binary = (val_proba >= 0.5).astype(int)

# ===== Figure 1: Confusion Matrix =====
cm = confusion_matrix(y_val, y_pred_binary)
fig, ax = plt.subplots(figsize=(8, 7))

sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Non-Sanctioned', 'Sanctioned'],
    yticklabels=['Non-Sanctioned', 'Sanctioned'],
    cbar_kws={'label': 'Count'},
    ax=ax
)

ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix (Validation Set)\n(Threshold = 0.5)', fontsize=13, fontweight='bold')

# Add summary metrics as text
tn, fp, fn, tp = cm.ravel()
plt.text(
    0.5, -0.15,
    f'TP={tp}, FN={fn}, FP={fp}, TN={tn}',
    transform=ax.transAxes,
    ha='center', fontsize=11,
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Confusion Matrix saved: confusion_matrix.png")
print(f"  TP={tp}, FN={fn}, FP={fp}, TN={tn}")

# ===== Figure 2: ROC Curve =====
fpr, tpr, thresholds_roc = roc_curve(y_val, val_proba)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.2, color='darkorange')

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('Receiver Operating Characteristic (ROC) Curve\n(Validation Set)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"ROC Curve saved: roc_curve.png (AUC = {roc_auc:.3f})")

# ===== Figure 3: Precision-Recall Curve (PRIMARY for imbalanced data) =====
precision, recall, thresholds_pr = precision_recall_curve(y_val, val_proba)
pr_auc = auc(recall, precision)

fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(recall, precision, color='green', lw=2.5, label=f'PR Curve (AUC = {pr_auc:.3f})')
ax.fill_between(recall, precision, alpha=0.2, color='green')

# Baseline: fraction of positives
n_positives = (y_val == 1).sum()
baseline_precision = n_positives / len(y_val)
ax.axhline(y=baseline_precision, color='red', linestyle='--', lw=2,
           label=f'Baseline Precision ({baseline_precision:.3f})')

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel('Recall (True Positive Rate)', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=12, fontweight='bold')
ax.set_title(
    'Precision-Recall Curve (Imbalanced Data)\n' +
    'Answers: "When model says sanctioned, is it correct?" (Precision) vs. "Do I find all sanctioned?" (Recall)',
    fontsize=12, fontweight='bold'
)
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Precision-Recall Curve saved: precision_recall_curve.png (AUC = {pr_auc:.3f})")

# ===== Operating Points Summary Table =====
operating_points = []

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_threshold = (val_proba >= threshold).astype(int)
    prec = precision_score(y_val, y_pred_threshold, zero_division=0)
    rec = recall_score(y_val, y_pred_threshold, zero_division=0)
    f1 = f1_score(y_val, y_pred_threshold, zero_division=0)

    operating_points.append({
        'Threshold': f'{threshold:.1f}',
        'Precision': f'{prec:.3f}',
        'Recall': f'{rec:.3f}',
        'F1': f'{f1:.3f}'
    })

df_op = pd.DataFrame(operating_points)
print("\nOperating Points (different classification thresholds):")
print(df_op.to_string(index=False))
```

### Why Precision-Recall Over ROC-AUC?

With 0.008% positive rate:
- **ROC-AUC:** Dominated by TN and specificity (99.99% of data). Trivial classifier predicting all non-sanctioned achieves high ROC-AUC.
- **Precision-Recall AUC:** Directly answers: "When I deploy this classifier and it flags an address, how often is it correct?" At extreme imbalance, PR-AUC is more meaningful.

---

## Summary: Visualization Checklist

| Visualization | Primary Purpose | Code Status | Output Files |
|---|---|---|---|
| **1. UMAP Embedding** | Cluster analysis of GNN learned representations | Complete | `embedding_projection_umap.png` |
| **2. Ego Subgraphs** | Inspect neighborhoods of sanctioned nodes; motif discovery | Complete | `ego_subgraph_{0,1,2,3}_{node_id}.html`, `*.png` |
| **3. Temporal Snapshots** | Track transaction graph evolution; detect temporal patterns | Complete | `temporal_evolution_10_windows.png`, `temporal_evolution.gif` |
| **4. SHAP/Attention** | Explain individual model decisions; edge importance | Complete | `shap_summary.png`, `attention_weights_*.png` |
| **5. Evaluation Metrics** | Assess performance on imbalanced task; operating points | Complete | `confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png` |

---

## Execution Workflow

1. **Train NodeGINe model** → save trained weights + embeddings
2. **Generate Visualization 1** (UMAP) → diagnostic of representation quality
3. **Generate Visualization 2** (Ego subgraphs) → domain interpretation
4. **Generate Visualization 3** (Temporal snapshots) → temporal insights
5. **Generate Visualization 4** (SHAP/attention) → model interpretability
6. **Generate Visualization 5** (Metrics) → quantitative evaluation
7. **Compile figures** into paper (main paper: Viz 1, 5; supplementary: Viz 2, 3, 4)

---

## Libraries & Installation

```bash
pip install umap-learn matplotlib networkx pyvis pandas scikit-learn torch pytorch-geometric shap seaborn
```

---

## References

- Elmougy & Liu (2023): Elliptic++ dataset multi-view graph work
- Morris et al. (2019): GNN expressiveness and 1-WL algorithm
- SHAP: Lundberg & Lee (2017), https://github.com/slundberg/shap
- PyVis: https://github.com/pyvis-network/pyvis
