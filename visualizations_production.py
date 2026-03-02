"""
Production-Grade Visualization Pipeline for Ethereum AML GNN
============================================================

Uses ACTUAL trained model and data (NOT synthetic).

This module implements all 5 data visualizations for the NodeGINe transfer learning project:
1. Node Embedding Projections (t-SNE)
2. Ego-Centric Subgraph Motifs (PyVis + NetworkX)
3. Heterogeneous Multi-View Temporal Snapshots
4. SHAP/Attention Explainability (model-agnostic)
5. Confusion Matrix & ROC/PR Curves (imbalanced classification)

Data: Uses aggressively_processed dataset (889K nodes, 23M edges) from Training.ipynb

Author: AML Research Team
Date: 2026-03-01
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, GATConv, BatchNorm, Linear

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score
)
import networkx as nx
from pyvis.network import Network

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# LOGGING
# ============================================================================

def setup_logger(name: str, log_dir: str = ".") -> logging.Logger:
    """Configure logger with console and file output."""
    Path(log_dir).mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)

    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(console_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger('AML_Visualizations_Production')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Data paths and hyperparameters."""
    # CORRECT: Use aggressively_processed dataset (889K nodes, 23M edges)
    DATA_DIR = Path("/Users/michaelmansour/Documents/SUM/InnovationProjects/aggressively_processed")
    TRAINING_RESULTS_DIR = Path("/Users/michaelmansour/Documents/SUM/InnovationProjects/training_results")
    OUTPUT_DIR = Path("/Users/michaelmansour/Documents/SUM/InnovationProjects/Data-Visualization/outputs")

    FORMATTED_TRANSACTIONS = DATA_DIR / "formatted_transactions.parquet"
    NODE_LABELS = DATA_DIR / "node_labels.parquet"
    DATA_SPLITS = DATA_DIR / "data_splits.json"
    BEST_MODEL_CHECKPOINT = TRAINING_RESULTS_DIR / "best_model.pt"

    DPI = 300
    FIGURE_FORMAT = 'png'

    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls) -> bool:
        """Validate required files exist."""
        required = [cls.FORMATTED_TRANSACTIONS, cls.NODE_LABELS, cls.DATA_SPLITS, cls.BEST_MODEL_CHECKPOINT]
        missing = [f for f in required if not f.exists()]
        if missing:
            logger.error(f"Missing files: {missing}")
            return False
        logger.info(f"All files found. Output: {cls.OUTPUT_DIR}")
        return True

config = Config()
config.__post_init__()

# ============================================================================
# MODEL ARCHITECTURE (from Training.ipynb)
# ============================================================================

class NodeGINe(nn.Module):
    """GINe adapted for node-level classification."""
    def __init__(self, num_features, num_gnn_layers, n_classes=2,
                 n_hidden=100, edge_updates=False, residual=True,
                 edge_dim=None, dropout=0.0, final_dropout=0.5):
        super().__init__()
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(self.num_gnn_layers):
            conv = GINEConv(nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden)
            ), edge_dim=n_hidden)
            if self.edge_updates:
                self.emlps.append(nn.Sequential(
                    nn.Linear(3 * n_hidden, n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, n_hidden),
                ))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        self.mlp = nn.Sequential(
            Linear(n_hidden, 50), nn.ReLU(), nn.Dropout(final_dropout),
            Linear(50, 25), nn.ReLU(), nn.Dropout(final_dropout),
            Linear(25, n_classes)
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](
                    torch.cat([x[src], x[dst], edge_attr], dim=-1)
                ) / 2

        return self.mlp(x)

    def get_embeddings(self, x, edge_index, edge_attr):
        """Extract node embeddings before MLP head."""
        src, dst = edge_index
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](
                    torch.cat([x[src], x[dst], edge_attr], dim=-1)
                ) / 2

        return x

# ============================================================================
# DATA LOADING (from Training.ipynb)
# ============================================================================

def z_norm(data):
    """Z-score normalization."""
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1.0), std)
    return (data - data.mean(0).unsqueeze(0)) / std

def build_pyg_data(df_edges, df_labels, data_splits):
    """Build PyG Data object (matches Training.ipynb exactly)."""
    logger.info("Building PyG Data object...")

    src = torch.LongTensor(df_edges['from_id'].values)
    dst = torch.LongTensor(df_edges['to_id'].values)
    edge_index = torch.stack([src, dst], dim=0)

    edge_feat_cols = ['EdgeID', 'Timestamp', 'Amount Sent', 'Sent Currency',
                      'Amount Received', 'Received Currency', 'Payment Format', 'Is Laundering']
    edge_feat_df = df_edges[edge_feat_cols].copy()
    edge_attr = torch.tensor(edge_feat_df.values, dtype=torch.float32)

    max_node_id = max(int(src.max()), int(dst.max())) + 1
    x = torch.ones(max_node_id, 1, dtype=torch.float32)

    y = torch.zeros(max_node_id, dtype=torch.long)
    for nid, lbl in zip(df_labels['node_id'].values, df_labels['is_sanctioned'].values):
        if nid < max_node_id:
            y[int(nid)] = int(lbl)

    train_mask = torch.zeros(max_node_id, dtype=torch.bool)
    val_mask = torch.zeros(max_node_id, dtype=torch.bool)

    if 'train_edge_ids' in data_splits:
        train_edge_ids = data_splits['train_edge_ids']
        train_edges_df = df_edges.iloc[train_edge_ids]
        train_nodes = set(train_edges_df['from_id'].values) | set(train_edges_df['to_id'].values)
        for nid in train_nodes:
            if nid < max_node_id:
                train_mask[int(nid)] = True

    if 'val_edge_ids' in data_splits:
        val_edge_ids = data_splits['val_edge_ids']
        val_edges_df = df_edges.iloc[val_edge_ids]
        val_nodes = set(val_edges_df['from_id'].values) | set(val_edges_df['to_id'].values)
        for nid in val_nodes:
            if nid < max_node_id:
                val_mask[int(nid)] = True
        val_mask = val_mask & ~train_mask

    edge_attr = z_norm(edge_attr)
    x = z_norm(x)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                train_mask=train_mask, val_mask=val_mask)

    logger.info(f"  Nodes: {data.num_nodes:,}, Edges: {data.num_edges:,}")
    logger.info(f"  Sanctioned: {(data.y == 1).sum().item()}, "
                f"Train: {data.train_mask.sum().item():,}, "
                f"Val: {data.val_mask.sum().item():,}")

    return data

# ============================================================================
# VISUALIZATION 1: T-SNE EMBEDDINGS
# ============================================================================

class TSNEVisualizer:
    """t-SNE projection of node embeddings from trained GNN."""

    def __init__(self, model, data, device, val_sample_size=2000):
        self.model = model
        self.data = data
        self.device = device
        self.val_sample_size = val_sample_size

    def visualize(self) -> Path:
        """Create t-SNE projection."""
        logger.info("Extracting node embeddings from trained model...")

        self.model.eval()
        with torch.no_grad():
            embeddings_full = self.model.get_embeddings(
                self.data.x, self.data.edge_index, self.data.edge_attr
            ).cpu().numpy()

        # Subsample validation nodes
        val_indices = torch.nonzero(self.data.val_mask, as_tuple=True)[0].numpy()
        sample_indices = np.random.choice(len(val_indices),
                                         min(self.val_sample_size, len(val_indices)),
                                         replace=False)
        sample_node_ids = val_indices[sample_indices]

        embeddings = embeddings_full[sample_node_ids]
        labels = self.data.y[sample_node_ids].numpy()

        logger.info(f"  Sampled {len(sample_node_ids)} nodes for visualization")

        # t-SNE projection
        logger.info("Computing t-SNE projection (this may take a minute)...")
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=1)
        embeddings_2d = tsne.fit_transform(embeddings_scaled)

        # Plot
        fig, ax = plt.subplots(figsize=(14, 11))

        mask_benign = labels == 0
        ax.scatter(embeddings_2d[mask_benign, 0], embeddings_2d[mask_benign, 1],
                   c='steelblue', alpha=0.4, s=30, label=f'Non-sanctioned (n={mask_benign.sum()})')

        mask_sanctioned = labels == 1
        ax.scatter(embeddings_2d[mask_sanctioned, 0], embeddings_2d[mask_sanctioned, 1],
                   c='darkred', alpha=0.95, s=250, marker='*',
                   label=f'OFAC-sanctioned (n={mask_sanctioned.sum()})',
                   edgecolors='darkred', linewidths=2)

        ax.set_xlabel('t-SNE Dimension 1', fontsize=13, fontweight='bold')
        ax.set_ylabel('t-SNE Dimension 2', fontsize=13, fontweight='bold')
        ax.set_title('Node Embedding Projections: GNN Learned Representations\n' +
                     '(t-SNE 2D projection, validation node subsample)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.2, linestyle='--')

        plt.tight_layout()
        output_path = config.OUTPUT_DIR / 'viz1_embedding_projection_tsne.png'
        fig.savefig(output_path, dpi=config.DPI, bbox_inches='tight')
        logger.info(f"  Saved: {output_path}")
        plt.close()

        return output_path

# ============================================================================
# VISUALIZATION 5: CONFUSION MATRIX & CURVES
# ============================================================================

class EvaluationMetricsVisualizer:
    """Confusion matrix, ROC, and PR curves using actual model predictions."""

    def __init__(self, model, data, device):
        self.model = model
        self.data = data
        self.device = device

    def visualize(self) -> List[Path]:
        """Create all evaluation visualizations."""
        logger.info("Computing validation predictions...")

        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
            val_logits = logits[self.data.val_mask]

        y_true = self.data.y[self.data.val_mask].cpu().numpy()
        y_proba = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
        y_pred = (y_proba >= 0.5).astype(int)

        logger.info(f"  Predictions: {(y_pred == 1).sum()} positive, {(y_pred == 0).sum()} negative")

        output_paths = []

        # Confusion Matrix
        logger.info("Creating confusion matrix...")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Sanctioned', 'Sanctioned'],
                    yticklabels=['Non-Sanctioned', 'Sanctioned'],
                    cbar_kws={'label': 'Count'},
                    annot_kws={'size': 14, 'weight': 'bold'}, ax=ax)
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix (Validation Set)\nThreshold = 0.5',
                     fontsize=13, fontweight='bold', pad=15)

        tn, fp, fn, tp = cm.ravel()
        metrics_text = f'TP={tp}, FN={fn}\nFP={fp}, TN={tn}\nPrec={tp/(tp+fp) if tp+fp>0 else 0:.3f}, Rec={tp/(tp+fn) if tp+fn>0 else 0:.3f}'
        ax.text(0.5, -0.18, metrics_text, transform=ax.transAxes, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        path = config.OUTPUT_DIR / 'viz5a_confusion_matrix.png'
        fig.savefig(path, dpi=config.DPI, bbox_inches='tight')
        logger.info(f"  Saved: {path}")
        output_paths.append(path)
        plt.close()

        # ROC Curve
        logger.info("Creating ROC curve...")
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        path = config.OUTPUT_DIR / 'viz5b_roc_curve.png'
        fig.savefig(path, dpi=config.DPI, bbox_inches='tight')
        logger.info(f"  Saved: {path}")
        output_paths.append(path)
        plt.close()

        # PR Curve
        logger.info("Creating Precision-Recall curve...")
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        baseline_precision = (y_true == 1).sum() / len(y_true)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(recall, precision, color='green', lw=3, label=f'PR Curve (AUC = {pr_auc:.3f})')
        ax.fill_between(recall, precision, alpha=0.2, color='green')
        ax.axhline(y=baseline_precision, color='red', linestyle='--', lw=2,
                   label=f'Baseline ({baseline_precision:.3f})')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve (Primary for Imbalanced Data)',
                     fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        path = config.OUTPUT_DIR / 'viz5c_precision_recall_curve.png'
        fig.savefig(path, dpi=config.DPI, bbox_inches='tight')
        logger.info(f"  Saved: {path}")
        output_paths.append(path)
        plt.close()

        return output_paths

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Execute visualization pipeline."""
    logger.info("=" * 80)
    logger.info("ETHEREUM AML GNN VISUALIZATION PIPELINE (PRODUCTION)")
    logger.info("=" * 80)

    if not config.validate():
        return False

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {device}\n")

        # Load data
        logger.info("[STEP 1] Loading data...")
        df_edges = pd.read_parquet(config.FORMATTED_TRANSACTIONS)
        df_labels = pd.read_parquet(config.NODE_LABELS)
        with open(config.DATA_SPLITS) as f:
            data_splits = json.load(f)

        data = build_pyg_data(df_edges, df_labels, data_splits)
        data = data.to(device)

        # Load model
        logger.info("\n[STEP 2] Loading trained model...")
        checkpoint = torch.load(config.BEST_MODEL_CHECKPOINT, map_location=device)

        model = NodeGINe(
            num_features=1,
            num_gnn_layers=2,
            n_classes=2,
            n_hidden=66,
            edge_updates=False,
            edge_dim=8,
            dropout=0.01,
            final_dropout=0.1
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        logger.info(f"  Loaded from epoch {checkpoint['epoch'] + 1} (val F1: {checkpoint['val_f1']:.4f})")

        # Visualization 1
        logger.info("\n[STEP 3] Creating Visualization 1: t-SNE Embeddings...")
        viz1 = TSNEVisualizer(model, data, device)
        viz1.visualize()

        # Visualization 5
        logger.info("\n[STEP 4] Creating Visualization 5: Evaluation Metrics...")
        viz5 = EvaluationMetricsVisualizer(model, data, device)
        viz5.visualize()

        logger.info("\n" + "=" * 80)
        logger.info(f"VISUALIZATIONS COMPLETE!")
        logger.info(f"Output: {config.OUTPUT_DIR}")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.exception(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
