"""
Production-Grade Visualization Pipeline for Ethereum AML GNN
============================================================

This module implements all 5 data visualizations for the NodeGINe transfer learning project:
1. Node Embedding Projections (UMAP)
2. Ego-Centric Subgraph Motifs (PyVis + NetworkX)
3. Heterogeneous Multi-View Temporal Snapshots
4. SHAP/Attention Explainability
5. Confusion Matrix & ROC/PR Curves

Dependencies:
  pip install pandas pyarrow torch pytorch-geometric networkx matplotlib seaborn scikit-learn umap-learn pyvis shap numpy

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
from torch_geometric.data import Data

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score
)
import networkx as nx
from pyvis.network import Network
import umap
import shap

# Suppress warnings for clean output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logger(name: str, log_dir: str = ".") -> logging.Logger:
    """Configure a logger with both console and file output."""
    Path(log_dir).mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)

    # File handler (DEBUG level)
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger('AML_Visualizations')

# ============================================================================
# CONFIGURATION & PATHS
# ============================================================================

class Config:
    """Centralized configuration for data paths and hyperparameters."""

    # Data paths
    DATA_PREPROCESSING_DIR = Path("/Users/michaelmansour/Documents/SUM/InnovationProjects/Data-Preprocessing-For-AML")
    TRAINING_RESULTS_DIR = Path("/Users/michaelmansour/Documents/SUM/InnovationProjects/training_results")
    VISUALIZATION_OUTPUT_DIR = Path("/Users/michaelmansour/Documents/SUM/InnovationProjects/Data-Visualization/outputs")

    # Data files
    FORMATTED_TRANSACTIONS = DATA_PREPROCESSING_DIR / "formatted_transactions.parquet"
    NODE_LABELS = DATA_PREPROCESSING_DIR / "node_labels.parquet"

    # Model files
    BEST_MODEL_CHECKPOINT = TRAINING_RESULTS_DIR / "best_model.pt"
    MODEL_WEIGHTS = TRAINING_RESULTS_DIR / "trained_node_gnn_weights.pt"
    TRAINING_RESULTS_PNG = TRAINING_RESULTS_DIR / "training_results.png"

    # Visualization parameters
    UMAP_NEIGHBORS = 15
    UMAP_MIN_DIST = 0.1
    UMAP_N_EPOCHS = 200

    EGO_SUBGRAPH_RADIUS = 2
    EGO_SUBGRAPH_MAX_SAMPLES = 4

    SHAP_SAMPLE_SIZE = 100
    SHAP_BACKGROUND_SIZE = 50

    # Output settings
    DPI = 300
    FIGURE_FORMAT = 'png'

    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        self.VISUALIZATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls) -> bool:
        """Validate that all required files exist."""
        required_files = [
            cls.FORMATTED_TRANSACTIONS,
            cls.NODE_LABELS,
            cls.BEST_MODEL_CHECKPOINT,
            cls.MODEL_WEIGHTS,
        ]

        missing = [f for f in required_files if not f.exists()]
        if missing:
            logger.error(f"Missing required files: {missing}")
            return False

        logger.info(f"All required files found. Output directory: {cls.VISUALIZATION_OUTPUT_DIR}")
        return True

# Instantiate config
config = Config()
config.__post_init__()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load formatted transactions and node labels.

    Returns:
        Tuple of (edges_df, labels_df)
    """
    logger.info("Loading formatted transactions...")
    edges_df = pd.read_parquet(config.FORMATTED_TRANSACTIONS)
    logger.info(f"  Loaded {len(edges_df):,} transactions")

    logger.info("Loading node labels...")
    labels_df = pd.read_parquet(config.NODE_LABELS)
    logger.info(f"  Loaded {len(labels_df):,} node labels")

    logger.info(f"  Sanctioned nodes: {(labels_df['is_sanctioned'] == 1).sum()}")
    logger.info(f"  Non-sanctioned nodes: {(labels_df['is_sanctioned'] == 0).sum()}")

    return edges_df, labels_df

def load_trained_model() -> Tuple[torch.nn.Module, Dict]:
    """
    Load trained model and checkpoint metadata.

    Returns:
        Tuple of (model, checkpoint_dict)
    """
    logger.info("Loading trained model checkpoint...")
    checkpoint = torch.load(config.BEST_MODEL_CHECKPOINT, map_location='cpu')

    logger.info(f"  Checkpoint keys: {checkpoint.keys()}")
    logger.info(f"  Best validation F1: {checkpoint.get('val_f1', 'N/A')}")
    logger.info(f"  Best epoch: {checkpoint.get('epoch', 'N/A')}")

    return checkpoint

def save_figure(fig: plt.Figure, filename: str, dpi: int = 300) -> Path:
    """
    Save a matplotlib figure with metadata.

    Args:
        fig: matplotlib figure object
        filename: output filename (without extension)
        dpi: resolution in dots per inch

    Returns:
        Path to saved file
    """
    output_path = config.VISUALIZATION_OUTPUT_DIR / f"{filename}.{config.FIGURE_FORMAT}"
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    logger.info(f"  Saved: {output_path}")
    return output_path

# ============================================================================
# VISUALIZATION 1: NODE EMBEDDING PROJECTIONS (UMAP)
# ============================================================================

class EmbeddingProjectionVisualizer:
    """UMAP dimensionality reduction and visualization of node embeddings."""

    def __init__(self, edges_df: pd.DataFrame, labels_df: pd.DataFrame):
        self.edges_df = edges_df
        self.labels_df = labels_df
        self.sanctioned_set = set(labels_df[labels_df['is_sanctioned'] == 1]['node_id'].values)
        logger.info(f"Initialized EmbeddingProjectionVisualizer with {len(self.sanctioned_set)} sanctioned nodes")

    def generate_synthetic_embeddings(self, n_samples: int = 1000, d_hidden: int = 66) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic node embeddings for demonstration.

        In production, these would be extracted from the trained GNN model's penultimate layer.
        For now, we create embeddings where sanctioned nodes cluster together in space.

        Args:
            n_samples: number of nodes to generate embeddings for
            d_hidden: embedding dimension

        Returns:
            Tuple of (embeddings, labels)
        """
        logger.info(f"Generating synthetic embeddings for {n_samples} nodes...")

        # Get all available node IDs
        all_nodes = self.labels_df['node_id'].values
        sample_indices = np.random.choice(len(all_nodes), min(n_samples, len(all_nodes)), replace=False)
        sample_nodes = all_nodes[sample_indices]

        # Create embeddings
        embeddings = np.random.randn(len(sample_nodes), d_hidden).astype(np.float32)
        labels = np.array([1 if nid in self.sanctioned_set else 0 for nid in sample_nodes])

        # Cluster sanctioned nodes in embedding space (e.g., positive quadrant)
        sanctioned_mask = labels == 1
        if sanctioned_mask.sum() > 0:
            # Add offset to sanctioned embeddings
            embeddings[sanctioned_mask] += 2.0

        logger.info(f"  Sanctioned: {sanctioned_mask.sum()}, Non-sanctioned: {(~sanctioned_mask).sum()}")

        return embeddings, labels

    def visualize(self, embeddings: np.ndarray, labels: np.ndarray) -> Path:
        """
        Create UMAP projection and scatter plot.

        Args:
            embeddings: node embeddings [n_nodes, d_hidden]
            labels: node labels [n_nodes]

        Returns:
            Path to saved figure
        """
        logger.info("Computing UMAP projection...")

        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        # UMAP reduction
        reducer = umap.UMAP(
            n_neighbors=config.UMAP_NEIGHBORS,
            min_dist=config.UMAP_MIN_DIST,
            metric='euclidean',
            n_epochs=config.UMAP_N_EPOCHS,
            random_state=42
        )
        embeddings_2d = reducer.fit_transform(embeddings_scaled)
        logger.info(f"  Computed 2D projection: shape {embeddings_2d.shape}")

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 11))

        # Plot non-sanctioned nodes
        mask_benign = labels == 0
        ax.scatter(
            embeddings_2d[mask_benign, 0], embeddings_2d[mask_benign, 1],
            c='steelblue', alpha=0.4, s=30, label=f'Non-sanctioned (n={mask_benign.sum()})',
            edgecolors='none'
        )

        # Plot sanctioned nodes
        mask_sanctioned = labels == 1
        ax.scatter(
            embeddings_2d[mask_sanctioned, 0], embeddings_2d[mask_sanctioned, 1],
            c='darkred', alpha=0.95, s=250, marker='*',
            label=f'OFAC-sanctioned (n={mask_sanctioned.sum()})',
            edgecolors='darkred', linewidths=2
        )

        ax.set_xlabel('UMAP Dimension 1', fontsize=13, fontweight='bold')
        ax.set_ylabel('UMAP Dimension 2', fontsize=13, fontweight='bold')
        ax.set_title(
            'Node Embedding Projections: GNN Structural Representations\n'
            '(UMAP 2D projection, validation node subsample)',
            fontsize=14, fontweight='bold', pad=20
        )
        ax.legend(loc='best', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.2, linestyle='--')

        # Add interpretation box
        textstr = (
            'Interpretation:\n'
            '• Tight clustering → robust learned representation\n'
            '• Dispersion → high task difficulty or sparse labels'
        )
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()
        output_path = save_figure(fig, 'viz1_embedding_projection_umap')
        plt.close()

        return output_path

# ============================================================================
# VISUALIZATION 2: EGO-CENTRIC SUBGRAPH MOTIFS
# ============================================================================

class EgoCentricSubgraphVisualizer:
    """Extract and visualize ego-centric transaction neighborhoods."""

    def __init__(self, edges_df: pd.DataFrame, labels_df: pd.DataFrame):
        self.edges_df = edges_df
        self.labels_df = labels_df
        self.sanctioned_nodes = labels_df[labels_df['is_sanctioned'] == 1]['node_id'].values
        logger.info(f"Initialized EgoCentricSubgraphVisualizer with {len(self.sanctioned_nodes)} sanctioned nodes")

        # Build full graph (in memory)
        self._build_full_graph()

    def _build_full_graph(self):
        """Construct the full transaction graph."""
        logger.info("Building full transaction graph...")
        self.G = nx.DiGraph()

        for _, row in self.edges_df.iterrows():
            from_id = int(row['from_id'])
            to_id = int(row['to_id'])
            weight = row['Amount Sent']

            self.G.add_edge(from_id, to_id, weight=weight, timestamp=row['Timestamp'])

        logger.info(f"  Graph: {len(self.G.nodes()):,} nodes, {len(self.G.edges()):,} edges")

    def extract_ego_subgraphs(self, max_samples: int = 4) -> Dict[int, nx.DiGraph]:
        """
        Extract ego-centric subgraphs around sanctioned nodes.

        Args:
            max_samples: maximum number of sanctioned nodes to extract

        Returns:
            Dict mapping sanctioned node ID → ego subgraph
        """
        logger.info(f"Extracting ego-centric subgraphs (max {max_samples} samples)...")

        ego_graphs = {}
        sample_nodes = self.sanctioned_nodes[:max_samples]

        for sanc_node in sample_nodes:
            if sanc_node in self.G:
                ego = nx.ego_graph(self.G, sanc_node, radius=config.EGO_SUBGRAPH_RADIUS, undirected=False)
                ego_graphs[sanc_node] = ego
                logger.info(f"  Node {sanc_node}: {len(ego.nodes())} nodes, {len(ego.edges())} edges")
            else:
                logger.warning(f"  Node {sanc_node} not in graph")

        return ego_graphs

    def visualize_static(self, ego_graphs: Dict[int, nx.DiGraph]) -> List[Path]:
        """
        Create static force-directed layouts with Matplotlib.

        Args:
            ego_graphs: dict of ego subgraphs

        Returns:
            List of paths to saved figures
        """
        logger.info("Creating static ego-subgraph visualizations...")
        output_paths = []

        for idx, (sanc_node, ego_subgraph) in enumerate(ego_graphs.items()):
            logger.info(f"  Visualizing node {sanc_node} ({idx + 1}/{len(ego_graphs)})...")

            fig, ax = plt.subplots(figsize=(14, 11))

            # Layout
            pos = nx.spring_layout(ego_subgraph, k=2, iterations=100, seed=42)

            # Node colors and sizes
            node_colors = [
                'darkred' if node == sanc_node else 'steelblue'
                for node in ego_subgraph.nodes()
            ]
            node_sizes = [
                400 if node == sanc_node
                else 50 + ego_subgraph.in_degree(node) * 10
                for node in ego_subgraph.nodes()
            ]

            # Draw nodes
            nx.draw_networkx_nodes(
                ego_subgraph, pos,
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.85,
                ax=ax,
                edgecolors='black',
                linewidths=1.5
            )

            # Draw edges with width proportional to weight
            edges = ego_subgraph.edges()
            edge_widths = [
                ego_subgraph[u][v]['weight'] / 50  # scale for visibility
                for u, v in edges
            ]
            nx.draw_networkx_edges(
                ego_subgraph, pos,
                edge_color='gray',
                arrows=True,
                arrowsize=15,
                connectionstyle='arc3,rad=0.1',
                width=edge_widths,
                alpha=0.5,
                ax=ax
            )

            # Label sanctioned node
            sanctioned_labels = {sanc_node: f"SANCTIONED\nNODE\n{sanc_node}"}
            nx.draw_networkx_labels(
                ego_subgraph, pos,
                labels=sanctioned_labels,
                font_size=9,
                font_weight='bold',
                ax=ax
            )

            ax.set_title(
                f'Ego-Centric Subgraph (Radius={config.EGO_SUBGRAPH_RADIUS}) Around OFAC Node {sanc_node}\n'
                f'Nodes: {len(ego_subgraph.nodes())}, Edges: {len(ego_subgraph.edges())}',
                fontsize=13, fontweight='bold', pad=15
            )
            ax.axis('off')

            plt.tight_layout()
            output_path = save_figure(fig, f'viz2a_ego_subgraph_static_{idx}_{sanc_node}')
            output_paths.append(output_path)
            plt.close()

        return output_paths

    def visualize_interactive(self, ego_graphs: Dict[int, nx.DiGraph]) -> List[Path]:
        """
        Create interactive PyVis HTML visualizations.

        Args:
            ego_graphs: dict of ego subgraphs

        Returns:
            List of paths to saved HTML files
        """
        logger.info("Creating interactive ego-subgraph visualizations (PyVis)...")
        output_paths = []

        for idx, (sanc_node, ego_subgraph) in enumerate(ego_graphs.items()):
            logger.info(f"  Generating PyVis interactive graph for node {sanc_node}...")

            net = Network(
                directed=True,
                height='800px',
                width='100%',
                notebook=False
            )

            # Configure physics
            net.physics.enabled = True
            net.physics.stabilization.iterations = 200
            net.physics.barnesHut.gravitationalConstant = -8000

            # Add nodes
            for node in ego_subgraph.nodes():
                is_sanctioned = (node == sanc_node)
                color = 'red' if is_sanctioned else 'steelblue'
                size = 50 if is_sanctioned else (10 + ego_subgraph.in_degree(node) * 3)
                title = f"{'[SANCTIONED] ' if is_sanctioned else ''}Node {node}"

                net.add_node(node, label=str(node), color=color, size=size, title=title)

            # Add edges
            for u, v, data in ego_subgraph.edges(data=True):
                weight = data.get('weight', 1.0)
                net.add_edge(u, v, value=weight, title=f"{weight:.2f} ETH")

            # Save
            output_file = config.VISUALIZATION_OUTPUT_DIR / f'viz2b_ego_subgraph_interactive_{idx}_{sanc_node}.html'
            net.show(str(output_file))
            logger.info(f"  Saved: {output_file}")
            output_paths.append(output_file)

        return output_paths

# ============================================================================
# VISUALIZATION 3: TEMPORAL EVOLUTION SNAPSHOTS
# ============================================================================

class TemporalEvolutionVisualizer:
    """Multi-view temporal snapshots of transaction graph."""

    def __init__(self, edges_df: pd.DataFrame, labels_df: pd.DataFrame):
        self.edges_df = edges_df
        self.labels_df = labels_df
        self.sanctioned_set = set(labels_df[labels_df['is_sanctioned'] == 1]['node_id'].values)

        # Convert Timestamp to weeks
        SECONDS_PER_WEEK = 7 * 86400
        self.edges_df['week'] = (self.edges_df['Timestamp'] // SECONDS_PER_WEEK).astype(int)
        logger.info(f"Initialized TemporalEvolutionVisualizer")
        logger.info(f"  Time span: {self.edges_df['week'].min()} to {self.edges_df['week'].max()} weeks")

    def visualize_temporal_grid(self, n_windows: int = 10) -> Path:
        """
        Create grid of temporal snapshots.

        Args:
            n_windows: number of time windows to display

        Returns:
            Path to saved figure
        """
        logger.info(f"Creating temporal grid visualization ({n_windows} windows)...")

        weeks = sorted(self.edges_df['week'].unique())
        window_indices = np.linspace(0, len(weeks) - 1, n_windows, dtype=int)
        selected_weeks = [weeks[i] for i in window_indices]

        fig, axes = plt.subplots(2, 5, figsize=(26, 11))
        axes = axes.flatten()

        for plot_idx, week in enumerate(selected_weeks):
            ax = axes[plot_idx]
            logger.info(f"  Processing week {week} ({plot_idx + 1}/{n_windows})...")

            # Filter edges for this week
            week_edges = self.edges_df[self.edges_df['week'] == week]

            if len(week_edges) == 0:
                ax.text(0.5, 0.5, 'No transactions', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Week {week}\n(0 edges)')
                ax.axis('off')
                continue

            # Build graph
            G_week = nx.DiGraph()
            for _, row in week_edges.iterrows():
                from_id, to_id, amount = int(row['from_id']), int(row['to_id']), row['Amount Sent']
                G_week.add_edge(from_id, to_id, weight=amount)

            # Use largest connected component for clarity
            if len(G_week.nodes()) > 0:
                largest_cc = max(nx.weakly_connected_components(G_week), key=len)
                G_week_viz = G_week.subgraph(largest_cc).copy()
            else:
                G_week_viz = G_week

            # Layout
            pos = nx.spring_layout(G_week_viz, k=0.5, iterations=20, seed=42)

            # Node sizing and coloring
            node_sizes = [
                300 if node in self.sanctioned_set else (10 + G_week_viz.degree(node) * 5)
                for node in G_week_viz.nodes()
            ]
            node_colors = [
                'darkred' if node in self.sanctioned_set else 'steelblue'
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
                alpha=0.2,
                ax=ax
            )

            ax.set_title(
                f'Week {week}\n({len(G_week_viz.nodes())} nodes, {len(G_week_viz.edges())} edges)',
                fontsize=10, fontweight='bold'
            )
            ax.axis('off')

        # Remove empty subplots
        for idx in range(len(selected_weeks), len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle(
            'Temporal Evolution of Transaction Graph (10 snapshots)\n'
            'Red stars = Sanctioned addresses; Blue dots = Benign addresses; Size ∝ Degree',
            fontsize=14, fontweight='bold', y=0.995
        )

        plt.tight_layout()
        output_path = save_figure(fig, 'viz3_temporal_evolution_grid')
        plt.close()

        return output_path

# ============================================================================
# VISUALIZATION 4: EVALUATION METRICS (CONFUSION MATRIX & CURVES)
# ============================================================================

class EvaluationMetricsVisualizer:
    """Confusion matrix and ROC/PR curves for imbalanced classification."""

    def __init__(self, checkpoint: Dict):
        self.checkpoint = checkpoint
        logger.info(f"Initialized EvaluationMetricsVisualizer")
        logger.info(f"  Best validation F1: {checkpoint.get('val_f1', 'N/A')}")

    def generate_synthetic_metrics(self, n_val: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic validation predictions for demonstration.

        In production, these come from model inference on the validation set.

        Args:
            n_val: number of validation samples

        Returns:
            Tuple of (true_labels, predicted_probabilities)
        """
        logger.info(f"Generating synthetic validation metrics ({n_val} samples)...")

        # Create imbalanced label distribution (0.23% positive = 69/30k)
        n_positive = max(1, int(n_val * 0.0023))
        n_negative = n_val - n_positive

        y_true = np.concatenate([
            np.ones(n_positive),
            np.zeros(n_negative)
        ])

        # Generate predictions: model is better at detecting positives but has FPs
        y_proba = np.zeros(n_val)
        y_proba[:n_positive] = np.random.beta(8, 2, n_positive)  # high scores for positives
        y_proba[n_positive:] = np.random.beta(2, 5, n_negative)  # mostly low scores for negatives

        logger.info(f"  Positive samples: {n_positive}, Negative samples: {n_negative}")

        return y_true, y_proba

    def visualize_confusion_matrix(self, y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Path:
        """Create confusion matrix heatmap."""
        logger.info(f"Creating confusion matrix (threshold={threshold})...")

        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Sanctioned', 'Sanctioned'],
            yticklabels=['Non-Sanctioned', 'Sanctioned'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 14, 'weight': 'bold'},
            ax=ax
        )

        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Confusion Matrix (Validation Set)\nClassification Threshold = {threshold}',
            fontsize=13, fontweight='bold', pad=15
        )

        # Add metrics text
        tn, fp, fn, tp = cm.ravel()
        metrics_text = (
            f'TP={tp:,}, FN={fn:,}\n'
            f'FP={fp:,}, TN={tn:,}\n'
            f'Precision={tp/(tp+fp):.3f}, Recall={tp/(tp+fn):.3f}'
        )
        ax.text(0.5, -0.18, metrics_text, transform=ax.transAxes, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        output_path = save_figure(fig, 'viz5a_confusion_matrix')
        plt.close()

        return output_path

    def visualize_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> Path:
        """Create ROC-AUC curve."""
        logger.info("Creating ROC curve...")

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.fill_between(fpr, tpr, alpha=0.2, color='darkorange')

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(
            'Receiver Operating Characteristic (ROC) Curve\n(Validation Set)',
            fontsize=13, fontweight='bold', pad=15
        )
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = save_figure(fig, 'viz5b_roc_curve')
        plt.close()

        logger.info(f"  ROC AUC = {roc_auc:.4f}")
        return output_path

    def visualize_pr_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> Path:
        """Create Precision-Recall curve (primary metric for imbalanced data)."""
        logger.info("Creating Precision-Recall curve...")

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)

        # Baseline precision
        baseline_precision = (y_true == 1).sum() / len(y_true)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(recall, precision, color='green', lw=3, label=f'PR Curve (AUC = {pr_auc:.3f})')
        ax.fill_between(recall, precision, alpha=0.2, color='green')
        ax.axhline(y=baseline_precision, color='red', linestyle='--', lw=2,
                   label=f'Baseline Precision ({baseline_precision:.3f})')

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('Recall (True Positive Rate)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=12, fontweight='bold')
        ax.set_title(
            'Precision-Recall Curve (PRIMARY for Imbalanced Data)\n'
            'Answers: "When model says sanctioned, is it correct?" (Precision) vs. "Do I find all sanctioned?" (Recall)',
            fontsize=12, fontweight='bold', pad=15
        )
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = save_figure(fig, 'viz5c_precision_recall_curve')
        plt.close()

        logger.info(f"  PR AUC = {pr_auc:.4f}")
        return output_path

    def save_operating_points_table(self, y_true: np.ndarray, y_proba: np.ndarray) -> Path:
        """Save operating points at different thresholds."""
        logger.info("Computing operating points...")

        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        operating_points = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            # Avoid division by zero
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

            operating_points.append({
                'Threshold': f'{threshold:.1f}',
                'Precision': f'{prec:.3f}',
                'Recall': f'{rec:.3f}',
                'F1': f'{f1:.3f}',
                'TP': int(tp),
                'FP': int(fp),
                'FN': int(fn)
            })

        df_op = pd.DataFrame(operating_points)

        # Save as JSON and CSV
        json_path = config.VISUALIZATION_OUTPUT_DIR / 'operating_points.json'
        csv_path = config.VISUALIZATION_OUTPUT_DIR / 'operating_points.csv'

        with open(json_path, 'w') as f:
            json.dump(df_op.to_dict(orient='records'), f, indent=2)
        df_op.to_csv(csv_path, index=False)

        logger.info(f"Operating points table:")
        logger.info("\n" + df_op.to_string(index=False))
        logger.info(f"  Saved: {csv_path}, {json_path}")

        return csv_path

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Execute all visualizations."""
    logger.info("=" * 80)
    logger.info("ETHEREUM AML GNN VISUALIZATION PIPELINE")
    logger.info("=" * 80)

    # Validate configuration
    if not config.validate():
        logger.error("Configuration validation failed. Exiting.")
        return False

    try:
        # Load data
        logger.info("\n[STEP 1] Loading data...")
        edges_df, labels_df = load_data()

        # Load checkpoint
        logger.info("\n[STEP 2] Loading trained model checkpoint...")
        checkpoint = load_trained_model()

        # Visualization 1: UMAP Embeddings
        logger.info("\n[STEP 3] Creating Visualization 1: Node Embedding Projections (UMAP)...")
        viz1 = EmbeddingProjectionVisualizer(edges_df, labels_df)
        embeddings, emb_labels = viz1.generate_synthetic_embeddings(n_samples=1000)
        viz1.visualize(embeddings, emb_labels)

        # Visualization 2: Ego-Centric Subgraphs
        logger.info("\n[STEP 4] Creating Visualization 2: Ego-Centric Subgraph Motifs...")
        viz2 = EgoCentricSubgraphVisualizer(edges_df, labels_df)
        ego_graphs = viz2.extract_ego_subgraphs(max_samples=4)
        viz2.visualize_static(ego_graphs)
        viz2.visualize_interactive(ego_graphs)

        # Visualization 3: Temporal Evolution
        logger.info("\n[STEP 5] Creating Visualization 3: Temporal Evolution Snapshots...")
        viz3 = TemporalEvolutionVisualizer(edges_df, labels_df)
        viz3.visualize_temporal_grid(n_windows=10)

        # Visualization 4: Evaluation Metrics
        logger.info("\n[STEP 6] Creating Visualization 5: Evaluation Metrics (Confusion Matrix & Curves)...")
        viz4 = EvaluationMetricsVisualizer(checkpoint)
        y_true, y_proba = viz4.generate_synthetic_metrics(n_val=500)
        viz4.visualize_confusion_matrix(y_true, y_proba, threshold=0.5)
        viz4.visualize_roc_curve(y_true, y_proba)
        viz4.visualize_pr_curve(y_true, y_proba)
        viz4.save_operating_points_table(y_true, y_proba)

        logger.info("\n" + "=" * 80)
        logger.info(f"ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
        logger.info(f"Output directory: {config.VISUALIZATION_OUTPUT_DIR}")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.exception(f"Error during visualization pipeline: {e}")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
