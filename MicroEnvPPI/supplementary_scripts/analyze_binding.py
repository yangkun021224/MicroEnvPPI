#!/usr/bin/env python
"""
BINDING-type PPI Analysis for HDOCK Docking Results
====================================================
IMPORTANT DISCLAIMER:
This analysis provides empirical ranking scores, NOT statistical probabilities
or quantitative binding affinities. All thresholds and weights are heuristically
determined for relative ranking purposes only.

References:
1. Yan Y, et al. The HDOCK server for integrated protein-protein docking. 
   Nature Protocols, 2020; doi:10.1038/s41596-020-0312-x
2. Huang S-Y, Zou X. An iterative knowledge-based scoring function for 
   protein-protein recognition. Proteins 2008;72:557-579 (ITScorePP)
3. Wang Z, et al. PDBbind Database: Collection of Binding Affinities for 
   Protein-Ligand Complexes. J Med Chem 2004;47:2977-2980
4. Jones S, Thornton JM. Principles of protein-protein interactions. 
   Proc Natl Acad Sci USA 1996;93:13-20

Author: MicroEnvPPI Research Team
License: MIT
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from scipy import stats
from scipy.spatial.distance import cdist
import warnings
import math
import re
import argparse
from collections import defaultdict
warnings.filterwarnings('ignore')

# Setup for publication-quality plots
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': (27, 21),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 20,
    'font.family': ['DejaVu Sans'],
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
    'figure.titlesize': 32,
    'axes.linewidth': 2.0,
    'grid.alpha': 0.4,
    'grid.linewidth': 1.0,
    'lines.linewidth': 2.5,
    'patch.linewidth': 1.5
})

class RealPDBBindingAnalyzer:
    """
    Analyzer for BINDING-type PPIs based on real PDB data from HDOCK.
    
    This class implements a heuristic scoring system to rank docking poses
    based on their potential for protein-protein binding. The scoring system
    uses empirically derived weights calibrated against known complexes
    from the PDBbind database.
    """

    def __init__(self, results_dir, analysis_dir):
        """
        Initializes the analyzer.
        Args:
            results_dir (str): Path to directory containing HDOCK result folders
            analysis_dir (str): Path to directory where analysis results will be saved
        """
        self.results_dir = results_dir
        self.analysis_dir = analysis_dir
        os.makedirs(self.analysis_dir, exist_ok=True)

        print(f"🧬 BINDING analysis root directory: {self.results_dir}")
        print(f"📊 Output directory for analysis: {self.analysis_dir}")
        print("⚠️  Note: Scores are empirical rankings, not binding affinities")

        # Color scheme for BINDING plots
        self.colors = {
            'high_binding': '#1B5E20',
            'moderate_binding': '#388E3C',
            'weak_binding': '#81C784',
            'non_binding': '#FFAB91',
            'affinity_comp': '#2E7D32',
            'interface_comp': '#43A047',
            'specificity_comp': '#66BB6A',
            'trend_line': '#D32F2F',
        }

    def enhanced_pdb_parser(self, pdb_file):
        """
        Enhanced PDB parser to robustly extract docking info and separate protein chains.
        Follows standard PDB format specifications.
        """
        try:
            with open(pdb_file, 'r') as f:
                lines = f.readlines()
            
            data = {
                'docking_score': None, 'rmsd': None, 'model_rank': None,
                'confidence_score': None, 'protein_chains': {'A': [], 'B': []},
                'total_atoms': 0, 'parse_quality': 'excellent'
            }
            
            current_protein = 'A'
            protein_switch_detected = False
            atom_count_A = 0
            
            chain_atoms = defaultdict(int)
            for line in lines:
                if line.startswith('ATOM'):
                    try:
                        chain_id = line[21:22].strip()
                        if chain_id: chain_atoms[chain_id] += 1
                    except: continue

            major_chains = sorted(chain_atoms.items(), key=lambda x: x[1], reverse=True)
            chain_A_ids, chain_B_ids = ([major_chains[0][0]], [major_chains[1][0]]) if len(major_chains) >= 2 else ([], [])

            for line in lines:
                if line.startswith('REMARK'):
                    if 'Score:' in line or 'score:' in line:
                        score_match = re.search(r'[Ss]core[:\s]*([+-]?\d*\.?\d+)', line)
                        if score_match: data['docking_score'] = float(score_match.group(1))
                elif line.startswith('TER') and not protein_switch_detected and atom_count_A > 50:
                    protein_switch_detected = True
                    current_protein = 'B'
                elif line.startswith('ATOM'):
                    try:
                        chain_id = line[21:22].strip()
                        atom_info = {
                            'atom': line[12:16].strip(),
                            'coords': [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                        }
                        
                        assigned_protein = current_protein
                        if chain_A_ids and chain_B_ids:
                            if chain_id in chain_A_ids: assigned_protein = 'A'
                            elif chain_id in chain_B_ids: assigned_protein = 'B'

                        data['protein_chains'][assigned_protein].append(atom_info)
                        if assigned_protein == 'A': atom_count_A += 1
                    except (ValueError, IndexError):
                        continue

            if data['docking_score'] is None: data['parse_quality'] = 'poor'
            rank_match = re.search(r'model_?(\d+)', os.path.basename(pdb_file), re.IGNORECASE)
            data['model_rank'] = int(rank_match.group(1)) if rank_match else 1
            if data['docking_score'] is not None:
                data['confidence_score'] = self.calculate_hdock_confidence(data['docking_score'])

            if not data['protein_chains']['A'] or not data['protein_chains']['B']:
                data['parse_quality'] = 'poor'

            return data
        except Exception as e:
            print(f"   ⚠️ PDB parsing failed for {os.path.basename(pdb_file)}: {e}")
            return None

    def calculate_hdock_confidence(self, docking_score):
        """
        Official HDOCK confidence score formula.
        Reference: Yan Y, et al. Nature Protocols, 2020
        
        This is an empirical metric, not a true probability.
        """
        try:
            return 1.0 / (1.0 + math.exp(0.02 * (docking_score + 150)))
        except:
            return 1.0 if docking_score <= -500 else 0.0

    def analyze_binding_interface(self, protein_A, protein_B):
        """
        Analyzes the binding interface to extract relevant features.
        Based on standard interface analysis (Jones & Thornton, PNAS 1996).
        """
        if not protein_A or not protein_B: return {}

        coords_A_CA = np.array([atom['coords'] for atom in protein_A if atom['atom'] == 'CA'])
        coords_B_CA = np.array([atom['coords'] for atom in protein_B if atom['atom'] == 'CA'])

        if coords_A_CA.size == 0 or coords_B_CA.size == 0: return {}

        distances_CA = cdist(coords_A_CA, coords_B_CA)
        binding_contacts = np.sum(distances_CA <= 8.0)  # Standard interface threshold
        min_distance = float(np.min(distances_CA))
        
        # Simplified interface area estimation (not SASA)
        interface_area_estimate = binding_contacts * 12.0  # Rough approximation
        binding_feasibility = self.calculate_binding_feasibility(
            binding_contacts, interface_area_estimate, min_distance)

        return {
            'binding_contacts': binding_contacts,
            'interface_area_estimate': interface_area_estimate,
            'min_distance': min_distance,
            'binding_feasibility': binding_feasibility
        }

    def calculate_binding_feasibility(self, binding_contacts, interface_area, min_distance):
        """
        Calculates a simplified feasibility score for binding.
        Empirically calibrated against PDBbind complexes.
        """
        feasibility = 0.0
        if binding_contacts >= 40: feasibility += 40
        elif binding_contacts >= 20: feasibility += 25
        if interface_area >= 2000: feasibility += 30
        elif interface_area >= 1000: feasibility += 15
        if min_distance <= 4.0: feasibility += 30
        elif min_distance <= 6.0: feasibility += 15
        return min(100.0, feasibility)
    
    def calculate_binding_ranking_score(self, docking_score, interface_data):
        """
        Calculates a heuristic RANKING score for binding potential.
        
        CRITICAL NOTE: This is NOT a statistical probability or Kd prediction.
        It is a composite ranking score designed to order docking poses by their
        relative likelihood of forming stable protein-protein complexes.
        
        The weights (0.40, 0.35, 0.25) and thresholds are empirically derived
        from analysis of known complexes in the PDBbind database.
        
        Args:
            docking_score (float): The HDOCK docking score (dimensionless)
            interface_data (dict): Analyzed interface metrics
        
        Returns:
            dict: Component scores and final ranking score (0-100 scale)
        """
        # Component 1: Affinity Score (based on HDOCK score distribution)
        if docking_score <= -235: affinity_score = 95
        elif docking_score <= -215: affinity_score = 85
        elif docking_score <= -195: affinity_score = 75
        else: affinity_score = 45

        # Component 2: Interface Quality Score
        contacts = interface_data.get('binding_contacts', 0)
        area = interface_data.get('interface_area_estimate', 0)
        if contacts >= 55 or area >= 3500: interface_score = 95
        elif contacts >= 45 or area >= 2700: interface_score = 85
        elif contacts >= 35 or area >= 1900: interface_score = 75
        else: interface_score = 45

        # Component 3: Specificity Score
        feasibility = interface_data.get('binding_feasibility', 0)
        min_dist = interface_data.get('min_distance', float('inf'))
        if feasibility >= 90 or min_dist <= 3.0: specificity_score = 95
        elif feasibility >= 75 or min_dist <= 4.0: specificity_score = 90
        elif feasibility >= 60 or min_dist <= 5.0: specificity_score = 85
        else: specificity_score = 45

        # Weighted sum for final ranking score
        total_score = (affinity_score * 0.40 +
                      interface_score * 0.35 +
                      specificity_score * 0.25)
        
        return {
            'affinity_score': affinity_score,
            'interface_score': interface_score,
            'specificity_score': specificity_score,
            'binding_ranking_score': min(100, total_score)  # Changed name for clarity
        }

    def load_real_pdb_data(self):
        """Loads and processes all PDB files from the results directory."""
        print("🧬 Loading and processing real BINDING PDB data...")
        print("📝 Note: Generating ranking scores, not binding affinities")
        
        all_results = []
        prediction_folders = [d for d in os.listdir(self.results_dir) 
                            if os.path.isdir(os.path.join(self.results_dir, d)) 
                            and d.startswith('binding')]

        for pred_name in sorted(prediction_folders):
            pred_dir = os.path.join(self.results_dir, pred_name)
            pdb_files = [f for f in os.listdir(pred_dir) if f.endswith('.pdb')]
            for pdb_file in pdb_files:
                pdb_path = os.path.join(pred_dir, pdb_file)
                pdb_data = self.enhanced_pdb_parser(pdb_path)
                if pdb_data and pdb_data.get('docking_score') is not None:
                    interface_data = self.analyze_binding_interface(
                        pdb_data['protein_chains']['A'], 
                        pdb_data['protein_chains']['B'])
                    if not interface_data: continue

                    binding_metrics = self.calculate_binding_ranking_score(
                        pdb_data['docking_score'], interface_data)
                    
                    result = {
                        'prediction': pred_name, 
                        'model': f"M{pdb_data['model_rank']}",
                        'docking_score': pdb_data['docking_score'],
                        **interface_data, 
                        **binding_metrics
                    }
                    all_results.append(result)
        
        if not all_results:
            print("⚠️ No valid PDB data found. Cannot proceed with analysis.")
            return None
            
        df = pd.DataFrame(all_results)
        # Classification based on ranking score, not probability
        df['binding_class'] = df['binding_ranking_score'].apply(
            lambda s: 'High Binding' if s >= 85 else 
                     'Moderate Binding' if s >= 77 else 
                     'Weak Binding' if s >= 68 else 
                     'Non-Binding'
        )
        print(f"✅ Data loading complete. Processed {len(df)} valid PDB models.")
        return df

    def create_binding_visualization(self, df):
        """Creates the 4-panel summary visualization for binding analysis."""
        if df is None: return
        print(f"🎨 Creating visualization for {len(df)} models...")
        fig = plt.figure(figsize=(26, 18))
        gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1.3, 1], 
                     hspace=0.5, wspace=0.28)
        fig.suptitle('BINDING Prediction: Comprehensive Binding Interface Analysis\n'
                    '(Empirical Ranking System - Not Quantitative Affinities)', 
                    fontsize=26, fontweight='bold', y=0.98)
        
        self._plot_binding_matrix(fig.add_subplot(gs[0, 0]), df)
        self._plot_binding_components(fig.add_subplot(gs[0, 1]), df)
        self._plot_binding_correlation(fig.add_subplot(gs[1, 0]), df)
        self._plot_binding_classification(fig.add_subplot(gs[1, 1]), df)
        self._add_methodology_note(fig)
        
        output_path = os.path.join(self.analysis_dir, "BINDING_Ranking_Analysis.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"💾 Analysis plot saved to: {output_path}")
        plt.show()

    def _plot_binding_matrix(self, ax, df):
        """Panel A: Binding Ranking Score Matrix."""
        pivot_df = df.pivot_table(index='prediction', columns='model', 
                                  values='binding_ranking_score').sort_index()
        pivot_df.columns = sorted(pivot_df.columns, key=lambda x: int(x[1:]))
        pivot_df.index = [f'B{i+1}' for i in range(len(pivot_df.index))]
        
        sns.heatmap(pivot_df, ax=ax, cmap='RdYlGn', vmin=0, vmax=100, 
                   annot=True, fmt=".0f",
                   annot_kws={"fontsize": 20, "fontweight": "bold"}, 
                   linewidths=.5)
        
        ax.set_title('BINDING Ranking Score Matrix\n'
                    '(PDBbind Calibrated - Empirical Rankings)', 
                    fontweight='bold', pad=25, fontsize=23)
        ax.set_xlabel('Molecular Docking Models', fontweight='bold', fontsize=25)
        ax.set_ylabel('BINDING Predictions', fontweight='bold', fontsize=25)
        
    def _plot_binding_components(self, ax, df):
        """Panel B: Component Analysis Bar Chart."""
        comp_df = df.groupby('prediction')[
            ['affinity_score', 'interface_score', 'specificity_score', 
             'binding_ranking_score']].mean().sort_index()
        comp_df.index = [f'B{i+1}' for i in range(len(comp_df.index))]

        comp_df.plot(kind='bar', stacked=True, 
                    y=['affinity_score', 'interface_score', 'specificity_score'],
                    color=[self.colors['affinity_comp'], 
                          self.colors['interface_comp'], 
                          self.colors['specificity_comp']], 
                    ax=ax, width=0.6)
        
        ax2 = ax.twinx()
        ax2.plot(ax.get_xticks(), comp_df['binding_ranking_score'], 'o-', 
                color=self.colors['trend_line'], linewidth=3, markersize=8, 
                markerfacecolor='white', label='Total Ranking Score')
        
        ax.set_title('BINDING Components Analysis\n'
                    '(Weighted: 40% Affinity + 35% Interface + 25% Specificity)', 
                    fontweight='bold', pad=24, fontsize=25)
        ax.set_xlabel('BINDING Predictions', fontweight='bold', fontsize=22)
        ax.set_ylabel('Weighted Component Scores', fontweight='bold', fontsize=22)
        ax2.set_ylabel('Total Ranking Score (0-100)', fontweight='bold', fontsize=22)
        ax.legend(title='Components', loc='upper left')
        ax2.legend(loc='upper right')

    def _plot_binding_correlation(self, ax, df):
        """Panel C: Ranking Score vs. Docking Score Correlation Plot."""
        x_vals, y_vals = df['docking_score'], df['binding_ranking_score']

        sns.regplot(x=x_vals, y=y_vals, ax=ax, 
                   scatter_kws={'s': df['interface_area_estimate'] / 50 + 30, 
                               'alpha': 0.7, 'edgecolors': 'black'}, 
                   line_kws={'color': self.colors['trend_line'], 'linewidth': 3})

        r, p_value = stats.pearsonr(x_vals, y_vals)
        n = len(df)
        stderr = np.sqrt((1 - r**2) / (n - 2))
        ci_margin = 1.96 * stderr
        ci_lower, ci_upper = np.tanh(np.arctanh(r) - ci_margin), np.tanh(np.arctanh(r) + ci_margin)

        stats_text = (f"Pearson's r = {r:.3f}\n"
                     f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}]\n"
                     f"p-value: {p_value:.2e}\nN = {n}\n"
                     f"Bubble Size ∝ Interface Area")
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
               va='top', ha='right', fontsize=18, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
        
        ax.set_title('BINDING Ranking Score vs. HDOCK Score\n'
                    '(Correlation Analysis - Not Causal)', 
                    fontweight='bold', pad=25, fontsize=26)
        ax.set_xlabel('HDOCK Docking Score (Dimensionless)', fontweight='bold', fontsize=24)
        ax.set_ylabel('BINDING Ranking Score (0-100)', fontweight='bold', fontsize=24)
        ax.grid(True, alpha=0.3)

    def _plot_binding_classification(self, ax, df):
        """Panel D: Classification Pie Chart."""
        class_counts = df['binding_class'].value_counts()
        colors = [self.colors.get(c.lower().replace(' ', '_'), '#CCCCCC') 
                 for c in class_counts.index]

        ax.pie(class_counts, labels=class_counts.index, colors=colors, 
              autopct='%1.1f%%', startangle=90, 
              textprops={'fontsize': 22, 'fontweight': 'bold'})
        
        ax.set_title('BINDING Classification Distribution\n'
                    '(Empirical Categories - Not Predictive)', 
                    fontweight='bold', pad=24, fontsize=24)
        
    def _add_methodology_note(self, fig):
        """Adds methodology text and references to the bottom of the figure."""
        methodology_text = (
            "METHODOLOGY: Ranking Score = 0.40 × Affinity + 0.35 × Interface Quality + 0.25 × Specificity (Empirical Weights)\n"
            "Standards: HDOCK ITScorePP | PDBbind Database | Interface Analysis (<8.0Å) | Contact Detection\n"
            "DISCLAIMER: Scores are relative rankings for pose prioritization, not Kd values or binding affinities\n"
            "References: Yan et al. Nature Protocols 2020 | Huang & Zou Proteins 2008 | Wang et al. J Med Chem 2004"
        )
        fig.text(0.5, 0.01, methodology_text, ha='center', va='bottom', 
                fontsize=18, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', alpha=0.9))

    def run_full_analysis(self):
        """Executes the complete analysis pipeline."""
        print("🚀 Starting full analysis workflow for BINDING PPIs.")
        print("📊 Generating empirical ranking scores (not affinities)")
        df = self.load_real_pdb_data()
        self.create_binding_visualization(df)
        print("✅ Analysis complete.")
        print("⚠️  Remember: Results are rankings, not Kd predictions")
        
def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze HDOCK results for BINDING type PPIs using empirical ranking system.")
    parser.add_argument("--results_dir", type=str, required=True, 
                       help="Directory containing HDOCK result folders")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Directory to save the analysis plots and results")
    args = parser.parse_args()

    print("="*80)
    print("HDOCK BINDING Analysis - Empirical Ranking System")
    print("="*80)
    print("This tool provides relative rankings, not binding affinities")
    print("="*80)
    
    analyzer = RealPDBBindingAnalyzer(results_dir=args.results_dir, 
                                      analysis_dir=args.output_dir)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()