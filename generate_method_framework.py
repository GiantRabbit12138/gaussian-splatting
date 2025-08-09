#!/usr/bin/env python3
"""
Framework Diagram Generator for Enhanced 3D Gaussian Splatting Method

This script generates a comprehensive framework diagram for the research paper's 
Method section, illustrating the four key innovations and their workflows.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle
import numpy as np
from pathlib import Path

class MethodFrameworkDiagram:
    """Generate publication-ready framework diagrams for the enhanced 3D Gaussian Splatting method."""
    
    def __init__(self, output_dir="./method_diagrams", figsize=(20, 14)):
        """
        Initialize the diagram generator.
        
        Args:
            output_dir: Directory to save generated diagrams
            figsize: Figure size for the main diagram
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.figsize = figsize
        
        # Define color scheme for different components
        self.colors = {
            'input': '#E8F4FD',      # Light blue for inputs
            'background': '#FFF2CC',  # Light yellow for background processing
            'noise': '#FFE6CC',      # Light orange for noise handling
            'texture': '#E1F5FE',    # Light cyan for texture optimization
            'viewpoint': '#F3E5F5',  # Light purple for viewpoint selection
            'output': '#E8F5E8',     # Light green for outputs
            'arrow': '#2196F3',      # Blue for arrows
            'text': '#333333'        # Dark gray for text
        }
        
        # Define text styling
        self.text_style = {
            'fontsize': 10,
            'ha': 'center',
            'va': 'center',
            'color': self.colors['text'],
            'fontweight': 'normal'
        }
        
        self.title_style = {
            'fontsize': 12,
            'ha': 'center', 
            'va': 'center',
            'color': self.colors['text'],
            'fontweight': 'bold'
        }

    def create_rounded_box(self, ax, xy, width, height, text, color, text_style=None, 
                          corner_radius=0.02, edge_color='black', linewidth=1.5):
        """Create a rounded rectangle box with text."""
        if text_style is None:
            text_style = self.text_style
            
        # Create rounded rectangle
        box = FancyBboxPatch(
            xy, width, height,
            boxstyle=f"round,pad=0.01,rounding_size={corner_radius}",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=linewidth,
            alpha=0.8
        )
        ax.add_patch(box)
        
        # Add text
        text_x = xy[0] + width/2
        text_y = xy[1] + height/2
        ax.text(text_x, text_y, text, **text_style, wrap=True)
        
        return box

    def create_arrow(self, ax, start, end, color=None, style='->'):
        """Create an arrow between two points."""
        if color is None:
            color = self.colors['arrow']
            
        arrow = ConnectionPatch(
            start, end, "data", "data",
            arrowstyle=style,
            shrinkA=5, shrinkB=5,
            mutation_scale=20,
            fc=color,
            ec=color,
            linewidth=2
        )
        ax.add_patch(arrow)
        return arrow

    def generate_main_framework(self):
        """Generate the main framework diagram showing all four innovations."""
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(5, 9.5, 'Enhanced 3D Gaussian Splatting Framework', 
               fontsize=16, fontweight='bold', ha='center', va='center')
        
        # Input Section (Top)
        self.create_rounded_box(ax, (0.5, 8), 2, 0.8, 
                               'Multi-view\nRGB Images', self.colors['input'], self.text_style)
        self.create_rounded_box(ax, (3, 8), 2, 0.8,
                               'Depth Maps\n(GT + Mono)', self.colors['input'], self.text_style)
        self.create_rounded_box(ax, (5.5, 8), 2, 0.8,
                               'Camera Poses\n& Parameters', self.colors['input'], self.text_style)
        self.create_rounded_box(ax, (8, 8), 1.5, 0.8,
                               'Scene Masks', self.colors['input'], self.text_style)
        
        # Innovation 1: Background Artifact Suppression
        self.create_rounded_box(ax, (0.5, 6.5), 3, 1,
                               'Background Artifact Suppression\n• Mask R-CNN Segmentation\n• Key Region Extraction',
                               self.colors['background'], self.title_style)
        
        # Innovation 2: Robustness Against Noise  
        self.create_rounded_box(ax, (4, 6.5), 3, 1,
                               'Robustness Against Noise\n• NR-IQA Optimization\n• Distortion-Resistant Network',
                               self.colors['noise'], self.title_style)
        
        # Innovation 3: Textureless Region Optimization
        self.create_rounded_box(ax, (0.5, 4.5), 3, 1,
                               'Textureless Region Optimization\n• Dense Point Cloud Init\n• Depth Information Fusion',
                               self.colors['texture'], self.title_style)
        
        # Innovation 4: Viewpoint Redundancy Reduction
        self.create_rounded_box(ax, (4, 4.5), 3, 1,
                               'Viewpoint Redundancy Reduction\n• Pose-based View Selection\n• Quality Assessment',
                               self.colors['viewpoint'], self.title_style)
        
        # Core 3D Gaussian Splatting
        self.create_rounded_box(ax, (7.5, 5.5), 2, 1.5,
                               '3D Gaussian\nSplatting\nOptimization',
                               self.colors['output'], self.title_style)
        
        # Output Section
        self.create_rounded_box(ax, (2, 2.5), 6, 1,
                               'High-Quality 3D Reconstruction\n• Enhanced Geometric Detail\n• Reduced Artifacts\n• Improved Fidelity',
                               self.colors['output'], self.title_style)
        
        # Arrows showing data flow
        # Input to processing modules
        self.create_arrow(ax, (1.5, 8), (2, 7.5))
        self.create_arrow(ax, (4, 8), (5.5, 7.5))
        self.create_arrow(ax, (6.5, 8), (8.5, 6.5))
        self.create_arrow(ax, (8.75, 8), (8.75, 7))
        
        # Processing modules to core optimization
        self.create_arrow(ax, (3.5, 6.5), (7.5, 6.25))
        self.create_arrow(ax, (7, 6.5), (7.5, 6.25))
        self.create_arrow(ax, (3.5, 4.5), (7.5, 5.75))
        self.create_arrow(ax, (7, 4.5), (7.5, 5.75))
        
        # Core to output
        self.create_arrow(ax, (8.5, 5.5), (5, 3.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'main_framework.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'main_framework.pdf', bbox_inches='tight')
        return fig

    def generate_detailed_workflow(self):
        """Generate detailed workflow diagram showing step-by-step process."""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Title
        ax.text(5, 11.5, 'Detailed Processing Workflow', 
               fontsize=16, fontweight='bold', ha='center', va='center')
        
        # Step 1: Input Processing
        ax.text(1, 10.5, 'Step 1: Input Processing', fontsize=14, fontweight='bold')
        self.create_rounded_box(ax, (0.5, 9.5), 1.8, 0.6, 'Raw Images', self.colors['input'])
        self.create_rounded_box(ax, (2.5, 9.5), 1.8, 0.6, 'Depth Maps', self.colors['input'])
        self.create_rounded_box(ax, (4.5, 9.5), 1.8, 0.6, 'Camera Data', self.colors['input'])
        
        # Step 2: Background Suppression
        ax.text(1, 8.5, 'Step 2: Background Artifact Suppression', fontsize=14, fontweight='bold')
        self.create_rounded_box(ax, (0.5, 7.5), 2.5, 0.8, 
                               'Mask R-CNN\nSegmentation', self.colors['background'])
        self.create_rounded_box(ax, (3.5, 7.5), 2.5, 0.8,
                               'Target Region\nExtraction', self.colors['background'])
        
        # Step 3: Quality Assessment & Noise Handling
        ax.text(1, 6.5, 'Step 3: Quality Assessment & Noise Handling', fontsize=14, fontweight='bold')
        self.create_rounded_box(ax, (0.5, 5.5), 2, 0.8,
                               'NR-IQA\nEvaluation', self.colors['noise'])
        self.create_rounded_box(ax, (3, 5.5), 2.5, 0.8,
                               'Distortion Network\nTraining', self.colors['noise'])
        self.create_rounded_box(ax, (6, 5.5), 2, 0.8,
                               'Noise\nSuppression', self.colors['noise'])
        
        # Step 4: Depth & Texture Optimization
        ax.text(1, 4.5, 'Step 4: Depth & Texture Enhancement', fontsize=14, fontweight='bold')
        self.create_rounded_box(ax, (0.5, 3.5), 2.2, 0.8,
                               'Dense Point Cloud\nInitialization', self.colors['texture'])
        self.create_rounded_box(ax, (3, 3.5), 2.5, 0.8,
                               'GT + Mono Depth\nFusion', self.colors['texture'])
        self.create_rounded_box(ax, (6, 3.5), 2, 0.8,
                               'Geometric Detail\nRecovery', self.colors['texture'])
        
        # Step 5: View Selection
        ax.text(1, 2.5, 'Step 5: Viewpoint Optimization', fontsize=14, fontweight='bold')
        self.create_rounded_box(ax, (0.5, 1.5), 3, 0.8,
                               'Pose-based View Selection', self.colors['viewpoint'])
        self.create_rounded_box(ax, (4, 1.5), 3, 0.8,
                               'Representative View Identification', self.colors['viewpoint'])
        
        # Step 6: Final Optimization
        ax.text(7.5, 2.5, 'Step 6: 3D Gaussian Optimization', fontsize=14, fontweight='bold')
        self.create_rounded_box(ax, (7.5, 1.5), 2, 0.8,
                               'Final 3D Model', self.colors['output'])
        
        # Arrows showing workflow
        # Step 1 to 2
        self.create_arrow(ax, (1.4, 9.5), (1.75, 8.3))
        self.create_arrow(ax, (3.4, 9.5), (4.75, 8.3))
        
        # Step 2 to 3
        self.create_arrow(ax, (1.75, 7.5), (1.5, 6.3))
        self.create_arrow(ax, (4.75, 7.5), (4.25, 6.3))
        
        # Step 3 to 4
        self.create_arrow(ax, (1.5, 5.5), (1.6, 4.3))
        self.create_arrow(ax, (4.25, 5.5), (4.25, 4.3))
        self.create_arrow(ax, (7, 5.5), (7, 4.3))
        
        # Step 4 to 5
        self.create_arrow(ax, (1.6, 3.5), (2, 2.3))
        self.create_arrow(ax, (4.25, 3.5), (5.5, 2.3))
        
        # Step 5 to 6
        self.create_arrow(ax, (5.5, 1.5), (7.5, 1.9))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'detailed_workflow.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'detailed_workflow.pdf', bbox_inches='tight')
        return fig

    def generate_innovation_details(self):
        """Generate detailed diagrams for each innovation."""
        # Innovation 1: Background Artifact Suppression
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 8)
        ax1.axis('off')
        
        ax1.text(5, 7.5, 'Innovation 1: Background Artifact Suppression', 
                fontsize=14, fontweight='bold', ha='center')
        
        # Input images
        self.create_rounded_box(ax1, (0.5, 6), 2, 0.8, 'Raw RGB\nImages', self.colors['input'])
        
        # Mask R-CNN processing
        self.create_rounded_box(ax1, (3.5, 6), 2.5, 0.8, 'Mask R-CNN\nSegmentation', self.colors['background'])
        
        # Mask extraction
        self.create_rounded_box(ax1, (7, 6), 2, 0.8, 'Object Masks', self.colors['background'])
        
        # Key region extraction
        self.create_rounded_box(ax1, (0.5, 4), 3, 0.8, 'Key Target Region\nExtraction', self.colors['background'])
        
        # Background suppression
        self.create_rounded_box(ax1, (4.5, 4), 3, 0.8, 'Background Artifact\nSuppression', self.colors['background'])
        
        # Output
        self.create_rounded_box(ax1, (2, 2), 4, 0.8, 'Clean Images with\nEliminated Background Noise', self.colors['output'])
        
        # Arrows
        self.create_arrow(ax1, (2.5, 6.4), (3.5, 6.4))
        self.create_arrow(ax1, (6, 6.4), (7, 6.4))
        self.create_arrow(ax1, (1.5, 6), (2, 4.8))
        self.create_arrow(ax1, (8, 6), (6, 4.8))
        self.create_arrow(ax1, (3, 4), (4, 2.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'innovation1_background_suppression.png', dpi=300, bbox_inches='tight')
        
        # Innovation 2: Robustness Against Noise
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 8)
        ax2.axis('off')
        
        ax2.text(5, 7.5, 'Innovation 2: Robustness Against Noise', 
                fontsize=14, fontweight='bold', ha='center')
        
        # Input
        self.create_rounded_box(ax2, (0.5, 6), 2, 0.8, 'Input Images', self.colors['input'])
        
        # NR-IQA
        self.create_rounded_box(ax2, (3.5, 6.5), 2.5, 0.6, 'NR-IQA Assessment\n(FISH Algorithm)', self.colors['noise'])
        
        # Distortion Network
        self.create_rounded_box(ax2, (3.5, 5.5), 2.5, 0.6, 'Distortion Network\nTraining', self.colors['noise'])
        
        # Noise types
        self.create_rounded_box(ax2, (7, 6.5), 2, 0.4, 'Gaussian Noise', self.colors['noise'])
        self.create_rounded_box(ax2, (7, 6), 2, 0.4, 'Blur & Artifacts', self.colors['noise'])
        self.create_rounded_box(ax2, (7, 5.5), 2, 0.4, 'JPEG Compression', self.colors['noise'])
        
        # Robust optimization
        self.create_rounded_box(ax2, (1, 3.5), 4, 0.8, 'Robust Optimization Process\nwith Noise Suppression', self.colors['noise'])
        
        # Output
        self.create_rounded_box(ax2, (6, 3.5), 3, 0.8, 'Noise-Resistant\n3D Reconstruction', self.colors['output'])
        
        # Arrows
        self.create_arrow(ax2, (2.5, 6.4), (3.5, 6.8))
        self.create_arrow(ax2, (2.5, 6.4), (3.5, 5.8))
        self.create_arrow(ax2, (6, 6.8), (7, 6.7))
        self.create_arrow(ax2, (6, 5.8), (7, 5.7))
        self.create_arrow(ax2, (4.75, 5.5), (3, 4.3))
        self.create_arrow(ax2, (5, 3.5), (6, 3.9))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'innovation2_noise_robustness.png', dpi=300, bbox_inches='tight')
        
        # Innovation 3: Textureless Region Optimization
        fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 8)
        ax3.axis('off')
        
        ax3.text(5, 7.5, 'Innovation 3: Optimization for Textureless Regions', 
                fontsize=14, fontweight='bold', ha='center')
        
        # Input depth maps
        self.create_rounded_box(ax3, (0.5, 6.5), 1.8, 0.6, 'Ground Truth\nDepth', self.colors['input'])
        self.create_rounded_box(ax3, (2.5, 6.5), 1.8, 0.6, 'Monocular\nDepth', self.colors['input'])
        
        # Depth fusion
        self.create_rounded_box(ax3, (1.5, 5.2), 2, 0.8, 'Depth Information\nFusion', self.colors['texture'])
        
        # Dense initialization
        self.create_rounded_box(ax3, (5, 6.5), 2.5, 0.6, 'Dense Point Cloud\nInitialization', self.colors['texture'])
        
        # Textureless detection
        self.create_rounded_box(ax3, (0.5, 3.8), 2.5, 0.8, 'Textureless Region\nDetection', self.colors['texture'])
        
        # Geometric enhancement
        self.create_rounded_box(ax3, (4, 3.8), 3, 0.8, 'Geometric Detail\nEnhancement', self.colors['texture'])
        
        # Enhanced regions
        self.create_rounded_box(ax3, (7.5, 3.8), 2, 0.8, 'Enhanced\nTextureless\nRegions', self.colors['texture'])
        
        # Output
        self.create_rounded_box(ax3, (2, 1.5), 4, 0.8, 'Improved Geometric\nDetail Recovery', self.colors['output'])
        
        # Arrows
        self.create_arrow(ax3, (1.4, 6.5), (2, 6))
        self.create_arrow(ax3, (3.4, 6.5), (3, 6))
        self.create_arrow(ax3, (2.5, 5.2), (1.75, 4.6))
        self.create_arrow(ax3, (6.25, 6.5), (5.5, 4.6))
        self.create_arrow(ax3, (3, 3.8), (4, 4.2))
        self.create_arrow(ax3, (7, 3.8), (7.5, 4.2))
        self.create_arrow(ax3, (5.5, 3.8), (4, 2.3))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'innovation3_textureless_optimization.png', dpi=300, bbox_inches='tight')
        
        # Innovation 4: Viewpoint Redundancy Reduction
        fig4, ax4 = plt.subplots(1, 1, figsize=(12, 8))
        ax4.set_xlim(0, 10)
        ax4.set_ylim(0, 8)
        ax4.axis('off')
        
        ax4.text(5, 7.5, 'Innovation 4: Viewpoint Redundancy Reduction', 
                fontsize=14, fontweight='bold', ha='center')
        
        # Input camera poses
        self.create_rounded_box(ax4, (0.5, 6.5), 2, 0.6, 'Camera Poses\n& Parameters', self.colors['input'])
        
        # All input views
        self.create_rounded_box(ax4, (3, 6.5), 2, 0.6, 'All Input\nViews', self.colors['input'])
        
        # Pose analysis
        self.create_rounded_box(ax4, (0.5, 5.2), 2.5, 0.8, 'Pose-based\nSimilarity Analysis', self.colors['viewpoint'])
        
        # Quality assessment
        self.create_rounded_box(ax4, (4, 5.2), 2.5, 0.8, 'Image Quality\nAssessment (FISH)', self.colors['viewpoint'])
        
        # Redundancy detection
        self.create_rounded_box(ax4, (7, 5.2), 2.5, 0.8, 'Redundant View\nDetection', self.colors['viewpoint'])
        
        # View selection
        self.create_rounded_box(ax4, (1, 3.5), 3.5, 0.8, 'Representative View\nSelection Algorithm', self.colors['viewpoint'])
        
        # Optimized views
        self.create_rounded_box(ax4, (5.5, 3.5), 3, 0.8, 'Optimized View Set\n(Reduced Redundancy)', self.colors['viewpoint'])
        
        # Output
        self.create_rounded_box(ax4, (2, 1.5), 4, 0.8, 'Efficient Training with\nOptimal Viewpoints', self.colors['output'])
        
        # Arrows
        self.create_arrow(ax4, (1.5, 6.5), (1.75, 6))
        self.create_arrow(ax4, (4, 6.5), (5.25, 6))
        self.create_arrow(ax4, (4, 6.5), (8.25, 6))
        self.create_arrow(ax4, (1.75, 5.2), (2.75, 4.3))
        self.create_arrow(ax4, (5.25, 5.2), (2.75, 4.3))
        self.create_arrow(ax4, (8.25, 5.2), (2.75, 4.3))
        self.create_arrow(ax4, (4.5, 3.5), (5.5, 3.9))
        self.create_arrow(ax4, (4, 3.5), (4, 2.3))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'innovation4_viewpoint_reduction.png', dpi=300, bbox_inches='tight')
        
        return [fig1, fig2, fig3, fig4]

    def generate_all_diagrams(self):
        """Generate all framework diagrams."""
        print("Generating Enhanced 3D Gaussian Splatting Framework Diagrams...")
        
        # Generate main framework
        print("Creating main framework diagram...")
        main_fig = self.generate_main_framework()
        
        # Generate detailed workflow
        print("Creating detailed workflow diagram...")
        workflow_fig = self.generate_detailed_workflow()
        
        # Generate innovation details
        print("Creating innovation detail diagrams...")
        innovation_figs = self.generate_innovation_details()
        
        print(f"All diagrams saved to: {self.output_dir}")
        print("Generated files:")
        for file in self.output_dir.glob("*.png"):
            print(f"  - {file.name}")
        for file in self.output_dir.glob("*.pdf"):
            print(f"  - {file.name}")
        
        return [main_fig, workflow_fig] + innovation_figs

def main():
    """Main function to generate all framework diagrams."""
    generator = MethodFrameworkDiagram()
    figures = generator.generate_all_diagrams()
    
    # Show the main framework diagram
    plt.show()

if __name__ == "__main__":
    main()