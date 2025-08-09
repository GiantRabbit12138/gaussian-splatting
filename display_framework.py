#!/usr/bin/env python3
"""
Display the main framework diagram for demonstration.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

def display_main_framework():
    """Display the main framework diagram."""
    diagram_path = Path("./method_diagrams/main_framework.png")
    
    if not diagram_path.exists():
        print("Main framework diagram not found. Please run generate_method_framework.py first.")
        return
    
    # Create a large figure for clear display
    plt.figure(figsize=(20, 14))
    
    # Load and display the image
    img = mpimg.imread(str(diagram_path))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Enhanced 3D Gaussian Splatting Framework', fontsize=18, fontweight='bold', pad=20)
    
    # Tight layout for better presentation
    plt.tight_layout()
    
    # Save as a demo screenshot
    plt.savefig('./framework_demo.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Framework diagram displayed and saved as 'framework_demo.png'")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    display_main_framework()