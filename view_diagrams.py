#!/usr/bin/env python3
"""
Simple diagram viewer to display the generated framework diagrams.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import argparse

def view_diagrams(diagram_dir="./method_diagrams"):
    """Display all generated diagrams in a grid."""
    diagram_path = Path(diagram_dir)
    
    # Find all PNG files
    png_files = list(diagram_path.glob("*.png"))
    
    if not png_files:
        print(f"No PNG files found in {diagram_path}")
        return
    
    print(f"Found {len(png_files)} diagrams:")
    for i, file in enumerate(png_files):
        print(f"  {i+1}. {file.name}")
    
    # Display diagrams one by one
    for i, png_file in enumerate(png_files):
        plt.figure(figsize=(16, 12))
        img = mpimg.imread(str(png_file))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{png_file.stem.replace('_', ' ').title()}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Ask user if they want to continue (except for the last image)
        if i < len(png_files) - 1:
            response = input(f"\nPress Enter to view next diagram or 'q' to quit: ")
            if response.lower() == 'q':
                break

def view_single_diagram(diagram_path):
    """Display a single diagram."""
    try:
        plt.figure(figsize=(16, 12))
        img = mpimg.imread(diagram_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{Path(diagram_path).stem.replace('_', ' ').title()}", 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error loading diagram: {e}")

def main():
    parser = argparse.ArgumentParser(description="View framework diagrams")
    parser.add_argument("--dir", default="./method_diagrams", 
                       help="Directory containing diagram files")
    parser.add_argument("--file", help="Specific diagram file to view")
    
    args = parser.parse_args()
    
    if args.file:
        view_single_diagram(args.file)
    else:
        view_diagrams(args.dir)

if __name__ == "__main__":
    main()