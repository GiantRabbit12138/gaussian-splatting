#!/usr/bin/env python3
"""
Test script to validate the framework diagram generation.
"""

import unittest
from pathlib import Path
import sys
import os

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from generate_method_framework import MethodFrameworkDiagram

class TestFrameworkDiagrams(unittest.TestCase):
    """Test the framework diagram generation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_output_dir = Path("./test_diagrams")
        self.generator = MethodFrameworkDiagram(output_dir=self.test_output_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove test files if they exist
        if self.test_output_dir.exists():
            for file in self.test_output_dir.glob("*"):
                file.unlink()
            self.test_output_dir.rmdir()
    
    def test_generator_initialization(self):
        """Test that the generator initializes correctly."""
        self.assertIsInstance(self.generator, MethodFrameworkDiagram)
        self.assertEqual(self.generator.output_dir, self.test_output_dir)
        self.assertTrue(self.test_output_dir.exists())
    
    def test_main_framework_generation(self):
        """Test main framework diagram generation."""
        fig = self.generator.generate_main_framework()
        self.assertIsNotNone(fig)
        
        # Check if files are created
        png_file = self.test_output_dir / "main_framework.png"
        pdf_file = self.test_output_dir / "main_framework.pdf"
        self.assertTrue(png_file.exists())
        self.assertTrue(pdf_file.exists())
        
        # Check file sizes (should be non-zero)
        self.assertGreater(png_file.stat().st_size, 0)
        self.assertGreater(pdf_file.stat().st_size, 0)
    
    def test_detailed_workflow_generation(self):
        """Test detailed workflow diagram generation."""
        fig = self.generator.generate_detailed_workflow()
        self.assertIsNotNone(fig)
        
        # Check if files are created
        png_file = self.test_output_dir / "detailed_workflow.png"
        pdf_file = self.test_output_dir / "detailed_workflow.pdf"
        self.assertTrue(png_file.exists())
        self.assertTrue(pdf_file.exists())
    
    def test_innovation_details_generation(self):
        """Test innovation detail diagrams generation."""
        figs = self.generator.generate_innovation_details()
        self.assertEqual(len(figs), 4)  # Should generate 4 innovation diagrams
        
        # Check if all innovation files are created
        expected_files = [
            "innovation1_background_suppression.png",
            "innovation2_noise_robustness.png",
            "innovation3_textureless_optimization.png",
            "innovation4_viewpoint_reduction.png"
        ]
        
        for filename in expected_files:
            file_path = self.test_output_dir / filename
            self.assertTrue(file_path.exists(), f"File {filename} was not created")
            self.assertGreater(file_path.stat().st_size, 0, f"File {filename} is empty")
    
    def test_all_diagrams_generation(self):
        """Test complete diagram generation."""
        figs = self.generator.generate_all_diagrams()
        self.assertGreater(len(figs), 0)
        
        # Check that all expected files exist
        expected_files = [
            "main_framework.png", "main_framework.pdf",
            "detailed_workflow.png", "detailed_workflow.pdf",
            "innovation1_background_suppression.png",
            "innovation2_noise_robustness.png", 
            "innovation3_textureless_optimization.png",
            "innovation4_viewpoint_reduction.png"
        ]
        
        for filename in expected_files:
            file_path = self.test_output_dir / filename
            self.assertTrue(file_path.exists(), f"Expected file {filename} was not created")
    
    def test_color_scheme(self):
        """Test that color scheme is properly defined."""
        expected_colors = ['input', 'background', 'noise', 'texture', 'viewpoint', 'output', 'arrow', 'text']
        for color_key in expected_colors:
            self.assertIn(color_key, self.generator.colors)
            self.assertIsInstance(self.generator.colors[color_key], str)
            self.assertTrue(self.generator.colors[color_key].startswith('#'))

def run_diagram_validation():
    """Run validation to ensure all diagrams are properly generated."""
    print("Running framework diagram validation...")
    
    # Check if main diagram directory exists and has expected files
    diagram_dir = Path("./method_diagrams")
    if not diagram_dir.exists():
        print("ERROR: method_diagrams directory does not exist!")
        return False
    
    expected_files = [
        "main_framework.png", "main_framework.pdf",
        "detailed_workflow.png", "detailed_workflow.pdf",
        "innovation1_background_suppression.png",
        "innovation2_noise_robustness.png",
        "innovation3_textureless_optimization.png", 
        "innovation4_viewpoint_reduction.png",
        "README.md"
    ]
    
    missing_files = []
    for filename in expected_files:
        file_path = diagram_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
        elif file_path.stat().st_size == 0:
            missing_files.append(f"{filename} (empty)")
    
    if missing_files:
        print(f"ERROR: Missing or empty files: {missing_files}")
        return False
    
    print("✅ All framework diagrams are properly generated!")
    print(f"📁 Location: {diagram_dir.absolute()}")
    print(f"📊 Total files: {len(list(diagram_dir.glob('*')))}")
    
    # Print file sizes for reference
    print("\nGenerated files:")
    for file_path in sorted(diagram_dir.glob("*")):
        size_kb = file_path.stat().st_size / 1024
        print(f"  - {file_path.name}: {size_kb:.1f} KB")
    
    return True

if __name__ == "__main__":
    # Run validation first
    validation_passed = run_diagram_validation()
    
    # Run unit tests
    print("\n" + "="*50)
    print("Running unit tests...")
    unittest.main(verbosity=2, exit=False)
    
    if validation_passed:
        print("\n🎉 Framework diagram generation completed successfully!")
        print("📋 Use the generated diagrams in your research paper's Method section.")
    else:
        print("\n❌ Framework diagram validation failed!")
        sys.exit(1)