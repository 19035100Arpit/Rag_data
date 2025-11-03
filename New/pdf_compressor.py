# 🗜️ PDF Compression Tool
import os
import subprocess
import sys
from pathlib import Path

def install_required_packages():
    """Install required packages for PDF compression"""
    packages = ["PyPDF2", "pikepdf", "Pillow"]
    
    for package in packages:
        try:
            __import__(package.lower())
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install packages if needed
install_required_packages()

import PyPDF2
from PIL import Image
import pikepdf

class PDFCompressor:
    def __init__(self):
        self.compression_methods = [
            "basic_compression",
            "advanced_compression", 
            "image_optimization"
        ]
    
    def basic_compression(self, input_path: str, output_path: str = None) -> str:
        """Basic PDF compression using PyPDF2"""
        if output_path is None:
            output_path = input_path.replace('.pdf', '_compressed_basic.pdf')
        
        print(f"🔧 Basic compression: {input_path}")
        
        try:
            with open(input_path, 'rb') as input_file:
                reader = PyPDF2.PdfReader(input_file)
                writer = PyPDF2.PdfWriter()
                
                # Copy pages with compression
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    
                    # Remove duplicate objects
                    if hasattr(page, 'compress_content_streams'):
                        page.compress_content_streams()
                    
                    writer.add_page(page)
                
                # Write compressed PDF
                with open(output_path, 'wb') as output_file:
                    writer.write(output_file)
            
            # Check compression ratio
            original_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(output_path)
            ratio = (1 - compressed_size / original_size) * 100
            
            print(f"✅ Basic compression complete!")
            print(f"📊 Original: {original_size:,} bytes")
            print(f"📊 Compressed: {compressed_size:,} bytes")
            print(f"📊 Reduction: {ratio:.1f}%")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Basic compression failed: {e}")
            return None
    
    def advanced_compression(self, input_path: str, output_path: str = None) -> str:
        """Advanced PDF compression using pikepdf"""
        if output_path is None:
            output_path = input_path.replace('.pdf', '_compressed_advanced.pdf')
        
        print(f"🚀 Advanced compression: {input_path}")
        
        try:
            with pikepdf.open(input_path) as pdf:
                # Optimize the PDF
                pdf.save(output_path, 
                        compress_streams=True,
                        recompress_flate=True,
                        optimize_images=True,
                        remove_duplicate_objects=True)
            
            # Check compression ratio
            original_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(output_path)
            ratio = (1 - compressed_size / original_size) * 100
            
            print(f"✅ Advanced compression complete!")
            print(f"📊 Original: {original_size:,} bytes")
            print(f"📊 Compressed: {compressed_size:,} bytes")
            print(f"📊 Reduction: {ratio:.1f}%")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Advanced compression failed: {e}")
            return None
    
    def compress_pdf(self, input_path: str, method: str = "advanced", quality: int = 85) -> str:
        """
        Compress PDF with specified method
        
        Args:
            input_path: Path to input PDF
            method: 'basic', 'advanced', or 'auto'
            quality: Image quality (1-100, higher = better quality)
        """
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")
        
        if not input_path.lower().endswith('.pdf'):
            raise ValueError("Input file must be a PDF")
        
        print(f"🗜️ Starting PDF compression...")
        print(f"📁 Input file: {input_path}")
        print(f"⚙️ Method: {method}")
        
        if method == "basic":
            return self.basic_compression(input_path)
        elif method == "advanced":
            return self.advanced_compression(input_path)
        elif method == "auto":
            # Try advanced first, fallback to basic
            result = self.advanced_compression(input_path)
            if result is None:
                print("🔄 Falling back to basic compression...")
                result = self.basic_compression(input_path)
            return result
        else:
            raise ValueError("Method must be 'basic', 'advanced', or 'auto'")

# Example usage
def compress_pdf_file(pdf_path: str, method: str = "auto"):
    """Easy function to compress a PDF file"""
    compressor = PDFCompressor()
    return compressor.compress_pdf(pdf_path, method)

if __name__ == "__main__":
    # Example usage
    print("🗜️ PDF Compressor Ready!")
    print("\n📋 Usage examples:")
    print("compress_pdf_file('document.pdf', 'advanced')")
    print("compress_pdf_file('large_file.pdf', 'basic')")
    print("compress_pdf_file('any_file.pdf', 'auto')")
    
    # Test if you have a PDF file
    test_files = ["swagger.pdf", "document.pdf", "test.pdf"]
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n🔍 Found {test_file} - ready to compress!")
            break