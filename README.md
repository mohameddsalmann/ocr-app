# Arabic OCR Pro - Optimized Version

A powerful and optimized Optical Character Recognition (OCR) application specifically designed for Arabic text extraction from PDF documents and images. Built with Python and featuring a modern GUI interface.

## 🌟 Features

### Core OCR Capabilities
- **Multi-language Support**: Arabic and English text recognition using EasyOCR
- **Multiple File Formats**: PDF documents and various image formats (PNG, JPG, JPEG, TIFF, BMP)
- **Batch Processing**: Efficient processing of multi-page PDFs with configurable batch sizes
- **Real-time Preview**: Live image preview during processing

### Performance Optimizations
- **Multi-threading**: Parallel processing using ThreadPoolExecutor for faster results
- **Memory Management**: Optimized memory usage with garbage collection and cleanup
- **Image Preprocessing**: Advanced image enhancement with bilateral filtering and Otsu thresholding
- **Smart Resizing**: Automatic image resizing to optimize processing speed

### User Interface
- **Modern GUI**: Clean, intuitive interface built with Tkinter
- **Progress Tracking**: Real-time progress bar and status updates
- **Preview Panel**: Image preview with automatic canvas fitting
- **Text Management**: Copy, save, and extract specific content

### Text Processing Tools
- **Number Extraction**: Automatic detection of Arabic and English numbers using regex
- **Text Export**: Save extracted text to files with UTF-8 encoding
- **Clipboard Integration**: Easy copying of results
- **Text Clearing**: Quick cleanup of results

## 📋 Requirements

### System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python Version**: 3.7 or higher
- **RAM**: Minimum 4GB (8GB recommended for large documents)
- **Storage**: 2GB free space for dependencies

### Hardware Recommendations
- **CPU**: Multi-core processor for optimal performance
- **GPU**: Optional (CPU processing is optimized)
- **Display**: 1200x800 minimum resolution

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ocrr
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python main.py
```

## 📖 Usage Guide

### Getting Started

1. **Launch the Application**
   - Run `python main.py` from the project directory
   - The application window will open with a modern interface

2. **Select a File**
   - Click "📂 Select File" button
   - Choose a PDF document or image file
   - Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP

3. **Process the Document**
   - Click "⚡ Process" to start OCR
   - Monitor progress in the progress bar
   - View real-time status updates

4. **View Results**
   - Extracted text appears in the right panel
   - Preview the processed image in the left panel
   - Use scroll to navigate through results

### Advanced Features

#### Batch Processing
- **Multi-page PDFs**: Automatically processes all pages in configurable batches
- **Progress Tracking**: Real-time progress updates with percentage display
- **Cancellation**: Stop processing at any time with "⏹ Cancel" button

#### Text Management
- **📋 Copy**: Copy all extracted text to clipboard
- **💾 Save**: Save text to a file (UTF-8 encoding)
- **🔢 Numbers**: Extract and display only Arabic and English numbers
- **🗑️ Clear**: Clear all extracted text

#### Image Preview
- **Auto-fit**: Images automatically resize to fit canvas dimensions
- **Quality**: High-quality image processing with bilateral filtering
- **Real-time**: Live preview during processing

## ⚙️ Configuration

### Performance Settings
The application includes several optimization settings that can be modified in the code:

```python
# In main.py - OptimizedArabicOCRApp.__init__()
self.max_workers = min(4, multiprocessing.cpu_count())  # Thread count
self.batch_size = 5                                     # PDF pages per batch
self.max_dimension = 1800                               # Max image dimension
```

### OCR Settings
```python
# Confidence threshold for text detection (line 280)
if res[2] > 0.3:  # Adjust this value (0.0 to 1.0)
```

### Image Processing Settings
```python
# PDF rendering DPI (line 250)
mat = fitz.Matrix(1.5, 1.5)  # Lower DPI for speed

# Image preprocessing (lines 200-210)
denoised = cv2.bilateralFilter(gray, 9, 75, 75)
_, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

## 🔧 Troubleshooting

### Common Issues

#### Installation Problems
- **Import Errors**: Ensure all dependencies are installed correctly
- **Version Conflicts**: Use a virtual environment
- **Missing Libraries**: Run `pip install -r requirements.txt`

#### Performance Issues
- **Slow Processing**: Reduce `max_workers` or `batch_size`
- **Memory Errors**: Close other applications or reduce `max_dimension`
- **Large Files**: Process in smaller batches

#### OCR Accuracy
- **Poor Results**: Ensure good image quality
- **Missing Text**: Lower confidence threshold (currently 0.3)
- **Wrong Language**: Check if text is clearly visible

### Error Messages

| Error | Solution |
|-------|----------|
| "OCR initialization failed" | Check internet connection for model download |
| "Failed to load image" | Verify file format and corruption |
| "Memory error" | Reduce batch size or image dimensions |

## 📊 Performance Tips

### For Best Results
1. **Image Quality**: Use high-resolution, clear images
2. **File Size**: Optimize images before processing
3. **Text Clarity**: Ensure text is clearly visible and not blurred
4. **Background**: Use documents with good contrast

### Optimization Settings
- **Small Documents**: Increase `batch_size` for faster processing
- **Large Documents**: Decrease `batch_size` to reduce memory usage
- **Low-end Systems**: Reduce `max_workers` to 2 or 1

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest

# Format code
black main.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **EasyOCR**: Core OCR functionality with Arabic language support
- **OpenCV**: Image processing capabilities (bilateral filtering, thresholding)
- **PyMuPDF**: PDF handling and page rendering
- **Tkinter**: GUI framework for the user interface

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

## 🔄 Version History

### v1.0.0
- Initial release
- Arabic and English OCR support using EasyOCR
- PDF and image processing with PyMuPDF and OpenCV
- Modern GUI interface with Tkinter
- Performance optimizations with multi-threading
- Memory management with garbage collection

---

**Note**: This application is optimized for Arabic text recognition but works well with English text as well. For best results with Arabic text, ensure proper text direction and clarity. The application uses EasyOCR with CPU processing for optimal compatibility. 