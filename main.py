import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import easyocr
import fitz
import os
import threading
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import gc
from pathlib import Path
import sys
import shutil

class OfflineArabicOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECA OCR - Fully Offline Text Recognition")
        self.root.geometry("1200x800")
        
        self.colors = {
            'bg': '#f0f0f0',
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'text': '#2c3e50',
            'white': '#ffffff',
            'light_gray': '#ecf0f1'
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        self.ocr_reader = None
        self.current_image = None
        self.processing = False
        self.cancel_processing = False
        self.selected_file_path = None
        
        self.max_workers = min(4, multiprocessing.cpu_count())
        self.batch_size = 5
        self.max_dimension = 1800
        
        # Setup model directories
        self.setup_model_paths()
        self.check_offline_models()
        self.setup_ui()
        
    def setup_model_paths(self):
        """Setup all possible model paths for offline operation"""
        # Get the directory where the script is located
        script_dir = Path(__file__).parent.absolute()
        
        # List of possible model locations
        possible_paths = [
            # Direct EasyOCR path from script location
            script_dir / "EasyOCR" / "model",
            script_dir / ".EasyOCR" / "model",
            # Primary path from your original code
            Path(r"X:\Internship 2025\July 2025\Economic Intelligence Department\ocr-app-main\.EasyOCR\model"),
            # Home directory paths
            Path.home() / '.EasyOCR' / 'model',
            Path.home() / 'EasyOCR' / 'model',
            # Current working directory
            Path.cwd() / 'EasyOCR' / 'model',
            Path.cwd() / '.EasyOCR' / 'model',
            # Local models directory
            script_dir / 'models',
        ]
        
        # Find the first existing path with model files
        self.models_dir = None
        for path in possible_paths:
            if path.exists():
                # Check if it contains .pth files
                pth_files = list(path.glob("*.pth"))
                if pth_files:
                    self.models_dir = path
                    print(f"Found models in: {path}")
                    break
        
        # If no path found, use the script directory path
        if self.models_dir is None:
            self.models_dir = script_dir / "EasyOCR" / "model"
            print(f"No models found. Expected location: {self.models_dir}")
    
    def check_offline_models(self):
        """Check if required models exist offline"""
        # Updated to match your actual model files
        self.required_models = {
            'craft_mlt_25k.pth': {
                'size': 83176120,  # Approximate size in bytes
                'description': 'Text detection model'
            },
            'english_g2.pth': {
                'size': 64000000,  # Approximate
                'description': 'English recognition model'
            },
            'arabic_g2.pth': {  # Fixed: was 'arabic.pth', now 'arabic_g2.pth'
                'size': 64000000,  # Approximate
                'description': 'Arabic recognition model'
            }
        }
        
        self.available_models = {}
        self.missing_models = {}
        
        if self.models_dir.exists():
            for model_name, model_info in self.required_models.items():
                model_path = self.models_dir / model_name
                if model_path.exists():
                    # Verify file size
                    file_size = model_path.stat().st_size
                    if file_size > 1000000:  # At least 1MB
                        self.available_models[model_name] = model_path
                    else:
                        self.missing_models[model_name] = model_info
                else:
                    self.missing_models[model_name] = model_info
        else:
            self.missing_models = dict(self.required_models)
        
        # Check if minimum models are available (detection + at least one language)
        self.models_exist = ('craft_mlt_25k.pth' in self.available_models and 
                            ('english_g2.pth' in self.available_models or 
                             'arabic_g2.pth' in self.available_models))
        
        print(f"Models directory: {self.models_dir}")
        print(f"Available models: {list(self.available_models.keys())}")
        print(f"Missing models: {list(self.missing_models.keys())}")
        print(f"Models ready: {self.models_exist}")
    
    def copy_models_from_usb(self):
        """Allow copying models from USB or external drive"""
        source_dir = filedialog.askdirectory(
            title="Select folder containing model files (.pth files)"
        )
        
        if source_dir:
            try:
                copied_count = 0
                source_path = Path(source_dir)
                
                # Ensure target directory exists
                self.models_dir.mkdir(parents=True, exist_ok=True)
                
                # Look for any .pth files in source
                for pth_file in source_path.glob("*.pth"):
                    dest_file = self.models_dir / pth_file.name
                    shutil.copy2(pth_file, dest_file)
                    copied_count += 1
                    self.status_label.config(
                        text=f"✅ Copied {pth_file.name}", 
                        fg=self.colors['success']
                    )
                    self.root.update()
                
                if copied_count > 0:
                    messagebox.showinfo("Success", f"Copied {copied_count} model files successfully!")
                    self.check_offline_models()
                    self.refresh_ui_after_model_update()
                else:
                    messagebox.showwarning("Warning", "No .pth model files found in selected directory")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to copy models: {str(e)}")
    
    def refresh_ui_after_model_update(self):
        """Refresh UI elements after models are updated"""
        if self.models_exist:
            status_text = f"✅ Models ready (Offline) - Available: {', '.join(self.available_models.keys())}"
            self.model_status_label.config(
                text=status_text, 
                fg=self.colors['success']
            )
            self.status_label.config(
                text="✅ Ready to process", 
                fg=self.colors['success']
            )
            self.process_btn.config(state=tk.NORMAL)
            
            # Hide download/copy buttons if they exist
            if hasattr(self, 'copy_btn'):
                self.copy_btn.pack_forget()
            if hasattr(self, 'manual_btn'):
                self.manual_btn.pack_forget()
    
    def setup_ui(self):
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_frame = tk.Frame(main_container, bg=self.colors['primary'], height=60)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, 
                text="📄 ECA OCR - Fully Offline Text Recognition", 
                font=('Arial', 20, 'bold'), 
                bg=self.colors['primary'], 
                fg=self.colors['white']).pack(expand=True)
        
        # Model status frame
        model_frame = tk.Frame(main_container, bg=self.colors['white'], relief='raised', bd=1)
        model_frame.pack(fill=tk.X, pady=(0, 15))
        
        model_content = tk.Frame(model_frame, bg=self.colors['white'])
        model_content.pack(fill=tk.X, padx=15, pady=10)
        
        model_row = tk.Frame(model_content, bg=self.colors['white'])
        model_row.pack(fill=tk.X)
        
        # Model status
        if self.models_exist:
            status_text = f"✅ Models ready (Offline) - Available: {', '.join(self.available_models.keys())}"
            status_color = self.colors['success']
        else:
            status_text = f"❌ Models missing: {', '.join(self.missing_models.keys()) if self.missing_models else 'No models found'}"
            status_color = self.colors['danger']
        
        self.model_status_label = tk.Label(model_row, 
                                          text=status_text,
                                          bg=self.colors['white'],
                                          fg=status_color,
                                          font=('Arial', 10, 'bold'))
        self.model_status_label.pack(side=tk.LEFT)
        
        # Show path info
        path_label = tk.Label(model_content, 
                             text=f"Path: {self.models_dir}",
                             bg=self.colors['white'],
                             fg=self.colors['text'],
                             font=('Arial', 8))
        path_label.pack(anchor='w', pady=(5, 0))
        
        if not self.models_exist:
            # Copy from USB button
            self.copy_btn = tk.Button(model_row, 
                     text="📁 Copy from USB/Folder", 
                     command=self.copy_models_from_usb,
                     bg=self.colors['secondary'],
                     fg=self.colors['white'],
                     font=('Arial', 9, 'bold'),
                     padx=15, pady=5)
            self.copy_btn.pack(side=tk.RIGHT, padx=(0, 5))
            
            # Manual guide button
            self.manual_btn = tk.Button(model_row, 
                     text="ℹ️ Offline Setup Guide", 
                     command=self.show_offline_setup_guide,
                     bg=self.colors['warning'],
                     fg=self.colors['white'],
                     font=('Arial', 9, 'bold'),
                     padx=10, pady=5)
            self.manual_btn.pack(side=tk.RIGHT)
        
        # File selection frame
        file_frame = tk.Frame(main_container, bg=self.colors['white'], relief='raised', bd=1)
        file_frame.pack(fill=tk.X, pady=(0, 15))
        
        file_content = tk.Frame(file_frame, bg=self.colors['white'])
        file_content.pack(fill=tk.X, padx=15, pady=15)
        
        file_row = tk.Frame(file_content, bg=self.colors['white'])
        file_row.pack(fill=tk.X)
        
        tk.Button(file_row, 
                 text="📂 Select File", 
                 command=self.select_file,
                 bg=self.colors['secondary'],
                 fg=self.colors['white'],
                 font=('Arial', 10, 'bold'),
                 padx=20, pady=8).pack(side=tk.LEFT, padx=(0, 10))
        
        self.file_label = tk.Label(file_row, 
                                  text="No file selected...",
                                  bg=self.colors['light_gray'],
                                  font=('Arial', 9),
                                  padx=10, pady=8)
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.process_btn = tk.Button(file_row, 
                                    text="⚡ Process", 
                                    command=self.process_file,
                                    bg=self.colors['success'],
                                    fg=self.colors['white'],
                                    font=('Arial', 10, 'bold'),
                                    padx=20, pady=8,
                                    state=tk.NORMAL if self.models_exist else tk.DISABLED)
        self.process_btn.pack(side=tk.RIGHT)
        
        self.cancel_btn = tk.Button(file_row, 
                                   text="⏹ Cancel", 
                                   command=self.cancel_process,
                                   bg=self.colors['danger'],
                                   fg=self.colors['white'],
                                   font=('Arial', 10, 'bold'),
                                   padx=20, pady=8)
        
        # Progress bar
        progress_frame = tk.Frame(file_content, bg=self.colors['white'])
        progress_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.progress_label = tk.Label(progress_frame, text="0%", bg=self.colors['white'])
        self.progress_label.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Status label
        self.status_label = tk.Label(file_content, 
                                    text="✅ Ready to process" if self.models_exist else "⚠️ Please install models first",
                                    bg=self.colors['white'],
                                    fg=self.colors['success'] if self.models_exist else self.colors['warning'],
                                    font=('Arial', 9))
        self.status_label.pack(fill=tk.X, pady=(5, 0))
        
        # Content frame
        content_frame = tk.Frame(main_container, bg=self.colors['bg'])
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Preview
        left_panel = tk.Frame(content_frame, bg=self.colors['white'], relief='raised', bd=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        
        tk.Label(left_panel, 
                text="🖼️ Preview", 
                font=('Arial', 12, 'bold'),
                bg=self.colors['warning'],
                fg=self.colors['white'],
                pady=8).pack(fill=tk.X)
        
        self.canvas = tk.Canvas(left_panel, bg=self.colors['light_gray'])
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right panel - Text
        right_panel = tk.Frame(content_frame, bg=self.colors['white'], relief='raised', bd=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))
        
        tk.Label(right_panel, 
                text="📝 Extracted Text", 
                font=('Arial', 12, 'bold'),
                bg=self.colors['success'],
                fg=self.colors['white'],
                pady=8).pack(fill=tk.X)
        
        text_frame = tk.Frame(right_panel, bg=self.colors['white'])
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.text_area = scrolledtext.ScrolledText(text_frame, 
                                                  wrap=tk.WORD, 
                                                  font=('Arial', 10),
                                                  bg=self.colors['white'],
                                                  fg=self.colors['text'])
        self.text_area.pack(fill=tk.BOTH, expand=True)
        
        # Button frame
        btn_frame = tk.Frame(text_frame, bg=self.colors['white'])
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Button(btn_frame, text="📋 Copy", command=self.copy_text,
                 bg=self.colors['secondary'], fg=self.colors['white'],
                 padx=15, pady=5).pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(btn_frame, text="💾 Save", command=self.save_text,
                 bg=self.colors['primary'], fg=self.colors['white'],
                 padx=15, pady=5).pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(btn_frame, text="🔢 Numbers", command=self.extract_numbers,
                 bg=self.colors['warning'], fg=self.colors['white'],
                 padx=15, pady=5).pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(btn_frame, text="🗑️ Clear", command=self.clear_text,
                 bg=self.colors['danger'], fg=self.colors['white'],
                 padx=15, pady=5).pack(side=tk.LEFT)
    
    def show_offline_setup_guide(self):
        """Show guide for offline setup"""
        help_text = f"""📋 Offline OCR Setup Guide

This application requires pre-downloaded model files to work offline.

🎯 REQUIRED MODEL FILES:
1. craft_mlt_25k.pth - Text detection (Required)
2. english_g2.pth - English recognition 
3. arabic_g2.pth - Arabic recognition

📁 CURRENT MODEL DIRECTORY:
{self.models_dir}

🔧 SETUP OPTIONS:

Option 1: Copy from USB/External Drive
1. Download models on a computer with internet
2. Copy the .pth files to a USB drive
3. Click "Copy from USB/Folder" button
4. Select the folder containing the models

Option 2: Manual Copy
1. Copy the model files manually to:
   {self.models_dir}
2. Restart the application

Option 3: Network Share
1. If models are on a network drive
2. Copy them to the model directory
3. Or update the path in the code

📊 MODEL STATUS:
✅ Available Models: {', '.join(self.available_models.keys()) if self.available_models else 'None'}
❌ Missing Models: {', '.join(self.missing_models.keys()) if self.missing_models else 'None'}

💡 TIPS:
- Models are typically 60-80 MB each
- Ensure files are not corrupted
- Check file permissions
- Detection model (craft_mlt_25k.pth) is mandatory
- At least one language model is required"""

        help_window = tk.Toplevel(self.root)
        help_window.title("Offline Setup Guide")
        help_window.geometry("700x600")
        help_window.configure(bg=self.colors['white'])
        
        text_frame = tk.Frame(help_window, bg=self.colors['white'])
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = scrolledtext.ScrolledText(
            text_frame, 
            wrap=tk.WORD, 
            font=('Consolas', 9),
            bg=self.colors['white'],
            fg=self.colors['text']
        )
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)
        
        btn_frame = tk.Frame(help_window, bg=self.colors['white'])
        btn_frame.pack(fill=tk.X, pady=10)
        
        def open_model_folder():
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(self.models_dir)
                elif os.name == 'posix':  # macOS and Linux
                    os.system(f'open "{self.models_dir}"' if sys.platform == 'darwin' else f'xdg-open "{self.models_dir}"')
            except:
                messagebox.showinfo("Path", f"Model directory: {self.models_dir}")
        
        tk.Button(btn_frame, text="📁 Open Model Folder", command=open_model_folder,
                 bg=self.colors['primary'], fg=self.colors['white'], padx=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="📁 Copy from USB/Folder", command=self.copy_models_from_usb,
                 bg=self.colors['secondary'], fg=self.colors['white'], padx=10).pack(side=tk.LEFT, padx=5)
    
    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select PDF or Image",
            filetypes=[
                ("All Supported", "*.pdf *.png *.jpg *.jpeg *.tiff *.bmp"),
                ("PDF files", "*.pdf"),
                ("Images", "*.png *.jpg *.jpeg *.tiff *.bmp")
            ]
        )
        
        if file_path:
            self.selected_file_path = file_path
            self.file_label.config(text=f"📄 {os.path.basename(file_path)}")
            self.status_label.config(text="✅ File selected", fg=self.colors['success'])
    
    def initialize_ocr(self):
        """Initialize OCR in fully offline mode"""
        if not self.models_exist:
            messagebox.showwarning("Warning", "Please install model files first")
            return False
            
        if self.ocr_reader is None:
            try:
                self.status_label.config(text="🔄 Initializing OCR (Offline)...", fg=self.colors['warning'])
                self.root.update()
                
                # Determine which languages to load based on available models
                languages = []
                if 'arabic_g2.pth' in self.available_models:
                    languages.append('ar')
                if 'english_g2.pth' in self.available_models:
                    languages.append('en')
                
                if not languages:
                    # Default to English if no language models found
                    languages = ['en']
                
                # CRITICAL: Copy models to EasyOCR's expected location
                # EasyOCR is hardcoded to look in user home directory
                user_easyocr_dir = Path.home() / ".EasyOCR"
                user_model_dir = user_easyocr_dir / "model"
                
                # Create directories if they don't exist
                user_model_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy all model files to user directory
                print(f"Copying models from {self.models_dir} to {user_model_dir}")
                for model_file in self.models_dir.glob("*.pth"):
                    dest_file = user_model_dir / model_file.name
                    if not dest_file.exists() or dest_file.stat().st_size != model_file.stat().st_size:
                        try:
                            shutil.copy2(model_file, dest_file)
                            print(f"Copied {model_file.name} to {dest_file}")
                        except Exception as e:
                            print(f"Error copying {model_file.name}: {e}")
                
                # Also copy to the exact path EasyOCR is looking for
                alt_path = Path("c:/Users/asus/Downloads/ocrr/EasyOCR/model")
                if not alt_path.exists():
                    alt_path.mkdir(parents=True, exist_ok=True)
                    for model_file in self.models_dir.glob("*.pth"):
                        dest_file = alt_path / model_file.name
                        try:
                            shutil.copy2(model_file, dest_file)
                            print(f"Also copied {model_file.name} to {dest_file}")
                        except:
                            pass
                
                # Set environment variables to point to user directory
                os.environ['EASYOCR_MODULE_PATH'] = str(user_easyocr_dir)
                os.environ['MODULE_PATH'] = str(user_easyocr_dir)
                
                # Initialize with offline mode - use user directory
                self.ocr_reader = easyocr.Reader(
                    languages, 
                    gpu=False, 
                    download_enabled=False,
                    model_storage_directory=str(user_easyocr_dir),
                    verbose=False
                )
                
                self.status_label.config(text="✅ OCR ready (Offline)", fg=self.colors['success'])
                return True
                
            except Exception as e:
                error_msg = f"OCR initialization failed: {str(e)}\n\n"
                error_msg += f"Models directory: {self.models_dir}\n"
                error_msg += f"Available models: {', '.join(self.available_models.keys())}\n\n"
                error_msg += "The app will now copy models to the required location and retry."
                
                # Try one more time after copying
                try:
                    # Ensure models are in user directory
                    user_model_dir = Path.home() / ".EasyOCR" / "model"
                    user_model_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Force copy all models
                    for model_name in ['craft_mlt_25k.pth', 'english_g2.pth', 'arabic_g2.pth']:
                        src_file = self.models_dir / model_name
                        if src_file.exists():
                            dest_file = user_model_dir / model_name
                            shutil.copy2(src_file, dest_file)
                    
                    # Try again with simpler initialization
                    self.ocr_reader = easyocr.Reader(
                        languages, 
                        gpu=False, 
                        download_enabled=False,
                        verbose=False
                    )
                    
                    self.status_label.config(text="✅ OCR ready (Offline)", fg=self.colors['success'])
                    return True
                    
                except Exception as e2:
                    messagebox.showerror("Error", f"OCR initialization failed: {str(e2)}")
                    self.status_label.config(text="❌ OCR initialization failed", fg=self.colors['danger'])
                    return False
        return True
    
    def resize_image(self, image):
        height, width = image.shape[:2]
        if max(height, width) > self.max_dimension:
            scale = self.max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image
    
    def preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        image = self.resize_image(image)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply bilateral filter for noise reduction while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply OTSU thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def display_image(self, image):
        try:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            pil_image = Image.fromarray(image)
            
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                pil_image.thumbnail((canvas_width-20, canvas_height-20), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(pil_image)
            self.canvas.delete("all")
            self.canvas.create_image(10, 10, anchor=tk.NW, image=photo)
            self.canvas.image = photo
        except Exception as e:
            print(f"Display error: {e}")
    
    def process_pdf_batch(self, pdf_path, start_idx, end_idx):
        """Process a batch of PDF pages"""
        results = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(start_idx, min(end_idx, len(doc))):
            if self.cancel_processing:
                break
                
            page = doc.load_page(page_num)
            mat = fitz.Matrix(1.5, 1.5)  # Scale for better quality
            pix = page.get_pixmap(matrix=mat)
            
            img_data = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img_data.reshape(pix.height, pix.width, pix.n)
            
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            processed = self.preprocess_image(img)
            
            try:
                ocr_results = self.ocr_reader.readtext(processed)
                text = "\n".join([res[1] for res in ocr_results if res[2] > 0.3])
                results.append((page_num, text, processed))
            except Exception as e:
                print(f"OCR error on page {page_num}: {e}")
                results.append((page_num, f"Error: {str(e)}", processed))
            
            pix = None
            gc.collect()
        
        doc.close()
        return results
    
    def process_file(self):
        if not self.models_exist:
            messagebox.showwarning("Warning", "Please install model files first")
            return
            
        if not self.selected_file_path:
            messagebox.showwarning("Warning", "Please select a file first")
            return
        
        self.processing = True
        self.cancel_processing = False
        self.process_btn.pack_forget()
        self.cancel_btn.pack(side=tk.RIGHT)
        
        thread = threading.Thread(target=self._process_file_thread)
        thread.daemon = True
        thread.start()
    
    def _process_file_thread(self):
        try:
            if not self.initialize_ocr():
                return
            
            self.text_area.delete(1.0, tk.END)
            
            if self.selected_file_path.lower().endswith('.pdf'):
                # Process PDF
                doc = fitz.open(self.selected_file_path)
                total_pages = len(doc)
                doc.close()
                
                self.status_label.config(text=f"📖 Processing {total_pages} pages (Offline)...", 
                                       fg=self.colors['warning'])
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    
                    for i in range(0, total_pages, self.batch_size):
                        if self.cancel_processing:
                            break
                        
                        future = executor.submit(
                            self.process_pdf_batch,
                            self.selected_file_path,
                            i,
                            i + self.batch_size
                        )
                        futures.append(future)
                    
                    completed = 0
                    for future in as_completed(futures):
                        if self.cancel_processing:
                            break
                        
                        batch_results = future.result()
                        
                        for page_num, text, img in batch_results:
                            if self.cancel_processing:
                                break
                            
                            self.text_area.insert(tk.END, f"\n=== Page {page_num + 1} ===\n")
                            self.text_area.insert(tk.END, text + "\n")
                            self.text_area.see(tk.END)
                            
                            if page_num == batch_results[-1][0]:
                                self.root.after(0, lambda i=img: self.display_image(i))
                            
                            completed += 1
                            progress = int((completed / total_pages) * 100)
                            self.root.after(0, self.update_progress, progress)
                
            else:
                # Process single image
                self.status_label.config(text="🔄 Processing image (Offline)...", fg=self.colors['warning'])
                
                img = cv2.imread(self.selected_file_path)
                if img is None:
                    raise Exception("Failed to load image")
                
                processed = self.preprocess_image(img)
                self.root.after(0, lambda: self.display_image(processed))
                
                results = self.ocr_reader.readtext(processed)
                text = "\n".join([res[1] for res in results if res[2] > 0.3])
                
                self.text_area.insert(tk.END, text)
                self.update_progress(100)
            
            if self.cancel_processing:
                self.status_label.config(text="❌ Cancelled", fg=self.colors['danger'])
            else:
                self.status_label.config(text="✅ Completed (Offline)!", fg=self.colors['success'])
                
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="❌ Error occurred", fg=self.colors['danger'])
        finally:
            self.processing = False
            self.cancel_btn.pack_forget()
            self.process_btn.pack(side=tk.RIGHT)
            self.progress['value'] = 0
            self.progress_label.config(text="0%")
            gc.collect()
    
    def update_progress(self, value):
        self.progress['value'] = value
        self.progress_label.config(text=f"{value}%")
        self.root.update_idletasks()
    
    def cancel_process(self):
        self.cancel_processing = True
        self.status_label.config(text="⏸ Cancelling...", fg=self.colors['warning'])
    
    def copy_text(self):
        text = self.text_area.get(1.0, tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_label.config(text="📋 Copied!", fg=self.colors['success'])
    
    def save_text(self):
        text = self.text_area.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "No text to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.status_label.config(text="💾 Saved!", fg=self.colors['success'])
            except Exception as e:
                messagebox.showerror("Error", f"Save failed: {str(e)}")
    
    def extract_numbers(self):
        text = self.text_area.get(1.0, tk.END)
        # Extract both Arabic and Western numerals
        numbers = re.findall(r'[\d\u0660-\u0669]+(?:[.,][\d\u0660-\u0669]+)*', text)
        
        if numbers:
            # Convert Arabic numerals to Western if needed
            converted_numbers = []
            arabic_to_western = {
                '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
                '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
            }
            
            for num in numbers:
                converted = num
                for ar, west in arabic_to_western.items():
                    converted = converted.replace(ar, west)
                converted_numbers.append(converted)
            
            result = "🔢 Found Numbers:\n\n"
            result += "Original:\n" + "\n".join(numbers) + "\n\n"
            result += "Converted (Western):\n" + "\n".join(converted_numbers)
            
            num_window = tk.Toplevel(self.root)
            num_window.title("Extracted Numbers")
            num_window.geometry("400x400")
            num_window.configure(bg=self.colors['white'])
            
            text_widget = scrolledtext.ScrolledText(
                num_window, 
                wrap=tk.WORD,
                font=('Arial', 10),
                bg=self.colors['white'],
                fg=self.colors['text']
            )
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text_widget.insert(tk.END, result)
            
            btn_frame = tk.Frame(num_window, bg=self.colors['white'])
            btn_frame.pack(fill=tk.X, pady=5)
            
            def copy_original():
                num_window.clipboard_clear()
                num_window.clipboard_append("\n".join(numbers))
                messagebox.showinfo("Success", "Original numbers copied!")
            
            def copy_converted():
                num_window.clipboard_clear()
                num_window.clipboard_append("\n".join(converted_numbers))
                messagebox.showinfo("Success", "Converted numbers copied!")
            
            tk.Button(btn_frame, text="Copy Original", command=copy_original,
                     bg=self.colors['secondary'], fg=self.colors['white'],
                     padx=10, pady=5).pack(side=tk.LEFT, padx=5)
            
            tk.Button(btn_frame, text="Copy Converted", command=copy_converted,
                     bg=self.colors['primary'], fg=self.colors['white'],
                     padx=10, pady=5).pack(side=tk.LEFT, padx=5)
        else:
            messagebox.showinfo("Info", "No numbers found")
    
    def clear_text(self):
        self.text_area.delete(1.0, tk.END)
        self.status_label.config(text="🗑️ Cleared", fg=self.colors['primary'])
        gc.collect()

def main():
    root = tk.Tk()
    app = OfflineArabicOCRApp(root)
    
    # Set window icon if available
    try:
        if os.path.exists("icon.ico"):
            root.iconbitmap("icon.ico")
    except:
        pass
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()

if __name__ == "__main__":
    main()