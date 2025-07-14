import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import easyocr
import fitz  # PyMuPDF
import os
import threading
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import gc

class OptimizedArabicOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arabic OCR Pro - Optimized")
        self.root.geometry("1200x800")
        
        # Colors
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
        
        # Core variables
        self.ocr_reader = None
        self.current_image = None
        self.processing = False
        self.cancel_processing = False
        self.selected_file_path = None
        
        # Optimization settings
        self.max_workers = min(4, multiprocessing.cpu_count())
        self.batch_size = 5
        self.max_dimension = 1800
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_frame = tk.Frame(main_container, bg=self.colors['primary'], height=60)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, 
                text="📄 Arabic OCR Pro - Optimized Version", 
                font=('Arial', 20, 'bold'), 
                bg=self.colors['primary'], 
                fg=self.colors['white']).pack(expand=True)
        
        # File selection section
        file_frame = tk.Frame(main_container, bg=self.colors['white'], relief='raised', bd=1)
        file_frame.pack(fill=tk.X, pady=(0, 15))
        
        file_content = tk.Frame(file_frame, bg=self.colors['white'])
        file_content.pack(fill=tk.X, padx=15, pady=15)
        
        # File row
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
                                    padx=20, pady=8)
        self.process_btn.pack(side=tk.RIGHT)
        
        self.cancel_btn = tk.Button(file_row, 
                                   text="⏹ Cancel", 
                                   command=self.cancel_process,
                                   bg=self.colors['danger'],
                                   fg=self.colors['white'],
                                   font=('Arial', 10, 'bold'),
                                   padx=20, pady=8)
        
        # Progress
        progress_frame = tk.Frame(file_content, bg=self.colors['white'])
        progress_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.progress_label = tk.Label(progress_frame, text="0%", bg=self.colors['white'])
        self.progress_label.pack(side=tk.RIGHT, padx=(10, 0))
        
        self.status_label = tk.Label(file_content, 
                                    text="✅ Ready to process",
                                    bg=self.colors['white'],
                                    fg=self.colors['success'],
                                    font=('Arial', 9))
        self.status_label.pack(fill=tk.X, pady=(5, 0))
        
        # Content area
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
        
        # Right panel - Results
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
        
        # Buttons
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
        if self.ocr_reader is None:
            try:
                self.status_label.config(text="🔄 Initializing OCR...", fg=self.colors['warning'])
                self.root.update()
                self.ocr_reader = easyocr.Reader(['ar', 'en'], gpu=False)
                self.status_label.config(text="✅ OCR ready", fg=self.colors['success'])
                return True
            except Exception as e:
                messagebox.showerror("Error", f"OCR initialization failed: {str(e)}")
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
        # Resize first
        image = self.resize_image(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Simple preprocessing for speed
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def display_image(self, image):
        try:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            pil_image = Image.fromarray(image)
            
            # Fit to canvas
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
            mat = fitz.Matrix(1.5, 1.5)  # Lower DPI for speed
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to numpy array
            img_data = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img_data.reshape(pix.height, pix.width, pix.n)
            
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # Preprocess
            processed = self.preprocess_image(img)
            
            # OCR
            try:
                ocr_results = self.ocr_reader.readtext(processed)
                text = "\n".join([res[1] for res in ocr_results if res[2] > 0.3])
                results.append((page_num, text, processed))
            except Exception as e:
                print(f"OCR error on page {page_num}: {e}")
                results.append((page_num, "", processed))
            
            # Cleanup
            pix = None
            gc.collect()
        
        doc.close()
        return results
    
    def process_file(self):
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
                # PDF processing
                doc = fitz.open(self.selected_file_path)
                total_pages = len(doc)
                doc.close()
                
                self.status_label.config(text=f"📖 Processing {total_pages} pages...", 
                                       fg=self.colors['warning'])
                
                # Process in batches using thread pool
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
                            
                            # Update UI
                            self.text_area.insert(tk.END, f"\n=== Page {page_num + 1} ===\n")
                            self.text_area.insert(tk.END, text + "\n")
                            self.text_area.see(tk.END)
                            
                            # Show last page of batch
                            if page_num == batch_results[-1][0]:
                                self.root.after(0, lambda i=img: self.display_image(i))
                            
                            completed += 1
                            progress = int((completed / total_pages) * 100)
                            self.root.after(0, self.update_progress, progress)
                
            else:
                # Single image processing
                self.status_label.config(text="🔄 Processing image...", fg=self.colors['warning'])
                
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
                self.status_label.config(text="✅ Completed!", fg=self.colors['success'])
                
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
        # Extract Arabic and English numbers
        numbers = re.findall(r'[\d\u0660-\u0669]+(?:[.,][\d\u0660-\u0669]+)*', text)
        
        if numbers:
            result = "🔢 Found Numbers:\n\n" + "\n".join(numbers)
            
            # Show in new window
            num_window = tk.Toplevel(self.root)
            num_window.title("Extracted Numbers")
            num_window.geometry("400x300")
            
            text_widget = scrolledtext.ScrolledText(num_window, wrap=tk.WORD)
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text_widget.insert(tk.END, result)
            
            def copy_nums():
                num_window.clipboard_clear()
                num_window.clipboard_append("\n".join(numbers))
                messagebox.showinfo("Success", "Numbers copied!")
            
            tk.Button(num_window, text="Copy Numbers", command=copy_nums).pack(pady=5)
        else:
            messagebox.showinfo("Info", "No numbers found")
    
    def clear_text(self):
        self.text_area.delete(1.0, tk.END)
        self.status_label.config(text="🗑️ Cleared", fg=self.colors['primary'])
        gc.collect()

def main():
    root = tk.Tk()
    app = OptimizedArabicOCRApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()