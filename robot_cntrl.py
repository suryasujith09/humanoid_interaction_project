#!/usr/bin/env python
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import subprocess
import threading

class AinexD6AController:
    def __init__(self, root):
        self.root = root
        self.root.title("Ainex Robot .d6a Controller")
        self.root.geometry("600x500")
        self.root.configure(bg='#2c3e50')
        
        # Default directory for .d6a files
        self.d6a_directory = os.path.expanduser("~/ainex_robot/d6a_files")
        self.d6a_files = []
        self.running_process = None
        
        self.create_widgets()
        self.load_d6a_files()
        
    def create_widgets(self):
        # Title
        title_label = tk.Label(
            self.root,
            text="Ainex Robot Controller",
            font=("Arial", 20, "bold"),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        title_label.pack(pady=20)
        
        # Directory selection frame
        dir_frame = tk.Frame(self.root, bg='#2c3e50')
        dir_frame.pack(pady=10, padx=20, fill='x')
        
        tk.Label(
            dir_frame,
            text="Directory:",
            font=("Arial", 10),
            bg='#2c3e50',
            fg='#ecf0f1'
        ).pack(side='left', padx=5)
        
        self.dir_entry = tk.Entry(dir_frame, font=("Arial", 10), width=30)
        self.dir_entry.insert(0, self.d6a_directory)
        self.dir_entry.pack(side='left', padx=5, fill='x', expand=True)
        
        browse_btn = tk.Button(
            dir_frame,
            text="Browse",
            command=self.browse_directory,
            bg='#3498db',
            fg='white',
            font=("Arial", 9),
            relief='flat',
            padx=10
        )
        browse_btn.pack(side='left', padx=5)
        
        refresh_btn = tk.Button(
            dir_frame,
            text="Refresh",
            command=self.load_d6a_files,
            bg='#27ae60',
            fg='white',
            font=("Arial", 9),
            relief='flat',
            padx=10
        )
        refresh_btn.pack(side='left', padx=5)
        
        # Listbox frame for .d6a files
        list_frame = tk.Frame(self.root, bg='#34495e')
        list_frame.pack(pady=20, padx=20, fill='both', expand=True)
        
        tk.Label(
            list_frame,
            text="Available .d6a Files:",
            font=("Arial", 12, "bold"),
            bg='#34495e',
            fg='#ecf0f1'
        ).pack(pady=5)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        # Listbox
        self.file_listbox = tk.Listbox(
            list_frame,
            font=("Arial", 11),
            bg='#ecf0f1',
            fg='#2c3e50',
            selectmode='single',
            yscrollcommand=scrollbar.set,
            height=10
        )
        self.file_listbox.pack(fill='both', expand=True, padx=10, pady=5)
        scrollbar.config(command=self.file_listbox.yview)
        
        # Button frame
        button_frame = tk.Frame(self.root, bg='#2c3e50')
        button_frame.pack(pady=20)
        
        self.execute_btn = tk.Button(
            button_frame,
            text="Execute Selected File",
            command=self.execute_d6a_file,
            bg='#e74c3c',
            fg='white',
            font=("Arial", 12, "bold"),
            relief='flat',
            padx=20,
            pady=10,
            width=20
        )
        self.execute_btn.pack(side='left', padx=10)
        
        self.stop_btn = tk.Button(
            button_frame,
            text="Stop Execution",
            command=self.stop_execution,
            bg='#c0392b',
            fg='white',
            font=("Arial", 12, "bold"),
            relief='flat',
            padx=20,
            pady=10,
            width=20,
            state='disabled'
        )
        self.stop_btn.pack(side='left', padx=10)
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Ready",
            font=("Arial", 10),
            bg='#2c3e50',
            fg='#2ecc71'
        )
        self.status_label.pack(pady=10)
        
    def browse_directory(self):
        directory = filedialog.askdirectory(initialdir=self.d6a_directory)
        if directory:
            self.d6a_directory = directory
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, directory)
            self.load_d6a_files()
            
    def load_d6a_files(self):
        self.d6a_directory = self.dir_entry.get()
        self.file_listbox.delete(0, tk.END)
        self.d6a_files = []
        
        if not os.path.exists(self.d6a_directory):
            self.status_label.config(
                text=f"Directory not found: {self.d6a_directory}",
                fg='#e74c3c'
            )
            return
        
        try:
            files = [f for f in os.listdir(self.d6a_directory) if f.endswith('.d6a')]
            self.d6a_files = sorted(files)
            
            if not self.d6a_files:
                self.status_label.config(
                    text="No .d6a files found in directory",
                    fg='#f39c12'
                )
            else:
                for file in self.d6a_files:
                    self.file_listbox.insert(tk.END, file)
                self.status_label.config(
                    text=f"Found {len(self.d6a_files)} .d6a file(s)",
                    fg='#2ecc71'
                )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load files: {str(e)}")
            
    def execute_d6a_file(self):
        selection = self.file_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a .d6a file to execute")
            return
        
        selected_file = self.d6a_files[selection[0]]
        file_path = os.path.join(self.d6a_directory, selected_file)
        
        # Execute in separate thread to avoid blocking UI
        thread = threading.Thread(target=self.run_d6a_file, args=(file_path, selected_file))
        thread.daemon = True
        thread.start()
        
    def run_d6a_file(self, file_path, filename):
        try:
            self.execute_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.status_label.config(
                text=f"Executing: {filename}",
                fg='#f39c12'
            )
            
            # Execute the .d6a file using ROS
            # Adjust this command based on how your Ainex robot processes .d6a files
            # Common options:
            # 1. If it's a ROS bag file: rosbag play file_path
            # 2. If it's a custom executable: python file_path or ./file_path
            # 3. If it needs a specific ROS node: rosrun package_name node_name file_path
            
            # Example command - modify based on your robot's requirements:
            cmd = ['python3', file_path]  # or ['rosbag', 'play', file_path]
            
            self.running_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = self.running_process.communicate()
            
            if self.running_process.returncode == 0:
                self.status_label.config(
                    text=f"Successfully executed: {filename}",
                    fg='#2ecc71'
                )
            else:
                self.status_label.config(
                    text=f"Execution failed: {filename}",
                    fg='#e74c3c'
                )
                if stderr:
                    messagebox.showerror("Error", stderr.decode())
                    
        except Exception as e:
            self.status_label.config(
                text=f"Error: {str(e)}",
                fg='#e74c3c'
            )
            messagebox.showerror("Execution Error", str(e))
        finally:
            self.running_process = None
            self.execute_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            
    def stop_execution(self):
        if self.running_process:
            self.running_process.terminate()
            self.status_label.config(
                text="Execution stopped",
                fg='#e74c3c'
            )
            self.running_process = None
            self.execute_btn.config(state='normal')
            self.stop_btn.config(state='disabled')

def main():
    root = tk.Tk()
    app = AinexD6AController(root)
    root.mainloop()

if __name__ == "__main__":
    main()
