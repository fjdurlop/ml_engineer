"""
GPU Profiling Utilities for Performance Analysis
Use these tools to identify bottlenecks in your TTS inference pipeline.
"""

import torch
import time
import psutil
import GPUtil
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from contextlib import contextmanager
from collections import defaultdict


class GPUProfiler:
    """Comprehensive GPU profiling utilities"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset_stats()
    
    def reset_stats(self):
        """Reset all profiling statistics"""
        self.timings = defaultdict(list)
        self.memory_usage = defaultdict(list)
        self.gpu_utilization_section = defaultdict(list)
        self.gpu_utilization = []
        self.batch_sizes = []
        self.throughput_data = []
        
    @contextmanager
    def profile_section(self, section_name: str):
        """Context manager to profile a code section"""
        # Clear GPU cache and synchronize
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Record initial state
        start_time = time.time()
        start_memory = self.get_gpu_memory_usage()
        
        try:
            yield
        finally:
            # Record final state
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            end_memory = self.get_gpu_memory_usage()
            
            # Store measurements
            self.timings[section_name].append(end_time - start_time)
            self.memory_usage[section_name].append(end_memory - start_memory)
            
    @contextmanager
    def profile_section_improved(self, section_name: str):
        """Context manager to profile a code section"""
        # Clear GPU cache and synchronize
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Record initial state
        start_time = time.time()
        start_memory = self.get_gpu_memory_usage()
        
        #start_mem = self.get_gpu_memory_usage()
        start_util = self.get_gpu_utilization()
        try:
            yield
        finally:
            # Record final state
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            end_memory = self.get_gpu_memory_usage()
            
            end_util = self.get_gpu_utilization()
            #    peak_mem = self.get_gpu_memory_usage()
            
            # Store measurements
            self.timings[section_name].append(end_time - start_time)
            self.memory_usage[section_name].append(end_memory - start_memory)
            
            self.gpu_utilization_section[section_name].append(max(start_util, end_util))
            
    
    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except:
            pass
        return 0.0
    
    def profile_inference_batch(self, inference_fn: Callable, inputs: List[Any], batch_sizes: List[int]):
        """Profile inference across different batch sizes"""
        results = {}
        
        for batch_size in batch_sizes:
            print(f"Profiling batch size: {batch_size}")
            
            # Prepare batched inputs
            batched_inputs = []
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                batched_inputs.append(batch)
            
            batch_times = []
            batch_memory = []
            batch_utilization = []
            
            for batch in batched_inputs:
                # Warm up
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                start_mem = self.get_gpu_memory_usage()
                start_util = self.get_gpu_utilization()
                
                print(time.time(),len(batch))
                with self.profile_section(f"batch_size_{batch_size}"):
                    _ = inference_fn(batch)
                
                end_util = self.get_gpu_utilization()
                peak_mem = self.get_gpu_memory_usage()
                
                batch_times.append(self.timings[f"batch_size_{batch_size}"][-1])
                batch_memory.append(peak_mem - start_mem)
                batch_utilization.append(max(start_util, end_util))
            
            # Calculate metrics
            avg_latency = np.mean(batch_times)
            throughput = batch_size / avg_latency
            avg_memory = np.mean(batch_memory)
            avg_utilization = np.mean(batch_utilization)
            
            results[batch_size] = {
                "latency": avg_latency,
                "throughput": throughput,
                "memory_usage": avg_memory,
                "gpu_utilization": avg_utilization,
                "samples": len(batch_times)
            }
        
        return results
    
        
    def plot_per_section(self, sections: Optional[List[str]] = None, name: str = "profiling_per_section.png"):
        """Plot timing, mem, gpu_util results for each profiled section"""
        import matplotlib.pyplot as plt
        
        sections = sections
        
        #sections = sections.remove("inference") if "inference" in sections else sections
        
        # compute average metrics per section
        avg_times = [np.mean(self.timings[s]) for s in sections]
        avg_mems = [np.mean(self.memory_usage[s]) for s in sections]
        avg_utils = [
            np.mean(self.gpu_utilization_section[s]) if self.gpu_utilization_section[s] else 0 
            for s in sections
        ]
        x = np.arange(len(sections))
        width = 0.6
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))
        # Plot timings
        axs[0].bar(x, avg_times, width, color='skyblue')
        axs[0].set_title('Average Execution Time per Section')
        axs[0].set_ylabel('Time (s)')
        # Plot memory usage
        axs[1].bar(x, avg_mems, width, color='lightgreen')
        axs[1].set_title('Average Memory Usage per Section')
        axs[1].set_ylabel('Memory (MB)')
        # Plot GPU utilization
        axs[2].bar(x, avg_utils, width, color='salmon')
        axs[2].set_title('Average GPU Utilization per Section')
        axs[2].set_ylabel('Utilization (%)')
        # Set x-axis labels
        for ax in axs:
            ax.set_xticks(x)
            ax.set_xticklabels(sections, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        plt.savefig(f'logs/{name}.png')