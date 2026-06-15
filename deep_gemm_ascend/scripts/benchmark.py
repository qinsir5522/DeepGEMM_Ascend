#!/usr/bin/env python3
# coding=utf-8
"""
GEMM Benchmark Tool for DeepGEMM on Ascend NPU

This module provides comprehensive benchmarking capabilities for GEMM operations
using DeepGEMM library on Ascend NPU devices. It includes:
- Parameter generation and filtering based on hardware constraints
- Multi-process distributed benchmark execution with checkpoint support
- Performance profiling using msprof tool
- Result collection, validation, and analysis
- Statistical analysis and visualization utilities

Author: DeepGEMM Team
Version: 2.0.0
"""

from typing import List, Dict, Any, Tuple, Optional, Iterator, Union
import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass, asdict, field
from pathlib import Path
from tqdm import tqdm
import os, sys, json, subprocess, math, argparse, re, time, logging, traceback
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
import statistics
import deep_gemm_ascend

torch.npu.config.allow_internal_format = False

relative_tol = 1.5e-6
absolute_tol = 1e-9
error_tol = 1e-4
error_tolerance = 1e-4

shape_group = [
    [4096, 4096, 4096], [8, 7168, 18432], [8, 18432, 7168],
    [64, 4096, 7168], [64, 7168, 18432], [64, 18432, 7168],
    [64, 24576, 1536], [64, 32768, 512], [64, 7168, 16384],
    [128, 4096, 7168], [128, 7168, 18432], [128, 18432, 7168],
    [1024, 4096, 7168], [1024, 18432, 7168], [2048, 4096, 7168],
    [1279, 5003, 7681], [3511, 6151, 8191], [5119, 6997, 9901]
]

MAX_CORES = 24
MAX_BLOCKS = 128
MAX_SHARED_MEMORY = 1024
MIN_TILE_SIZE = 16
DEFAULT_TIMEOUT = 60
MAX_ERROR_DISPLAY = 10


class BenchmarkLogger:
    """Centralized logging system with structured output and file support."""
    
    def __init__(self, name: str = "GEMMBenchmark", log_file: Optional[str] = None,
                 level: int = logging.INFO, console_output: bool = True):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(
                '[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
            self.logger.addHandler(console_handler)
        
        if log_file:
            os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
            self.logger.addHandler(file_handler)
    
    def info(self, msg: str): self.logger.info(msg)
    def warning(self, msg: str): self.logger.warning(msg)
    def error(self, msg: str): self.logger.error(msg)
    def debug(self, msg: str): self.logger.debug(msg)
    def exception(self, msg: str): self.logger.exception(msg)
    def critical(self, msg: str): self.logger.critical(msg)
    
    def log_section(self, title: str, level: str = 'info'):
        separator = "=" * 80
        message = f"\n{separator}\n{title}\n{separator}"
        getattr(self, level)(message)


class ConstraintValidator:
    """Hardware constraint validator for GEMM parameter generation."""
    
    def __init__(self, max_cores: int = MAX_CORES, max_blocks: int = MAX_BLOCKS,
                 max_shared_mem: int = MAX_SHARED_MEMORY, min_tile: int = MIN_TILE_SIZE):
        self.max_cores = max_cores
        self.max_blocks = max_blocks
        self.max_shared_mem = max_shared_mem
        self.min_tile = min_tile
        self.logger = BenchmarkLogger("ConstraintValidator")
    
    def validate_mn_sections(self, m_sections: int, n_sections: int) -> bool:
        """Validate section partition: m_sections × n_sections <= max_cores."""
        return m_sections * n_sections <= self.max_cores
    
    def validate_o_blocks(self, m_blocks: int, n_blocks: int, k_blocks: int, db_blocks: int) -> bool:
        """Validate block constraints for shared memory and block count limits."""
        checks = [
            m_blocks * n_blocks <= self.max_blocks,
            (m_blocks + n_blocks) * k_blocks * 2 < self.max_shared_mem,
            m_blocks * db_blocks <= self.max_blocks,
            n_blocks * db_blocks <= self.max_blocks
        ]
        return all(checks)
    
    def validate_shape_bounds(self, shape: List[int], params: Dict[str, int]) -> bool:
        """Validate parameter bounds relative to matrix dimensions."""
        M, N, K = shape
        
        max_m_sec = math.ceil(M / self.min_tile)
        max_n_sec = math.ceil(N / self.min_tile)
        min_m_blk = min(2, math.ceil(M / self.min_tile))
        min_n_blk = min(2, math.ceil(N / self.min_tile))
        min_k_blk = min(2, math.ceil(K / self.min_tile))
        max_m_blk = math.ceil(M / self.min_tile)
        max_n_blk = math.ceil(N / self.min_tile)
        max_k_blk = math.ceil(K / self.min_tile)
        
        checks = [
            params['m_sections'] <= max_m_sec,
            params['n_sections'] <= max_n_sec,
            params['m_sec_o_blocks'] >= min_m_blk,
            params['n_sec_o_blocks'] >= min_n_blk,
            params['k_o_iter_blocks'] >= min_k_blk,
            params['m_sec_o_blocks'] <= max_m_blk,
            params['n_sec_o_blocks'] <= max_n_blk,
            params['k_o_iter_blocks'] <= max_k_blk
        ]
        return all(checks)
    
    def get_max_k_blocks(self, m_blocks: int, n_blocks: int) -> int:
        """Calculate maximum K-axis block count based on shared memory."""
        return (self.max_shared_mem - 1) // ((m_blocks + n_blocks) * 2)
    
    def get_max_db_blocks(self, m_blocks: int, n_blocks: int) -> int:
        """Calculate maximum double buffer block count."""
        return min(self.max_blocks // m_blocks, self.max_blocks // n_blocks)
    
    def get_constraint_report(self) -> Dict[str, Any]:
        """Generate constraint configuration report."""
        return {
            'max_cores': self.max_cores,
            'max_blocks': self.max_blocks,
            'max_shared_memory': self.max_shared_mem,
            'min_tile_size': self.min_tile,
            'section_constraint': 'm_sections × n_sections <= max_cores',
            'block_constraint_1': 'm_blocks × n_blocks <= max_blocks',
            'block_constraint_2': '(m_blocks + n_blocks) × k_blocks × 2 < max_shared_mem',
            'block_constraint_3': 'm_blocks × db_blocks <= max_blocks',
            'block_constraint_4': 'n_blocks × db_blocks <= max_blocks'
        }


class ParameterGenerator(ABC):
    """Abstract base class for parameter generation strategies."""
    
    @abstractmethod
    def generate_mn_sections(self) -> Iterator[Tuple[int, int]]:
        """Generate valid M/N section combinations."""
        pass
    
    @abstractmethod
    def generate_block_params(self) -> Iterator[Tuple[int, int, int, int]]:
        """Generate valid block parameter combinations."""
        pass
    
    @abstractmethod
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about generated parameter space."""
        pass


class GridSearchGenerator(ParameterGenerator):
    """Grid search generator using predefined discrete value sets."""
    
    def __init__(self, validator: ConstraintValidator):
        self.validator = validator
        
        self.m_sections_values = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24]
        self.n_sections_values = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24]
        self.m_blocks_values = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
        self.n_blocks_values = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
        self.k_blocks_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        self.db_blocks_values = [1, 2, 4, 8, 16, 32, 64]
    
    def generate_mn_sections(self) -> Iterator[Tuple[int, int]]:
        """Generate M/N section combinations from predefined grid."""
        for m_sec in self.m_sections_values:
            max_n_sec = self.validator.max_cores // m_sec
            valid_n_sec = [n for n in self.n_sections_values if n <= max_n_sec]
            for n_sec in valid_n_sec:
                yield m_sec, n_sec
    
    def generate_block_params(self) -> Iterator[Tuple[int, int, int, int]]:
        """Generate block combinations from predefined grid with constraint validation."""
        for m_blk in self.m_blocks_values:
            for n_blk in self.n_blocks_values:
                if m_blk * n_blk > self.validator.max_blocks:
                    continue
                
                max_k = self.validator.get_max_k_blocks(m_blk, n_blk)
                for k_blk in self.k_blocks_values:
                    if k_blk > max_k:
                        continue
                    
                    max_db = min(self.validator.get_max_db_blocks(m_blk, n_blk), k_blk)
                    for db_blk in self.db_blocks_values:
                        if db_blk > max_db:
                            continue
                        
                        if self.validator.validate_o_blocks(m_blk, n_blk, k_blk, db_blk):
                            yield m_blk, n_blk, k_blk, db_blk
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get grid search generation statistics."""
        mn_count = len(list(self.generate_mn_sections()))
        block_count = len(list(self.generate_block_params()))
        
        return {
            'generator_type': 'grid_search',
            'mn_sections_grid_size': len(self.m_sections_values) * len(self.n_sections_values),
            'valid_mn_combinations': mn_count,
            'block_params_grid_size': len(self.m_blocks_values) * len(self.n_blocks_values) * 
                                      len(self.k_blocks_values) * len(self.db_blocks_values),
            'valid_block_combinations': block_count,
            'total_combinations': mn_count * block_count
        }


class ExhaustiveGenerator(ParameterGenerator):
    """Exhaustive generator for complete parameter space coverage."""
    
    def __init__(self, validator: ConstraintValidator):
        self.validator = validator
    
    def generate_mn_sections(self) -> Iterator[Tuple[int, int]]:
        """Generate all valid M/N section combinations."""
        for m_sec in range(1, self.validator.max_cores + 1):
            max_n_sec = min(self.validator.max_cores, self.validator.max_cores // m_sec)
            for n_sec in range(1, max_n_sec + 1):
                yield m_sec, n_sec
    
    def generate_block_params(self) -> Iterator[Tuple[int, int, int, int]]:
        """Generate all valid block parameter combinations."""
        for m_blk in range(1, self.validator.max_blocks + 1):
            for n_blk in range(1, self.validator.max_blocks + 1):
                if m_blk * n_blk > self.validator.max_blocks:
                    continue
                
                max_k = self.validator.get_max_k_blocks(m_blk, n_blk)
                for k_blk in range(1, max_k + 1):
                    max_db = self.validator.get_max_db_blocks(m_blk, n_blk)
                    for db_blk in range(1, max_db + 1):
                        if self.validator.validate_o_blocks(m_blk, n_blk, k_blk, db_blk):
                            yield m_blk, n_blk, k_blk, db_blk
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get exhaustive generation statistics."""
        mn_count = len(list(self.generate_mn_sections()))
        block_count = len(list(self.generate_block_params()))
        
        return {
            'generator_type': 'exhaustive',
            'mn_sections_range': f'[1, {self.validator.max_cores}]',
            'valid_mn_combinations': mn_count,
            'block_params_range': f'[1, {self.validator.max_blocks}]',
            'valid_block_combinations': block_count,
            'total_combinations': mn_count * block_count
        }


class Parameter:
    """Parameter manager for GEMM benchmark configuration."""
    
    def __init__(self, generator_type: str = "grid"):
        self.validator = ConstraintValidator()
        self.logger = BenchmarkLogger("ParameterManager")
        
        if generator_type == "grid":
            self.generator = GridSearchGenerator(self.validator)
        elif generator_type == "exhaustive":
            self.generator = ExhaustiveGenerator(self.validator)
        else:
            raise ValueError(f"Unknown generator type: {generator_type}. "
                           f"Supported types: 'grid', 'exhaustive'")
        
        self.grid_parameters = self._generate_all_parameters()
        self._log_generation_stats()
    
    def _generate_all_parameters(self) -> List[Dict[str, int]]:
        """Generate complete parameter list."""
        mn_sections = list(self.generator.generate_mn_sections())
        block_params = list(self.generator.generate_block_params())
        
        parameters = []
        for m_sec, n_sec in mn_sections:
            for m_blk, n_blk, k_blk, db_blk in block_params:
                parameters.append({
                    'm_sections': m_sec,
                    'n_sections': n_sec,
                    'm_sec_o_blocks': m_blk,
                    'n_sec_o_blocks': n_blk,
                    'k_o_iter_blocks': k_blk,
                    'db_o_blocks': db_blk
                })
        
        return parameters
    
    def _log_generation_stats(self):
        """Log parameter generation statistics."""
        stats = self.generator.get_generation_stats()
        self.logger.info(f"Parameter generation statistics:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")
    
    def filter_parameters(self, shape: List[int]) -> List[Dict[str, int]]:
        """Filter parameters based on matrix shape constraints."""
        filtered = [p for p in self.grid_parameters 
                   if self.validator.validate_shape_bounds(shape, p)]
        self.logger.info(f"Filtered {len(filtered)} parameters for shape {shape}")
        return filtered
    
    def get_params_with_idx(self, shape: List[int], idx: int) -> Dict[str, int]:
        """Get specific parameter by index."""
        params = self.filter_parameters(shape)
        if idx < 0 or idx >= len(params):
            raise IndexError(f"Parameter index {idx} out of range [0, {len(params)})")
        return params[idx]
    
    def get_parameter_summary(self, shape: Optional[List[int]] = None) -> Dict[str, Any]:
        """Get parameter summary statistics."""
        params = self.grid_parameters if shape is None else self.filter_parameters(shape)
        
        if not params:
            return {'total_count': 0}
        
        summary = {
            'total_count': len(params),
            'm_sections_range': (min(p['m_sections'] for p in params),
                                max(p['m_sections'] for p in params)),
            'n_sections_range': (min(p['n_sections'] for p in params),
                                max(p['n_sections'] for p in params)),
            'm_blocks_range': (min(p['m_sec_o_blocks'] for p in params),
                              max(p['m_sec_o_blocks'] for p in params)),
            'n_blocks_range': (min(p['n_sec_o_blocks'] for p in params),
                              max(p['n_sec_o_blocks'] for p in params)),
            'k_blocks_range': (min(p['k_o_iter_blocks'] for p in params),
                              max(p['k_o_iter_blocks'] for p in params)),
            'db_blocks_range': (min(p['db_o_blocks'] for p in params),
                               max(p['db_o_blocks'] for p in params))
        }
        
        return summary


@dataclass
class BenchmarkResult:
    """Data class for benchmark execution results."""
    idx: int
    M: int
    N: int
    K: int
    time: float
    diff: float
    negative: bool
    parameters: Dict[str, int] = field(default_factory=lambda: {
        'm_sections': 0, 'n_sections': 0, 'm_sec_o_blocks': 0,
        'n_sec_o_blocks': 0, 'k_o_iter_blocks': 0, 'db_o_blocks': 0
    })
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BenchmarkResult':
        """Create result from dictionary."""
        return cls(
            idx=data['idx'], M=data['M'], N=data['N'], K=data['K'],
            time=data['time'], diff=data['diff'], negative=data['negative'],
            parameters=data.get('parameters', {})
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def is_valid(self) -> bool:
        """Check if result indicates valid computation."""
        return self.time > 0 and self.time < float('inf') and self.diff < error_tolerance
    
    def get_tflops(self) -> float:
        """Calculate TFLOPS performance."""
        if self.time <= 0 or self.time == float('inf'):
            return 0.0
        operations = 2 * self.M * self.N * self.K
        time_seconds = self.time / 1e6
        return operations / time_seconds / 1e12
    
    def get_summary(self) -> Dict[str, Any]:
        """Get result summary."""
        return {
            'index': self.idx,
            'shape': f"[{self.M}, {self.N}, {self.K}]",
            'time_us': self.time,
            'error_ratio': self.diff,
            'tflops': self.get_tflops(),
            'is_valid': self.is_valid(),
            'has_negative': self.negative
        }


class DataGenerator:
    """Matrix data generator for benchmark testing."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.logger = BenchmarkLogger("DataGenerator")
    
    def generate_heavy_tail_data(self, shape: Tuple[int, int], mean: float = 1.0,
                                  sigma: float = 1.2, clip_min: float = 1.0,
                                  clip_max: float = 10.0) -> np.ndarray:
        """Generate matrix with heavy-tail distribution."""
        data = self.rng.lognormal(mean=mean, sigma=sigma, size=shape)
        clipped = np.clip(data, clip_min, clip_max).astype(np.float16)
        self.logger.debug(f"Generated {shape} heavy-tail data")
        return clipped
    
    def generate_test_matrices(self, M: int, N: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate input matrices A and B."""
        mat_a = self.generate_heavy_tail_data((M, K))
        mat_b = self.generate_heavy_tail_data((K, N))
        return mat_a, mat_b
    
    def compute_golden_result(self, mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
        """Compute golden reference using float32."""
        return np.matmul(mat_a.astype(np.float32), 
                        mat_b.astype(np.float32)).astype(np.float32)
    
    def save_binary_data(self, mat_a: np.ndarray, mat_b: np.ndarray,
                         output_dir: str = "./input"):
        """Save matrices to binary files."""
        os.makedirs(output_dir, exist_ok=True)
        mat_a.tofile(os.path.join(output_dir, "x1_gm.bin"))
        mat_b.tofile(os.path.join(output_dir, "x2_gm.bin"))
        self.logger.debug(f"Saved binary data to {output_dir}")
    
    def generate_full_test_data(self, M: int, N: int, K: int,
                                device: str = "npu") -> Tuple[Tensor, Tensor, np.ndarray]:
        """Generate complete test data with NPU tensors."""
        if device == "npu" and not torch.npu.is_available():
            raise AssertionError("NPU device requested but not available")
        
        target_device = torch.device("npu" if device == "npu" else "cpu")
        
        mat_a, mat_b = self.generate_test_matrices(M, N, K)
        golden = self.compute_golden_result(mat_a, mat_b)
        self.save_binary_data(mat_a, mat_b)
        
        os.makedirs("output", exist_ok=True)
        golden.tofile("./output/golden.bin")
        
        tensor_a = torch.tensor(mat_a, device=target_device, dtype=torch.float16)
        tensor_b = torch.tensor(mat_b, device=target_device, dtype=torch.float16)
        
        self.logger.info(f"Generated test data for [{M}, {N}, {K}] on {device}")
        return tensor_a, tensor_b, golden
    
    def get_data_statistics(self, mat_a: np.ndarray, mat_b: np.ndarray) -> Dict[str, Any]:
        """Get statistical summary of generated matrices."""
        stats = {
            'matrix_a': {
                'shape': mat_a.shape,
                'dtype': mat_a.dtype,
                'mean': float(np.mean(mat_a)),
                'std': float(np.std(mat_a)),
                'min': float(np.min(mat_a)),
                'max': float(np.max(mat_a))
            },
            'matrix_b': {
                'shape': mat_b.shape,
                'dtype': mat_b.dtype,
                'mean': float(np.mean(mat_b)),
                'std': float(np.std(mat_b)),
                'min': float(np.min(mat_b)),
                'max': float(np.max(mat_b))
            }
        }
        return stats


class PerformanceProfiler:
    """Performance profiler using msprof tool for Ascend NPU."""
    
    def __init__(self, msp_bench_path: str, msp_output_dir: str = "./msp",
                 timeout: int = DEFAULT_TIMEOUT):
        self.msp_bench_path = msp_bench_path
        self.msp_output_dir = msp_output_dir
        self.timeout = timeout
        self.logger = BenchmarkLogger("PerformanceProfiler")
        os.makedirs(msp_output_dir, exist_ok=True)
    
    def profile_operation(self, rank_id: int, shape: List[int],
                         parameters: Dict[str, int]) -> float:
        """Execute profiling and extract execution time."""
        param_str = self._build_param_string(rank_id, shape, parameters)
        cmd_str = self._build_profiling_command(param_str)
        
        try:
            result = subprocess.run(
                cmd_str, shell=True, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=self.timeout
            )
            return self._parse_profiling_result(result.stdout.decode('utf-8'))
            
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Profiling timeout for shape {shape}")
            return float('inf')
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Profiling command failed: {e}")
            return float('inf')
        except Exception as e:
            self.logger.exception(f"Unexpected profiling error: {e}")
            return float('inf')
    
    def _build_param_string(self, rank_id: int, shape: List[int],
                            params: Dict[str, int]) -> str:
        """Build parameter string for profiling command."""
        M, N, K = shape
        return f"{rank_id} {M} {N} {K} " \
               f"{params['m_sections']} {params['n_sections']} " \
               f"{params['m_sec_o_blocks']} {params['n_sec_o_blocks']} " \
               f"{params['k_o_iter_blocks']} {params['db_o_blocks']}"
    
    def _build_profiling_command(self, param_str: str) -> str:
        """Build complete profiling command."""
        return f"msprof op --output={self.msp_output_dir} " \
               f"--aic-metrics='PipeUtilization' " \
               f"--kernel-name='mmad' {self.msp_bench_path} {param_str}"
    
    def _parse_profiling_result(self, output: str) -> float:
        """Parse profiler output to extract execution time."""
        pattern = r'Task Duration\(us\): (\d+\.\d+)'
        match = re.search(pattern, output)
        
        if match:
            time_us = float(match.group(1))
            self.logger.debug(f"Parsed execution time: {time_us} us")
            return time_us
        
        self.logger.warning(f"Failed to parse result: {output[:200]}")
        return 999999999.0
    
    def get_profiler_config(self) -> Dict[str, Any]:
        """Get profiler configuration."""
        return {
            'msp_bench_path': self.msp_bench_path,
            'output_directory': self.msp_output_dir,
            'timeout_seconds': self.timeout,
            'profiling_metrics': 'PipeUtilization',
            'kernel_name': 'mmad'
        }


class ResultValidator:
    """Result correctness validator."""
    
    def __init__(self, rel_tol: float = relative_tol, abs_tol: float = absolute_tol,
                 err_tol: float = error_tol):
        self.relative_tol = rel_tol
        self.absolute_tol = abs_tol
        self.error_tol = err_tol
        self.logger = BenchmarkLogger("ResultValidator")
    
    def validate_result(self, computed: Tensor, golden: np.ndarray) -> Tuple[bool, float]:
        """Validate computed result against golden reference."""
        computed_np = computed.cpu().numpy()
        computed_flat = computed_np.reshape(-1)
        golden_flat = golden.reshape(-1)
        
        if computed_flat.size != golden_flat.size:
            self.logger.error(f"Size mismatch: computed {computed_flat.size}, "
                            f"golden {golden_flat.size}")
            return False, 1.0
        
        close_elements = np.isclose(
            computed_flat, golden_flat,
            rtol=self.relative_tol, atol=self.absolute_tol, equal_nan=True
        )
        
        diff_indices = np.where(~close_elements)[0]
        error_ratio = float(diff_indices.size) / golden_flat.size
        
        is_valid = error_ratio <= self.error_tol
        self.logger.debug(f"Validation: valid={is_valid}, error_ratio={error_ratio:.6f}")
        
        return is_valid, error_ratio
    
    def analyze_errors(self, computed: Tensor, golden: np.ndarray,
                       max_display: int = MAX_ERROR_DISPLAY) -> Dict[str, Any]:
        """Analyze error patterns in result."""
        computed_np = computed.cpu().numpy()
        computed_flat = computed_np.reshape(-1)
        golden_flat = golden.reshape(-1)
        
        close = np.isclose(computed_flat, golden_flat,
                          rtol=self.relative_tol, atol=self.absolute_tol, equal_nan=True)
        diff_indices = np.where(~close)[0]
        
        errors = []
        for idx in diff_indices[:max_display]:
            expected = golden_flat[idx]
            actual = computed_flat[idx]
            rel_diff = abs(actual - expected) / abs(expected) if expected != 0 else abs(actual)
            errors.append({
                'index': idx, 'expected': expected, 'actual': actual, 'relative_diff': rel_diff
            })
        
        return {
            'error_count': diff_indices.size,
            'total_count': golden_flat.size,
            'error_ratio': float(diff_indices.size) / golden_flat.size,
            'max_relative_error': max(e['relative_diff'] for e in errors) if errors else 0,
            'error_samples': errors
        }
    
    def check_negative_values(self, result: Tensor) -> Tuple[bool, int]:
        """Check for unexpected negative values."""
        has_negative = torch.any(result < 0).item()
        
        if has_negative:
            neg_indices = torch.where(result < 0)
            neg_count = len(neg_indices[0])
            self.logger.warning(f"Found {neg_count} negative values")
            return True, neg_count
        
        return False, 0
    
    def get_tolerance_config(self) -> Dict[str, float]:
        """Get tolerance configuration."""
        return {
            'relative_tolerance': self.relative_tol,
            'absolute_tolerance': self.absolute_tol,
            'error_threshold': self.error_tol
        }


class CheckpointManager:
    """Checkpoint manager for fault-tolerant execution."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.logger = BenchmarkLogger("CheckpointManager")
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def get_checkpoint_path(self, shape: List[int], rank_id: int) -> str:
        """Generate checkpoint file path."""
        shape_str = '_'.join(map(str, shape))
        filename = f'shape_{shape_str}_rank_{rank_id}_checkpoint.json'
        return os.path.join(self.checkpoint_dir, filename)
    
    def save_checkpoint(self, checkpoint_path: str, last_idx: int,
                        metadata: Optional[Dict] = None):
        """Save checkpoint with progress information."""
        data = {
            'last_process_idx': last_idx,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metadata': metadata or {}
        }
        
        try:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=3, ensure_ascii=False)
            self.logger.debug(f"Saved checkpoint: idx={last_idx}")
        except IOError as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint to get last completed index."""
        if not os.path.exists(checkpoint_path):
            self.logger.debug("No existing checkpoint")
            return -1
        
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            last_idx = data.get('last_process_idx', -1)
            self.logger.info(f"Loaded checkpoint: idx={last_idx}")
            return last_idx
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return -1
    
    def clear_checkpoint(self, checkpoint_path: str):
        """Clear checkpoint after completion."""
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                self.logger.debug("Checkpoint cleared")
            except IOError as e:
                self.logger.warning(f"Failed to clear checkpoint: {e}")
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Get detailed checkpoint information."""
        if not os.path.exists(checkpoint_path):
            return None
        
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to read checkpoint info: {e}")
            return None


class ResultStorage:
    """Result storage manager with caching and aggregation support."""
    
    def __init__(self, result_dir: str = "./results", cache_size: int = 100):
        self.result_dir = result_dir
        self.logger = BenchmarkLogger("ResultStorage")
        self.cache: List[Dict] = []
        self.cache_size = cache_size
        os.makedirs(result_dir, exist_ok=True)
    
    def get_result_path(self, shape: List[int], rank_id: int) -> str:
        """Generate result file path."""
        shape_str = '_'.join(map(str, shape))
        filename = f'shape_{shape_str}_rank_{rank_id}.jsonl'
        return os.path.join(self.result_dir, filename)
    
    def save_result(self, result: BenchmarkResult, result_path: str, use_cache: bool = True):
        """Save benchmark result."""
        result_dict = result.to_dict()
        
        if use_cache:
            self.cache.append(result_dict)
            if len(self.cache) >= self.cache_size:
                self._flush_cache(result_path)
        else:
            self._write_single(result_dict, result_path)
    
    def _flush_cache(self, result_path: str):
        """Flush cached results to file."""
        try:
            Path(result_path).parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, 'a', encoding='utf-8') as f:
                for res in self.cache:
                    json.dump(res, f, ensure_ascii=False)
                    f.write('\n')
            self.logger.debug(f"Flushed {len(self.cache)} results")
            self.cache.clear()
        except IOError as e:
            self.logger.error(f"Flush failed: {e}")
    
    def _write_single(self, result_dict: Dict, result_path: str):
        """Write single result."""
        try:
            with open(result_path, 'a', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False)
                f.write('\n')
        except IOError as e:
            self.logger.error(f"Write failed: {e}")
    
    def finalize(self, result_path: str):
        """Finalize by flushing remaining cache."""
        if self.cache:
            self._flush_cache(result_path)
    
    def load_results(self, result_path: str) -> List[BenchmarkResult]:
        """Load all results from file."""
        results = []
        if not os.path.exists(result_path):
            self.logger.warning(f"Result file not found: {result_path}")
            return results
        
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        results.append(BenchmarkResult.from_dict(json.loads(line)))
            self.logger.info(f"Loaded {len(results)} results")
        except Exception as e:
            self.logger.error(f"Load failed: {e}")
        
        return results
    
    def aggregate_results_by_shape(self, results: List[BenchmarkResult]) -> Dict[str, List[BenchmarkResult]]:
        """Aggregate results by matrix shape."""
        grouped = defaultdict(list)
        for result in results:
            key = f"[{result.M}, {result.N}, {result.K}]"
            grouped[key].append(result)
        return dict(grouped)
    
    def get_result_statistics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Get statistical summary of results."""
        if not results:
            return {'total_count': 0}
        
        valid_results = [r for r in results if r.is_valid()]
        times = [r.time for r in valid_results if r.time != float('inf')]
        
        stats = {
            'total_count': len(results),
            'valid_count': len(valid_results),
            'invalid_count': len(results) - len(valid_results),
            'min_time_us': min(times) if times else 0,
            'max_time_us': max(times) if times else 0,
            'avg_time_us': statistics.mean(times) if times else 0,
            'median_time_us': statistics.median(times) if times else 0,
            'std_time_us': statistics.stdev(times) if len(times) > 1 else 0
        }
        
        return stats


class GEMMBenchmarkRunner:
    """Main benchmark runner for distributed GEMM testing."""
    
    def __init__(self, shape_group: List[List[int]], rank_id: int, num_processes: int,
                 msp_bench_path: str, result_dir: str = "./results", msp_dir: str = "./msp",
                 generator_type: str = "grid"):
        self.shape_group = shape_group
        self.rank_id = rank_id
        self.num_processes = num_processes
        
        self.parameters = Parameter(generator_type)
        self.data_generator = DataGenerator()
        self.profiler = PerformanceProfiler(msp_bench_path, msp_dir)
        self.validator = ResultValidator()
        self.checkpoint_manager = CheckpointManager(result_dir)
        self.result_storage = ResultStorage(result_dir)
        
        log_file = os.path.join(result_dir, f"rank_{rank_id}.log")
        self.logger = BenchmarkLogger(f"GEMMBenchmarkRunner_rank{rank_id}", log_file=log_file)
        
        self._log_initialization_info()
    
    def _log_initialization_info(self):
        """Log initialization configuration."""
        self.logger.info(f"Benchmark Runner Initialized")
        self.logger.info(f"  Rank ID: {self.rank_id}")
        self.logger.info(f"  Total Processes: {self.num_processes}")
        self.logger.info(f"  Shape Group Size: {len(self.shape_group)}")
        self.logger.info(f"  Generator Type: {self.parameters.generator.__class__.__name__}")
    
    def benchmark_shape(self, shape: List[int]):
        """Execute benchmark for a specific matrix shape."""
        M, N, K = shape
        self.logger.info(f"Starting benchmark for shape [{M}, {N}, {K}]")
        
        result_path = self.result_storage.get_result_path(shape, self.rank_id)
        checkpoint_path = self.checkpoint_manager.get_checkpoint_path(shape, self.rank_id)
        
        filter_params = self.parameters.filter_parameters(shape)
        total_tasks = len(filter_params)
        tasks_per_process = math.ceil(total_tasks / self.num_processes)
        
        start_idx = self.rank_id * tasks_per_process
        end_idx = min(start_idx + tasks_per_process, total_tasks)
        process_params = filter_params[start_idx:end_idx]
        process_task_count = len(process_params)
        
        self.logger.info(f"Task assignment: {process_task_count} tasks [{start_idx}-{end_idx-1}]")
        
        if process_task_count == 0:
            self.logger.warning(f"No tasks assigned to rank {self.rank_id}")
            return
        
        last_idx = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        start_local = 0
        
        if last_idx >= start_idx:
            start_local = last_idx - start_idx
            if start_local >= process_task_count:
                self.logger.info(f"All tasks already completed for rank {self.rank_id}")
                return
        
        a_npu, b_npu, golden = self.data_generator.generate_full_test_data(M, N, K)
        completed = max(0, last_idx - start_idx) if last_idx >= start_idx else 0
        
        with tqdm(total=process_task_count, initial=completed,
                  desc=f"Rank {self.rank_id} [{M},{N},{K}]",
                  postfix={"Processed": completed}) as pbar:
            
            local_idx = start_local
            while local_idx < process_task_count:
                global_idx = start_idx + local_idx
                params = process_params[local_idx]
                
                if global_idx == last_idx:
                    self.logger.warning(f"Skipping problematic index: {global_idx}")
                    self._save_error_result(shape, global_idx, params, result_path)
                    local_idx += 1
                    pbar.update(1)
                    continue
                
                self.checkpoint_manager.save_checkpoint(
                    checkpoint_path, global_idx, {'shape': shape, 'rank': self.rank_id})
                
                try:
                    output, _ = self._execute_gemm(a_npu, b_npu, params)
                    is_valid, error_ratio = self.validator.validate_result(output, golden)
                    has_negative, _ = self.validator.check_negative_values(output)
                    
                    if is_valid:
                        time_us = self.profiler.profile_operation(self.rank_id, shape, params)
                    else:
                        time_us = float('inf')
                        self.logger.warning(f"Invalid result at idx {global_idx}")
                    
                    result = BenchmarkResult(
                        idx=global_idx, M=M, N=N, K=K, time=time_us,
                        diff=error_ratio, negative=has_negative, parameters=params
                    )
                    self.result_storage.save_result(result, result_path)
                    
                except Exception as e:
                    self.logger.exception(f"Error executing task {global_idx}: {e}")
                    self._save_error_result(shape, global_idx, params, result_path)
                
                local_idx += 1
                pbar.update(1)
                pbar.set_postfix({'Processed': local_idx, 'Global': global_idx})
        
        self.result_storage.finalize(result_path)
        if local_idx >= process_task_count:
            self.checkpoint_manager.clear_checkpoint(checkpoint_path)
        
        self.logger.info(f"Completed benchmark for shape [{M}, {N}, {K}]")
    
    def _execute_gemm(self, a_npu: Tensor, b_npu: Tensor,
                      params: Dict[str, int]) -> Tuple[Tensor, Tensor]:
        """Execute GEMM operation."""
        param_list = list(params.values()) + [0] * 22
        param_npu = torch.tensor(param_list, device='npu', dtype=torch.int32)
        
        z_shape = [a_npu.size(0), b_npu.size(1)]
        z_npu = torch.empty(z_shape, device='npu', dtype=torch.float32)
        
        deep_gemm_ascend.run_mmad_bench(a_npu, b_npu, z_npu, param_npu)
        return z_npu, param_npu
    
    def _save_error_result(self, shape: List[int], idx: int,
                           params: Dict[str, int], result_path: str):
        """Save error result."""
        error_result = BenchmarkResult(
            idx=idx, M=shape[0], N=shape[1], K=shape[2],
            time=-1, diff=-1, negative=True, parameters=params
        )
        self.result_storage.save_result(error_result, result_path)
    
    def run_benchmarks(self):
        """Execute benchmarks for all shapes."""
        self.logger.log_section("STARTING GEMM BENCHMARK")
        
        for i, shape in enumerate(self.shape_group):
            self.logger.info(f"Processing shape {i+1}/{len(self.shape_group)}: {shape}")
            try:
                self.benchmark_shape(shape)
            except Exception as e:
                self.logger.exception(f"Failed to benchmark shape {shape}: {e}")
                continue
        
        self.logger.log_section("COMPLETED GEMM BENCHMARK")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='GEMM Benchmark Tool for DeepGEMM on Ascend NPU',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--rank-id', type=int, default=0,
                       help='Process rank identifier for distributed execution')
    parser.add_argument('--num-processes', type=int, default=1,
                       help='Total number of parallel processes')
    parser.add_argument('--msp-bench-path', type=str, required=True,
                       help='Path to msprof benchmark binary')
    parser.add_argument('--result-dir', type=str, default='./results',
                       help='Directory for result files')
    parser.add_argument('--msp-dir', type=str, default='./msp',
                       help='Directory for profiler output')
    parser.add_argument('--generator-type', type=str, choices=['grid', 'exhaustive'],
                       default='grid', help='Parameter generation strategy')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    runner = GEMMBenchmarkRunner(
        shape_group=shape_group,
        rank_id=args.rank_id,
        num_processes=args.num_processes,
        msp_bench_path=args.msp_bench_path,
        result_dir=args.result_dir,
        msp_dir=args.msp_dir,
        generator_type=args.generator_type
    )
    
    runner.run_benchmarks()


if __name__ == "__main__":
    main()