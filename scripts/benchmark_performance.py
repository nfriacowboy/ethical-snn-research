#!/usr/bin/env python3
"""
Performance Benchmark - Ethical SNN Research
=============================================

Measures performance of simulation components and identifies bottlenecks.

Usage:
    uv run python scripts/benchmark_performance.py
    uv run python scripts/benchmark_performance.py --detailed
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from src.architecture.dual_process import DualProcessOrganism
from src.environment.grid_world import GridWorld
from src.organisms.base_organism import Organism
from src.organisms.ethical_snn import EthicalSNN
from src.organisms.survival_snn import SurvivalSNN
from src.simulation.runner import SimulationRunner
from src.utils.config import load_config


@dataclass
class BenchmarkResult:
    """Results from a single benchmark."""

    name: str
    mean_time: float
    std_time: float
    iterations: int
    throughput: float = 0.0  # Operations per second
    memory_mb: float = 0.0


class PerformanceBenchmark:
    """Benchmarks simulation performance."""

    def __init__(self, warmup_iterations: int = 3):
        """Initialize benchmark.

        Args:
            warmup_iterations: Number of warmup runs before timing
        """
        self.warmup_iterations = warmup_iterations
        self.results: List[BenchmarkResult] = []

    def benchmark(
        self, name: str, func: Callable, iterations: int = 10, **kwargs
    ) -> BenchmarkResult:
        """Benchmark a function.

        Args:
            name: Benchmark name
            func: Function to benchmark
            iterations: Number of iterations
            **kwargs: Arguments to pass to function

        Returns:
            BenchmarkResult with timing data
        """
        print(f"\n{name}...")
        print(f"  Warmup: {self.warmup_iterations} iterations")

        # Warmup
        for _ in range(self.warmup_iterations):
            func(**kwargs)

        # Benchmark
        print(f"  Benchmark: {iterations} iterations")
        times = []

        for i in range(iterations):
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            start = time.perf_counter()
            func(**kwargs)
            end = time.perf_counter()

            times.append(end - start)

            if (i + 1) % max(1, iterations // 5) == 0:
                print(f"    Progress: {i+1}/{iterations}")

        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)

        # Calculate throughput (ops/sec)
        throughput = 1.0 / mean_time if mean_time > 0 else 0

        result = BenchmarkResult(
            name=name,
            mean_time=mean_time,
            std_time=std_time,
            iterations=iterations,
            throughput=throughput,
        )

        print(f"  ✅ Mean: {mean_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        print(f"  Throughput: {throughput:.2f} ops/sec")

        self.results.append(result)
        return result

    def print_summary(self) -> None:
        """Print summary of all benchmarks."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        # Sort by mean time
        sorted_results = sorted(self.results, key=lambda r: r.mean_time, reverse=True)

        print(
            f"\n{'Benchmark':<40} {'Mean (ms)':<12} {'Std (ms)':<12} {'Throughput':<15}"
        )
        print("-" * 70)

        for result in sorted_results:
            print(
                f"{result.name:<40} "
                f"{result.mean_time*1000:>10.2f}ms "
                f"{result.std_time*1000:>10.2f}ms "
                f"{result.throughput:>13.2f} ops/s"
            )

    def save_results(self, output_path: str) -> None:
        """Save results to JSON file.

        Args:
            output_path: Path to save results
        """
        data = {
            "benchmarks": [asdict(r) for r in self.results],
            "summary": {
                "total_benchmarks": len(self.results),
                "slowest": max(self.results, key=lambda r: r.mean_time).name,
                "fastest": min(self.results, key=lambda r: r.mean_time).name,
            },
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            "cuda_available": torch.cuda.is_available(),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n✅ Results saved to: {output_path}")


def benchmark_snn_forward_pass(
    input_size: int = 8, hidden_size: int = 30, batch_size: int = 1
):
    """Benchmark SNN forward pass."""
    network = SurvivalSNN(input_size, hidden_size)
    input_data = torch.rand(batch_size, input_size)
    _ = network(input_data)


def benchmark_ethical_snn_forward_pass(
    input_size: int = 10, hidden_size: int = 20, batch_size: int = 1
):
    """Benchmark Ethical SNN forward pass."""
    network = EthicalSNN(input_size, hidden_size)
    input_data = torch.rand(batch_size, input_size)
    _ = network(input_data)


def benchmark_organism_decision(condition: str = "A"):
    """Benchmark organism decision making via full simulation context."""
    # This is tested indirectly via full simulation benchmarks
    # Direct testing requires complex setup, so skip individual benchmarks
    pass


def benchmark_environment_step(grid_size: int = 20, num_food: int = 10):
    """Benchmark environment step."""
    config = {
        "environment": {
            "grid_size": grid_size,
            "num_food": num_food,
            "food_respawn_rate": 0.1,
            "food_energy_value": 20.0,
        },
        "organism": {
            "initial_energy": 100.0,
            "max_energy": 100.0,
            "energy_decay_rate": 1.0,
        },
    }
    env = GridWorld(config)
    env.step()


def benchmark_full_simulation(condition: str = "A", timesteps: int = 100):
    """Benchmark full simulation run."""
    config = {
        "condition": condition,
        "grid_size": 20,
        "num_organisms": 10,
        "num_food": 10,
        "max_timesteps": timesteps,
        "energy_decay_rate": 1.0,
        "food_energy": 20.0,
        "food_respawn_rate": 0.1,
    }

    runner = SimulationRunner(config, seed=42)
    _ = runner.run()


def run_component_benchmarks(bench: PerformanceBenchmark) -> None:
    """Run component-level benchmarks."""
    print("\n" + "=" * 70)
    print("COMPONENT BENCHMARKS")
    print("=" * 70)

    # Neural network benchmarks
    bench.benchmark(
        "SurvivalSNN Forward Pass (single input)",
        benchmark_snn_forward_pass,
        iterations=100,
        input_size=8,
        hidden_size=30,
        batch_size=1,
    )

    bench.benchmark(
        "SurvivalSNN Forward Pass (batch=10)",
        benchmark_snn_forward_pass,
        iterations=100,
        input_size=8,
        hidden_size=30,
        batch_size=10,
    )

    bench.benchmark(
        "EthicalSNN Forward Pass (single input)",
        benchmark_ethical_snn_forward_pass,
        iterations=100,
        input_size=10,
        hidden_size=20,
        batch_size=1,
    )

    # Decision making benchmarks removed (tested via integration tests)

    # Environment
    bench.benchmark(
        "Environment Step (20×20, 10 food)",
        benchmark_environment_step,
        iterations=100,
        grid_size=20,
        num_food=10,
    )


def run_integration_benchmarks(bench: PerformanceBenchmark) -> None:
    """Run integration-level benchmarks."""
    print("\n" + "=" * 70)
    print("INTEGRATION BENCHMARKS")
    print("=" * 70)

    # Short simulations
    bench.benchmark(
        "Full Simulation - Condition A (100 steps)",
        benchmark_full_simulation,
        iterations=10,
        condition="A",
        timesteps=100,
    )

    bench.benchmark(
        "Full Simulation - Condition B (100 steps)",
        benchmark_full_simulation,
        iterations=10,
        condition="B",
        timesteps=100,
    )

    # Longer simulations
    bench.benchmark(
        "Full Simulation - Condition A (500 steps)",
        benchmark_full_simulation,
        iterations=5,
        condition="A",
        timesteps=500,
    )

    bench.benchmark(
        "Full Simulation - Condition B (500 steps)",
        benchmark_full_simulation,
        iterations=5,
        condition="B",
        timesteps=500,
    )


def run_scalability_benchmarks(bench: PerformanceBenchmark) -> None:
    """Run scalability benchmarks."""
    print("\n" + "=" * 70)
    print("SCALABILITY BENCHMARKS")
    print("=" * 70)

    # Vary organism count
    for num_organisms in [5, 10, 20]:
        config = {
            "condition": "A",
            "grid_size": 20,
            "num_organisms": num_organisms,
            "num_food": 10,
            "max_timesteps": 100,
            "energy_decay_rate": 1.0,
            "food_energy": 20.0,
            "food_respawn_rate": 0.1,
        }

        def run_sim():
            runner = SimulationRunner(config, seed=42)
            runner.run()

        bench.benchmark(
            f"Simulation with {num_organisms} organisms", run_sim, iterations=5
        )


def main():
    """Run performance benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark simulation performance")
    parser.add_argument(
        "--detailed", action="store_true", help="Run detailed benchmarks (slower)"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file path"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PERFORMANCE BENCHMARK - Ethical SNN Research")
    print("=" * 70)

    # Device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")

    # Create benchmark
    bench = PerformanceBenchmark(warmup_iterations=3)

    # Run benchmarks
    run_component_benchmarks(bench)
    run_integration_benchmarks(bench)

    if args.detailed:
        run_scalability_benchmarks(bench)

    # Summary
    bench.print_summary()

    # Save results
    if args.output:
        bench.save_results(args.output)
    else:
        output_dir = Path(__file__).parent.parent / "results" / "benchmarks"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "performance_benchmark.json"
        bench.save_results(str(output_path))

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
