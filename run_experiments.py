#!/usr/bin/env python3
"""Hyperparameter exploration system for running experiments."""
from itertools import product
from pathlib import Path
from typing import Any
import argparse
import subprocess

HYPERPARAMETER_GRID = {
    "learning_rate": [1e-4, 5e-5, 1e-5],
    "batch_size": [2, 4, 8, 16],
}


def generate_commands(
    grid: dict[str, list[Any]], 
    base_command: str = "uv run main.py"
) -> list[str]:
    """Generate all command combinations from hyperparameter grid."""
    if not grid:
        return [base_command]
    
    keys = list(grid.keys())
    values = list(grid.values())
    
    commands = []
    for combination in product(*values):
        args = " ".join(f"--{k}={v}" for k, v in zip(keys, combination))
        commands.append(f"{base_command} {args}")
    
    return commands


def write_shell_script(
    commands: list[str], 
    output_path: Path = Path("run_all_experiments.sh")
) -> None:
    """Write commands to a shell script."""
    script_content = "#!/bin/bash\n\n"
    script_content += "# Auto-generated experiment runner\n"
    script_content += f"# Total experiments: {len(commands)}\n\n"
    
    for i, cmd in enumerate(commands, 1):
        script_content += f"echo 'Running experiment {i}/{len(commands)}'\n"
        script_content += f"{cmd}\n\n"
    
    output_path.write_text(script_content)
    output_path.chmod(0o755)
    print(f"✓ Created executable script: {output_path}")
    print(f"  Run with: ./{output_path}")


def run_experiments(commands: list[str]) -> None:
    """Execute all experiments sequentially."""
    total = len(commands)
    print(f"Running {total} experiments...\n")
    
    for i, cmd in enumerate(commands, 1):
        print(f"[{i}/{total}] {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"✗ Experiment {i} failed with error: {e}")
            if input("Continue with next experiment? (y/n): ").lower() != 'y':
                break
        except KeyboardInterrupt:
            print("\n✗ Interrupted by user")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter exploration system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and run all experiments
  python run_experiments.py --run

  # Generate shell script
  python run_experiments.py --script

  # Just preview commands
  python run_experiments.py
        """
    )
    parser.add_argument(
        "--run", 
        action="store_true", 
        help="Execute all experiments immediately"
    )
    parser.add_argument(
        "--script", 
        action="store_true", 
        help="Generate shell script"
    )
    parser.add_argument(
        "--output", 
        type=Path,
        default=Path("run_all_experiments.sh"),
        help="Output path for shell script (default: run_all_experiments.sh)"
    )
    
    args = parser.parse_args()
    
    commands = generate_commands(HYPERPARAMETER_GRID)
    
    print(f"Generated {len(commands)} experiment(s):\n")
    for i, cmd in enumerate(commands, 1):
        print(f"  {i}. {cmd}")
    print()
    
    if args.run:
        run_experiments(commands)
    elif args.script:
        write_shell_script(commands, args.output)
    else:
        print("Preview mode. Use --run to execute or --script to generate shell script.")


if __name__ == "__main__":
    main()
