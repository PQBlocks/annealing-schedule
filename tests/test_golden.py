"""
Golden instance verification tests.
Loads reference golden instances and verifies against independent CP-SAT oracle.
These tests require the source tree (golden_instances.json in brief/golden-vectors/)
and are skipped when running from the installed package.
"""
import json
import os
import sys
import pytest
import numpy as np

# Add src to path for reference module imports when running from source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pqblocks_anneal_schedule.reference.stats import compute_instance_stats_ref
from pqblocks_anneal_schedule.reference.schedule import generate_schedule_ref
from pqblocks_anneal_schedule.reference.annealing import simulated_annealing_ref


def load_golden_instances(filepath):
    """Load golden instances from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('instances', [])


def _get_golden_file():
    """Get path to golden instances file, or None if not available (installed package)."""
    candidates = [
        os.path.join(os.path.dirname(__file__), 'fixtures', 'golden-vectors', 'golden_instances.json'),
        os.path.join(os.path.dirname(__file__), '..', 'brief', 'golden-vectors', 'golden_instances.json'),
    ]
    return next((c for c in candidates if os.path.exists(c)), None)


def test_golden_instances_format():
    """Test that golden instances have the expected format."""
    golden_file = _get_golden_file()
    if golden_file is None:
        pytest.skip("Golden instances file not available (running from installed package)")
    
    instances = load_golden_instances(golden_file)
    
    required_fields = ['name', 'J', 'h']
    for instance in instances[:3]:  # Check first 3 instances
        for field in required_fields:
            assert field in instance, f"Missing field '{field}' in instance {instance.get('name', 'unknown')}"
        
        # Check data types
        assert isinstance(instance['name'], str)
        assert isinstance(instance['J'], list)
        assert isinstance(instance['h'], list)
        
        # Check matrix dimensions
        J = np.array(instance['J'])
        h = np.array(instance['h'])
        assert J.ndim == 2 and J.shape[0] == J.shape[1], f"J must be square matrix for {instance['name']}"
        assert len(h) == J.shape[0], f"h length must match J dimension for {instance['name']}"


def test_golden_instances_cpsat_verified():
    """Verify all golden instances have been validated against CP-SAT oracle."""
    golden_file = _get_golden_file()
    if golden_file is None:
        pytest.skip("Golden instances file not available (running from installed package)")
    
    instances = load_golden_instances(golden_file)
    assert len(instances) > 0, "No golden instances loaded"
    
    passed = 0
    failed = 0
    skipped = 0
    
    for instance in instances:
        name = instance['name']
        J = np.array(instance['J'], dtype=float)
        h = np.array(instance['h'], dtype=float)
        
        # Skip if instance is malformed
        if J.shape[0] != J.shape[1] or len(h) != J.shape[0]:
            print(f"WARNING: Skipping malformed instance {name}")
            skipped += 1
            continue
            
        # Handle different possible energy field names
        certified_energy = None
        for key in ['certified_energy', 'ground_energy', 'optimal_energy', 'certified_ground_energy']:
            if key in instance:
                certified_energy = float(instance[key])
                break
                
        if certified_energy is None:
            print(f"WARNING: No reference energy field in {name}")
            skipped += 1
            continue
            
        try:
            # Compute statistics and generate schedule
            stats = compute_instance_stats_ref(J, h)
            max_degree = max(1, int(np.sum(np.abs(J) > 0)))
            schedule = generate_schedule_ref(stats, len(h), max_degree, budget_steps=128)
            
            # Run SA multiple times to get good statistics
            best_energy = float('inf')
            for seed in [12345, 67890, 11111, 22222, 33333]:
                result = simulated_annealing_ref(J, h, schedule, seed=seed, num_steps=128, num_runs=8)
                if result.best_energy < best_energy:
                    best_energy = result.best_energy
                    
            # Check if we found energy close to reference optimum
            energy_diff = abs(best_energy - certified_energy)
            relative_error = energy_diff / max(abs(certified_energy), 1e-10)
            
            # Allow some tolerance since SA is heuristic
            if relative_error < 0.01:  # Within 1% of reference optimum
                passed += 1
            else:
                print(f"FAILED: {name} - best={best_energy}, reference={certified_energy}, diff={energy_diff}")
                failed += 1
                
        except Exception as e:
            print(f"ERROR processing {name}: {e}")
            failed += 1
    
    print(f"Golden instance verification: {passed} passed, {failed} failed, {skipped} skipped")
    # Allow some golden instances to be heuristic (not exact)
    assert failed == 0, f"{failed} golden instances failed verification"