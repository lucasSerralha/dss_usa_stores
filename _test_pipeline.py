"""Smoke test: run_moead_unsga3_o1.py with N_MAX_GEN=30"""
import sys, importlib.util
sys.path.insert(0, 'src')

spec = importlib.util.spec_from_file_location("run_o1", "scripts/run_moead_unsga3_o1.py")
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.N_MAX_GEN = 30
print("Running O1 smoke test (N_MAX_GEN=30)...")
mod.main()
print("PASSED")
