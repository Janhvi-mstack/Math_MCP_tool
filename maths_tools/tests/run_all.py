"""Run every test suite and report one total."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
suites = ["test_stage12.py", "test_stage3.py", "test_stage56.py",
          "test_stage4.py", "test_stage7.py",
          "test_vendored.py", "test_metrics.py", "test_mcp.py",
          # storage: backend selection, then the AOSS backend itself. All
          # offline -- the AOSS suites drive an in-memory stand-in, so they
          # need no credentials, no network and cost nothing.
          "test_backend.py", "test_aoss_signing.py", "test_aoss_mappings.py",
          "test_aoss_store.py", "test_aoss_doctests.py"]
total_p = total_f = 0
bad = []
for s in suites:
    r = subprocess.run([sys.executable, os.path.join(HERE, s)],
                       capture_output=True, text=True)
    out = r.stdout.strip().splitlines()
    p = next((int(l.split()[1]) for l in out if l.startswith("PASS ")), 0)
    f = next((int(l.split()[1]) for l in out if l.startswith("FAILED ")), 0)
    total_p += p; total_f += f
    print("%-18s %3d pass  %d fail" % (s, p, f))
    if f:
        bad.append((s, "\n".join(l for l in out if l.startswith("FAIL") or l.startswith("     "))))
print("-" * 40)
print("TOTAL %d pass, %d fail" % (total_p, total_f))
for s, msg in bad:
    print("\n--- %s ---\n%s" % (s, msg))
sys.exit(1 if total_f else 0)
