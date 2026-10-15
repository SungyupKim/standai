"""getup_mpc.py on prone-side lying starts (pitch <= 0, torso < 0.5); prints results, does NOT write the .npz."""
import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/analysis')
import time
from multiprocessing import Pool
import getup_cem as g
import getup_mpc as gm
seeds = [int(x) for x in sys.argv[1:]]
print("prone starts:", seeds, flush=True)
t0 = time.time()
with Pool(6, initializer=g.init_worker) as p:
    for r in p.imap_unordered(gm.run_seed, [(s,) for s in seeds]):
        print(f"seed {r['seed']}: peak z(foot) {r['peak_sup_z']:.2f}, z_end {r['zend']:.2f}, pitch_end {r['pend']:.0f}d, head_end {r['hend']:.2f}, "
              f"stood {r['stood_frac']*100:.0f}%, first upright t {r['first_upright']}, UPRIGHT-STOOD(real) {r['real_ok']}  [{time.time()-t0:.0f}s]", flush=True)
print(f"done in {time.time()-t0:.0f}s")
