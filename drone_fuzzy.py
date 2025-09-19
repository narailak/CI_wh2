import os
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ----------------- สร้างคอนโทรลเลอร์ Mamdani (ทั้งหมด12 กฎ) -----------------
def build_mamdani_controller():

    # Define input variables
    e = ctrl.Antecedent(np.linspace(-3, 3, 601), 'error')
    v = ctrl.Antecedent(np.linspace(-3, 3, 601), 'velocity')
    # Define output variable
    u = ctrl.Consequent(np.linspace(0, 100, 601), 'throttle')

    # ===== Memberships  =====
    # Memberships error
    e['VN'] = fuzz.trapmf(e.universe, [-3.0, -2.8, -2.0, -1.0])
    e['N']  = fuzz.trimf(e.universe,  [-2.0, -1.0,  0.0])
    e['Z']  = fuzz.trimf(e.universe,  [-0.7,  0.0,  0.7])
    e['P']  = fuzz.trimf(e.universe,  [ 0.0,  1.0,  2.0])
    e['VP'] = fuzz.trapmf(e.universe, [ 1.0,  2.0,  2.8,  3.0])

    # Memberships velocity
    v['DF'] = fuzz.trapmf(v.universe, [-3.0, -2.8, -2.0, -1.0])
    v['D']  = fuzz.trimf(v.universe,  [-2.0, -1.0,  0.0])
    v['Z']  = fuzz.trimf(v.universe,  [-0.7,  0.0,  0.7])
    v['U']  = fuzz.trimf(v.universe,  [ 0.0,  1.0,  2.0])
    v['UF'] = fuzz.trapmf(v.universe, [ 1.0,  2.0,  2.8,  3.0])

    # Memberships throttle (output)
    u['Low']     = fuzz.trapmf(u.universe, [ 0,  5, 15, 30])
    u['MedLow']  = fuzz.trimf(u.universe,  [20, 35, 50])
    u['Hover']   = fuzz.trimf(u.universe,  [40, 50, 60])
    u['MedHigh'] = fuzz.trimf(u.universe,  [50, 65, 80])
    u['High']    = fuzz.trapmf(u.universe, [70, 85, 95, 100])

    # ===== 12 กฎสำคัญ =====
    rules = [
        ctrl.Rule(e['VP'] & v['DF'], u['High']),
        ctrl.Rule(e['VP'] & v['D'],  u['High']),
        ctrl.Rule(e['VN'] & v['UF'], u['Low']),
        ctrl.Rule(e['VN'] & v['U'],  u['Low']),
        ctrl.Rule(e['P']  & v['DF'], u['High']),
        ctrl.Rule(e['N']  & v['UF'], u['Low']),
        ctrl.Rule(e['Z']  & v['Z'],  u['Hover']),
        ctrl.Rule(e['Z']  & v['D'],  u['MedHigh']),
        ctrl.Rule(e['Z']  & v['U'],  u['MedLow']),
        ctrl.Rule(e['P']  & v['U'],  u['Hover']),
        ctrl.Rule(e['P']  & v['Z'],  u['MedHigh']),
        ctrl.Rule(e['N']  & v['Z'],  u['MedLow']),
    ]
    sys = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(sys)
    return sim, e, v, u

# -------------- เคสทดสอบสำหรับ 12 กฎ (จะใช้ค่ากลางแต่ละเซต) --------------
def test_cases():
    return [
        ("VP & DF -> High",    2.0,  -1.5, "VP", "DF", "High"),
        ("VP & D  -> High",    2.0,  -1.0, "VP", "D",  "High"),
        ("VN & UF -> Low",    -1.5,   2.0, "VN", "UF", "Low"),
        ("VN & U  -> Low",    -1.5,   1.0, "VN", "U",  "Low"),
        ("P  & DF -> High",    1.0,  -1.5, "P",  "DF", "High"),
        ("N  & UF -> Low",    -1.0,   2.0, "N",  "UF", "Low"),
        ("Z  & Z  -> Hover",   0.0,   0.0, "Z",  "Z",  "Hover"),
        ("Z  & D  -> MedHigh", 0.0,  -1.0, "Z",  "D",  "MedHigh"),
        ("Z  & U  -> MedLow",  0.0,   1.0, "Z",  "U",  "MedLow"),
        ("P  & U  -> Hover",   1.0,   1.0, "P",  "U",  "Hover"),
        ("P  & Z  -> MedHigh", 1.0,   0.0, "P",  "Z",  "MedHigh"),
        ("N  & Z  -> MedLow", -1.0,   0.0, "N",  "Z",  "MedLow"),
    ]

# -------------------------- ตัวช่วย plot --------------------------
def plot_one_rule(idx, title, e_in, v_in, e_term, v_term, u_term, e, v, u, outdir):
    mu_e = float(fuzz.interp_membership(e.universe, e.terms[e_term].mf, e_in))
    mu_v = float(fuzz.interp_membership(v.universe, v.terms[v_term].mf, v_in))
    alpha = min(mu_e, mu_v)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    fig.suptitle(f"[{idx:02d}] {title} | alpha={alpha:.2f}", y=1.03, fontsize=12)

    # ---- ส่วน 1: error ----
    ax = axes[0]
    for name, term in e.terms.items():
        ax.plot(e.universe, term.mf, label=name, linewidth=2)
    ax.fill_between(e.universe, 0, np.minimum(e.terms[e_term].mf, alpha), alpha=0.35)
    ax.axvline(e_in, linewidth=2)
    ax.axhline(alpha, ls="--")
    ax.set_title("error (m)")
    ax.set_xlabel("error"); ax.set_ylabel("membership")
    ax.set_ylim(-0.05, 1.05); ax.grid(True); ax.legend(loc="lower right", fontsize=8)

    # ---- ส่วน 2: velocity ----
    ax = axes[1]
    for name, term in v.terms.items():
        ax.plot(v.universe, term.mf, label=name, linewidth=2)
    ax.fill_between(v.universe, 0, np.minimum(v.terms[v_term].mf, alpha), alpha=0.35)
    ax.axvline(v_in, linewidth=2)
    ax.axhline(alpha, ls="--")
    ax.set_title("velocity (m/s)")
    ax.set_xlabel("velocity")
    ax.set_ylim(-0.05, 1.05); ax.grid(True); ax.legend(loc="lower right", fontsize=8)

    # ---- ส่วน 3: output  ----
    ax = axes[2]
    for name, term in u.terms.items():
        ax.plot(u.universe, term.mf, label=name, linewidth=2)
    clipped = np.minimum(u.terms[u_term].mf, alpha)
    ax.fill_between(u.universe, 0, clipped, alpha=0.35)
    # centroid
    area = np.trapezoid(clipped, u.universe)
    if area > 1e-12:
        c = np.trapezoid(u.universe * clipped, u.universe) / area
        ax.axvline(c, linewidth=2)
        ax.text(c, 0.02, f"centroid≈{c:.1f}%", ha="center", va="bottom")
    ax.axhline(alpha, ls="--")
    ax.set_title(f"output: {u_term}")
    ax.set_xlabel("throttle (%)")
    ax.set_ylim(-0.05, 1.05); ax.grid(True); ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"rule_{idx:02d}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path

# -------------------------- main --------------------------
def main():
    sim, e, v, u = build_mamdani_controller()
    cases = test_cases()
    outdir = "./rule_plots"
    paths = []
    for i, (lbl, e_in, v_in, e_term, v_term, u_term) in enumerate(cases, 1):
        p = plot_one_rule(i, lbl, e_in, v_in, e_term, v_term, u_term, e, v, u, outdir)
        paths.append(p)
    print(f"Saved {len(paths)} figures to: {os.path.abspath(outdir)}")
    for p in paths:
        print(" -", p)

if __name__ == "__main__":
    main()
