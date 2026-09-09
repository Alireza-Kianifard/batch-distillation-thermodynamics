import os
import csv
import numpy as np

from batch_distillation.models.component import Component
from batch_distillation.models.mixture import Mixture
from batch_distillation.solvers.rachford_rice import flash
from batch_distillation.solvers.bubble_point import calculate_bubble_temperature
from batch_distillation.core.eos_pr import solve_pr_eos_for_z
from batch_distillation.core.enthalpy import calculate_residual_enthalpy
from batch_distillation.simulation.batch_column import simulate_batch_distillation


def build_mixture() -> Mixture:
    c1 = Component(
        name="n-Hexane",
        mw=86.18,
        tc=234.7 + 273.15,
        pc=30.32 * 101325.0,
        omega=0.3007,
        v_liq=0.0001316,
        antoine_coeffs={"a": 70.4265, "b": -6055.6, "c": 0.0, "d": -8.37865, "e": 0.00000662, "f": 2.0},
        cpig_coeffs=[-4.241300, 49.73113, -1.786503, 0.283208, 0.187393],
        hf_298=-167200.0,
    )

    c2 = Component(
        name="Ethanol",
        mw=46.07,
        tc=240.8 + 273.15,
        pc=61.47 * 101325.0,
        omega=0.6444,
        v_liq=0.0000584,
        antoine_coeffs={"a": 86.486, "b": -7931.1, "c": 0.0, "d": -10.2498, "e": 0.00000638949, "f": 2.0},
        cpig_coeffs=[2.783504, 22.03823, -0.678101, 0.078801, -0.012503],
        hf_298=-235300.0,
    )

    c3 = Component(
        name="Methylcyclopentane",
        mw=84.16,
        tc=259.6 + 273.15,
        pc=37.9 * 101325.0,
        omega=0.2389,
        v_liq=0.0001124,
        antoine_coeffs={"a": 71.3356, "b": -6029.95, "c": 0.0, "d": -8.57235, "e": 0.00000716495, "f": 2.0},
        cpig_coeffs=[-10.03820, 53.43210, -2.112900, 0.311700, 0.209200],
        hf_298=-106700.0,
    )

    c4 = Component(
        name="Benzene",
        mw=78.11,
        tc=288.9 + 273.15,
        pc=49.24 * 101325.0,
        omega=0.2150,
        v_liq=0.0000891,
        antoine_coeffs={"a": 169.65, "b": -10314.8, "c": 0.0, "d": -23.5895, "e": 0.0000209, "f": 2.0},
        cpig_coeffs=[-3.392100, 47.39000, -3.017200, 0.713000, 0.190100],
        hf_298=82930.0,
    )

    kij_matrix = [
        [0.0000, 0.0000, 0.0002835, 0.001674],
        [0.0000, 0.0000, 0.0000, 0.0000],
        [0.0002835, 0.0000, 0.0000, 0.0005808],
        [0.001674, 0.0000, 0.0005808, 0.0000],
    ]

    nrtl_tau = [
        [0.0, 859.964, -258.386, 567.411],
        [1081.56, 0.0, 1564.85, 991.507],
        [347.479, 644.072, 0.0, 328.459],
        [-194.042, 334.152, -69.688, 0.0],
    ]

    nrtl_alpha = [
        [0.0, 0.383, 0.306, 0.299],
        [0.383, 0.0, 0.432, 0.291],
        [0.306, 0.432, 0.0, 0.301],
        [0.299, 0.291, 0.301, 0.0],
    ]

    return Mixture(
        components=[c1, c2, c3, c4],
        kij_matrix=kij_matrix,
        nrtl_tau=nrtl_tau,
        nrtl_alpha=nrtl_alpha,
    )


def main() -> None:
    mixture = build_mixture()
    feed = np.array([0.162, 0.068, 0.656, 0.114], dtype=float)
    pressure_psi = 45.0
    temp_f = 220.0
    bubble_t_guess_f = 0.0
    component_names = [c.name for c in mixture.components]
    short_names = ["n-Hexane", "Ethanol", "MCP", "Benzene"]

    print("Phase 1: Flash Calculation")
    k_vals, x_flash, y_flash, nv = flash(temp_f, pressure_psi, feed, mixture)
    print(f"Flash Results at T = {temp_f}°F, P = {pressure_psi} psi:")
    print(f"Vapor Mole Fraction (nv) = {nv:.6f}")
    print(f"{'Component':<20} {'x (Liquid)':<15} {'y (Vapor)':<15} {'K-value':<15}")
    print("-" * 65)
    for i, name in enumerate(component_names):
        print(f"{name:<20} {x_flash[i]:<15.6f} {y_flash[i]:<15.6f} {k_vals[i]:<15.6f}")

    print("\nPhase 2: Bubble Temperature Calculation")
    bubble_t_f, y_bubble, k_vals_bubble = calculate_bubble_temperature(
        pressure_psi, feed, bubble_t_guess_f, mixture
    )
    bubble_t_c = (bubble_t_f - 32.0) * 5.0 / 9.0
    print(f"Bubble Temperature Results at P = {pressure_psi} psi:")
    print(f"Calculated Bubble Temperature = {bubble_t_f:.4f} °F ({bubble_t_c:.4f} °C)")
    print(f"{'Component':<20} {'x (Liquid)':<15} {'y (Vapor)':<15}")
    print("-" * 50)
    for i, name in enumerate(component_names):
        print(f"{name:<20} {feed[i]:<15.6f} {y_bubble[i]:<15.6f}")

    initial_moles_kmol = 9.159
    initial_moles_mol = initial_moles_kmol * 1000.0
    t_initial_k = ((205.2198 - 32.0) * 5.0 / 9.0) + 273.15
    p_initial_pa = pressure_psi * 6894.76

    z_l_initial, _ = solve_pr_eos_for_z(t_initial_k, p_initial_pa, feed, mixture)
    _, z_v_initial = solve_pr_eos_for_z(t_initial_k, p_initial_pa, y_bubble, mixture)
    hr_l_initial = calculate_residual_enthalpy(t_initial_k, p_initial_pa, feed, z_l_initial, mixture)
    hr_v_initial = calculate_residual_enthalpy(t_initial_k, p_initial_pa, y_bubble, z_v_initial, mixture)
    delta_h_vap_initial = hr_v_initial - hr_l_initial

    vapor_rate_per_hr = 2.484 * 1000.0
    heat_duty_watt = (vapor_rate_per_hr / 3600.0) * delta_h_vap_initial
    total_time_sec = 5980.0
    dt_sec = 20.0

    print("\nPhase 3: Batch Distillation Simulation (Rigorous Energy Balance)")
    print(f"--- Using a constant heat duty of {heat_duty_watt / 1000.0:.2f} kW ---")

    results = simulate_batch_distillation(
        initial_moles=initial_moles_mol,
        initial_comp=feed,
        pressure_psi=pressure_psi,
        heat_duty_watt=heat_duty_watt,
        total_time_sec=total_time_sec,
        dt_sec=dt_sec,
        mixture=mixture,
    )

    header_display = f"{'Time (s)':<10} {'Temp (C)':<10} {'Vapor Out':<12}"
    for name in short_names:
        header_display += f"{'x_' + name:<12}"
    for name in short_names:
        header_display += f"{'y_' + name:<12}"
    print(header_display)
    print("-" * len(header_display))

    # Print first 5 and last 5 rows for brevity in stdout
    n_rows = len(results["time_sec"])
    indices = list(range(min(5, n_rows))) + ([i for i in range(n_rows-3, n_rows) if i >= 5])
    for i in indices:
        time_s = results["time_sec"][i]
        temp_c = results["temp_C"][i]
        moles_v_out = results["total_vapor_out"][i]

        row_str = f"{time_s:<10.0f} {temp_c:<10.2f} {moles_v_out:<12.4f}"
        for comp_frac in results["liquid_comp"][i]:
            row_str += f"{comp_frac:<12.4f}"
        for comp_frac in results["vapor_comp"][i]:
            row_str += f"{comp_frac:<12.4f}"
        print(row_str)

    output_filename = os.path.join("distillation_results.csv")
    print(f"\nWriting detailed results to '{output_filename}'...")

    try:
        csv_header = ["Time (s)", "Temp (C)", "Liq Lvl (%)", "total_vapor_out"]
        csv_header += [f"x_{name}" for name in component_names]
        csv_header += [f"y_{name}" for name in component_names]

        with open(output_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)

            for i in range(len(results["time_sec"])):
                time_s = results["time_sec"][i]
                temp_c = results["temp_C"][i]
                moles_l = results["moles_in_still"][i]
                liq_lvl = (moles_l / initial_moles_mol) * 100.0
                moles_v_out = results["total_vapor_out"][i]

                row_data = [time_s, temp_c, liq_lvl, moles_v_out]
                row_data.extend(results["liquid_comp"][i])
                row_data.extend(results["vapor_comp"][i])
                writer.writerow(row_data)

        print(f"Successfully created '{output_filename}'. Total rows written: {len(results['time_sec'])}.")
    except Exception as exc:
        print(f"An error occurred while writing the file: {exc}")


if __name__ == "__main__":
    main()
