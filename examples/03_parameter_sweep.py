"""Example: Running a parameter sweep across adapters and models.

Generates a SLURM array job that sweeps over all combinations,
following the pattern from OpenEEG-Bench's job_sweep.slurm.
"""

from neurolab.jobs import SweepConfig, submit_sweep, generate_sweep_script


def main():
    # ── Define the sweep ────────────────────────────────────────────
    sweep = SweepConfig(
        name="adapter_model_sweep",
        command_template=(
            "python scripts/train.py "
            "adapter={adapter} model={model} data={data} "
            "experiment.name={adapter}_{model}_{data}"
        ),
        parameters={
            "adapter": ["lora", "ia3", "dora", "full_finetune"],
            "model": ["labram", "eegpt", "biot"],
            "data": ["bcic2a", "physionet"],
        },
        cluster="expanse",
        max_concurrent=4,  # Run at most 4 jobs simultaneously
        time_limit="04:00:00",
        gpus=1,
    )

    print(f"Sweep: {sweep.n_jobs} total combinations")
    print(f"Parameters: {list(sweep.parameters.keys())}")
    print(f"Max concurrent: {sweep.max_concurrent}")

    # ── Preview the generated script ────────────────────────────────
    script = generate_sweep_script(sweep)
    print("\n--- Generated SLURM array script ---")
    print(script)

    # ── Submit (uncomment when ready) ───────────────────────────────
    # job_id = submit_sweep(sweep, script_dir="slurm/generated")
    # print(f"Submitted sweep array job: {job_id}")


if __name__ == "__main__":
    main()
