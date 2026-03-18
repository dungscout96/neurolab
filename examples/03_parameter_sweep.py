"""Example: Running a parameter sweep across adapters and models.

Generates a SLURM array job that sweeps over all combinations,
following the pattern from OpenEEG-Bench's job_sweep.slurm.
"""

from neurolab.jobs import Job, SweepConfig


def main():
    # ── Define the base job ──────────────────────────────────────────
    base = Job(
        name="adapter_model_sweep",
        cluster="expanse",
        repo_path="/expanse/projects/nemar/dtyoung/OpenEEG-Bench",
        command="",  # overridden by command_template
        venv="/expanse/projects/nemar/dtyoung/conda_envs/adapter-finetuning",
        time_limit="04:00:00",
        gpus=1,
    )

    # ── Define the sweep ────────────────────────────────────────────
    sweep = SweepConfig(
        base=base,
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
        max_concurrent=4,  # Run at most 4 jobs simultaneously
    )

    print(f"Sweep: {sweep.n_jobs} total combinations")
    print(f"Parameters: {list(sweep.parameters.keys())}")
    print(f"Max concurrent: {sweep.max_concurrent}")

    # ── Preview the generated script ────────────────────────────────
    script = sweep.generate_script()
    print("\n--- Generated SLURM array script ---")
    print(script)

    # ── Submit (uncomment when ready) ───────────────────────────────
    # job_id = sweep.submit()
    # print(f"Submitted sweep array job: {job_id}")


if __name__ == "__main__":
    main()
