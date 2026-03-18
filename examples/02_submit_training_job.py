"""Example: Submitting a training job to Expanse via neurolab-jobs.

Demonstrates how to create and submit a Job that runs
an OpenEEG-Bench training experiment, monitor it, and retrieve logs.
"""

from neurolab.jobs import Job, monitor_jobs, get_logs


def main():
    # ── Define the job ──────────────────────────────────────────────
    job = Job(
        name="train_labram_lora_bcic2a",
        cluster="expanse",
        repo_path="/expanse/projects/nemar/dtyoung/OpenEEG-Bench",
        command="python scripts/train.py adapter=lora model=labram data=bcic2a",
        venv="/expanse/projects/nemar/dtyoung/conda_envs/adapter-finetuning",
        branch="main",
        time_limit="04:00:00",
        gpus=1,
        env_vars={
            "WANDB_MODE": "online",  # Override default offline for this run
        },
    )

    # ── Dry run to inspect the generated script ─────────────────────
    script = job.submit(dry_run=True)
    print("--- Generated SLURM script ---")
    print(script)

    # ── Submit (uncomment when ready) ───────────────────────────────
    # job_id = job.submit()
    # print(f"Submitted job: {job_id}")
    #
    # # ── Monitor ──────────────────────────────────────────────────
    # statuses = monitor_jobs([job_id], cluster="expanse")
    # for s in statuses:
    #     print(f"Job {s.job_id}: {s.state.value} ({s.elapsed})")
    #
    # # ── Get logs for debugging ───────────────────────────────────
    # result = get_logs(job_id, cluster="expanse", log_dir="logs")
    # if result.has_errors:
    #     print("Errors found:")
    #     for entry in result.errors():
    #         print(f"  [{entry.source}:{entry.line_number}] {entry.line}")


if __name__ == "__main__":
    main()
