"""Example: Submitting a training job to Expanse via neurolab-jobs.

Demonstrates how to create and submit a SLURM job that runs
an OpenEEG-Bench training experiment, monitor it, and retrieve logs.
"""

from neurolab.jobs import SlurmJob, submit_job, monitor_jobs, get_logs


def main():
    # ── Define the job ──────────────────────────────────────────────
    job = SlurmJob(
        name="train_labram_lora_bcic2a",
        command="python scripts/train.py adapter=lora model=labram data=bcic2a",
        cluster="expanse",
        time_limit="04:00:00",
        gpus=1,
        working_dir="/expanse/projects/nemar/adapter_finetuning",
        log_dir="logs",
        env_overrides={
            "WANDB_MODE": "online",  # Override default offline for this run
        },
    )

    # ── Dry run to inspect the generated script ─────────────────────
    script_path = submit_job(job, dry_run=True, script_dir="slurm/generated")
    print(f"Generated script: {script_path}")
    with open(script_path) as f:
        print(f.read())

    # ── Submit (uncomment when ready) ───────────────────────────────
    # job_id = submit_job(job)
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
