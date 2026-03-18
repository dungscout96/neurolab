# neurolab-jobs: Job Submission Guide

This guide covers how to submit, monitor, and manage experiments across local development machines and HPC clusters using `neurolab-jobs`.

## Installation

```bash
cd neurolab
pip install -e ".[dev]"
```

## Architecture Overview

`neurolab-jobs` uses an SSH-first design. Every remote operation — submitting jobs, checking status, fetching logs — goes through SSH. For 2FA-protected HPC clusters, this relies on SSH ControlMaster connections that you authenticate once and reuse for hours. For clusters without 2FA (like a personal dev machine), it connects directly via IP.

The flow when you call `job.submit()`:

1. Look up the cluster profile (YAML) to get SLURM defaults, modules, paths, env vars
2. Build a shell preamble: `cd` into repo, `git checkout`, load modules, activate venv, export env vars
3. Render a complete script (SLURM batch script or backgrounded shell script)
4. Pipe the script to `ssh <target> sbatch` (SLURM) or `ssh <target> bash -ls` (direct)
5. Parse and return the job ID (SLURM job ID or remote PID)

No files are copied to the remote system. Scripts are piped through stdin. Both SLURM and direct submission return immediately — SLURM queues the job, while direct submission backgrounds it with `nohup`.

## SSH Setup

### HPC Clusters (2FA Required)

For clusters with two-factor authentication (Expanse, Delta), set up SSH ControlMaster in `~/.ssh/config`. This creates a persistent, authenticated connection that avoids repeated 2FA prompts:

```
Host expanse
    HostName login.expanse.sdsc.edu
    User dtyoung
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 12h

Host delta
    HostName login.delta.ncsa.illinois.edu
    User dtyoung
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 12h
```

Create the socket directory and authenticate once:

```bash
mkdir -p ~/.ssh/sockets
ssh expanse   # authenticate once with 2FA
# Now all neurolab-jobs commands reuse this connection
```

The cluster profile's `ssh_alias` field maps to the `Host` entry name (e.g. `ssh_alias: expanse`).

### Direct SSH (No 2FA)

For clusters without 2FA (like a personal dev GPU), no ControlMaster is needed. The profile uses `login_host` with the IP directly and leaves `ssh_alias` empty:

```yaml
# jamming profile
login_host: 100.113.196.11
ssh_alias: ""  # connects via IP directly
```

The `ssh_target` property resolves to `ssh_alias` if set, otherwise falls back to `login_host`.

## Cluster Profiles

Each cluster has a YAML profile defining its paths, SLURM defaults, modules, and environment. Built-in profiles live in `neurolab/jobs/profiles/`.

### Built-in Clusters

**expanse** — SDSC Expanse HPC with SLURM, gpu-shared partition, CUDA 12.2 modules, conda envs at `/expanse/projects/nemar/dtyoung/conda_envs/`. Connects via `ssh expanse` (ControlMaster alias, 2FA).

**delta** — NCSA Delta HPC with SLURM, gpuA100x4 partition, anaconda3_gpu module. Connects via `ssh delta` (ControlMaster alias, 2FA).

**jamming** — Personal dev GPU workstation at `100.113.196.11`. No SLURM — jobs run directly via SSH. System CUDA, no module loading. Connects via IP directly (no 2FA).

**local** — Fallback for local development. No SSH, no SLURM.

### Listing and Inspecting Clusters

```python
from neurolab.jobs import list_clusters, get_cluster

print(list_clusters())  # ['delta', 'expanse', 'jamming', 'local']

expanse = get_cluster("expanse")
print(expanse.slurm.partition)   # "gpu-shared"
print(expanse.slurm.account)     # "csd403"
print(expanse.ssh_target)        # "expanse" (alias)
print(expanse.modules)           # ["gpu", "cuda12.2/toolkit/12.2.2"]

jamming = get_cluster("jamming")
print(jamming.ssh_target)        # "100.113.196.11" (IP fallback)
print(jamming.is_hpc)            # False
```

### Auto-Detection

`auto_detect_cluster()` checks (in order): the `NEUROLAB_CLUSTER` env var, hostname substring matching, then falls back to `local`.

```python
from neurolab.jobs import auto_detect_cluster

cluster = auto_detect_cluster()
# On Expanse login nodes → returns expanse config
# On your laptop → returns local config
# With NEUROLAB_CLUSTER=expanse → returns expanse config
```

### Custom Profiles

Register a custom cluster from a YAML file:

```python
from neurolab.jobs import register_cluster_profile

cfg = register_cluster_profile("./my_cluster.yaml")
```

YAML format:

```yaml
name: my_cluster
hostname_patterns: [myhost]
scheduler: slurm        # or "local" for non-SLURM
login_host: 10.0.0.1   # IP or hostname for direct SSH
ssh_alias: my_cluster   # ~/.ssh/config Host entry (leave empty to use login_host)
paths:
  data: /data/root
  scratch: /scratch
  results: /results
  conda_envs: /conda
  cache: /cache
slurm:
  partition: gpu
  account: my-account
  time_limit: "04:00:00"
modules: [cuda/12.0]
conda_env: /path/to/conda/env
env_vars:
  MY_VAR: value
```

## Submitting Jobs

### The Job Dataclass

Every job is defined with a `Job`:

```python
from neurolab.jobs import Job

job = Job(
    name="train_labram_lora",          # job name + log file prefix
    cluster="expanse",                  # which cluster profile to use
    repo_path="/expanse/projects/nemar/dtyoung/OpenEEG-Bench",  # remote repo
    command="python scripts/train.py adapter=lora model=labram", # entry point
)
```

#### Required Fields

| Field | Description |
|-------|-------------|
| `name` | Job name (used for SLURM `--job-name` and log filenames) |
| `cluster` | Cluster profile name (`"expanse"`, `"delta"`, `"jamming"`, etc.) |
| `repo_path` | Absolute path to the project repo on the remote system |
| `command` | The command to run (e.g. `"python scripts/train.py --lr 1e-4"`) |

#### Optional Fields

| Field | Default | Description |
|-------|---------|-------------|
| `branch` | `"main"` | Git branch to checkout before running. Set to `""` to skip. |
| `venv` | `""` | Path to conda/venv on remote. Empty = use cluster default. `"__none__"` = skip activation. |
| `env_vars` | `{}` | Extra env vars to export. Merged with cluster profile defaults (job wins on conflicts). |
| `time_limit` | cluster default | Wall clock limit (e.g. `"04:00:00"`) |
| `gpus` | `1` | Number of GPUs |
| `partition` | cluster default | SLURM partition override |
| `cpus_per_task` | cluster default | CPUs per task |
| `mem_gb` | cluster default | Memory in GB |
| `log_dir` | `"logs"` | Log directory (relative to `repo_path`) |
| `array_spec` | `""` | SLURM array spec (e.g. `"0-23%4"`) |
| `dependency` | `""` | SLURM dependency (e.g. `"afterok:12345"`) |

### Dry Run (Preview the Script)

Always preview before submitting:

```python
script = job.submit(dry_run=True)
print(script)
```

This prints the complete script that would be piped to the remote cluster, without actually running anything.

### Submit for Real

```python
job_id = job.submit()
print(f"Submitted: {job_id}")
# SLURM clusters: returns SLURM job ID (e.g. "12345678")
# Direct clusters: returns remote PID (e.g. "48231")
```

Both return immediately. The job continues running on the remote system.

### What Happens on the Cluster

When you call `job.submit()`, the following sequence executes on the remote system:

**For SLURM clusters (Expanse, Delta):**

The script is piped to `ssh expanse sbatch` via stdin. SLURM queues it and returns a job ID.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=train_labram_lora
#SBATCH --partition=gpu-shared
#SBATCH --account=csd403
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_labram_lora_%j.out
#SBATCH --error=logs/train_labram_lora_%j.err

cd /expanse/projects/nemar/dtyoung/OpenEEG-Bench
git checkout main
git pull --ff-only 2>/dev/null || true
module purge
module load gpu
module load cuda12.2/toolkit/12.2.2
source activate /expanse/projects/nemar/dtyoung/conda_envs/neurolab
export HF_HOME="/expanse/projects/nemar/dtyoung/huggingface_cache"
export MNE_DATA="/expanse/projects/nemar/dtyoung/mne_data"
export WANDB_MODE="offline"

mkdir -p logs

echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date) ==="

python scripts/train.py adapter=lora model=labram

echo "=== Job $SLURM_JOB_ID finished at $(date) ==="
```

**For non-SLURM clusters (jamming):**

The script is piped to `ssh 100.113.196.11 bash -ls`. The job is backgrounded with `nohup` so it survives SSH disconnection, and the remote PID is returned immediately.

```bash
set -euo pipefail
cd ~/projects/OpenEEG-Bench
git checkout main
git pull --ff-only 2>/dev/null || true
export HF_HOME="~/.cache/huggingface"
export MNE_DATA="~/mne_data"
export WANDB_MODE="online"

mkdir -p logs

echo "=== Job train_labram_lora on $(hostname) at $(date) ===" | tee logs/train_labram_lora.out

nohup bash -c 'python scripts/train.py adapter=lora model=labram' >> logs/train_labram_lora.out 2> logs/train_labram_lora.err &
BGPID=$!
disown $BGPID
echo "$BGPID"
```

Stdout goes to `logs/<name>.out`, stderr to `logs/<name>.err`. You can tail these files over SSH to monitor progress.

## Environment Variables

Environment variables come from two sources, merged at submission time:

1. **Cluster profile defaults** — defined in the YAML profile (e.g. `HF_HOME`, `WANDB_MODE`)
2. **Job-specific overrides** — defined in `Job.env_vars`

Job overrides win when there are conflicts:

```python
# Expanse profile sets WANDB_MODE="offline" by default
job = Job(
    name="train",
    cluster="expanse",
    repo_path="/expanse/projects/nemar/dtyoung/MyProject",
    command="python train.py",
    env_vars={
        "WANDB_MODE": "online",       # overrides cluster default "offline"
        "CUSTOM_SEED": "42",           # job-specific addition
    },
)
```

The generated script will contain:

```bash
export CUSTOM_SEED="42"
export HF_HOME="/expanse/projects/nemar/dtyoung/huggingface_cache"
export MNE_DATA="/expanse/projects/nemar/dtyoung/mne_data"
export WANDB_MODE="online"         # job override won
```

## Virtual Environment Handling

The virtual environment is resolved with this priority:

1. **`job.venv`** — if set, activates this specific environment
2. **Cluster profile `conda_env`** — if `job.venv` is empty, uses the cluster default
3. **`"__none__"`** — set `job.venv = "__none__"` to explicitly skip activation (useful when the command manages its own environment, e.g. `uv run`)

```python
# Uses cluster default conda env
job1 = Job(name="t", cluster="expanse", repo_path="/repo", command="python train.py")

# Uses a specific env for this job
job2 = Job(name="t", cluster="expanse", repo_path="/repo", command="python train.py",
           venv="/expanse/projects/nemar/dtyoung/conda_envs/adapter-finetuning")

# Skips activation entirely (e.g. using uv run which manages its own env)
job3 = Job(name="t", cluster="jamming", repo_path="/repo",
           command="uv run experiments/main.py", venv="__none__")
```

## Git Branch Management

By default, every job does:

```bash
git checkout main
git pull --ff-only 2>/dev/null || true
```

Customize with the `branch` field:

```python
# Use a feature branch
job = Job(..., branch="feature/new-adapter")

# Skip branch management entirely (use whatever's checked out)
job = Job(..., branch="")
```

The `git pull --ff-only` silently skips if the branch is already up to date or the repo has no remote configured.

## Parameter Sweeps

`SweepConfig` generates SLURM array jobs from a parameter grid:

```python
from neurolab.jobs import Job, SweepConfig

base = Job(
    name="adapter_sweep",
    cluster="expanse",
    repo_path="/expanse/projects/nemar/dtyoung/OpenEEG-Bench",
    command="",  # overridden by command_template
    venv="/expanse/projects/nemar/dtyoung/conda_envs/adapter-finetuning",
)

sweep = SweepConfig(
    base=base,
    command_template="python scripts/train.py adapter={adapter} model={model}",
    parameters={
        "adapter": ["lora", "ia3", "dora"],
        "model": ["labram", "eegpt"],
    },
    max_concurrent=4,
)

print(f"Total combinations: {sweep.n_jobs}")  # 6
script = sweep.generate_script()               # preview
sweep_id = sweep.submit(dry_run=True)          # or submit()
```

The generated script uses a `SLURM_ARRAY_TASK_ID`-indexed `EXPERIMENTS` array:

```bash
#SBATCH --array=0-5%4

EXPERIMENTS=(
    "lora,labram"
    "lora,eegpt"
    "ia3,labram"
    "ia3,eegpt"
    "dora,labram"
    "dora,eegpt"
)

IFS=',' read -r ADAPTER MODEL <<< "${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
python scripts/train.py adapter=$ADAPTER model=$MODEL
```

## Monitoring Jobs

`monitor_jobs` automatically picks the right strategy based on cluster type: `squeue`/`sacct` for SLURM, `ps` for PID-based direct jobs.

### SLURM Clusters (Expanse, Delta)

```python
from neurolab.jobs import monitor_jobs, wait_for_jobs, JobState

statuses = monitor_jobs(["12345678", "12345679"], cluster="expanse")
for s in statuses:
    print(f"{s.job_id}: {s.state.value} ({s.elapsed})")
```

### Direct Clusters (Jamming)

```python
# job.submit() returned a PID
pid = job.submit()

statuses = monitor_jobs([pid], cluster="jamming")
for s in statuses:
    print(f"PID {s.job_id}: {s.state.value} ({s.elapsed})")
    # RUNNING with elapsed time if still alive
    # COMPLETED if process has exited
```

### Waiting for Completion

`wait_for_jobs` works for both cluster types:

```python
final = wait_for_jobs(
    [job_id],
    cluster="expanse",  # or "jamming"
    poll_interval=30,
    timeout=7200,  # 2 hour max wait
    on_update=lambda statuses: print(f"Still running: {sum(1 for s in statuses if not s.is_done)}"),
)

for s in final:
    if s.state == JobState.COMPLETED:
        print(f"Job {s.job_id} succeeded")
    elif s.state == JobState.FAILED:
        print(f"Job {s.job_id} failed (exit {s.exit_code})")
```

## Log Retrieval

`get_logs` picks the right log file naming convention per cluster type:

- SLURM: `logs/{name}_{job_id}.out`, `slurm-{job_id}.out`
- Direct: `logs/{name}.out`, `logs/{name}.err`

### SLURM Logs

```python
from neurolab.jobs import get_logs, tail_logs

result = get_logs("12345678", cluster="expanse", log_dir="logs",
                  repo_path="/expanse/projects/nemar/dtyoung/OpenEEG-Bench")

print(result.stdout)           # full stdout
print(result.stderr)           # full stderr
print(result.has_errors)       # True if stderr has content

# Search for errors
for entry in result.errors():
    print(f"[{entry.source}:{entry.line_number}] {entry.line}")

# Search with custom pattern
for entry in result.search(r"epoch \d+.*loss"):
    print(entry.line)

# Quick tail
print(result.tail(n=20))

# One-liner tail
print(tail_logs("12345678", cluster="expanse", n_lines=50))
```

### Direct Job Logs

For non-SLURM jobs, pass the `name` parameter (the job name used when creating the `Job`):

```python
result = get_logs(pid, cluster="jamming", name="test_jepa",
                  log_dir="logs", repo_path="/home/dung/Documents/eb_jepa_eeg")

print(result.tail(n=30))

# Or one-liner
print(tail_logs(pid, cluster="jamming", name="test_jepa",
                repo_path="/home/dung/Documents/eb_jepa_eeg", n_lines=50))
```

## Ad-Hoc SSH Commands

`ssh_run` lets you run any command on a remote cluster:

```python
from neurolab.jobs import ssh_run

# Check disk usage
result = ssh_run("expanse", "du -sh /expanse/projects/nemar/dtyoung/data")
print(result.stdout)

# List running jobs
result = ssh_run("expanse", "squeue -u dtyoung")
print(result.stdout)

# Check GPU status on jamming
result = ssh_run("jamming", "nvidia-smi")
print(result.stdout)

# Check if a backgrounded job is still running
result = ssh_run("jamming", "ps -p 48231 -o pid,stat,etime,comm")
print(result.stdout)
```

## CLI Tools

Three command-line entry points are installed:

```bash
# Submit a job (reads from a YAML job spec)
neurolab-submit --config job.yaml --cluster expanse

# Check job status
neurolab-status 12345678 --cluster expanse

# Tail job logs
neurolab-logs 12345678 --cluster expanse --tail 100
```

## Complete Example: Training on Expanse

```python
from neurolab.jobs import Job, monitor_jobs, get_logs, wait_for_jobs

# 1. Define the job
job = Job(
    name="train_labram_lora_bcic2a",
    cluster="expanse",
    repo_path="/expanse/projects/nemar/dtyoung/OpenEEG-Bench",
    command="python scripts/train.py adapter=lora model=labram data=bcic2a",
    venv="/expanse/projects/nemar/dtyoung/conda_envs/adapter-finetuning",
    branch="main",
    time_limit="04:00:00",
    gpus=1,
    env_vars={"WANDB_MODE": "online"},
)

# 2. Preview
print(job.submit(dry_run=True))

# 3. Submit
job_id = job.submit()
print(f"Submitted: {job_id}")

# 4. Wait for completion
final = wait_for_jobs([job_id], cluster="expanse", poll_interval=60)

# 5. Check results
if final[0].is_success:
    print("Training complete!")
    logs = get_logs(job_id, cluster="expanse",
                    repo_path=job.repo_path, log_dir=job.log_dir)
    print(logs.tail(n=20))
else:
    print(f"Job failed: {final[0].state.value}")
    logs = get_logs(job_id, cluster="expanse",
                    repo_path=job.repo_path, log_dir=job.log_dir)
    for err in logs.errors():
        print(f"  {err.line}")
```

## Complete Example: Training on Jamming

```python
from neurolab.jobs import Job, monitor_jobs, get_logs, wait_for_jobs

# 1. Define the job
job = Job(
    name="test_jepa",
    cluster="jamming",
    repo_path="/home/dung/Documents/eb_jepa_eeg",
    command="PYTHONPATH=. uv run experiments/eeg_jepa/main.py",
    branch="",  # skip git checkout, use whatever's checked out
    env_vars={"environment": "development"},
)

# 2. Preview
print(job.submit(dry_run=True))

# 3. Submit (returns PID immediately, job runs in background)
pid = job.submit()
print(f"Running with PID: {pid}")

# 4. Monitor (uses ps to check the PID)
statuses = monitor_jobs([pid], cluster="jamming")
print(f"Status: {statuses[0].state.value} ({statuses[0].elapsed})")

# 5. Wait for completion
final = wait_for_jobs([pid], cluster="jamming", poll_interval=10)

# 6. Get logs
logs = get_logs(pid, cluster="jamming", name=job.name,
                repo_path=job.repo_path, log_dir=job.log_dir)
print(logs.tail(n=30))
if logs.has_errors:
    for err in logs.errors():
        print(f"  {err.line}")
```

## Complete Example: Same Job on Different Clusters

```python
from neurolab.jobs import Job

# Same experiment, different clusters
for cluster in ["expanse", "jamming"]:
    job = Job(
        name="train_labram_lora",
        cluster=cluster,
        repo_path={
            "expanse": "/expanse/projects/nemar/dtyoung/OpenEEG-Bench",
            "jamming": "~/projects/OpenEEG-Bench",
        }[cluster],
        command="python scripts/train.py adapter=lora model=labram data=bcic2a",
        branch="main",
    )
    script = job.submit(dry_run=True)
    print(f"\n=== {cluster} ===")
    print(script)
    # Expanse: SLURM batch script piped to ssh expanse sbatch
    # Jamming: nohup background script piped to ssh 100.113.196.11 bash -ls
```
