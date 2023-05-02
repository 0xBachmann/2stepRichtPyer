import resource
import os

mem_per_cpu = int(os.getenv("SLURM_MEM_PER_CPU", -1))
if mem_per_cpu != -1:
    tot_mem = mem_per_cpu * int(os.environ["SLURM_CPUS_PER_TASK"]) * int(os.environ["SLURM_NTASKS"])
    resource.setrlimit(resource.RLIMIT_AS, (mem_per_cpu, mem_per_cpu))
