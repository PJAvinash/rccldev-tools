import os
import subprocess
import json
from pathlib import Path
from typing import Optional
import re
import io
import numpy as np
from typing import Optional, Dict
import pandas as pd
import plotly.graph_objects as go
from typing import List

def get_last_n_commit_hashes(repo_path, n):
    if not os.path.isdir(repo_path):
        raise ValueError(f"'{repo_path}' is not a valid directory.")
    try:
        # Get commit hashes using git log
        result = subprocess.run(
            ["git", "-C", repo_path, "log", "develop", f"-n{n}", "--pretty=format:%h", "--reverse"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        # Split into list of commit hashes
        return result.stdout.strip().splitlines()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git command failed: {e.stderr.strip()}") from e
    

def clone_rccl(scratch_dir: str, repo_url: str = "https://github.com/ROCm/rccl.git") -> Path:
    """
    Clone the RCCL GitHub repository into a specified scratch directory.

    Args:
        scratch_dir (str): Directory where RCCL will be cloned.
        repo_url (str): URL of the RCCL git repository (default is official RCCL).

    Returns:
        Path: Path to the cloned RCCL repository.

    Raises:
        subprocess.CalledProcessError: If git clone fails.
    """
    scratch_path = Path(scratch_dir).resolve()
    rccl_path = scratch_path / "rccl"

    if rccl_path.exists():
        print(f"📁 RCCL already exists at {rccl_path}, skipping clone.")
        return rccl_path

    scratch_path.mkdir(parents=True, exist_ok=True)

    print(f"🔽 Cloning RCCL into {rccl_path}...")
    subprocess.run(["git", "clone", repo_url, str(rccl_path)], check=True)

    return rccl_path

def clone_rccl_tests(scratch_dir: str, repo_url: str = "https://github.com/ROCm/rccl-tests.git") -> Path:
    """
    Clone the rccl-tests GitHub repository into a specified scratch directory.

    Args:
        scratch_dir (str): Directory where rccl-tests will be cloned.
        repo_url (str): URL of the RCCL git repository 

    Returns:
        Path: Path to the cloned RCCL repository.

    Raises:
        subprocess.CalledProcessError: If git clone fails.
    """
    scratch_path = Path(scratch_dir).resolve()
    rccl_tests_path = scratch_path / "rccl-tests"

    if rccl_tests_path.exists():
        print(f"📁 rccl-tests already exists at {rccl_tests_path}, skipping clone.")
        return rccl_tests_path

    scratch_path.mkdir(parents=True, exist_ok=True)

    print(f"🔽 Cloning rccl-tests into {rccl_tests_path}...")
    subprocess.run(["git", "clone", repo_url, str(rccl_tests_path)], check=True)

    return rccl_tests_path


def build_rccl(rccl_dir: str, commit_hash: Optional[str] = None, jobs: int = 32) -> str:
    """
    Clone a specific commit of RCCL and build it using the install script.

    Args:
        rccl_dir (str): Path to the RCCL repository.
        commit_hash (Optional[str]): Git commit hash to check out before build.
        jobs (int): Number of parallel jobs for the build (default 32).

    Raises:
        subprocess.CalledProcessError: If any command fails.
        FileNotFoundError: If rccl_dir or install.sh doesn't exist.
    """
    rccl_path = Path(rccl_dir).resolve()
    if not rccl_path.exists():
        raise FileNotFoundError(f"RCCL directory not found: {rccl_path}")
    os.chdir(rccl_path)
    if commit_hash:
        subprocess.run(["git", "checkout", commit_hash], check=True)

    install_script = rccl_path / "install.sh"
    if not install_script.exists():
        raise FileNotFoundError(f"install.sh not found in {rccl_path}")
    print(f"🔨 Building RCCL with {jobs} jobs...")
    try:
        env = os.environ.copy()
        env["ONLY_FUNCS"] = "AllReduce|Reduce"
        result = subprocess.run(["bash","install.sh", "-l","--debug",f"-j{jobs}"],env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True, check=True) 
        print("✅ RCCL build completed.")
    except subprocess.CalledProcessError as e:
        print(e.output)
    return os.path.join(rccl_path,"build","debug","librccl.so.1.0") 
    


def build_rccl_tests(
    rccl_tests_dir: str,
    mpi_dir: str = "/opt/ompi5",
    rocm_path: str = "/opt/rocm",
    custom_rccl: str = "/opt/rocm/lib/librccl.so.1",
    rccl_install: Path = "/opt/rocm/include/rccl"
):
    """
    Build rccl-tests using custom RCCL and MPI installations.

    Args:
        rccl_tests_dir (str): Path to rccl-tests source directory.
        mpi_dir (str): Path to MPI installation.
        rocm_path (str): Path to ROCm installation.
        custom_rccl (str): Path to custom librccl.so.
        rccl_install (Path): Path to RCCL build directory.
    """
    os.chdir(rccl_tests_dir)
    subprocess.run(["make", "clean"], check=True)
    subprocess.run(["rm", "-rf", "build"], check=True)
    subprocess.run([
        "make",
        "CXXFLAGS=-g -O0",
        "CFLAGS=-g -O0",
        "MPI=0",
        f"HIP_HOME={rocm_path}/include/hip",
        f"MPI_HOME={mpi_dir}",
        f"CUSTOM_RCCL_LIB={custom_rccl}",
        f"NCCL_HOME={rccl_install}",
        "-j"
    ], check=True)
    return os.path.join(rccl_tests_dir,"build")

# def build_TransferBench():

def run_rccl_test(
    coll: str,
    tag: str,
    total_ranks: int = 8,
    workdir: Optional[str] = None,
    mpi_install_dir: str = "/opt/ompi5",
    rocm_path: str = "/opt/rocm",
    rccl_test_bin_subdir: str = "rccl-tests/build",
    rt_args_dict:Optional[Dict[str, str]] = None
) -> str:
    """
    Run RCCL test for a specified collective and save the debug log.

    Args:
        coll (str): Collective name, e.g., 'all_reduce'.
        tag (str): Custom tag for output filename.
        total_ranks (int): Number of MPI ranks (default 8).
        workdir (Optional[str]): Working directory. Uses current if None.
        mpi_install_dir (str): Path to MPI installation.
        rocm_path (str): Path to ROCm installation.
        rccl_test_bin_subdir (str): path to RCCL test binaries directory.

    Returns:
        generated log
    """
    MPI = 0
    g = 8
    if MPI:
        g = 1    
    default_args = {
        "-z": "1",
        "-b": "1",
        "-e": "16G",
        "-f": "2",
        "-g": f"{g}",
        "-t": "1",
        "-R": "1",
        "-n": "1",
        "-w": "5",
        "-d": "float"
    }
    # Merge user args (they override defaults)
    merged_args = default_args.copy()
    if rt_args_dict:
        merged_args.update(rt_args_dict)
    
    workdir = workdir or os.getcwd()
    rccl_test_binary = os.path.join(rccl_test_bin_subdir,f"{coll}_perf")
    env = os.environ.copy()
    env_path = f"{mpi_install_dir}/bin:{rocm_path}/bin:{os.environ['PATH']}"
    env_ld = f"{mpi_install_dir}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
    cmd = []
    if MPI:
        cmd = [
            f"{mpi_install_dir}/bin/mpirun",
            "-np", str(total_ranks),
            "--bind-to", "numa",
            "-x", "NCCL_DEBUG=version",
            "-x", f"PATH={env_path}",
            "-x", f"LD_LIBRARY_PATH={env_ld}"
        ]
    else:
        cmd = [str(rccl_test_binary)]
    for flag, val in merged_args.items():
        cmd.extend([flag, val])
    result = None
    try:
        result = subprocess.run(cmd,env=env, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True, check=True) 
        return str(result.stdout)
    except subprocess.CalledProcessError as e:
        return str(e.output)

def parse_rccl_tests_output(rccl_tests_log_str):
    data = []
    regexstr1 = r"\s*(-?\d+)\s+(-?\d+)\s+(\S+)\s+(\S+)\s+(-?\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+((-?\d+)|N/A)"
    regexstr2 = r"\s*(-?\d+)\s+(-?\d+)\s+(\S+)\s+(-?\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+((-?\d+)|N/A)"
    pattern1 = re.compile(regexstr1)
    pattern2 = re.compile(regexstr2)
    for line in io.StringIO(rccl_tests_log_str):
        # Skip lines that start with '##' or don't match the regex pattern
        if line.startswith('##') or not (pattern1.match(line) or pattern2.match(line)):
            continue

        # Split line into columns (handle variable spacing)
        columns = line.split()
            
        # Ensure the line has the expected number of fields
        if len(columns) >= 12:
            i = 1 if pattern1.match(line) else 0
            entry = {
                "size": np.int64(columns[0]),
                "elements": np.int64(columns[1]),
                "type": columns[2],
                "redop": columns[2+i] if i else 'none',
                "root": np.int64(columns[3+i]),
                "op_time(us)": float(columns[4+i]),
                "op_algbw(GB/s)": float(columns[5+i]),
                "op_busbw(GB/s)": float(columns[6+i]),
                "op_wrong": np.int64(columns[7+i]) if columns[7+i].isdigit() or (columns[7+i][1:].isdigit() if columns[7+i].startswith('-') else False) else 0,
                "ip_time(us)": float(columns[8+i]),
                "ip_algbw(GB/s)": float(columns[9+i]),
                "ip_busbw(GB/s)": float(columns[10+i]),
                "ip_wrong":np.int64(columns[11+i]) if columns[11+i].isdigit() or (columns[11+i][1:].isdigit() if columns[11+i].startswith('-') else False) else 0,
            }
        data.append(entry)          
    return data


def generate_rccl_3d_plot(
    json_path: str,
    html_output: str,
    metrics: List[str] = None
) -> None:
    """
    Reads JSON performance data and outputs an HTML file with an interactive 3D scatter plot.
    
    json_path: path to JSON file (commit_hash -> list of dicts)
    html_output: path for the output .html plot
    metrics: list of metric field names for Z-axis dropdown
    """
    # Load data
    with open(json_path) as f:
        data = json.load(f)

    # Flatten into DataFrame
    rows = []
    for commit, entries in data.items():
        for e in entries:
            flat = {'commit': commit, 'size': float(e['size'])}
            for k, v in e.items():
                if k not in ('size',):
                    try:
                        flat[k] = float(v)
                    except (ValueError, TypeError):
                        flat[k] = v
            rows.append(flat)
    df = pd.DataFrame(rows)

    # Default metrics if not provided
    if metrics is None:
        metrics = [
            "op_time(us)",
            "op_algbw(GB/s)",
            "op_busbw(GB/s)",
            "ip_time(us)",
            "ip_algbw(GB/s)",
            "ip_busbw(GB/s)",
        ]

    # Build traces: one per commit, but we map across all points per commit
    traces = []
    for commit in sorted(df['commit'].unique()):
        sub = df[df['commit'] == commit]
        trace = go.Scatter3d(
            x=sub['size'],
            y=[commit] * len(sub),  # treat commit as categorical axis
            z=sub[metrics[0]],       # placeholder, will be updated
            mode='markers',
            name=commit,
            marker=dict(size=4),
            visible=True
        )
        traces.append(trace)

    # Build layout with dropdown
    buttons = []
    for m in metrics:
        buttons.append(dict(
            args=[
                # Update Z data array in each trace
                {'z': [df[df['commit']==c][m] for c in sorted(df['commit'].unique())]},
                {'scene': {'zaxis': {'title': m}}}
            ],
            label=m,
            method='update'
        ))

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='Size'),
            yaxis=dict(title='Commit Hash'),
            zaxis=dict(title=metrics[0]),
        ),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            showactive=True,
            x=0,
            y=1.1,
            buttons=buttons
        )],
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.write_html(html_output, full_html=True)
    print(f"✅ 3D plot saved to {html_output}")


    
# def parse_TransferBench_output():
# def append_PerfResults():






    