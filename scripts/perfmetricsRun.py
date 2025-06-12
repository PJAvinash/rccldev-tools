from common import *
import json

def getRcclPath(scratch_workdir):
    return os.path.join(scratch_workdir,"rccl")
def getRcclTestsPath(scratch_workdir):
    return  os.path.join(scratch_workdir,"rccl-tests")
def getLibrcclPath(scratch_workdir):
    return os.path.join(scratch_workdir,"rccl","build","debug","librccl.so.1.0") 
def getRcclTestsBinDir(scratch_workdir):
    return os.path.join(scratch_workdir,"rccl-tests","build")

def write_to_log(message: str, file_path: str):
    try:
        log_dir = os.path.dirname(file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(message + '\n')
    except Exception as e:
        print(f"Error writing to log file '{file_path}': {e}")

if __name__ == "__main__":
    scratch_workdir = "/home/apotnuru/SWDEV-535655/Temp"
    CLONE_RCCL = 0
    CLONE_RT = 0
    BUILD_RCCL = 1
    BUILD_RT = 0
    rccl_path:str = getRcclPath(scratch_workdir)
    rccl_tests_path:str = getRcclTestsPath(scratch_workdir)
    librccl:str = getLibrcclPath(scratch_workdir)
    rccltests_binaries_path:str = getRcclTestsBinDir(scratch_workdir)

    if CLONE_RCCL:
        rccl_path = clone_rccl(scratch_workdir)  
    if CLONE_RT:
        rccl_tests_path= clone_rccl_tests(scratch_workdir)
        
    lastNCommits = get_last_n_commit_hashes(rccl_path, 200)
    results = []
    output_json = os.path.join(scratch_workdir,"results.json")
    for idx, commit in enumerate(lastNCommits):
        if BUILD_RCCL:
            librccl = build_rccl(rccl_path,commit_hash=commit)
        if BUILD_RT:
            rccltests_binaries_path = build_rccl_tests(rccl_tests_path,custom_rccl=librccl,rccl_install=os.path.join(rccl_path,"build","debug"))
        rt_args = {"-n":"2"}
        outputlog = run_rccl_test("all_reduce",0,8,scratch_workdir,rccl_test_bin_subdir=rccltests_binaries_path,rt_args_dict=rt_args)
        data = parse_rccl_tests_output(outputlog)
        results.append({ "index": idx,"commit": commit,"data": data})
        write_to_log(outputlog,os.path.join(scratch_workdir,"backup",f"{commit}.log"))
        #checkpointing
        if idx%4 == 0:
            with open(output_json, "w") as f:
                json.dump(results, f, default=str, indent=2)
    with open(output_json, "w") as f:
                json.dump(results, f, default=str, indent=2)
#3717829.PJsession
