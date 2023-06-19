from merge_meta_class import MergeMetaclass


class default_setup(metaclass=MergeMetaclass):
    timerAnalysis = {
        "T0 -> BRISC FW start": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "timerID": 0},
            "end": {"risc": "BRISC", "timerID": 1},
        },
        "TRISC0 kernel start -> TRISC0 kernel end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "TRISC_0", "timerID": 2},
            "end": {"risc": "TRISC_0", "timerID": 3},
        },
        "TRISC1 kernel start -> TRISC1 kernel end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "TRISC_1", "timerID": 2},
            "end": {"risc": "TRISC_1", "timerID": 3},
        },
        "TRISC2 kernel start -> TRISC2 kernel end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "TRISC_2", "timerID": 2},
            "end": {"risc": "TRISC_2", "timerID": 3},
        },
        "BRISC kernel start -> BRISC kernel end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "timerID": 2},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "NCRISC kernel start -> NCRISC kernel end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 2},
            "end": {"risc": "NCRISC", "timerID": 3},
        },
        "ANY RISC FW start -> ANY RISC FW end": {
            "across": "core",
            "type": "launch_first_last",
            "start": {"risc": "ANY", "timerID": 1},
            "end": {"risc": "ANY", "timerID": 4},
        },
        "ANY RISC FW end -> BRISC FW start": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "ANY", "timerID": 4},
            "end": {"risc": "BRISC", "timerID": 1},
        },
        "T0 -> ANY RISC FW end": {
            "across": "core",
            "type": "session_first_last",
            "start": {"risc": "BRISC", "timerID": 0},
            "end": {"risc": "ANY", "timerID": 4},
        },
        "BRISC FW start -> ANY RISC FW end": {
            "across": "core",
            "type": "session_first_last",
            "start": {"risc": "ANY", "timerID": 1},
            "end": {"risc": "ANY", "timerID": 4},
        },
        "T0 -> ANY CORE ANY RISC FW end": {
            "across": "device",
            "type": "session_first_last",
            "start": {"core": "ANY", "risc": "ANY", "timerID": 1},
            "end": {"core": "ANY", "risc": "ANY", "timerID": 4},
        },
    }

    riscsData = {
        "BRISC": {"color": "light:g"},
        "NCRISC": {"color": "light:r"},
        "TRISC_0": {"color": "light:gray"},
        "TRISC_1": {"color": "light:gray"},
        "TRISC_2": {"color": "light:gray"},
        "TENSIX": {"color": "light:b"},
    }

    riscs = [
        "BRISC",
        "NCRISC",
        "TRISC_0",
        "TRISC_1",
        "TRISC_2",
        # "TENSIX",
    ]

    timerIDLabels = [(0, "Start"), (1, "Firmware Start"), (2, "Kernel start"), (3, "Kernel End"), (4, "Firmware End")]

    coreFreq = 1.2  # GHz

    displayStats = ["Count", "Average", "Max", "Median", "Min"]

    plotBaseHeight = 200
    plotPerCoreHeight = 100

    webappPort = 8050

    outputFolder = "output/device"
    deviceInputLog = "logs/profile_log_device.csv"
    deviceRearranged = "device_rearranged_timestamps.csv"
    deviceAnalysisData = "device_analysis_data.json"
    devicePerfHTML = "timeline.html"
    deviceStatsTXT = "device_stats.txt"
    deviceTarball = "device_perf_results.tgz"


class test_matmul_multi_core_multi_dram(default_setup):
    timerAnalysis = {
        "Compute~": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 6},
            "end": {"risc": "BRISC", "timerID": 5},
        }
    }


class test_matmul_multi_core_multi_dram_in0_mcast(default_setup):
    timerAnalysis = {
        "NCRISC start sender -> BRISC kernel end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 10},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "NCRISC start reciever -> BRISC kernel end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 7},
            "end": {"risc": "BRISC", "timerID": 3},
        },
    }


class test_matmul_multi_core_multi_dram_in1_mcast(default_setup):
    timerAnalysis = {
        "NCRISC start sender -> BRISC kernel end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 20},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "NCRISC start reciever -> BRISC kernel end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 16},
            "end": {"risc": "BRISC", "timerID": 3},
        },
    }


class test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast(default_setup):
    timerAnalysis = {
        "NC_in0_s_in1_r -> B_end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 24},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "NC_in0_s_in1_s -> B_end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 29},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "NC_in0_r_in1_r -> B_end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 34},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "NC_in0_r_in1_s -> B_end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 39},
            "end": {"risc": "BRISC", "timerID": 3},
        },
    }


class test_full_buffer(default_setup):
    timerAnalysis = {
        "Marker Repeat": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "ANY", "timerID": 2},
            "end": {"risc": "ANY", "timerID": 2},
        }
    }
