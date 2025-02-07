# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import importlib.machinery
import sys
import os
import io
import subprocess
import time

from loguru import logger

from tt_metal.tools.profiler.common import TT_METAL_HOME, PROFILER_BIN_DIR, PROFILER_LOGS_DIR

TRACY_MODULE_PATH = TT_METAL_HOME / "tt_metal/third_party/tracy"
TRACY_FILE_NAME = "tracy_profile_log_host.tracy"
TRACY_CSV_FILE_NAME = "tracy_profile_log_host.csv"

TRACY_CAPTURE_TOOL = "capture"
TRACY_CSVEXPROT_TOOL = "csvexport"

import tracy_state


class Profiler:
    def __init__(self):
        from tracy_tt_lib import tracy_marker_func, tracy_marker_line, finish_all_zones

        self.doProfile = tracy_state.doPartial and sys.gettrace() is None and sys.getprofile() is None
        self.doLine = tracy_state.doLine

        self.lineMarker = tracy_marker_line
        self.funcMarker = tracy_marker_func
        self.finishZones = finish_all_zones

    def enable(self):
        if self.doProfile:
            if self.doLine:
                sys.settrace(self.lineMarker)
            else:
                sys.setprofile(self.funcMarker)

    def disable(self):
        if self.doProfile:
            sys.settrace(None)
            sys.setprofile(None)
            self.finishZones()


def runctx(cmd, globals, locals, partialProfile):
    from tracy_tt_lib import tracy_marker_func, finish_all_zones

    if not partialProfile:
        sys.setprofile(tracy_marker_func)

    try:
        exec(cmd, globals, locals)
    finally:
        sys.setprofile(None)
        finish_all_zones()


def confirmTracyToolInstall(tool):
    ret = True
    if not os.path.exists(PROFILER_BIN_DIR / tool):
        toolTracyPath = TRACY_MODULE_PATH / tool / "build/unix"
        try:
            logger.info(f"Building tracy profiling tool: {tool}")
            subprocess.run(
                f"mkdir -p {PROFILER_BIN_DIR}",
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                subprocess.run(
                    f"cd {toolTracyPath}; make",
                    shell=True,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError as e:
                subprocess.run(
                    f"cd {toolTracyPath}; make clean; make",
                    shell=True,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            subprocess.run(
                f"cp {toolTracyPath}/{tool}-release {PROFILER_BIN_DIR / tool}",
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            ret = False
    return ret


def run_report_setup(verbose):
    toolsReady = True

    logger.info("Verifying tracy profiling tools")
    toolsReady &= confirmTracyToolInstall(TRACY_CAPTURE_TOOL)
    toolsReady &= confirmTracyToolInstall(TRACY_CSVEXPROT_TOOL)

    captureProcess = None
    if toolsReady:
        subprocess.run(
            f"rm -rf {PROFILER_LOGS_DIR}; mkdir -p {PROFILER_LOGS_DIR}",
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        captureCommand = (f"{PROFILER_BIN_DIR / TRACY_CAPTURE_TOOL} -o {PROFILER_LOGS_DIR / TRACY_FILE_NAME} -f",)
        if verbose:
            logger.info(f"Capture command: {captureCommand}")
            captureProcess = subprocess.Popen(captureCommand, shell=True)
        else:
            captureProcess = subprocess.Popen(
                captureCommand, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    else:
        logger.warning(
            "Perf report generation is skipped!! Tracy tools can't be installed,  make sure tt_metal dev packages are installed"
        )

    return toolsReady, captureProcess


def generate_report():
    tracyOutFile = PROFILER_LOGS_DIR / TRACY_FILE_NAME
    if not os.path.exists(tracyOutFile):
        logger.warning(
            f"tracy capture output file {tracyOutFile} was not generated. Run in verbose (-v) mode to see tracy capture info"
        )
        return
    with open(PROFILER_LOGS_DIR / TRACY_CSV_FILE_NAME, "w") as csvFile:
        subprocess.run(
            f"{PROFILER_BIN_DIR / TRACY_CSVEXPROT_TOOL} -u {PROFILER_LOGS_DIR / TRACY_FILE_NAME}",
            shell=True,
            check=True,
            stdout=csvFile,
            stderr=subprocess.DEVNULL,
        )

    logger.info(f"Host side profiling report generated at {PROFILER_LOGS_DIR / TRACY_CSV_FILE_NAME}")


def main():
    from optparse import OptionParser

    usage = "tracy_python.py [-m module | scriptfile] [arg] ..."
    parser = OptionParser(usage=usage)
    parser.allow_interspersed_args = False
    parser.add_option("-m", dest="module", action="store_true", help="Profile a library module.", default=False)
    parser.add_option("-p", dest="partial", action="store_true", help="Only profile enabled zones", default=False)
    parser.add_option("-l", dest="lines", action="store_true", help="Profile every line of python code", default=False)
    parser.add_option("-r", dest="report", action="store_true", help="Generate ops report", default=False)
    parser.add_option("-v", dest="verbose", action="store_true", help="More info is printed to stdout", default=False)

    if not sys.argv[1:]:
        parser.print_usage()
        sys.exit(2)

    originalArgs = sys.argv.copy()

    (options, args) = parser.parse_args()
    sys.argv[:] = args

    if len(args) > 0:
        doReport = False
        if options.report:
            doReport, captureProcess = run_report_setup(options.verbose)

        if not doReport:
            if options.module:
                import runpy

                code = "run_module(modname, run_name='__main__')"
                globs = {
                    "run_module": runpy.run_module,
                    "modname": args[0],
                }
            else:
                progname = args[0]
                sys.path.insert(0, os.path.dirname(progname))
                with io.open_code(progname) as fp:
                    code = compile(fp.read(), progname, "exec")
                spec = importlib.machinery.ModuleSpec(name="__main__", loader=None, origin=progname)
                globs = {
                    "__spec__": spec,
                    "__file__": spec.origin,
                    "__name__": spec.name,
                    "__package__": None,
                    "__cached__": None,
                }

            if options.partial:
                tracy_state.doPartial = True

            if options.lines:
                tracy_state.doLine = True

            try:
                runctx(code, globs, None, options.partial)
            except BrokenPipeError as exc:
                # Prevent "Exception ignored" during interpreter shutdown.
                sys.stdout = None
                sys.exit(exc.errno)
        else:
            originalArgs.remove("-r")
            osCmd = " ".join(originalArgs[1:])

            testCommand = f"python -m tracy {osCmd}"
            testProcess = subprocess.Popen([testCommand], shell=True, env=dict(os.environ))

            testProcess.communicate()
            logger.info(f"Test fully finished. Waiting for tracy capture tool to finish ...")
            captureProcess.communicate()

            generate_report()

    else:
        parser.print_usage()
    return parser


# When invoked as main program, invoke the profiler on a script
if __name__ == "__main__":
    main()
