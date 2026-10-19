import subprocess
import sys

import pytest

from drone_scene.common import PipelineError, Runner, checked_output
from drone_scene.reconstruct import gpu_flag


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.parametrize("namespace", ["SiftExtraction", "FeatureExtraction"])
def test_gpu_flag_detects_both_help_streams(monkeypatch, tmp_path, stream, namespace):
    flag = f"--{namespace}.use_gpu"
    output = {"stdout": "", "stderr": ""}
    output[stream] = f"Options:\n  {flag} arg (=1)\n"
    monkeypatch.setattr("drone_scene.common.subprocess.run", lambda *a, **kw:
                        subprocess.CompletedProcess(a[0], 0, **output))
    assert gpu_flag("feature_extractor", "--SiftExtraction.use_gpu",
                    "--FeatureExtraction.use_gpu", Runner(tmp_path)) == flag


def test_checked_output_keeps_stdout_default_and_opt_in_stderr():
    command = [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"]
    assert checked_output(command) == "out\n"
    assert checked_output(command, include_stderr=True) == "out\nerr\n"


def test_failed_help_preserves_command_error():
    with pytest.raises(PipelineError, match="broken help"):
        checked_output([sys.executable, "-c",
                        "import sys; print('broken help', file=sys.stderr); sys.exit(2)"],
                       include_stderr=True)


def test_missing_flags_gives_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr("drone_scene.reconstruct.checked_output", lambda *a, **kw: "Other options")
    with pytest.raises(PipelineError, match="which colmap"):
        gpu_flag("matches_importer", "--SiftMatching.use_gpu", "--FeatureMatching.use_gpu",
                 Runner(tmp_path))


def test_dry_run_does_not_probe_help(monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        pytest.fail("Dry run must not execute COLMAP")
    monkeypatch.setattr("drone_scene.reconstruct.checked_output", unexpected)
    assert gpu_flag("feature_extractor", "old", "new", Runner(tmp_path, dry_run=True)) == "old"