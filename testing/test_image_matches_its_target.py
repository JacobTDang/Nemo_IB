"""The image compose deploys must be built for the host it will run on.

Proxmox VE is x86-64 only. This repo is developed on Apple Silicon, and
`docker build` without `--platform` produces linux/arm64 — which will not run
there. The Dockerfile comment and deploy/README both say so.

Nothing enforced it. Seven images were built, deployed and verified green
during one session, all linux/arm64, none of which could have run on the
target. Every "gate 7/7" was true and measured the wrong artifact.

The second half of the trap: the documented build tags `nemo-data:amd64` while
compose hardcoded `nemo-data:local`, so following the instructions produced an
image compose would never deploy.

`NEMO_TARGET_ARCH` declares where the image is going. Unset, it defaults to
this machine, so local development is unaffected — it is only when you say
"this is for the homelab" that a mismatch becomes a failure.
"""
import json
import os
import platform
import re
import shutil
import subprocess

import pytest

COMPOSE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "deploy", "docker-compose.yml")

_LOCAL_ARCH = {"arm64": "arm64", "aarch64": "arm64",
               "x86_64": "amd64", "amd64": "amd64"}.get(platform.machine(),
                                                        platform.machine())


def _compose_image() -> str:
    """The image reference compose would use, with its default applied."""
    text = open(COMPOSE).read()
    match = re.search(r"^\s*image:\s*(\S+)", text, re.M)
    assert match, "no image reference found in docker-compose.yml"
    ref = match.group(1)
    var = re.fullmatch(r"\$\{(\w+):-([^}]+)\}", ref)
    if var:
        return os.environ.get(var.group(1)) or var.group(2)
    return ref


def _image_arch(ref: str):
    if not shutil.which("docker"):
        return None
    out = subprocess.run(["docker", "image", "inspect", ref],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return json.loads(out.stdout)[0].get("Architecture")


def test_the_compose_image_is_overridable():
    """A hardcoded tag means the documented amd64 build is never deployed."""
    text = open(COMPOSE).read()
    assert "${NEMO_IMAGE" in text, (
        "docker-compose.yml pins one image tag, so the homelab cannot deploy "
        "the amd64 build the docs tell you to make")


def test_the_image_matches_the_declared_target():
    ref = _compose_image()
    arch = _image_arch(ref)
    if arch is None:
        pytest.skip(f"image {ref!r} is not present on this machine")

    target = os.environ.get("NEMO_TARGET_ARCH", _LOCAL_ARCH)
    assert arch == target, (
        f"compose would deploy {ref!r}, built for linux/{arch}, but the target "
        f"is linux/{target}. Build it with: docker buildx build --platform "
        f"linux/{target} -t nemo-data:{target} -f Dockerfile --load .")


def test_the_homelab_target_is_documented_as_x86():
    readme = open(os.path.join(os.path.dirname(COMPOSE), "README.md")).read()
    assert "amd64" in readme and "x86" in readme.lower(), (
        "the deploy notes no longer state the homelab architecture")
