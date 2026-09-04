"""Every installed file belongs to exactly one distribution.

`html5==0.0.9` sat one line above `html5lib==1.1` in the dependency list.
Nothing imported it, and its install wrote a years-old copy of the `html5lib`
package tree over the real one: 42 files, two owners, and whichever
distribution installed last decided whether `import html5lib` worked. On
this machine the real one won; on the CI runner the stale one did, and the
governance-extractor test failed with an import error from 2020.

Two owners for one path is never intended. It is what a stray dependency, a
fork installed beside its origin, or a vendored copy looks like from the
outside, and the symptom is a test that passes here and fails there.
"""
import collections
import pathlib
import sysconfig


def _owners():
    """path -> {distribution: recorded hash}, for every installed file."""
    site = pathlib.Path(sysconfig.get_paths()["purelib"])
    owners = collections.defaultdict(dict)
    for record in site.glob("*.dist-info/RECORD"):
        dist = record.parent.name.removesuffix(".dist-info")
        for line in record.read_text(encoding="utf-8").splitlines():
            path, _, rest = line.partition(",")
            if not path or path.endswith(".pyc") or path.startswith(dist):
                continue
            owners[path][dist] = rest.partition(",")[0]
    return owners


def test_no_installed_file_is_two_different_files():
    """Two owners shipping different bytes for one path: whichever installed
    last is the one that runs. Identical bytes under two names is an
    uninstall hazard but not a substitution, and is not what failed CI."""
    clashes = {p: sorted(o) for p, o in _owners().items()
               if len(o) > 1 and len(set(o.values())) > 1}
    by_pair = collections.Counter(tuple(d) for d in clashes.values())

    assert not clashes, (
        f"{len(clashes)} installed files differ between their owners; by pair: "
        f"{dict(by_pair)}; first: {sorted(clashes)[:3]}")
