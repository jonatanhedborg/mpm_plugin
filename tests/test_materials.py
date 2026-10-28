import pytest

from mpm_core.materials import MaterialParams, list_presets, load_preset


def test_list_presets():
    assert list_presets() == ["jelly", "sand_dry", "snow_packed", "snow_powder", "whipped_cream"]


def test_load_cream():
    p = load_preset("whipped_cream")
    assert p.label == "Whipped cream"
    assert p.material.model == "von_mises" and p.material.yield_stress == 2500
    assert p.look.display == "mesh"


def test_lame():
    mu, lam = MaterialParams("m", "elastic", 1.0e5, 0.3, 1000).lame()
    assert mu == pytest.approx(1.0e5 / 2.6) and lam == pytest.approx(1.0e5 * 0.3 / (1.3 * 0.4))


def test_roundtrip_dict():
    m = load_preset("sand_dry").material
    assert MaterialParams.from_dict(m.to_dict()) == m


@pytest.mark.parametrize("kw", [{"model": "goo"}, {"poisson_ratio": 0.5}, {"youngs_modulus": 0}, {"density": -1}])
def test_invalid(kw):
    base = dict(name="m", model="elastic", youngs_modulus=1e5, poisson_ratio=0.3, density=1000)
    with pytest.raises(ValueError):
        MaterialParams(**{**base, **kw})


def test_unknown_preset():
    with pytest.raises(KeyError):
        load_preset("custard")
