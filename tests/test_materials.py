from dataclasses import replace

import pytest

from mpm_core.materials import (MaterialParams, delete_user_preset, list_presets, load_preset,
                                save_user_preset)


def test_list_presets():
    assert list_presets() == ["clay", "gravel", "jelly", "mashed_potato", "mud", "rubber", "sand_dry", "snow_packed",
                              "snow_powder", "snow_wet", "soil_dry", "sugar", "toothpaste", "whipped_cream"]


@pytest.mark.parametrize("preset_id", list_presets())
def test_builtin_presets_describe_themselves(preset_id):
    p = load_preset(preset_id)
    assert p.description and not p.is_user


def test_user_presets(tmp_path):
    cream = load_preset("whipped_cream")
    firm = replace(cream.material, yield_stress=5000.0)
    saved = save_user_preset(tmp_path, " Firm cream ", firm, cream.look)
    assert saved.id == "user:firm_cream" and saved.is_user and saved.label == "Firm cream"
    assert list_presets(tmp_path)[-1] == "user:firm_cream" and "user:firm_cream" not in list_presets()
    loaded = load_preset("user:firm_cream", tmp_path)
    assert loaded.material == replace(firm, name="user:firm_cream") and loaded.look == cream.look
    save_user_preset(tmp_path, "FIRM CREAM", cream.material, cream.look)  # same id: replaces
    assert len(list(tmp_path.iterdir())) == 1 and load_preset("user:firm_cream", tmp_path).label == "FIRM CREAM"
    delete_user_preset(tmp_path, "user:firm_cream")
    assert list_presets(tmp_path) == list_presets()


def test_user_presets_sorted_by_label(tmp_path):
    look = load_preset("jelly").look
    for label in ("b", "A", "c"):
        save_user_preset(tmp_path, label, load_preset("jelly").material, look)
    assert list_presets(tmp_path)[-3:] == ["user:a", "user:b", "user:c"]


@pytest.mark.parametrize("label", ["Whipped cream", "whipped CREAM", "", "  ", "!!"])
def test_user_preset_bad_names(tmp_path, label):
    with pytest.raises(ValueError):
        save_user_preset(tmp_path, label, load_preset("jelly").material, load_preset("jelly").look)


def test_cannot_delete_builtin(tmp_path):
    with pytest.raises(ValueError):
        delete_user_preset(tmp_path, "jelly")


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
