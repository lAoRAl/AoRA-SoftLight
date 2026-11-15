"""Tests CPU de aora_soft_light.py (version 2) — images synthétiques, aucun GPU, aucun fichier écrit.

Lancer / run :  python -m pytest test_aora_soft_light.py -q -p no:cacheprovider

Les conservations sont mesurées sur la SORTIE NUMÉRIQUE retournée (décodée en
linéaire), pas sur le texte de contrôle.
"""

import math

import pytest
import torch

import aora_soft_light as asl

WHITE = (1.0, 1.0, 1.0)
PINK = (1.0, 0.24, 0.65)
CYAN = (0.0, 0.9, 1.0)


def sphere_normals(size: int = 96):
    """Normal map d'une sphère vue de face (convention « R opposé à +x »), masque binaire, grille x."""
    ys, xs = torch.meshgrid(torch.linspace(1, -1, size), torch.linspace(-1, 1, size), indexing="ij")
    r2 = xs ** 2 + ys ** 2
    inside = r2 < 0.95
    nz = torch.sqrt((1 - r2).clamp_min(0))
    n = torch.stack([-xs, ys, nz], dim=-1)
    n = torch.where(inside.unsqueeze(-1), n, torch.tensor([0.0, 0.0, 1.0]))
    return ((n + 1) / 2).unsqueeze(0), inside.float().unsqueeze(0), xs


def light(x, y, z, k=0.3, color=WHITE):
    return {"dir": (x, y, z), "strength": k, "color": color}


def run(image, normal_map, mask, lights, wrap=0.75, linear=True, preserve=True):
    return asl.relight(image, normal_map, mask, lights, wrap, True, False, linear, preserve)


def total_linear_luminance(srgb_image):
    """Somme de luminance linéaire par image du lot."""
    return asl.luminance(asl.srgb_to_linear(srgb_image[..., :3])).sum(dim=(1, 2))


def gpt_counterexample():
    """Bande de 64 px : masque 0→1, gris sRGB 0,1→0,65, normales de −1,4 à +1,4 rad (relecture §212)."""
    t = torch.linspace(0, 1, 64)
    image = (0.1 + 0.55 * t).view(1, 1, 64, 1).expand(1, 1, 64, 3).contiguous()
    angle = -1.4 + 2.8 * t
    n = torch.stack([-torch.sin(angle), torch.zeros(64), torch.cos(angle)], dim=-1)  # R opposé à +x
    normal_map = ((n + 1) / 2).view(1, 1, 64, 3)
    mask = t.view(1, 1, 64).clone()
    return image, normal_map, mask


# ---------- terme doux ----------

@pytest.mark.parametrize("wrap", [0.0, 0.5, 0.75, 1.0])
def test_exactly_opposed_face_gets_zero_for_every_allowed_wrap(wrap):
    back = torch.tensor([[[[0.0, 0.0, -1.0]]]])
    assert float(asl.soft_term(back, torch.tensor([0.0, 0.0, 1.0]), wrap)) == 0.0


@pytest.mark.parametrize("wrap", [-0.1, 1.01, 2.0, float("nan")])
def test_wrap_outside_zero_one_is_refused(wrap):
    front = torch.tensor([[[[0.0, 0.0, 1.0]]]])
    with pytest.raises(ValueError):
        asl.soft_term(front, torch.tensor([0.0, 0.0, 1.0]), wrap)


@pytest.mark.parametrize("wrap", [0.0, 0.75, 1.0])
def test_soft_term_is_monotonic_and_bounded(wrap):
    angles = torch.linspace(0, math.pi, 200)
    normals = torch.stack([torch.sin(angles), torch.zeros(200), torch.cos(angles)], dim=-1)
    s = asl.soft_term(normals.view(1, 1, 200, 3), torch.tensor([0.0, 0.0, 1.0]), wrap).flatten()
    assert torch.all(s[1:] <= s[:-1] + 1e-7)
    assert float(s.min()) >= 0.0 and float(s.max()) <= 1.0 + 1e-6
    assert float(s[0]) == pytest.approx(1.0)


def test_wrap_zero_is_squared_clamped_lambert():
    n = torch.tensor([[[[math.sin(1.0), 0.0, math.cos(1.0)]]]])
    s = float(asl.soft_term(n, torch.tensor([0.0, 0.0, 1.0]), 0.0))
    assert s == pytest.approx(math.cos(1.0) ** 2, rel=1e-5)


# ---------- direction ----------

def test_light_from_the_right_brightens_the_right_side_and_conversely():
    normal_map, mask, xs = sphere_normals()
    image = torch.full((1, 96, 96, 3), 0.4)
    inside = mask[0] > 0
    luma = asl.luminance(run(image, normal_map, mask, [light(1.0, 0.0, 0.3)])[0])[0]
    assert luma[inside & (xs > 0.3)].mean() > luma[inside & (xs < -0.3)].mean() + 0.02
    luma = asl.luminance(run(image, normal_map, mask, [light(-1.0, 0.0, 0.3)])[0])[0]
    assert luma[inside & (xs < -0.3)].mean() > luma[inside & (xs > 0.3)].mean() + 0.02


# ---------- conservation, mesurée sur la sortie retournée ----------

def test_soft_mask_counterexample_from_review_is_now_conserved():
    image, normal_map, mask = gpt_counterexample()
    out, _, info = run(image, normal_map, mask, [light(1.0, 0.0, 0.3, 1.0)])
    assert "pixels écrêtés 0.000 %" in info and "gain négatif ramené à 0 0.000 %" in info
    before, after = total_linear_luminance(image), total_linear_luminance(out)
    assert float(after) == pytest.approx(float(before), rel=2e-5)
    assert not torch.allclose(out, image, atol=1e-3)          # l'image a bien été modifiée


@pytest.mark.parametrize("lights", [
    [light(-0.6, 0.5, 0.5, 0.3, WHITE)],
    [light(-0.6, 0.5, 0.5, 0.3, PINK)],
    [light(-1.0, 0.2, 0.2, 0.3, PINK), light(1.0, 0.2, 0.2, 0.3, CYAN)],
])
@pytest.mark.parametrize("mask_kind", ["binary", "soft", "none"])
def test_total_luminance_is_conserved_on_returned_output(lights, mask_kind):
    torch.manual_seed(0)
    normal_map, binary, xs = sphere_normals()
    image = torch.rand((1, 96, 96, 3)) * 0.4 + 0.1            # loin de l'écrêtage
    mask = {"binary": binary, "soft": (binary * (0.5 + 0.5 * xs.abs())).clamp(0, 1), "none": None}[mask_kind]
    out, _, info = run(image, normal_map, mask, lights)
    assert "pixels écrêtés 0.000 %" in info
    assert float(total_linear_luminance(out)) == pytest.approx(float(total_linear_luminance(image)), rel=3e-5)


def test_conservation_also_holds_when_working_in_srgb_space():
    torch.manual_seed(1)
    normal_map, mask, _ = sphere_normals()
    image = torch.rand((1, 96, 96, 3)) * 0.4 + 0.1
    out, _, _ = run(image, normal_map, mask, [light(-0.6, 0.5, 0.5)], linear=False)
    assert float(asl.luminance(out).sum()) == pytest.approx(float(asl.luminance(image).sum()), rel=3e-5)


def test_without_recentring_the_luminance_changes():
    normal_map, mask, _ = sphere_normals()
    image = torch.full((1, 96, 96, 3), 0.4)
    out, _, _ = run(image, normal_map, mask, [light(-0.6, 0.5, 0.5)], preserve=False)
    assert float(total_linear_luminance(out)) > float(total_linear_luminance(image)) * 1.02


def test_empty_mask_returns_the_image():
    normal_map, mask, _ = sphere_normals()
    image = torch.rand((1, 96, 96, 3))
    out, _, info = run(image, normal_map, torch.zeros_like(mask), [light(1.0, 0.0, 0.3, 1.0)])
    assert torch.allclose(out, image, atol=2e-4)
    assert "zone traitée 0.00 %" in info


def test_pixels_outside_a_binary_mask_are_untouched():
    normal_map, mask, _ = sphere_normals()
    image = torch.rand((1, 96, 96, 3))
    out, _, _ = run(image, normal_map, mask, [light(1.0, 0.0, 0.3, 0.8)])
    outside = mask[0] == 0
    assert torch.allclose(out[0][outside], image[0][outside], atol=2e-4)


def test_zero_strength_returns_the_image():
    normal_map, mask, _ = sphere_normals()
    image = torch.rand((1, 96, 96, 3))
    out, _, _ = run(image, normal_map, mask, [light(1.0, 0.0, 0.3, 0.0)])
    assert torch.allclose(out, image, atol=2e-4)


# ---------- lots ----------

def test_one_normal_map_is_shared_and_luminance_is_conserved_per_image():
    torch.manual_seed(2)
    normal_map, mask, _ = sphere_normals()
    images = torch.rand((3, 96, 96, 3)) * 0.4 + 0.1
    images[1] *= 0.5                                           # images différentes : mu différent
    out, _, info = run(images, normal_map, mask, [light(-0.6, 0.5, 0.5, 0.3, PINK)])
    before, after = total_linear_luminance(images), total_linear_luminance(out)
    for b in range(3):
        assert float(after[b]) == pytest.approx(float(before[b]), rel=3e-5)
    assert "image 3/3" in info


def test_matching_batches_use_each_normal_map():
    normal_map, mask, _ = sphere_normals()
    flat = torch.tensor([0.5, 0.5, 1.0]).view(1, 1, 1, 3).expand(1, 96, 96, 3)
    maps = torch.cat([normal_map, flat], dim=0)
    images = torch.full((2, 96, 96, 3), 0.4)
    out, _, _ = run(images, maps, mask, [light(1.0, 0.0, 0.3, 0.8)])
    assert not torch.allclose(out[0], images[0], atol=1e-3)    # sphère : relief visible
    assert torch.allclose(out[1], images[1], atol=2e-4)        # normales plates : s constant, recentré à 0


def test_inconsistent_batches_are_refused():
    normal_map, mask, _ = sphere_normals()
    images = torch.full((3, 96, 96, 3), 0.4)
    with pytest.raises(ValueError, match="normal_map"):
        run(images, normal_map.expand(2, -1, -1, -1), mask, [light(1.0, 0.0, 0.3)])
    with pytest.raises(ValueError, match="mask"):
        run(images, normal_map, mask.expand(2, -1, -1), [light(1.0, 0.0, 0.3)])


# ---------- entrées invalides ----------

def test_non_finite_and_malformed_inputs_are_refused():
    normal_map, mask, _ = sphere_normals()
    image = torch.full((1, 96, 96, 3), 0.4)
    bad = image.clone()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non finies"):
        run(bad, normal_map, mask, [light(1.0, 0.0, 0.3)])
    bad_n = normal_map.clone()
    bad_n[0, 3, 3, 1] = float("inf")
    with pytest.raises(ValueError, match="non finies"):
        run(image, bad_n, mask, [light(1.0, 0.0, 0.3)])
    with pytest.raises(ValueError, match="forme"):
        run(image[0], normal_map, mask, [light(1.0, 0.0, 0.3)])
    with pytest.raises(ValueError, match="forme"):
        run(image, normal_map[..., :2], mask, [light(1.0, 0.0, 0.3)])


def test_degenerate_normals_are_excluded_and_reported():
    normal_map, mask, _ = sphere_normals()
    image = torch.rand((1, 96, 96, 3)) * 0.4 + 0.1
    broken = normal_map.clone()
    broken[0, 40:56, 40:56, :] = 0.5                            # 2·0,5 − 1 = vecteur nul
    out, _, info = run(image, broken, mask, [light(1.0, 0.0, 0.3, 1.0)])
    assert torch.allclose(out[0, 40:56, 40:56], image[0, 40:56, 40:56], atol=2e-4)
    assert "normales invalides exclues 0.000 %" not in info
    assert float(total_linear_luminance(out)) == pytest.approx(float(total_linear_luminance(image)), rel=3e-5)


def test_normal_map_and_mask_are_resized():
    normal_map, mask, _ = sphere_normals(48)
    image = torch.full((1, 96, 96, 3), 0.4)
    out, s, _ = run(image, normal_map, mask, [light(1.0, 0.0, 0.3)])
    assert out.shape == (1, 96, 96, 3) and s.shape == (1, 96, 96)


# ---------- nœud ----------

def test_node_interface_and_wrap_widget_range():
    normal_map, mask, _ = sphere_normals()
    image = torch.full((1, 96, 96, 3), 0.4)
    node = asl.AoRASoftLight()
    out, s_image, info = node.apply(image, normal_map, 0.75, -1.0, 0.2, 0.2, 0.3, "#ff3ea5",
                                    True, 1.0, 0.2, 0.2, 0.3, "#00e5ff", True, False, True, True,
                                    mask=mask)
    assert out.shape == image.shape and s_image.shape == image.shape
    assert "source 2" in info
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    wrap_spec = asl.AoRASoftLight.INPUT_TYPES()["required"]["wrap"][1]
    assert wrap_spec["min"] == 0.0 and wrap_spec["max"] == 1.0


def test_invalid_colour_is_refused():
    with pytest.raises(ValueError):
        asl.parse_color("rose")
    assert asl.parse_color("#fff") == (1.0, 1.0, 1.0)
