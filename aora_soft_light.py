"""
AoRA Soft Light — nœud ComfyUI expérimental / experimental ComfyUI node.
Version 2 (2026-11-15), après la relecture GPT de §212.

FR — Éclairage STYLISÉ à partir d'une normal map et de une ou deux sources
choisies à la main. Ce nœud ne calcule NI ombre portée NI rebond de lumière :
il module l'ombrage déjà présent dans l'image, il ne le retire pas.

EN — STYLISED lighting from a normal map and one or two hand-picked sources.
No cast shadows, no bounced light: it modulates the shading already painted in
the image, it does not remove it.

Formules / formulas
-------------------
1. Normales :  n = normalize(2 * rgb - 1), inversion optionnelle de X / Y.
   Une normale dont la longueur avant normalisation est < MIN_NORMAL_LENGTH
   (frontière entre deux surfaces après interpolation, pixel vide) est
   déclarée invalide : ce pixel n'est ni éclairé ni compté dans le recentrage.
2. Direction : L = normalize(x, y, max(z, 0)). C'est une DIRECTION, pas une
   position : +x = vers la droite de l'image, +y = vers le haut, +z = vers
   la caméra. z = 1 n'est « de face » que si x = y = 0.
3. Terme doux, wrap w dans [0, 1] :
               s = clamp((n.L + w) / (1 + w), 0, 1) ** 2
   s décroît de façon monotone quand la surface se détourne ; le clamp avant
   le carré évite la remontée d'une valeur négative. Pour w <= 1 une face
   exactement opposée (n.L = -1) reçoit 0. Pour 0 < w <= 1 des surfaces
   partiellement détournées reçoivent volontairement de la lumière.
   w = 0 donne max(n.L, 0)², ce n'est pas un Lambert simple.
4. Poids effectif :  m_eff = masque * validité des normales
5. Par source i (couleur c_i, force k_i), dans l'espace de travail
   (linéaire si linear_light) :
               Yc_i = luminance(c_i * image)
               mu_i = somme(m_eff * Yc_i * s_i) / somme(m_eff * Yc_i)   (par image)
               gain = 1 + somme_i k_i * c_i * (s_i - mu_i)
6. Composition DANS LE MÊME ESPACE, avant le retour en sRGB :
               mixte = image * (1 + m_eff * (max(gain, 0) - 1))
               sortie = vers_sRGB(clamp(mixte, 0, 1))

Contrat de conservation / conservation contract
------------------------------------------------
Tant que gain >= 0 et avant l'écrêtage à [0, 1], la SOMME DE LUMINANCE DE
L'IMAGE COMPLÈTE (espace de travail) est conservée, y compris avec un masque
doux. Ce n'est PAS la moyenne « repondérée par le masque » de la sortie : pour
un masque doux, cette dernière grandeur peut changer. La teinte et les
moyennes R, G, B ne sont pas conservées avec une lumière colorée.
Le texte de contrôle est calculé sur la sortie réellement retournée, par image.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_LUMA = (0.2126, 0.7152, 0.0722)  # Rec. 709, lumière linéaire
_EPS = 1e-6
MIN_NORMAL_LENGTH = 0.25


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    """[..., 3] -> [...]"""
    w = torch.tensor(_LUMA, dtype=rgb.dtype, device=rgb.device)
    return (rgb * w).sum(dim=-1)


def parse_color(text: str) -> tuple[float, float, float]:
    """'#ff3ea5' ou 'ff3ea5' -> (r, g, b) entre 0 et 1, valeurs sRGB."""
    t = str(text).strip().lstrip("#")
    if len(t) == 3:
        t = "".join(ch * 2 for ch in t)
    if len(t) != 6:
        raise ValueError(f"Couleur invalide / invalid colour: {text!r} (attendu '#rrggbb')")
    try:
        return tuple(int(t[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError as error:
        raise ValueError(f"Couleur invalide / invalid colour: {text!r}") from error


def _check_image(name: str, tensor, min_channels: int = 3) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4 or tensor.shape[-1] < min_channels:
        shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else type(tensor).__name__
        raise ValueError(f"{name} : forme attendue [lot, hauteur, largeur, >= {min_channels} canaux], reçu {shape}.")
    if tensor.shape[0] < 1 or tensor.shape[1] < 1 or tensor.shape[2] < 1:
        raise ValueError(f"{name} : image vide {tuple(tensor.shape)}.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} : contient des valeurs non finies (NaN ou infini).")
    return tensor


def _match_batch(name: str, tensor: torch.Tensor, batch: int) -> torch.Tensor:
    """Lot de taille 1 (partagé explicitement) ou exactement égal au lot d'images."""
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch, *tensor.shape[1:])
    raise ValueError(
        f"{name} : {tensor.shape[0]} élément(s) pour {batch} image(s). "
        f"Fournir 1 élément (partagé) ou exactement {batch}."
    )


def decode_normals(normal_map: torch.Tensor, height: int, width: int,
                   flip_x: bool, flip_y: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """IMAGE [B, h, w, >=3] 0..1 -> (normales unitaires [B, H, W, 3], validité [B, H, W])."""
    n = normal_map[..., :3].float()
    if n.shape[1] != height or n.shape[2] != width:
        n = F.interpolate(n.permute(0, 3, 1, 2), size=(height, width),
                          mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    n = n * 2.0 - 1.0
    if flip_x:
        n = n * torch.tensor([-1.0, 1.0, 1.0], dtype=n.dtype, device=n.device)
    if flip_y:
        n = n * torch.tensor([1.0, -1.0, 1.0], dtype=n.dtype, device=n.device)
    length = n.norm(dim=-1, keepdim=True)
    valid = (length.squeeze(-1) >= MIN_NORMAL_LENGTH).to(n.dtype)
    return n / length.clamp_min(_EPS), valid


def light_direction(x: float, y: float, z: float, device, dtype) -> torch.Tensor:
    v = torch.tensor([float(x), float(y), max(float(z), 0.0)], dtype=dtype, device=device)
    if not torch.isfinite(v).all():
        raise ValueError("Direction de lumière non finie.")
    length = v.norm()
    if float(length) < _EPS:
        return torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    return v / length


def soft_term(normals: torch.Tensor, direction: torch.Tensor, wrap: float) -> torch.Tensor:
    """s = clamp((n.L + w) / (1 + w), 0, 1) ** 2, avec 0 <= w <= 1  ->  [B, H, W]"""
    w = float(wrap)
    if not (0.0 <= w <= 1.0):
        raise ValueError(f"wrap doit être entre 0 et 1, reçu {wrap!r}.")
    ndotl = (normals * direction).sum(dim=-1)
    return ((ndotl + w) / (1.0 + w)).clamp(0.0, 1.0) ** 2


def prepare_mask(mask, batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    """MASK optionnel -> [B, H, W] ; sans masque : tout à 1."""
    if mask is None:
        return torch.ones((batch, height, width), dtype=dtype, device=device)
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"mask : tenseur attendu, reçu {type(mask).__name__}.")
    m = mask.float().to(device)
    if m.dim() == 2:
        m = m.unsqueeze(0)
    if m.dim() != 3:
        raise ValueError(f"mask : forme attendue [hauteur, largeur] ou [lot, hauteur, largeur], reçu {tuple(mask.shape)}.")
    if not torch.isfinite(m).all():
        raise ValueError("mask : contient des valeurs non finies (NaN ou infini).")
    if m.shape[1] != height or m.shape[2] != width:
        m = F.interpolate(m.unsqueeze(1), size=(height, width),
                          mode="bilinear", align_corners=False).squeeze(1)
    m = _match_batch("mask", m, batch)
    return m.clamp(0.0, 1.0).to(dtype)


def relight(image: torch.Tensor, normal_map: torch.Tensor, mask, lights: list[dict],
            wrap: float, flip_x: bool, flip_y: bool, linear_light: bool,
            preserve_luminance: bool) -> tuple[torch.Tensor, torch.Tensor, str]:
    """
    image   : [B, H, W, >=3] sRGB 0..1
    lights  : [{"dir": (x, y, z), "strength": k, "color": (r, g, b) sRGB}, ...]
    Retour  : image sRGB [B, H, W, 3], carte s de la première source [B, H, W], texte de contrôle.
    """
    img = _check_image("image", image)[..., :3].float()
    batch, height, width, _ = img.shape
    device, dtype = img.device, img.dtype

    normal_map = _match_batch("normal_map", _check_image("normal_map", normal_map).to(device), batch)
    normals, valid = decode_normals(normal_map, height, width, flip_x, flip_y)
    m_eff = prepare_mask(mask, batch, height, width, device, dtype) * valid   # [B, H, W]

    work = srgb_to_linear(img) if linear_light else img.clamp(0.0, 1.0)
    gain = torch.ones_like(work)
    first_s = None
    notes = []

    for index, light in enumerate(lights, start=1):
        strength = float(light["strength"])
        direction = light_direction(*light["dir"], device=device, dtype=dtype)
        s = soft_term(normals, direction, wrap)
        if first_s is None:
            first_s = s
        if abs(strength) < _EPS:
            notes.append(f"source {index} : force 0, ignorée")
            continue
        color = torch.tensor(light["color"], dtype=dtype, device=device)
        if linear_light:
            color = srgb_to_linear(color)
        if preserve_luminance:
            weight = m_eff * luminance(work * color)                    # [B, H, W]
            total = weight.sum(dim=(1, 2), keepdim=True)
            mu = (weight * s).sum(dim=(1, 2), keepdim=True) / total.clamp_min(_EPS)
            # Image sans poids (masque vide, image noire) : pas de recentrage possible,
            # et de toute façon aucune modification visible.
            mu = torch.where(total > _EPS, mu, torch.zeros_like(mu))
        else:
            mu = torch.zeros((batch, 1, 1), dtype=dtype, device=device)
        gain = gain + strength * color * (s - mu).unsqueeze(-1)
        notes.append(f"source {index} : mu=" + ", ".join(f"{float(v):.4f}" for v in mu.flatten()))

    me = m_eff.unsqueeze(-1)
    mixed_raw = work * (1.0 + me * (gain.clamp_min(0.0) - 1.0))   # composition dans l'espace de travail
    mixed = mixed_raw.clamp(0.0, 1.0)
    out = linear_to_srgb(mixed) if linear_light else mixed

    # --- Contrôle, par image, sur la sortie réellement retournée ---
    pixels = float(height * width)
    back = srgb_to_linear(out) if linear_light else out
    before = luminance(work).mean(dim=(1, 2))
    after_raw = luminance(mixed_raw).mean(dim=(1, 2))
    after = luminance(back).mean(dim=(1, 2))
    clipped = (mixed_raw > 1.0).any(dim=-1).float().mean(dim=(1, 2))
    negative = ((gain < 0).any(dim=-1).float() * (m_eff > 0).float()).mean(dim=(1, 2))
    invalid = (1.0 - valid).mean(dim=(1, 2))
    covered = m_eff.sum(dim=(1, 2)) / pixels
    space = "linéaire" if linear_light else "sRGB"
    lines = []
    for b in range(batch):
        lines.append(
            f"image {b + 1}/{batch} — luminance moyenne de l'image complète ({space}) : "
            f"avant {float(before[b]):.6f} | avant écrêtage {float(after_raw[b]):.6f} | "
            f"sortie retournée {float(after[b]):.6f} | zone traitée {float(covered[b]) * 100:.2f} % | "
            f"pixels écrêtés {float(clipped[b]) * 100:.3f} % | gain négatif ramené à 0 "
            f"{float(negative[b]) * 100:.3f} % | normales invalides exclues {float(invalid[b]) * 100:.3f} %"
        )
    info = "\n".join(lines + [" ; ".join(notes)])
    if first_s is None:
        first_s = torch.zeros((batch, height, width), dtype=dtype, device=device)
    return out.clamp(0.0, 1.0), first_s, info


class AoRASoftLight:
    """Éclairage stylisé par demi-Lambert borné, sans ombre portée ni rebond."""

    @classmethod
    def INPUT_TYPES(cls):
        direction = {"min": -1.0, "max": 1.0, "step": 0.05}
        return {
            "required": {
                "image": ("IMAGE",),
                "normal_map": ("IMAGE", {"tooltip": "Normal map RGB, même cadrage que l'image. 1 carte ou une par image du lot."}),
                "wrap": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "w entre 0 et 1. 0 = max(N.L, 0) au carré (dur) ; 0,5 à 1 = transition douce."}),
                "light1_x": ("FLOAT", {"default": -0.6, **direction, "tooltip": "Direction : +x = vers la droite de l'image."}),
                "light1_y": ("FLOAT", {"default": 0.5, **direction, "tooltip": "Direction : +y = vers le haut."}),
                "light1_z": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "Direction : +z = vers la caméra. 0 = rasante. Le vecteur est normalisé."}),
                "light1_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01,
                                              "tooltip": "k. Premier essai : 0,3."}),
                "light1_color": ("STRING", {"default": "#ffffff"}),
                "enable_light_2": ("BOOLEAN", {"default": False}),
                "light2_x": ("FLOAT", {"default": 0.6, **direction}),
                "light2_y": ("FLOAT", {"default": 0.5, **direction}),
                "light2_z": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "light2_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01}),
                "light2_color": ("STRING", {"default": "#ffffff"}),
                "flip_normal_x": ("BOOLEAN", {"default": True,
                                              "tooltip": "Hypothèse NON vérifiée pour DSINE : canal R opposé à +x. À contrôler avec un balayage gauche / droite."}),
                "flip_normal_y": ("BOOLEAN", {"default": False, "tooltip": "Non vérifié. À contrôler avec un balayage haut / bas."}),
                "linear_light": ("BOOLEAN", {"default": True,
                                             "tooltip": "Calcul et composition en lumière linéaire (recommandé)."}),
                "preserve_luminance": ("BOOLEAN", {"default": True,
                                                   "tooltip": "Recentrage pondéré par la luminance : conserve la luminance totale de l'image avant écrêtage."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Zone à éclairer (peut être douce). Sans masque : toute l'image."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "carte_s_source_1", "controle")
    FUNCTION = "apply"
    CATEGORY = "AoRA/lumière"

    def apply(self, image, normal_map, wrap, light1_x, light1_y, light1_z, light1_strength,
              light1_color, enable_light_2, light2_x, light2_y, light2_z, light2_strength,
              light2_color, flip_normal_x, flip_normal_y, linear_light,
              preserve_luminance, mask=None):
        lights = [{"dir": (light1_x, light1_y, light1_z), "strength": light1_strength,
                   "color": parse_color(light1_color)}]
        if enable_light_2:
            lights.append({"dir": (light2_x, light2_y, light2_z), "strength": light2_strength,
                           "color": parse_color(light2_color)})
        out, s, info = relight(image, normal_map, mask, lights, wrap, bool(flip_normal_x),
                               bool(flip_normal_y), bool(linear_light), bool(preserve_luminance))
        print(f"[AoRA Soft Light]\n{info}")
        s_image = s.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
        return (out, s_image, info)


NODE_CLASS_MAPPINGS = {"AoRASoftLight": AoRASoftLight}
NODE_DISPLAY_NAME_MAPPINGS = {"AoRASoftLight": "AoRA Soft Light (lumière douce par normales)"}
