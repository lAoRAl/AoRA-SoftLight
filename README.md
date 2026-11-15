# AoRA Soft Light

**Experimental normal-map lighting adjustment for ComfyUI.**

Adjust the apparent lighting of an existing image with one or two manually controlled directional lights. Soft transitions, optional feathered masks, and per-image luminance preservation before clipping. No diffusion model or model weights are bundled.

This is a finishing tool under evaluation, not a physically based renderer or an automatic scene-lighting system. It modulates the shading already present; it does not remove it.

## Install

1. Download this repository and put its contents in `ComfyUI/custom_nodes/AoRA_SoftLight/`. The `__init__.py` file must be directly inside that folder.
2. The node uses PyTorch, normally already installed with ComfyUI. It needs no additional model download. Do not replace ComfyUI's existing PyTorch installation just for this node.
3. Restart ComfyUI and search for **AoRA Soft Light** (category `AoRA/lumière`).

## Use

- Connect your source image to `image` and an RGB normal map of the same framing to `normal_map`.
- Optionally connect a `MASK`: white is affected, black is preserved; soft edges are supported. Without a mask, the whole image is affected.
- Start with one white light, `light1_strength = 0.3`, `wrap = 0.75`, `linear_light = true`, and `preserve_luminance = true`.
- `x` points right, `y` points up, and `z` points toward the camera. These define a **direction**, not a lamp position. The direction is normalized. The all-zero direction falls back to a frontal light.
- Normal-map conventions vary: check left/right and up/down, then adjust `flip_normal_x` / `flip_normal_y`. The defaults were consistent with one tested DSINE scene; they are not a universal convention guarantee.
- A normal map or mask can be shared across the batch (one item) or provided per image (exactly matching batch size). Other batch sizes are rejected.

Outputs: adjusted RGB image, first light's grayscale response map (repeated over RGB channels), and diagnostic text. The response map is not the final masked composite.

## What has been checked

- **36 CPU tests passed** on synthetic inputs: feathered/empty masks, one and two colored lights, batch handling, degenerate normals, resizing, and invalid inputs.
- One illustrated scene was tested with DSINE normals and a soft person mask, using horizontal/vertical white-light sweeps. At strength 0.3, the recorded full-image linear-luminance drift was at most about 0.016%, with fewer than 0.2% clipped pixels. Higher-strength entries in the journal reached about 0.41% drift and 1.183% clipped pixels; their exact setting-to-entry mapping was not recorded.
- Visual changes were subtle at 0.3 and more visible at stronger settings. This is a limited experiment, not evidence that the tool improves every image or style.

The luminance figures above are node diagnostics on the returned floating-point output, before saving files. They do not measure PNG quantization.

## Limits

- No cast shadows, occlusion calculation, indirect light, material estimation, or automatic light-source detection.
- Existing highlights and shadows remain, and can conflict with the new adjustment.
- Luminance preservation applies to the full image, per batch item, in the selected calculation space, before clipping and provided no negative gain is clamped. It does not preserve each character separately or each RGB channel. Use linear-light mode for linear luminance.
- Strong settings may clip highlights, darken regions excessively, or create unnatural color shifts.
- Near-zero decoded normals are excluded using a length threshold of 0.25 before normalization. This catches degenerate vectors; it does **not** certify the accuracy of remaining normals or detect every interpolation boundary error.
- This node does not make Anima follow a lighting map. Separate exploratory LLLite experiments did not produce usable directional-light control in the tested configuration; that research is on hold.
- Normal-map generators are separate tools with their own dependencies and licenses. Their weights are not redistributed here.

## CPU tests

In a development environment with PyTorch and pytest installed, run from this folder:

```sh
python -m pytest test_aora_soft_light.py -q -p no:cacheprovider
```

Tests use synthetic CPU tensors and do not run image generation.

## License

MIT, copyright 2026 Anthony / AoRA. See [LICENSE](LICENSE). This license covers this repository's code and documentation, not third-party models, weights, or user images.

## Feedback

Useful reports include the node settings, normal-map provider and axis convention, and whether the effect helped or harmed the image. Share images only if you have permission, and remove private workflow metadata first.

Project: Anthony / AoRA. Developed with AI coding assistance, with CPU tests and a limited real-image experiment. This standalone node is separate from AoRA Maker.

---

## Français

**Retouche expérimentale de lumière pour ComfyUI, à partir d'une normal map.** Une ou deux sources directionnelles réglées à la main, masque doux facultatif, contrôle de luminance et diagnostics par image.

Installation : placer les fichiers dans `ComfyUI/custom_nodes/AoRA_SoftLight/`, puis redémarrer ComfyUI. Aucun poids de modèle n'est fourni. PyTorch est normalement déjà présent dans ComfyUI.

Brancher l'image et ses normales, puis éventuellement un masque (blanc = zone traitée). Commencer avec une source blanche, une force de 0,3 et un wrap de 0,75. Vérifier les axes avec les options d'inversion : les conventions dépendent du générateur de normales.

**Ce qui est vérifié :** 36 tests CPU synthétiques réussis et un essai sur une scène illustrée. L'effet reste discret à faible force. Ce résultat ne garantit pas une amélioration sur toutes les scènes.

**Ce que le nœud ne fait pas :** retirer l'ancien éclairage, calculer des ombres portées ou des rebonds, comprendre automatiquement les lampes du décor, ou guider la lumière pendant la génération Anima. Le contrôle LLLite étudié séparément n'a pas donné de résultat exploitable dans les conditions testées ; cette piste est en veille.

La conservation concerne la luminance de l'image complète avant écrêtage, tant qu'aucun gain négatif n'est ramené à zéro. Elle ne s'applique pas séparément à chaque personnage. Le filtre des normales presque nulles ne garantit pas la justesse des autres normales.
