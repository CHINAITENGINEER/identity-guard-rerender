# Realism research decisions

This compositor uses a pixel-height map for each product. Every opaque pixel
is associated with the lowest opaque pixel in its own image column (its
footpoint); the hard-shadow displacement is proportional to that pixel height.
Consequently a contact pixel has zero displacement and a high pixel casts the
longest ray. The support polygon clips the result to the actual desk.

The deterministic renderer uses two shadow scales:

- tight footprint ambient occlusion, derived from each product's real alpha;
- a directional pixel-height hard shadow, softened only after projection and
  clipped to the support plane and protected regions.

Research informs optional modules but does not relax the pixel contract:

- Depth Anything V2 may propose depth/support-plane evidence.  It must not
  replace the background; a human/geometry check selects the plane.
- Single-view relighting may produce a low-frequency illumination proxy only.
  Its geometry, alpha, text, pattern and high-frequency pixels are discarded.
- Shadow diffusion may be used only as an ROI residual for a pre-approved
  contact/shadow mask.  The original product RGBA and background outside that
  mask are restored afterward.

No generative result is a deliverable image by itself.  The final renderer
always re-composites original RGBA, hard-restores non-authorized background
pixels, and runs the bit-exact background audit.
