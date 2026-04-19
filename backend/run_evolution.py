""" 
Run the GA for EACH tile

genome: [fractal_id, crop_cx, crop_cy, crop_scale, brightness_shift]

  - fractal_id:         discrete (for crossover / mutation)
  - crop_cx:            continuous (0-1)
  - crop_cy:            continuous (0-1)
  - crop_scale:         continuous (min - max))
  - brightness_shift:   continuous (-0.4 - 0.4)
"""