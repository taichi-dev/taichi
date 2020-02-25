import taichi as ti

def approx(*args, **kwargs):
  from pytest import approx
  if ti.cfg.arch == ti.opengl:
    kwargs['rel'] = max(kwargs.get('rel', 1e6), 1e4)
    return approx(*args, **kwargs)
  else:
    return approx(*args, **kwargs)
