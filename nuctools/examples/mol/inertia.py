import nuctools


LABELS = ('o', 'h', 'h')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))

inertia = nuctools.mol.inertia_tensor(labels=LABELS, coords=COORDS)
print(inertia.round(13))
