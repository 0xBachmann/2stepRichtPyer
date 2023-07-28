# 2stepRichtPyer

Bachelor Thesis [Structure preserving Lax-Wendroff schemes for low Mach number flows](Structure-preserving-Lax-Wendroff-schemes-for-low-Mach-number-flows) from [Jonas Bachmann](mailto:jonas.bachmann@inf.ethz.ch).

## Setup
Simply execute `pip install -e .` to set up the project.
For storing trajectories instead of keeping them in memory, FFmpeg is required.


## Usage
Example usage of evolving the Euler equations in 2D with the linearized solutions from 1D (`Euler.waves`).

```python
from src.PDE_Types import Euler
from src.plotter import Plotter
from src.richtmyer_two_step_scheme import Richtmyer2step
from src.two_step_richtmyer_util import Dimension
import numpy as np

DIM = Dimension.twoD
PDE = Euler(gamma=5. / 3, dim=DIM)

resolution = np.array([100, 100])

Lx = 1
Ly = 1
stepper = Richtmyer2step(PDE, np.array([Lx, Ly]), resolution)

initial_condition = PDE.waves(0, np.array([1, 1, 1]), amp=1e-3)
stepper.initial_cond(initial_condition)

plotter = Plotter(PDE, action="show", writeout=10, dim=stepper.dim)

stepper.step_for(1., callback=lambda S: plotter.write(stepper.grid_no_ghost))

plotter.finalize()
```

## License

[MIT](https://choosealicense.com/licenses/mit/)

