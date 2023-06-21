### finite volume
-[x] conservation laws
-[x] flux functions
-[x] numerical fluxes -> why important
  - riemann problem -> maybe a bit more
-[ ] scalar vs vector valued
  - ex linear advection, burgers -> euler
-[ ] 1st ord vs 2nd ord
  - more about godunov theorem

### euler
-[x] conserved vs primitive
  - [ ] show conversion explicit
-[x] optional passive scalar
-[x] diagonalization -> "analytical" solutions

### my scheme
-[ ] layout
-[ ] implicit
  - lsa
  - howto solve in code
    - subspace iter to not need jacobian
-[ ] show benefits
  - conservation of rotation ...
  - low mach numbers
-[ ] show caviats
  - oscillations ...
-[ ] rusanov
  - show where s comes from
-[ ] lerp
-[ ] visc ?

### results

# Tests
-[ ] diagonlize then "analytic solution"
  - explained in euler, now show
-[ ] other init conditions for tests